"""Extract I3D features from a quadrant of side-by-side videos for FVD.

Reads video_*_side_by_side.mp4 files (1280x704), crops to one of four quadrants
(GT/Gen x Bot1/Bot2), extracts I3D features (400-dim logits), and saves
features + mu/sigma for FVD computation.

Decoding runs in worker processes (DataLoader) overlapped with GPU inference.

Uses the standard TF-Hub I3D (InceptionV1, Kinetics-400) ported to TorchScript
by the StyleGAN-V authors — the canonical model for FVD.

Quadrant layout (1280x704 -> four 640x352 crops):
  0 = GT  Bot1  (top-left)      1 = Gen Bot1 (top-right)
  2 = GT  Bot2  (bottom-left)   3 = Gen Bot2 (bottom-right)

Sample modes (--sample_mode):
  all            : feed all frames (T = video length).
  uniform        : np.linspace-strided downsample to --num_frames.
  random_window  : contiguous random window of length --num_frames, with start
                   offset deterministic per-video (seeded by --seed). The same
                   --seed across runs/models produces the same offsets, so
                   different models are evaluated on the same chunks.
"""

import argparse
import pathlib
import pprint
import re
import shutil
import time

import av
import huggingface_hub
import numpy as np
import torch
import tqdm

# (name, row_slice, col_slice) — each quadrant is 640x352
QUADRANTS: dict[int, tuple[str, slice, slice]] = {
    0: ("gt_bot1",  slice(0, 352), slice(0, 640)),
    1: ("gen_bot1", slice(0, 352), slice(640, 1280)),
    2: ("gt_bot2",  slice(352, 704), slice(0, 640)),
    3: ("gen_bot2", slice(352, 704), slice(640, 1280)),
}


def discover_videos(data_dir: pathlib.Path) -> list[pathlib.Path]:
    """Glob video_*_side_by_side.mp4 and sort by numeric index."""
    videos = list(data_dir.glob("video_*_side_by_side.mp4"))
    assert len(videos) > 0, f"No video_*_side_by_side.mp4 found in {data_dir}"
    videos.sort(key=lambda p: int(re.search(r"video_(\d+)_", p.name).group(1)))
    return videos


def decode_and_crop(
    video_path: pathlib.Path,
    row_slice: slice,
    col_slice: slice,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """Decode + crop frames. If `indices` is given, only convert/keep those
    frames (in order) and stop decoding past max(indices) — saving the RGB
    conversion for non-target frames and the decode work past the window.

    Returns [T_out, 352, 640, 3] uint8.
    """
    container = av.open(str(video_path))
    container.streams.video[0].thread_type = "AUTO"

    if indices is None:
        out = [
            frame.to_ndarray(format="rgb24")[row_slice, col_slice]
            for frame in container.decode(video=0)
        ]
        container.close()
        return np.stack(out)

    indices_int = [int(i) for i in indices]
    last_needed = max(indices_int)
    needed_to_pos: dict[int, list[int]] = {}
    for pos, i in enumerate(indices_int):
        needed_to_pos.setdefault(i, []).append(pos)
    out_arr: list[np.ndarray | None] = [None] * len(indices_int)

    for i, frame in enumerate(container.decode(video=0)):
        if i in needed_to_pos:
            cropped = frame.to_ndarray(format="rgb24")[row_slice, col_slice]
            for pos in needed_to_pos[i]:
                out_arr[pos] = cropped
        if i >= last_needed:
            break
    container.close()
    return np.stack(out_arr)


def select_indices(
    num_video_frames: int,
    num_frames: int,
    sample_mode: str,
    rng_offset: int,
) -> np.ndarray | None:
    """Compute frame indices for a given video, or None to keep all frames.

    Args:
        num_video_frames: T of the source clip.
        num_frames: target T (ignored if sample_mode == "all").
        sample_mode: "all" | "uniform" | "random_window".
        rng_offset: precomputed start frame for "random_window".
    """
    if sample_mode == "all":
        return None
    assert num_frames > 0, "num_frames must be >0 for non-'all' sample_mode"
    if sample_mode == "uniform":
        return np.linspace(0, num_video_frames - 1, num=num_frames, dtype=int)
    if sample_mode == "random_window":
        max_start = num_video_frames - num_frames
        assert max_start >= 0, (
            f"video has {num_video_frames} frames < num_frames={num_frames}"
        )
        start = rng_offset % (max_start + 1)
        return np.arange(start, start + num_frames)
    raise ValueError(f"unknown sample_mode: {sample_mode}")


class QuadrantDataset(torch.utils.data.Dataset):
    """Per-video decode + crop + temporal sample + resize + normalize."""

    def __init__(
        self,
        videos: list[pathlib.Path],
        row_slice: slice,
        col_slice: slice,
        num_frames: int,
        sample_mode: str,
        resolution: int,
        offsets: np.ndarray,
        video_length: int,
    ) -> None:
        self.videos = videos
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.num_frames = num_frames
        self.sample_mode = sample_mode
        self.resolution = resolution
        self.offsets = offsets
        self.video_length = video_length

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.sample_mode == "all":
            indices = None
        else:
            indices = select_indices(
                num_video_frames=self.video_length,
                num_frames=self.num_frames,
                sample_mode=self.sample_mode,
                rng_offset=int(self.offsets[idx]),
            )

        frames = decode_and_crop(
            video_path=self.videos[idx],
            row_slice=self.row_slice,
            col_slice=self.col_slice,
            indices=indices,
        )

        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        tensor = torch.nn.functional.interpolate(
            tensor, size=(self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        tensor = (2.0 * tensor - 1.0).permute(1, 0, 2, 3)  # [3, T, H, W]
        return tensor


def precompute_offsets(num_videos: int, seed: int) -> np.ndarray:
    """Deterministic per-video random offsets shared across runs / models."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=2**31 - 1, size=num_videos, dtype=np.int64)


def load_i3d_model(
    weights_path: pathlib.Path,
    device: torch.device,
) -> torch.jit.ScriptModule:
    """Load I3D TorchScript model, downloading from HuggingFace if absent."""
    if not weights_path.exists():
        print(f"Downloading I3D weights to {weights_path} ...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        cached = huggingface_hub.hf_hub_download(
            repo_id="flateon/FVD-I3D-torchscript",
            filename="i3d_torchscript.pt",
        )
        shutil.copy2(src=cached, dst=weights_path)
        print(f"Downloaded ({weights_path.stat().st_size / 1e6:.1f} MB)")
    return torch.jit.load(str(weights_path), map_location=device).eval()


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(features, axis=0).astype(np.float64)
    sigma = np.cov(features, rowvar=False).astype(np.float64)
    return mu, sigma


def main() -> None:
    """Extract I3D features from a quadrant of side-by-side videos."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with video_*_side_by_side.mp4 files")
    parser.add_argument("--quadrant", type=int, choices=[0, 1, 2, 3], required=True,
                        help="0=GT Bot1 (TL), 1=Gen Bot1 (TR), "
                             "2=GT Bot2 (BL), 3=Gen Bot2 (BR)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--i3d_weights", type=str, default="models/i3d_torchscript.pt",
                        help="Path to I3D TorchScript weights (auto-downloads if missing)")
    parser.add_argument("--sample_mode", type=str, default="all",
                        choices=["all", "uniform", "random_window"],
                        help="Temporal sampling strategy")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Frames per video (required for uniform/random_window)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random_window offsets (deterministic across models)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Spatial resolution for I3D input")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for I3D inference")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for video decode")
    args = parser.parse_args()

    quad_name, row_slice, col_slice = QUADRANTS[args.quadrant]

    print("=" * 60)
    print("FVD Feature Extraction (side-by-side quadrant)")
    print("=" * 60)
    pprint.pprint(vars(args))
    print(f"Quadrant {args.quadrant}: {quad_name} "
          f"(rows {row_slice.start}:{row_slice.stop}, "
          f"cols {col_slice.start}:{col_slice.stop})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = pathlib.Path(args.data_dir)
    output_path = pathlib.Path(args.output)

    videos = discover_videos(data_dir=data_dir)
    print(f"\n{len(videos)} videos discovered")

    with av.open(str(videos[0])) as c:
        s = c.streams.video[0]
        video_length = s.frames
        print(f"Video: {s.width}x{s.height}, codec={s.codec_context.name}, "
              f"frames={video_length}")
    print(f"Crop: {col_slice.stop - col_slice.start}x{row_slice.stop - row_slice.start}")

    # I3D model
    print(f"\nLoading I3D from {args.i3d_weights} ...")
    t0 = time.time()
    model = load_i3d_model(
        weights_path=pathlib.Path(args.i3d_weights),
        device=device,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    # Per-video offsets (deterministic given --seed and N)
    offsets = precompute_offsets(num_videos=len(videos), seed=args.seed)

    # DataLoader
    dataset = QuadrantDataset(
        videos=videos,
        row_slice=row_slice,
        col_slice=col_slice,
        num_frames=args.num_frames,
        sample_mode=args.sample_mode,
        resolution=args.resolution,
        offsets=offsets,
        video_length=video_length,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        shuffle=False,
        drop_last=False,
    )

    frames_desc = "all" if args.sample_mode == "all" else f"{args.num_frames} ({args.sample_mode})"
    print(f"\nExtracting {quad_name} features, {frames_desc} frames/video, "
          f"batch_size={args.batch_size}, workers={args.num_workers} ...")

    all_features: np.ndarray | None = None
    t_start = time.time()
    cursor = 0

    pbar = tqdm.tqdm(total=len(videos), desc="Videos")
    for batch in loader:
        batch = batch.to(device=device, non_blocking=True)
        with torch.inference_mode():
            feats = model(
                batch, rescale=False, resize=False, return_features=True,
            )  # [B, 400]
        feats_np = feats.float().cpu().numpy()
        del batch

        if all_features is None:
            all_features = np.zeros((len(videos), feats_np.shape[1]), dtype=np.float32)
            print(f"\nFeature dim: {feats_np.shape[1]}")

        all_features[cursor : cursor + feats_np.shape[0]] = feats_np
        cursor += feats_np.shape[0]
        pbar.update(feats_np.shape[0])
    pbar.close()

    elapsed = time.time() - t_start
    assert all_features is not None and cursor == len(videos)
    print(f"\n{len(videos)} videos in {elapsed:.1f}s ({len(videos) / elapsed:.1f} vid/s)")
    print(f"Features shape: {all_features.shape}")

    mu, sigma = compute_statistics(features=all_features)
    print(f"mu: shape={mu.shape}, ||mu||={np.linalg.norm(mu):.4f}")
    print(f"sigma: shape={sigma.shape}, trace={np.trace(sigma):.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=all_features,
        mu=mu,
        sigma=sigma,
        sample_mode=np.array(args.sample_mode),
        num_frames=np.int64(args.num_frames),
        seed=np.int64(args.seed),
    )
    print(f"\nSaved -> {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
