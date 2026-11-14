#!/usr/bin/env python3
"""Generate deterministic eval_ids JSON files for eval dataloaders.

This script is intended for the Solaris repo layout and writes files in
`src/data/eval_ids/eval_ids_<dataset_name>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.batch_sampler import BatchSampler, BatchSamplerMultiplayer
from src.data.dataset import Dataset, DatasetMultiplayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare eval_ids JSON for a dataset."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name used in configs (e.g. duet).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=257,
        help="Number of frames per eval segment.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024,
        help="Number of eval segments to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0, 16],
        help="Sampler seed list, e.g. --seed 0 16.",
    )
    parser.add_argument(
        "--multiplayer",
        action="store_true",
        help="Generate multiplayer segment IDs [ep,b0s,b0e,...,bNs,bNe].",
    )
    parser.add_argument(
        "--bot-names",
        nargs="+",
        default=["Alpha", "Bravo"],
        help="Bot names for multiplayer dataset discovery (space-separated, e.g. Alpha Bravo).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: "
            "src/data/eval_ids/eval_ids_<dataset-name>.json"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["auto", "test", "train", "none"],
        default="auto",
        help=(
            "Dataset split folder selection when --data-dir points to a root with "
            "train/test subfolders. 'auto' prefers test, then train."
        ),
    )
    return parser.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if args.split == "none":
        return data_dir
    if args.split in ("test", "train"):
        split_dir = data_dir / args.split
        if split_dir.exists():
            return split_dir
        raise FileNotFoundError(
            f"Requested split folder does not exist: {split_dir}"
        )

    # auto
    for split_name in ("test", "train"):
        split_dir = data_dir / split_name
        if split_dir.exists():
            return split_dir
    return data_dir


def build_dataset(args: argparse.Namespace):
    resolved_data_dir = resolve_data_dir(args)
    logging.info("Using data directory: %s", resolved_data_dir)
    if args.multiplayer:
        ds = DatasetMultiplayer(
            data_dir=str(resolved_data_dir),
            dataset_name=args.dataset_name,
            bot_names=args.bot_names,
            converters=[],
            obs_resize=None,
            shuffle_bots=False,
        )
        if ds.num_episodes == 0:
            raise ValueError(
                "Detected zero multiplayer episodes. Check --data-dir/--split and "
                "--bot-names against file naming."
            )
        return ds
    ds = Dataset(
        data_dir=str(resolved_data_dir),
        dataset_name=args.dataset_name,
        converters=[],
        obs_resize=None,
    )
    if ds.num_episodes == 0:
        raise ValueError(
            "Detected zero episodes. Check --data-dir/--split and dataset layout."
        )
    return ds


def build_sampler(args: argparse.Namespace, ds):
    if args.multiplayer:
        return BatchSamplerMultiplayer(
            ds,
            rank=0,
            num_replicas=1,
            batch_size=1,
            num_frames=args.num_frames,
            seed=args.seed,
        )
    return BatchSampler(
        ds,
        rank=0,
        num_replicas=1,
        batch_size=1,
        num_frames=args.num_frames,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    sampler = build_sampler(args, dataset)

    ids = []
    while len(ids) < args.num_samples:
        segment_ids = sampler.sample()
        ids.extend([list(map(int, seg.to_list())) for seg in segment_ids])
    ids = ids[: args.num_samples]

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("src/data/eval_ids") / f"eval_ids_{args.dataset_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)

    print(
        f"Wrote {len(ids)} eval IDs to {output_path} "
        f"(multiplayer={args.multiplayer}, num_frames={args.num_frames})"
    )


if __name__ == "__main__":
    main()
