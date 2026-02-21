#!/usr/bin/env python3
import argparse
import os
import tarfile
from pathlib import Path


def safe_extract(tar: tarfile.TarFile, path: Path):
    base = str(path.resolve())

    for m in tar.getmembers():
        # Skip links/devices for safety
        if m.islnk() or m.issym() or m.isdev():
            continue

        target = (path / m.name).resolve()
        if os.path.commonpath([base, str(target)]) != base:
            raise RuntimeError(f"Blocked path traversal in tar member: {m.name}")

    tar.extractall(path)


def main():
    ap = argparse.ArgumentParser(
        description="Extract tar shards produced by shard_datasets.py into a dataset folder."
    )
    ap.add_argument(
        "--shards",
        type=Path,
        required=True,
        help="Local folder containing .tar shards (output of shard_datasets.py)",
    )
    ap.add_argument("--out", type=Path, required=True, help="Where to reconstruct the dataset folder")
    ap.add_argument(
        "--pattern",
        default="*.tar",
        help="Glob pattern for shard files (default: *.tar)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    shards_dir = args.shards.resolve()
    if not shards_dir.is_dir():
        raise SystemExit(f"Shards folder does not exist: {shards_dir}")

    shard_paths = sorted(shards_dir.glob(args.pattern))
    if not shard_paths:
        raise SystemExit(f"No shards matched pattern: {args.pattern} in {shards_dir}")

    for sp in shard_paths:
        print(f"Extracting {sp} -> {args.out}")
        with tarfile.open(sp, "r") as tf:
            safe_extract(tf, args.out)

    print("Done.")

if __name__ == "__main__":
    main()
