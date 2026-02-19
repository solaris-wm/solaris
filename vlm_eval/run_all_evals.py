#!/usr/bin/env python3
"""Run evaluations for all models and datasets.

Note: per-(model, dataset) JSON outputs are written by `run_eval.py`, which now
clears any existing `trial_*.json` and `stats.json` in the resolved output directory
before starting a new set of trials (i.e., reruns overwrite cleanly).
"""

import argparse
import os
import subprocess
import concurrent.futures
import time
from pathlib import Path

# Base directories
GENERATIONS_DIR = Path("./../output")
DATASET_BASE = Path("./../datasets/eval/")

EVAL_TYPE_MAPPING = {
    "translation": "translationEval",
    "rotation": "rotationEval",
    "structure": "structureEval",
    "turn_to_look": "turnToLookEval",
    "turn_to_look_opposite": "turnToLookOppositeEval",
    "one_looks_away": "oneLooksAwayEval",
    "both_look_away": "bothLookAwayEval",
}

# Which eval types to actually run (comment out to skip)
ENABLED_EVAL_TYPES = [
    "translation",
    "rotation",
    "structure",
    "turn_to_look",
    "turn_to_look_opposite",
    "one_looks_away",
    "both_look_away",
]


def _normalize_eval_types_arg(values: list[str] | None) -> list[str] | None:
    """Normalize --eval-types input.

    Supports:
    - space-separated values: --eval-types translation rotation
    - comma-separated values: --eval-types translation,rotation
    - mixed: --eval-types translation,rotation structure
    - special: --eval-types all
    """
    if not values:
        return None

    normalized: list[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        normalized.extend(parts)

    if not normalized:
        return None

    if any(v.lower() == "all" for v in normalized):
        return sorted(EVAL_TYPE_MAPPING.keys())

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in normalized:
        if v not in seen:
            out.append(v)
            seen.add(v)

    return out


def _normalize_models_arg(values: list[str] | None) -> list[str] | None:
    """Normalize --models input.

    Supports:
    - space-separated values: --models MODEL_A MODEL_B
    - comma-separated values: --models MODEL_A,MODEL_B
    - mixed: --models MODEL_A,MODEL_B MODEL_C
    - special: --models all
    """
    if not values:
        return None

    normalized: list[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        normalized.extend(parts)

    if not normalized:
        return None

    if any(v.lower() == "all" for v in normalized):
        return None  # None means "no filtering"

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in normalized:
        if v not in seen:
            out.append(v)
            seen.add(v)

    return out


def extract_eval_type(folder_name: str) -> str | None:
    _, _, suffix = folder_name.partition("eval_")
    if not suffix:
        return None


    # Match against base keys only
    base_keys = [k for k in EVAL_TYPE_MAPPING]
    for key in sorted(base_keys, key=len, reverse=True):
        if suffix == key or suffix.startswith(f"{key}_"):
            return key

    return None


def _run_one_model(
    model_dir: Path,
    enabled_eval_types: list[str],
    dataset_base: Path,
    dry_run: bool,
    limit: int | None,
    num_trials: int,
    results_dir: Path | None,
) -> tuple[str, str, bool]:
    model_name = model_dir.name
    out_lines: list[str] = []
    any_failed = False

    # --- Copy-paste of the original per-model main-loop logic (buffered) ---
    out_lines.append(f"{'=' * 80}")
    out_lines.append(f"Model: {model_name}")
    out_lines.append(f"{'=' * 80}")

    # Check which evaluation types this model has generated videos for
    eval_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    available_eval_types: dict[str, Path] = {}

    for eval_subdir in eval_subdirs:
        eval_type = extract_eval_type(eval_subdir.name)
        if eval_type:
            available_eval_types[eval_type] = eval_subdir

    out_lines.append(f"Available eval types: {list(available_eval_types.keys())}")

    # Run evaluation for each enabled eval type
    for eval_type in enabled_eval_types:
        dataset_name = EVAL_TYPE_MAPPING[eval_type]
        dataset_path = dataset_base / dataset_name

        if not dataset_path.exists():
            out_lines.append(f"⊘ Skipping {eval_type} - dataset not found: {dataset_path}")
            continue

        if eval_type not in available_eval_types:
            out_lines.append(f"⊘ Skipping {eval_type} - no generated videos for this model")
            continue

        # Construct the command
        cmd = [
            "python",
            "run_eval.py",
            str(dataset_path),
            "--generated",
            str(model_dir),
        ]
        if results_dir is not None:
            cmd.extend(["--results-dir", str(results_dir)])
        if limit:
            cmd.extend(["--limit", str(limit)])
        if num_trials != 1:
            cmd.extend(["--num-trials", str(num_trials)])

        out_lines.append(f"\n{'[DRY RUN] Would run' if dry_run else 'Running'}: {' '.join(cmd)}")

        if dry_run:
            out_lines.append(f"  → Dataset: {dataset_path}")
            out_lines.append(f"  → Generated subdir: {available_eval_types[eval_type].name}")
            continue

        # Run the command
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            out_lines.append(f"✓ Success for {eval_type}")
            if result.stdout:
                # Print just the summary section
                lines = result.stdout.split("\n")
                in_summary = False
                for line in lines:
                    if "EVALUATION SUMMARY" in line:
                        in_summary = True
                    if in_summary:
                        out_lines.append(line)
        except subprocess.CalledProcessError as e:
            any_failed = True
            out_lines.append(f"✗ Error for {eval_type}")
            out_lines.append(f"Return code: {e.returncode}")
            if e.stdout:
                out_lines.append("STDOUT: " + e.stdout[-1000:])  # Last 1000 chars
            if e.stderr:
                out_lines.append("STDERR: " + e.stderr[-1000:])

    out_lines.append("")
    # --- end original logic ---

    return model_name, "\n".join(out_lines), any_failed


def main():
    parser = argparse.ArgumentParser(description="Run evaluations for all models and datasets")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be run without executing")
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help=(
            "Extract frames from eval datasets for visual inspection. "
            "If --generations-dir contains model folders, also extract generated frames and write "
            "side-by-side GT+generated comparisons; otherwise extract ground-truth only."
        ),
    )
    parser.add_argument(
        "--generations-dir",
        type=Path,
        default=GENERATIONS_DIR,
        help="Path to generations/ directory containing model folders",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DATASET_BASE,
        help="Path to base dataset directory containing eval datasets (e.g., translationEval, rotationEval, ...)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for auto-organized JSON outputs (forwarded to run_eval.py --results-dir). "
            "If omitted, run_eval.py defaults to ./results_json relative to the working directory."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of episodes/queries to process (passed to run_eval.py)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation trials to run per (model, dataset) pair (passed to run_eval.py).",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Max number of models to evaluate in parallel. "
            "Defaults to min(num_models, CPU count). Set to 1 for sequential behavior."
        ),
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        help=(
            "Override which eval types to run. "
            "Examples: --eval-types translation rotation | --eval-types translation,rotation | --eval-types all. "
            f"Valid keys: {', '.join(sorted(EVAL_TYPE_MAPPING.keys()))}"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Only run these model folder name(s) under --generations-dir. "
            "Examples: --models MODEL_A MODEL_B | --models MODEL_A,MODEL_B | --models all."
        ),
    )
    args = parser.parse_args()
    if args.num_trials < 1:
        raise SystemExit("--num-trials must be >= 1")

    generations_dir: Path = args.generations_dir
    dataset_base: Path = args.dataset_base

    enabled_eval_types = _normalize_eval_types_arg(args.eval_types) or ENABLED_EVAL_TYPES
    unknown = [t for t in enabled_eval_types if t not in EVAL_TYPE_MAPPING]
    if unknown:
        valid = ", ".join(sorted(EVAL_TYPE_MAPPING.keys()))
        raise SystemExit(f"Unknown eval type(s): {unknown}. Valid keys: {valid}")

    selected_models = _normalize_models_arg(args.models)

    print(f"Enabled eval types: {enabled_eval_types}")
    print(f"Dataset base: {dataset_base}")
    if selected_models is not None:
        print(f"Model filter: {selected_models}")
    if args.dry_run:
        print("DRY RUN - commands will not be executed")
    if args.extract_frames:
        print("EXTRACT FRAMES MODE - extracting from ground-truth videos (no --generated)")
    print()

    # --extract-frames mode: extract frames without VLM queries
    if args.extract_frames:
        # Check if generations_dir exists - if so, create side-by-side comparisons
        if generations_dir.exists():
            # Get all model directories
            model_dirs = [d for d in generations_dir.iterdir() if d.is_dir()]
            if selected_models is not None:
                available = {d.name for d in model_dirs}
                missing = [m for m in selected_models if m not in available]
                if missing:
                    raise SystemExit(
                        f"Unknown model(s) under {generations_dir}: {missing}. "
                        f"Available: {sorted(available)}"
                    )
                wanted = set(selected_models)
                model_dirs = [d for d in model_dirs if d.name in wanted]
            if model_dirs:
                print(f"{'=' * 80}")
                print("Extracting frames with side-by-side comparisons (GT + generated)")
                print(f"Found {len(model_dirs)} model(s)")
                print(f"{'=' * 80}")

                for model_dir in sorted(model_dirs):
                    model_name = model_dir.name
                    print(f"\n{'=' * 80}")
                    print(f"Model: {model_name}")
                    print(f"{'=' * 80}")

                    # Check which evaluation types this model has generated videos for
                    eval_subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
                    available_eval_types: dict[str, Path] = {}

                    for eval_subdir in eval_subdirs:
                        eval_type = extract_eval_type(eval_subdir.name)
                        if eval_type:
                            available_eval_types[eval_type] = eval_subdir

                    print(f"Available eval types: {list(available_eval_types.keys())}")

                    for eval_type in enabled_eval_types:
                        dataset_name = EVAL_TYPE_MAPPING[eval_type]
                        dataset_path = dataset_base / dataset_name

                        if not dataset_path.exists():
                            print(f"⊘ Skipping {eval_type} - dataset not found: {dataset_path}")
                            continue

                        if eval_type not in available_eval_types:
                            print(f"⊘ Skipping {eval_type} - no generated videos for this model")
                            continue

                        # Construct the command with both --extract-frames and --generated
                        cmd = [
                            "python",
                            "run_eval.py",
                            str(dataset_path),
                            "--extract-frames",
                            "--generated",
                            str(model_dir),
                        ]
                        if args.results_dir is not None:
                            cmd.extend(["--results-dir", str(args.results_dir)])
                        if args.limit:
                            cmd.extend(["--limit", str(args.limit)])

                        print(f"\n{'[DRY RUN] Would run' if args.dry_run else 'Running'}: {' '.join(cmd)}")

                        if args.dry_run:
                            print(f"  → Dataset: {dataset_path}")
                            print(f"  → Generated subdir: {available_eval_types[eval_type].name}")
                            continue

                        # Run the command
                        try:
                            result = subprocess.run(
                                cmd,
                                check=True,
                                capture_output=False,  # Show output in real-time
                                text=True
                            )
                            print(f"✓ Success for {eval_type}")
                        except subprocess.CalledProcessError as e:
                            print(f"✗ Error for {eval_type}")
                            print(f"Return code: {e.returncode}")

                return 0

        # No generations_dir or it doesn't exist - extract GT frames only
        print(f"{'=' * 80}")
        print("Extracting frames from ground-truth videos only (no --generations-dir found)")
        print(f"{'=' * 80}")

        for eval_type in enabled_eval_types:
            dataset_name = EVAL_TYPE_MAPPING[eval_type]
            dataset_path = dataset_base / dataset_name

            if not dataset_path.exists():
                print(f"⊘ Skipping {eval_type} - dataset not found: {dataset_path}")
                continue

            # Construct the command (no --generated, add --extract-frames)
            cmd = [
                "python",
                "run_eval.py",
                str(dataset_path),
                "--extract-frames",
            ]
            if args.results_dir is not None:
                cmd.extend(["--results-dir", str(args.results_dir)])
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])

            print(f"\n{'[DRY RUN] Would run' if args.dry_run else 'Running'}: {' '.join(cmd)}")

            if args.dry_run:
                print(f"  → Dataset: {dataset_path}")
                continue

            # Run the command
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=False,  # Show output in real-time
                    text=True
                )
                print(f"✓ Success for {eval_type}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error for {eval_type}")
                print(f"Return code: {e.returncode}")

        return 0

    # Normal mode: evaluate generated videos
    if not generations_dir.exists():
        print(f"Error: Generations directory not found: {generations_dir}")
        return 1

    # Get all model directories
    model_dirs = [d for d in generations_dir.iterdir() if d.is_dir()]
    if selected_models is not None:
        available = {d.name for d in model_dirs}
        missing = [m for m in selected_models if m not in available]
        if missing:
            raise SystemExit(
                f"Unknown model(s) under {generations_dir}: {missing}. "
                f"Available: {sorted(available)}"
            )
        wanted = set(selected_models)
        model_dirs = [d for d in model_dirs if d.name in wanted]

    if not model_dirs:
        print(f"No model directories found in {generations_dir}")
        return 1

    print(f"Found {len(model_dirs)} model(s) to evaluate")
    print(f"Generations dir: {generations_dir}")
    if args.max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(len(model_dirs), cpu)
    else:
        max_workers = args.max_workers
    if max_workers < 1:
        raise SystemExit("--max-workers must be >= 1")
    print(f"Parallelism: {max_workers} worker(s)")

    started = time.time()
    sorted_models = sorted(model_dirs)

    any_failed = False
    completed = 0

    if max_workers == 1:
        for model_dir in sorted_models:
            model_name, text, failed = _run_one_model(
                model_dir=model_dir,
                enabled_eval_types=enabled_eval_types,
                dataset_base=dataset_base,
                dry_run=args.dry_run,
                limit=args.limit,
                num_trials=args.num_trials,
                results_dir=args.results_dir,
            )
            completed += 1
            print(text)
            any_failed = any_failed or failed
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_model = {
                ex.submit(
                    _run_one_model,
                    model_dir,
                    enabled_eval_types,
                    dataset_base,
                    args.dry_run,
                    args.limit,
                    args.num_trials,
                    args.results_dir,
                ): model_dir
                for model_dir in sorted_models
            }

            for fut in concurrent.futures.as_completed(fut_to_model):
                model_name, text, failed = fut.result()
                completed += 1
                print(text)
                any_failed = any_failed or failed

    elapsed = time.time() - started
    print(f"{'=' * 80}")
    print(f"Done in {elapsed:.1f}s. Failures: {'yes' if any_failed else 'no'}")
    print(f"{'=' * 80}")

    # Preserve historical behavior: keep exit code 0 even if some evals fail.
    # (If you prefer non-zero on any failure, we can add a flag and flip this.)
    return 0


if __name__ == "__main__":
    exit(main())
