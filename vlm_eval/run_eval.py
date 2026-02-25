#!/usr/bin/env python3
"""
Unified evaluation script for all mc_multiplayer datasets.

Usage:
  python run_eval.py <folder> [options]

The folder name determines which handler to use:
  - mc_multiplayer_eval_dev     -> MinecraftMultiplayerHandler (movement)
  - mc_multiplayer_eval_dev_2   -> MinecraftCameraRotationHandler (camera rotation)
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional
import json
import statistics

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_utils import (
    VideoPair,
    KeyframeQuery,
    extract_frame,
    query_vlm,
    save_results,
    EvalResult,
    extract_quadrant,
    find_generated_video_subdir,
    extract_frame_from_generated,
    VLM_MODEL_NAME
)
from visualization_helper import (
    get_side_by_side_output_dir,
    create_side_by_side_comparison,
)


def get_frame_output_dir(dataset_name: str, model_name: str, query_type: Optional[str]) -> Path:
    """
    Get the output directory for extracted frames.
    
    Structure: frame_extraction/TASK/[_real | MODEL_VARIANT]/[QUERY_TYPE | default]/
    
    Args:
        dataset_name: Name of the dataset (e.g., "rotationEval")
        model_name: Model name, or "ground_truth" for real videos
        query_type: Query type from metadata, or None for default
        
    Returns:
        Path to the output directory
    """
    variant = "_real" if model_name == "ground_truth" else model_name
    query_folder = query_type if query_type and query_type != "default" else "default"
    return Path("frame_extraction") / dataset_name / variant / query_folder


def extract_query_frames(
    query: KeyframeQuery,
    generated_subdir: Optional[Path],
    current_video_id: int,
    frame1_idx: int,
) -> dict:
    """
    Extract frames for a query and return frame bytes with filename suffixes.
    
    This is the single source of truth for frame extraction, ensuring that
    the exact same frames are saved to disk and sent to the VLM.
    
    Args:
        query: The KeyframeQuery object
        generated_subdir: Path to generated video subdirectory, or None for GT videos
        current_video_id: Current video ID for generated videos
        frame1_idx: Reference frame index for generated video offset calculation
        
    Returns:
        Dict mapping suffix to frame bytes: {"frame": bytes} for single-frame,
        {"frame1": bytes, "frame2": bytes} for two-frame,
        {"alpha_frame": bytes, "bravo_frame": bytes} for turn_to_look
    """
    meta = query.metadata
    is_turn_to_look = meta.get('is_turn_to_look', False)
    
    frames = {}
    
    if is_turn_to_look:
        # Turn to look: extract from both alpha and bravo perspectives
        alpha_video = Path(meta['alpha_video'])
        bravo_video = Path(meta['bravo_video'])
        alpha_frame_idx = meta['alpha_frame']
        bravo_frame_idx = meta['bravo_frame']
        
        if generated_subdir:
            generated_video = generated_subdir / f"video_{current_video_id}_side_by_side.mp4"
            if not generated_video.exists():
                raise FileNotFoundError(f"Generated video not found: {generated_video.name}")
            frames["alpha_frame"] = extract_frame_from_generated(generated_video, alpha_frame_idx, frame1_idx, "alpha")
            frames["bravo_frame"] = extract_frame_from_generated(generated_video, bravo_frame_idx, frame1_idx, "bravo")
        else:
            frames["alpha_frame"] = extract_frame(alpha_video, alpha_frame_idx)
            frames["bravo_frame"] = extract_frame(bravo_video, bravo_frame_idx)
    
    elif query.second_frame_index is not None:
        # Two-frame comparison query (e.g., translation): needs both frames
        frame2_idx = query.second_frame_index
        variant = meta['variant']
        
        if generated_subdir:
            generated_video = generated_subdir / f"video_{current_video_id}_side_by_side.mp4"
            if not generated_video.exists():
                raise FileNotFoundError(f"Generated video not found: {generated_video.name}")
            # Use frame1_idx + 1 as first frame since generated video starts there
            frames["frame1"] = extract_frame_from_generated(generated_video, frame1_idx + 1, frame1_idx, variant)
            frames["frame2"] = extract_frame_from_generated(generated_video, frame2_idx, frame1_idx, variant)
        else:
            frames["frame1"] = extract_frame(query.video_path, query.frame_index)
            frames["frame2"] = extract_frame(query.video_path, frame2_idx)
    
    else:
        # Single-frame handlers: use query.frame_index
        variant = meta['variant']
        
        if generated_subdir:
            generated_video = generated_subdir / f"video_{current_video_id}_side_by_side.mp4"
            if not generated_video.exists():
                raise FileNotFoundError(f"Generated video not found: {generated_video.name}")
            frames["frame"] = extract_frame_from_generated(generated_video, query.frame_index, frame1_idx, variant)
        else:
            frames["frame"] = extract_frame(query.video_path, query.frame_index)
    
    return frames


def find_mc_video_pairs(folder: Path) -> List[VideoPair]:
    """
    Find video pairs in mc_multiplayer format (with _camera.mp4 suffix).

    Matches: {episode}_{Alpha|Bravo}_instance_{instance}_camera.mp4
    """
    pattern = re.compile(r'(\d+)_(Alpha|Bravo)_instance_(\d+)_camera\.mp4')

    files = {}
    for file in folder.iterdir():
        match = pattern.match(file.name)
        if match:
            episode_num, variant, instance_num = match.groups()
            key = (episode_num, instance_num)

            if key not in files:
                files[key] = {}

            if variant == "Alpha":
                files[key]["alpha_video"] = file
                json_file = folder / f"{episode_num}_Alpha_instance_{instance_num}.json"
                if json_file.exists():
                    files[key]["alpha_json"] = json_file
            else:
                files[key]["bravo_video"] = file
                json_file = folder / f"{episode_num}_Bravo_instance_{instance_num}.json"
                if json_file.exists():
                    files[key]["bravo_json"] = json_file

    pairs = []
    for (episode_num, instance_num), file_dict in files.items():
        required_keys = ["alpha_video", "bravo_video", "alpha_json", "bravo_json"]
        if all(key in file_dict for key in required_keys):
            pairs.append(VideoPair(
                episode_num=episode_num,
                instance_num=instance_num,
                alpha_video=file_dict["alpha_video"],
                bravo_video=file_dict["bravo_video"],
                alpha_json=file_dict["alpha_json"],
                bravo_json=file_dict["bravo_json"]
            ))

    return sorted(pairs, key=lambda p: (p.episode_num, p.instance_num))


def identify_handler(folder_name: str, summary_json_path: str = None):
    """
    Identify which handler to use based on folder name.

    Uses exact dataset name matching based on handler's DATASET_NAMES attribute.

    Args:
        folder_name: Name of the dataset folder
        summary_json_path: Optional path to structure_building_summary.json (for structure handler)

    Returns:
        Handler instance
    """
    from handlers import (
        MinecraftTranslationHandler,
        MinecraftRotationHandler,
        MinecraftLooksAwayHandler,
        MinecraftBothLookAwayHandler,
        MinecraftStructureBuildingHandler,
        MinecraftTurnToLookHandler,
        MinecraftTurnToLookOppositeHandler,
    )

    # List of all handler classes (order doesn't matter for exact matching)
    handler_classes = [
        MinecraftTranslationHandler,
        MinecraftRotationHandler,
        MinecraftLooksAwayHandler,
        MinecraftBothLookAwayHandler,
        MinecraftTurnToLookHandler,
        MinecraftTurnToLookOppositeHandler,
    ]

    # Structure handlers require summary_json_path
    structure_handler_classes = [
        MinecraftStructureBuildingHandler,
    ]

    # Check each handler for exact dataset name match
    for handler_class in handler_classes:
        if folder_name in handler_class.DATASET_NAMES:
            return handler_class()

    # Special case for structure handlers (require summary_json_path)
    for handler_class in structure_handler_classes:
        if folder_name in handler_class.DATASET_NAMES:
            if not summary_json_path:
                # Use correct default summary JSON based on handler type
                summary_json_path = str(Path(__file__).parent / "assets" / "hard_coded_gt" / "structure_building_summary.json")
            return handler_class(summary_json_path)

    # No exact match found
    all_dataset_names = []
    for handler_class in handler_classes:
        all_dataset_names.extend(handler_class.DATASET_NAMES)
    for handler_class in structure_handler_classes:
        all_dataset_names.extend(handler_class.DATASET_NAMES)

    raise ValueError(
        f"Cannot identify handler for dataset: {folder_name}\n"
        f"No handler found with matching DATASET_NAMES.\n"
        f"Available dataset names: {sorted(all_dataset_names)}"
    )


def dry_run(handler, video_pairs: List[VideoPair], limit: Optional[int] = None):
    """
    View keyframe info without extracting frames or querying VLM.

    Args:
        handler: Episode type handler
        video_pairs: List of video pairs
        limit: Optional limit on number of episodes
    """
    if limit:
        video_pairs = video_pairs[:limit]

    print(f"\n{'='*80}")
    print(f"DRY RUN: Keyframe Detection")
    print(f"Handler: {handler.__class__.__name__}")
    print(f"Episodes: {len(video_pairs)}")
    print(f"{'='*80}\n")

    for i, pair in enumerate(video_pairs, 1):
        print(f"Episode {i}: {pair.episode_num} (instance {pair.instance_num})")
        print("-" * 80)

        queries = handler.extract_keyframes(pair)

        if not queries:
            print("  ⚠ No movement/rotation found in this episode\n")
            continue

        # Print info from first query (they share same metadata)
        meta = queries[0].metadata
        
        # Handle different handler types
        if 'builder' in meta:
            # Structure handler
            print(f"  Builder bot: {meta['builder'].upper()}")
            print(f"  Observer bot: {meta['variant'].upper()}")
            print(f"  Structure: {meta['structure']}")
        elif 'moving_bot' in meta:
            # Translation handler
            print(f"  Moving bot: {meta['moving_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Movement frame: {meta['movement_frame']}")
            print(f"  Movement direction: {meta['movement_direction']}")
        elif 'rotating_bot' in meta:
            # Rotation/looks_away handlers
            print(f"  Rotating bot: {meta['rotating_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Rotation frame: {meta['rotation_frame']}")
            print(f"  Rotation direction: {meta['rotation_direction']}")
            if 'yaw1' in meta and 'yaw2' in meta:
                import math
                yaw_diff = math.degrees(meta['yaw2'] - meta['yaw1'])
                print(f"  Yaw difference: {yaw_diff:.2f}°")

        print(f"  Frame 1: {meta['frame1']}")
        if queries[0].second_frame_index is not None:
            print(f"  Frame 2: {queries[0].second_frame_index}")
        print(f"  Query frame: {queries[0].frame_index}")
        print(f"  Expected answer: {queries[0].expected_answer}")
        print(f"  Perspectives: {len(queries)} (both Alpha and Bravo)\n")


def run_evaluation(handler, video_pairs: List[VideoPair], output_file: Optional[str] = None, limit: Optional[int] = None, generated_path: Optional[Path] = None, dataset_name: Optional[str] = None, model_name: str = "ground_truth", extract_only: bool = False):
    """
    Run VLM evaluation with frame extraction.
    
    Frames are always extracted and saved to disk. If extract_only=True, 
    VLM queries are skipped (useful for visual inspection of frames).

    Args:
        handler: Episode type handler
        video_pairs: List of video pairs
        output_file: Path to save results JSON (not used if extract_only=True)
        limit: Optional limit on number of episodes
        generated_path: Optional path to generated videos directory
        dataset_name: Dataset name (required)
        model_name: Name of our video generation model being evaluated, or "ground_truth" for GT videos
        extract_only: If True, only extract frames without VLM queries
    """
    # Apply episode limit first
    original_count = len(video_pairs)
    if limit:
        video_pairs = video_pairs[:limit]

    # Check API key (only needed if not extract_only)
    enable_thinking = False
    if not extract_only:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("\n⚠ ERROR: GEMINI_API_KEY environment variable not set!")
            print("Please set it with: export GEMINI_API_KEY='your-api-key'")
            raise ValueError("GEMINI_API_KEY environment variable not set")
        # Use handler's enable_vlm_thinking property
        enable_thinking = handler.enable_vlm_thinking

    thinking_status = "default" if enable_thinking else "disabled"
    mode_str = "FRAME EXTRACTION ONLY" if extract_only else "VLM EVALUATION"

    print(f"\n{'='*80}")
    print(f"{mode_str}")
    print(f"Handler: {handler.__class__.__name__}")
    if not extract_only:
        print(f"Model: {VLM_MODEL_NAME} (thinking {thinking_status})")
        print(f"Output: {output_file}")
    if limit and len(video_pairs) < original_count:
        print(f"Episodes: {len(video_pairs)} (limited from {original_count})")
    else:
        print(f"Episodes: {len(video_pairs)}")
    if generated_path:
        print(f"Using generated videos from: {generated_path}")
    print(f"Frame output: frame_extraction/{dataset_name}/{('_real' if model_name == 'ground_truth' else model_name)}/")
    print(f"{'='*80}\n")

    # Find generated video subdirectory if using generated videos
    generated_subdir = None
    if generated_path:
        generated_subdir = find_generated_video_subdir(generated_path, dataset_name)
        if not generated_subdir:
            print(f"Error: Could not find generated video subdirectory for dataset '{dataset_name}'")
            return []
        print(f"Found generated video subdirectory: {generated_subdir.name}")

        # Count available generated videos
        generated_videos = list(generated_subdir.glob("video_*_side_by_side.mp4"))
        num_generated = len(generated_videos)
        print(f"Found {num_generated} generated videos")

        # Limit video pairs to number of generated videos available
        if num_generated < len(video_pairs):
            print(f"Limiting evaluation to first {num_generated} video pairs (fewer generated videos than GT)\n")
            video_pairs = video_pairs[:num_generated]
        else:
            print()

    # Extract keyframes
    print("Extracting keyframe queries...")
    all_queries = []
    for pair in video_pairs:
        queries = handler.extract_keyframes(pair)

        # For rotation and one_looks_away, only query from rotating bot's perspective
        # For both_look_away, keep both perspectives since both bots rotate
        # For structure, only query from observer's perspective (already filtered by handler)
        handler_name = handler.__class__.__name__
        if "Rotation" in handler_name and "Both" not in handler_name and queries:
            rotating_bot = queries[0].metadata['rotating_bot']
            queries = [q for q in queries if q.metadata['variant'] == rotating_bot]
        elif "LooksAway" in handler_name and "Both" not in handler_name and queries:
            rotating_bot = queries[0].metadata['rotating_bot']
            queries = [q for q in queries if q.metadata['variant'] == rotating_bot]
        # Structure handler already filters to observer only, no additional filtering needed

        all_queries.extend(queries)

    print(f"Total queries: {len(all_queries)}")

    # Print section header
    if extract_only:
        print(f"\n{'='*80}")
        print("EXTRACTING FRAMES")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("QUERYING VLM")
        print(f"{'='*80}\n")

    results = []
    vlm_errors = []  # Track VLM errors separately (quota/throttling issues)
    vlm_error_warned = False  # Only warn once about VLM errors

    # Track which video pair we're currently evaluating
    # (needed because filtering may reduce queries per pair)
    current_video_id = -1
    last_episode_instance = None

    for i, query in enumerate(all_queries, 1):
        meta = query.metadata

        # Get query-specific prompt if handler supports it
        query_type = meta.get('query_type', 'default')
        if hasattr(handler, 'get_prompt') and 'query_type' in meta:
            prompt = handler.get_prompt(query_type)
        else:
            prompt = handler.get_prompt()

        query_type_display = f" ({query_type})" if query_type != 'default' else ""
        print(f"[{i}/{len(all_queries)}] Episode {meta['episode']}, Instance {meta['instance']}, {meta['variant'].upper()}{query_type_display}")
        if not extract_only:
            print(f"  Expected: {query.expected_answer}", end="... ")

        # Track video ID for generated videos
        episode_instance = (meta['episode'], meta['instance'])
        if episode_instance != last_episode_instance:
            current_video_id += 1
            last_episode_instance = episode_instance

        frame1_idx = meta['frame1']

        # Extract frames using the unified helper
        frames = extract_query_frames(
            query=query,
            generated_subdir=generated_subdir,
            current_video_id=current_video_id,
            frame1_idx=frame1_idx,
        )

        # Save frames to disk
        output_dir = get_frame_output_dir(dataset_name, model_name, query_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        episode = meta['episode']
        instance = meta['instance']
        variant = meta['variant']
        
        for suffix, frame_bytes in frames.items():
            frame_path = output_dir / f"ep{episode}_inst{instance}_{variant}_{suffix}.png"
            with open(frame_path, 'wb') as f:
                f.write(frame_bytes)

        # If using generated videos, also create side-by-side comparison with GT
        if generated_subdir is not None:
            # Extract GT frames (same query but without generated_subdir)
            gt_frames = extract_query_frames(
                query=query,
                generated_subdir=None,
                current_video_id=current_video_id,
                frame1_idx=frame1_idx,
            )
            
            # Create side-by-side comparison
            sbs_output_dir = get_side_by_side_output_dir(dataset_name, model_name, query_type)
            create_side_by_side_comparison(
                gt_frames=gt_frames,
                gen_frames=frames,
                query=query,
                output_path=sbs_output_dir,
                episode=episode,
                instance=instance,
                variant=variant,
            )

        if extract_only:
            # Just report success and continue
            if generated_subdir is not None:
                print(f"  Saved {len(frames)} frame(s) + side-by-side comparison")
            else:
                print(f"  Saved {len(frames)} frame(s) to {output_dir}")
            continue

        # Query VLM with the extracted frames
        # Only wrap VLM query in try/except since it can fail due to quota/rate limits
        is_turn_to_look = meta.get('is_turn_to_look', False)
        try:
            if is_turn_to_look:
                vlm_response = query_vlm(prompt, frames["alpha_frame"], frames["bravo_frame"], enable_thinking=enable_thinking)
            elif "frame1" in frames and "frame2" in frames:
                vlm_response = query_vlm(prompt, frames["frame1"], frames["frame2"], enable_thinking=enable_thinking)
            else:
                vlm_response = query_vlm(prompt, frames["frame"], enable_thinking=enable_thinking)
        except Exception as e:
            error_str = str(e)
            print(f"✗ VLM Error: {e}")
            
            # Warn user on first VLM error (likely quota/throttling issue)
            if not vlm_error_warned:
                vlm_error_warned = True
                print(f"\n{'!'*80}")
                print("WARNING: VLM API error encountered. This may indicate quota exhaustion or throttling.")
                print("VLM errors will be tracked separately and will NOT count as incorrect responses.")
                print("Check your API quota at: https://ai.dev/rate-limit")
                print(f"{'!'*80}\n")
            
            # Track error separately - don't add to results (doesn't count as incorrect)
            vlm_errors.append({
                "query": query,
                "error": error_str,
                "metadata": meta,
            })
            continue

        # Validate response
        is_correct = handler.validate_response(vlm_response, query.expected_answer)

        result = EvalResult(
            query=query,
            vlm_response=vlm_response,
            is_correct=is_correct,
            metadata={
                "prompt": prompt,
                "handler": handler.__class__.__name__,
                "using_generated": generated_path is not None,
                **meta
            }
        )
        results.append(result)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"Got: '{vlm_response}' {status}")

    # For extract_only mode, just print summary and return
    if extract_only:
        print(f"\n{'='*80}")
        print("FRAME EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total queries processed: {len(all_queries)}")
        print(f"Frames saved to: frame_extraction/{dataset_name}/{('_real' if model_name == 'ground_truth' else model_name)}/")
        print(f"{'='*80}")
        return []

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    save_results(results, output_file, VLM_MODEL_NAME, model_name, thinking_enabled=enable_thinking, vlm_errors=vlm_errors)

    # Print summary
    total_attempted = len(all_queries)
    total_successful = len(results)  # Only queries that got VLM responses
    num_vlm_errors = len(vlm_errors)
    correct = sum(1 for r in results if r.is_correct)
    incorrect = total_successful - correct
    accuracy = correct / total_successful * 100 if total_successful > 0 else 0

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries: {total_successful}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Show VLM errors prominently if any occurred
    if num_vlm_errors > 0:
        print(f"\n{'!'*80}")
        print(f"VLM ERRORS: {num_vlm_errors} queries failed due to API errors (not counted as incorrect)")
        print(f"{'!'*80}")

    # Break down by query type if multiple types exist
    query_types = set(r.metadata.get('query_type', 'default') for r in results)
    if len(query_types) > 1:
        print(f"\n{'='*80}")
        print("BREAKDOWN BY QUERY TYPE")
        print(f"{'='*80}")
        for qtype in sorted(query_types):
            type_results = [r for r in results if r.metadata.get('query_type', 'default') == qtype]
            type_total = len(type_results)
            type_correct = sum(1 for r in type_results if r.is_correct)
            type_incorrect = type_total - type_correct
            type_accuracy = type_correct / type_total * 100 if type_total > 0 else 0

            print(f"\nQuery Type: {qtype}")
            print(f"  Total: {type_total}")
            print(f"  Correct: {type_correct}")
            print(f"  Incorrect: {type_incorrect}")
            print(f"  Accuracy: {type_accuracy:.2f}%")

    print(f"{'='*80}")

    return results


def _resolve_output_dir(
    output_arg: str,
    results_dir: Path,
    dataset_name: str,
    generated_path: Optional[Path],
    model_name: str,
) -> Path:
    """Resolve output directory path for per-trial outputs.

    Behavior:
    - If output_arg is the default "eval_results.json", we auto-organize into:
      - {results_dir}/generated/{model_name}_{dataset_name}/
      - {results_dir}/real/{dataset_name}/
    - If output_arg ends with ".json", we treat it as a legacy file path and convert it to a
      directory path by stripping the suffix (e.g., "foo.json" -> "foo/").
    - Otherwise, we treat output_arg as a directory path.
    """
    if output_arg == "eval_results.json":
        if generated_path:
            return results_dir / "generated" / f"{model_name}_{dataset_name}"
        return results_dir / "real" / dataset_name

    p = Path(output_arg)
    if p.suffix.lower() == ".json":
        return p.with_suffix("")
    return p


def _read_episode_accuracy(output_json_path: Path) -> float:
    """Read episode_level_accuracy.episode_accuracy from a saved trial JSON."""
    with open(output_json_path, "r") as f:
        data = json.load(f)

    try:
        return float(data["episode_level_accuracy"]["episode_accuracy"])
    except Exception as e:
        raise KeyError(
            f"Missing episode_level_accuracy.episode_accuracy in {output_json_path}"
        ) from e


def _write_stats_json(output_dir: Path, accuracies: list[float]) -> None:
    """Write aggregate stats for episode-level accuracy across trials."""
    if not accuracies:
        stats = {
            "metric": "episode_level_accuracy.episode_accuracy",
            "num_trials": 0,
            "trials": [],
            "mean": None,
            "median": None,
            "std": None,
        }
    else:
        stats = {
            "metric": "episode_level_accuracy.episode_accuracy",
            "num_trials": len(accuracies),
            "trials": [
                {"trial": i + 1, "episode_accuracy": acc} for i, acc in enumerate(accuracies)
            ],
            "mean": statistics.mean(accuracies),
            "median": statistics.median(accuracies),
            # Use population stddev so num_trials=1 yields 0.0
            "std": statistics.pstdev(accuracies),
        }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to: {stats_path}")


def _clear_existing_trial_files(output_dir: Path) -> None:
    """Remove any existing trial outputs in output_dir.

    We intentionally keep this narrowly-scoped (only deletes files we own: trial_*.json and
    stats.json) to avoid surprising data loss if the user points --output at a shared dir.
    """
    removed: list[Path] = []

    stats_path = output_dir / "stats.json"
    if stats_path.exists() and stats_path.is_file():
        stats_path.unlink()
        removed.append(stats_path)

    for p in output_dir.glob("trial_*.json"):
        if p.is_file():
            p.unlink()
            removed.append(p)

    if removed:
        # Print a stable, compact message for logs.
        removed_names = ", ".join(sorted(p.name for p in removed))
        print(f"Removed existing trial file(s) in {output_dir}: {removed_names}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified evaluation script for mc_multiplayer datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "folder",
        help="Path to dataset folder (e.g., datasets/eval/turnToLookEval). The /test subdirectory is automatically appended."
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="View keyframe info without extraction or evaluation"
    )
    mode_group.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames only (no VLM queries). Frames saved to frame_extraction/TASK/[_real|MODEL]/[QUERY_TYPE]/"
    )

    # Common options
    parser.add_argument(
        "--output", "-o",
        default="eval_results.json",
        help=(
            "Output path. If omitted, auto-organized into "
            "results_json/generated/{model}_{dataset}/ or results_json/real/{dataset}/ "
            "(override the results_json root with --results-dir). "
            "If a .json file path is provided, it will be converted into a directory by stripping the suffix."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_json"),
        help=(
            "Root directory for auto-organized JSON outputs when --output is left as the default. "
            "Example: --results-dir /tmp/mc_eval_results (writes to /tmp/mc_eval_results/{real,generated}/...)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of episodes to process"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation trials to run (default: 1). Produces trial_*.json plus stats.json.",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key (optional, will use GEMINI_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--summary-json",
        help="Path to structure_building_summary.json (only needed for structure dataset)"
    )
    parser.add_argument(
        "--generated",
        help="Path to generated videos directory (e.g., output/solaris/)"
    )

    args = parser.parse_args()

    if args.num_trials < 1:
        print("Error: --num-trials must be >= 1")
        return 1

    # Set API key if provided
    if args.api_key:
        os.environ['GEMINI_API_KEY'] = args.api_key

    # Get folder path - automatically append /test if not already present
    folder = Path(args.folder)
    if folder.name != "test":
        dataset_name = folder.name
        folder = folder / "test"
    else:
        dataset_name = folder.parent.name

    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        return 1

    try:
        handler = identify_handler(dataset_name, summary_json_path=args.summary_json)
        print(f"Dataset: {dataset_name}")
        print(f"Handler: {handler.__class__.__name__}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Find video pairs
    video_pairs = find_mc_video_pairs(folder)
    print(f"Found: {len(video_pairs)} video pairs")

    if not video_pairs:
        print("Error: No video pairs found")
        return 1

    # Apply default limit of 32 video pairs if no limit specified
    DEFAULT_VIDEO_PAIR_LIMIT = 32
    if len(video_pairs) > DEFAULT_VIDEO_PAIR_LIMIT and args.limit is None:
        print(f"Limiting to first {DEFAULT_VIDEO_PAIR_LIMIT} video pairs (use --limit to override)")
        video_pairs = video_pairs[:DEFAULT_VIDEO_PAIR_LIMIT]

    # Parse generated path if provided
    generated_path = None
    model_name = "ground_truth"
    if args.generated:
        generated_path = Path(args.generated)
        if not generated_path.exists():
            print(f"Error: Generated videos path not found: {generated_path}")
            return 1
        model_name = generated_path.name

    # Determine output directory for evaluation trials
    output_dir: Optional[Path] = None
    if not args.dry_run and not args.extract_frames:
        output_dir = _resolve_output_dir(
            output_arg=args.output,
            results_dir=args.results_dir,
            dataset_name=dataset_name,
            generated_path=generated_path,
            model_name=model_name,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {output_dir}")

    # Execute based on mode
    if args.dry_run:
        dry_run(handler, video_pairs, limit=args.limit)

    elif args.extract_frames:
        # Extract frames only mode (no VLM queries)
        run_evaluation(
            handler,
            video_pairs,
            output_file=None,
            limit=args.limit,
            generated_path=generated_path,
            dataset_name=dataset_name,
            model_name=model_name,
            extract_only=True,
        )

    else:
        # Run full evaluation num_trials times, save per-trial JSON + aggregate stats.
        assert output_dir is not None
        _clear_existing_trial_files(output_dir)
        trial_accuracies: list[float] = []

        for trial_idx in range(1, args.num_trials + 1):
            trial_output = output_dir / f"trial_{trial_idx}.json"
            print(f"\n{'=' * 80}")
            print(f"TRIAL {trial_idx}/{args.num_trials}")
            print(f"Output: {trial_output}")
            print(f"{'=' * 80}\n")

            run_evaluation(
                handler,
                video_pairs,
                str(trial_output),
                limit=args.limit,
                generated_path=generated_path,
                dataset_name=dataset_name,
                model_name=model_name,
            )

            if not trial_output.exists():
                raise RuntimeError(f"Expected trial output not found: {trial_output}")

            trial_accuracies.append(_read_episode_accuracy(trial_output))

        _write_stats_json(output_dir, trial_accuracies)

    return 0


if __name__ == "__main__":
    exit(main())
