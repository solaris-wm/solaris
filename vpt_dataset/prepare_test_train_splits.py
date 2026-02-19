#!/usr/bin/env python3
"""
Script to prepare test/train splits by moving episodes to respective subdirectories.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def move_episode_files(episode, dataset_path, target_dir):
    """
    Move video and actions files for an episode to the target directory,
    preserving the relative subfolder structure (e.g., v10/, v9/, etc.).
    
    Args:
        episode: Dictionary with 'video_path' and 'actions_path' keys
        dataset_path: Base path to the dataset
        target_dir: Target directory (test/ or train/)
    """
    video_path = episode["video_path"]
    actions_path = episode["actions_path"]
    
    # Source paths (relative to dataset_path)
    src_video = os.path.join(dataset_path, video_path)
    src_actions = os.path.join(dataset_path, actions_path)
    
    # Target paths (preserve relative structure under target_dir)
    dst_video = os.path.join(dataset_path, target_dir, video_path)
    dst_actions = os.path.join(dataset_path, target_dir, actions_path)
    
    # Create target directories if they don't exist
    os.makedirs(os.path.dirname(dst_video), exist_ok=True)
    os.makedirs(os.path.dirname(dst_actions), exist_ok=True)
    
    # Move files if they exist
    if os.path.exists(src_video):
        shutil.move(src_video, dst_video)
    else:
        print(f"Warning: Video file not found: {src_video}")
    
    if os.path.exists(src_actions):
        shutil.move(src_actions, dst_actions)
    else:
        print(f"Warning: Actions file not found: {src_actions}")


def process_split(episodes_info_path, dataset_path, target_dir, split_name):
    """
    Process a split (test or train) by moving episodes and copying episode info file.
    
    Args:
        episodes_info_path: Path to the episodes_info_{split}.json file
        dataset_path: Base path to the dataset
        target_dir: Target directory name (test/ or train/)
        split_name: Name of the split (test or train) for suffix removal
    """
    print(f"\nProcessing {split_name} split...")
    
    # Read episode info file
    with open(episodes_info_path, "r") as f:
        data = json.load(f)
    
    episodes = data.get("episodes", [])
    print(f"Found {len(episodes)} episodes in {split_name} split")
    
    # Create target directory
    target_path = os.path.join(dataset_path, target_dir)
    os.makedirs(target_path, exist_ok=True)
    
    # Move episode files
    moved_count = 0
    for episode in episodes:
        try:
            move_episode_files(episode, dataset_path, target_dir)
            moved_count += 1
        except Exception as e:
            print(f"Error moving episode {episode.get('episode_id', 'unknown')}: {e}")
    
    print(f"Moved {moved_count} episodes to {target_dir}/")
    
    # Copy and rename episode info file
    # Remove _test or _train suffix from filename
    base_name = os.path.basename(episodes_info_path)
    if base_name.endswith(f"_{split_name}.json"):
        new_name = base_name.replace(f"_{split_name}.json", ".json")
    else:
        # Fallback: just use episodes_info.json
        new_name = "episodes_info.json"
    
    dst_info_path = os.path.join(target_path, new_name)
    shutil.copy2(episodes_info_path, dst_info_path)
    print(f"Copied episode info file to {dst_info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare test/train splits by moving episodes to respective subdirectories"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    episodes_info_dir = script_dir
    
    # Paths to episode info files
    test_info_path = episodes_info_dir / "episodes_info_test.json"
    train_info_path = episodes_info_dir / "episodes_info_train.json"
    
    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return 1
    
    if not test_info_path.exists():
        print(f"Error: Test episodes info file not found: {test_info_path}")
        return 1
    
    if not train_info_path.exists():
        print(f"Error: Train episodes info file not found: {train_info_path}")
        return 1
    
    print(f"Dataset path: {dataset_path}")
    print(f"Test episodes info: {test_info_path}")
    print(f"Train episodes info: {train_info_path}")
    
    # Process test split
    process_split(
        str(test_info_path),
        str(dataset_path),
        "test",
        "test"
    )
    
    # Process train split
    process_split(
        str(train_info_path),
        str(dataset_path),
        "train",
        "train"
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
