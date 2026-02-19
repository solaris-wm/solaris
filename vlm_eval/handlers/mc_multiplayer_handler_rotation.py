#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_rotation dataset (camera rotation task).
"""

import json
from pathlib import Path
from typing import List
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk, find_end_of_first_rotation_chunk, calculate_position_answer, get_yaw_difference


class MinecraftRotationHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft rotation evaluation.

    Similar to translation evaluation, but detects camera rotation instead of
    directional movement.
    """

    DATASET_NAMES = ["rotationEval"]

    def get_prompt(self) -> str:
        return (
            "Here is a Minecraft screenshot potentially showing another player on the screen. "
            "Where is the player located on the screen? "
            "Answer with a single word from \"left\", \"right\", \"center\". "
            "If there is no player on the screen, answer \"no player\"."
        )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on sneak and camera rotation.

        Logic is the same as movement detection, but looks for camera
        rotation instead of forward/back/left/right.
        """
        queries = []

        # Load JSON data
        with open(video_pair.alpha_json) as f:
            alpha_data = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo_data = json.load(f)

        # Determine which bot is rotating (has sneak)
        alpha_sneak_frame = find_end_of_first_sneak_chunk(alpha_data)
        bravo_sneak_frame = find_end_of_first_sneak_chunk(bravo_data)

        if alpha_sneak_frame is not None:
            rotating_data = alpha_data
            sneak_frame = alpha_sneak_frame
            variant = "alpha"
        elif bravo_sneak_frame is not None:
            rotating_data = bravo_data
            sneak_frame = bravo_sneak_frame
            variant = "bravo"
        else:
            raise ValueError(f"No sneak frame found in episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Find the frame after the first rotation chunk ends (+ 20 frame buffer)
        frame2_idx = find_end_of_first_rotation_chunk(rotating_data, sneak_frame, buffer=20)
        if frame2_idx is None:
            raise ValueError(f"No rotation found in episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Calculate keyframe indices
        frame1_idx = sneak_frame

        # Calculate expected answer based on yaw difference
        try:
            expected_answer = calculate_position_answer(
                rotating_data, frame1_idx, frame2_idx
            )
        except ValueError as e:
            # Skip this episode if yaw difference doesn't match expected patterns
            print(f"  ⚠ Skipping episode {video_pair.episode_num} instance {video_pair.instance_num}: {e}")
            return queries

        # Compute delta_yaw and rotation_direction for metadata
        delta_yaw = get_yaw_difference(rotating_data, frame1_idx, frame2_idx)
        rotation_direction = "left" if delta_yaw > 0 else "right"

        # Create keyframe queries for BOTH bot perspectives
        # Note: Only frame2 is used for the VLM query (single-frame query)
        # frame1 is kept as reference for generated video offset calculation
        for video_path, video_variant in [
            (video_pair.alpha_video, "alpha"),
            (video_pair.bravo_video, "bravo")
        ]:
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame2_idx,
                expected_answer=expected_answer,
                metadata={
                    "variant": video_variant,
                    "rotating_bot": variant,
                    "rotation_direction": rotation_direction,
                    "delta_yaw": delta_yaw,
                    "frame1": frame1_idx,
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

        return queries


# Reuse the test functions from mc_multiplayer_handler but with new handler
if __name__ == "__main__":
    import argparse
    import re
    # VideoPair already imported at top of file

    def find_mc_video_pairs(folder: Path):
        """Find video pairs in mc_multiplayer format."""
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

    def test_and_extract(test_folder: str, num_episodes: int, extract_frames: bool, output_dir: str):
        """Test keyframe extraction and optionally extract frames."""
        import cv2

        test_path = Path(test_folder)
        handler = MinecraftRotationHandler()

        video_pairs = find_mc_video_pairs(test_path)
        video_pairs = video_pairs[:num_episodes]

        print(f"Testing keyframe extraction on {len(video_pairs)} episodes")
        print("=" * 80)

        results = []
        for i, pair in enumerate(video_pairs, 1):
            print(f"\nEpisode {i}: {pair.episode_num} (instance {pair.instance_num})")
            print("-" * 80)

            queries = handler.extract_keyframes(pair)

            if not queries:
                print("  ⚠ No camera rotation found in this episode")
                continue

            meta = queries[0].metadata
            print(f"  Rotating bot: {meta['rotating_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Rotation frame: {meta['rotation_frame']}")
            print(f"  Rotation direction: {meta['rotation_direction']}")
            print(f"  Keyframe 1: frame {meta['frame1']}")
            print(f"  Keyframe 2: frame {meta['frame2']}")
            print(f"  Expected answer: {queries[0].expected_answer}")
            print(f"  Extracting from: BOTH Alpha and Bravo perspectives")

            for query in queries:
                query_meta = query.metadata
                results.append({
                    "episode": pair.episode_num,
                    "instance": pair.instance_num,
                    "variant": query_meta['variant'],
                    "frame1": query_meta['frame1'],
                    "frame2": query_meta['frame2'],
                    "rotation": query_meta['rotation_direction'],
                    "expected": query.expected_answer,
                    "video_path": str(query.video_path)
                })

        # Extract frames if requested
        if extract_frames and results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 80}")
            print(f"Extracting frames to: {output_path}")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                video_path = result['video_path']
                frame1_idx = result['frame1']
                frame2_idx = result['frame2']
                episode = result['episode']
                instance = result['instance']
                variant = result['variant']

                print(f"\nEpisode {i}: {episode} (instance {instance}) - {variant}")

                cap = cv2.VideoCapture(video_path)

                # Extract frame 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
                ret1, frame1 = cap.read()

                # Extract frame 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
                ret2, frame2 = cap.read()

                cap.release()

                if ret1 and ret2:
                    frame1_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame1.png"
                    frame2_path = output_path / f"ep{episode}_inst{instance}_{variant}_frame2.png"

                    cv2.imwrite(str(frame1_path), frame1)
                    cv2.imwrite(str(frame2_path), frame2)

                    print(f"  ✓ Saved frame 1 (idx {frame1_idx}): {frame1_path.name}")
                    print(f"  ✓ Saved frame 2 (idx {frame2_idx}): {frame2_path.name}")
                    print(f"  Expected answer: {result['expected']}")
                else:
                    print(f"  ✗ Failed to extract frames")

            print(f"\n{'=' * 80}")
            print(f"Frames saved to: {output_path.absolute()}")
            print("=" * 80)

    parser = argparse.ArgumentParser(
        description="Test Minecraft camera rotation keyframe extraction"
    )
    parser.add_argument(
        "--test-folder",
        default="/home/dl3957/Documents/mp_eval_datasets/mc_multiplayer_eval_dev_2/test",
        help="Path to test folder"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to test"
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract and save actual frames for visual inspection"
    )
    parser.add_argument(
        "--output-dir",
        default="test_frames_dev2",
        help="Output directory for extracted frames"
    )

    args = parser.parse_args()

    test_and_extract(args.test_folder, args.num_episodes, args.extract_frames, args.output_dir)
