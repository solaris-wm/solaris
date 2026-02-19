#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_translation dataset evaluation.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Add the parent directory to the path to import vlm_utils
sys.path.insert(0, str(Path(__file__).parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk


class MinecraftTranslationHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft translation (movement) evaluation.

    The task evaluates whether the VLM can correctly identify the relative
    motion of another player between two frames.
    """

    DATASET_NAMES = ["translationEval"]

    def get_prompt(self) -> str:
        return (
            "Here are Minecraft screenshots showing another player on the screen. "
            "Between the first frame and the second frame, did the player being shown "
            "move closer, farther, to the left, or to the right on-screen? "
            "Answer with a single word from \"closer\", \"farther\", \"left\", \"right\", or \"no motion\"."
        )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on the sneak and movement actions.

        Logic:
        1. Find which video (Alpha or Bravo) has "sneak": true
        2. Find the first frame with "sneak": true (sneak_frame)
        3. Find the first frame after sneak with a directional movement (movement_frame)
        4. Extract two keyframes:
           - Frame 1: sneak_frame + SNEAK_FRAME_START_DELAY
           - Frame 2: movement_frame + 40
        5. Determine expected answer based on movement direction:
           - "left" → "right" (player moves left, appears to go right on screen)
           - "right" → "left"
           - "forward" → "closer"
           - "back" → "farther"
        """
        queries = []

        # Load JSON data
        with open(video_pair.alpha_json) as f:
            alpha_data = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo_data = json.load(f)

        # Determine which bot is moving (has sneak)
        alpha_sneak_frame = find_end_of_first_sneak_chunk(alpha_data)
        bravo_sneak_frame = find_end_of_first_sneak_chunk(bravo_data)

        if alpha_sneak_frame is not None:
            moving_data = alpha_data
            moving_video = video_pair.alpha_video
            sneak_frame = alpha_sneak_frame
            variant = "alpha"
        elif bravo_sneak_frame is not None:
            moving_data = bravo_data
            moving_video = video_pair.bravo_video
            sneak_frame = bravo_sneak_frame
            variant = "bravo"
        elif alpha_sneak_frame is not None and bravo_sneak_frame is not None:
            raise ValueError(f"Both bots have sneak frames in episode {video_pair.episode_num} instance {video_pair.instance_num}")
        else:
            # No movement found
            return queries

        # Find the movement frame (first frame after sneak with directional input)
        movement_frame, movement_direction = self._find_movement_frame(
            moving_data, sneak_frame
        )

        if movement_frame is None or movement_direction is None:
            # No directional movement found
            return queries

        # Calculate keyframe indices
        frame1_idx = sneak_frame
        frame2_idx = movement_frame + 40

        # Determine expected answer based on movement direction
        expected_answer = self._get_expected_answer(movement_direction)

        # Create keyframe queries for BOTH bot perspectives
        # Both Alpha and Bravo cameras show the same scene at the same time
        # This is a two-frame comparison query (frame1 vs frame2)
        for video_path, video_variant in [
            (video_pair.alpha_video, "alpha"),
            (video_pair.bravo_video, "bravo")
        ]:
            queries.append(KeyframeQuery(
                video_path=video_path,
                frame_index=frame1_idx,
                second_frame_index=frame2_idx,  # Two-frame comparison
                expected_answer=expected_answer,
                metadata={
                    "variant": video_variant,
                    "moving_bot": variant,
                    "sneak_frame": sneak_frame,
                    "movement_frame": movement_frame,
                    "movement_direction": movement_direction,
                    "frame1": frame1_idx,  # Kept for generated video offset calculation
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num
                }
            ))

        return queries

    def _find_movement_frame(
        self, data: List[dict], start_frame: int
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the first frame after start_frame with directional movement.

        Returns:
            Tuple of (frame_index, direction) where direction is one of:
            "forward", "back", "left", "right"
        """
        for i in range(start_frame, len(data)):
            action = data[i].get("action", {})
            for direction in ["forward", "back", "left", "right"]:
                if action.get(direction, False):
                    return i, direction
        return None, None

    def _get_expected_answer(self, movement_direction: str) -> str:
        """
        Map movement direction to expected VLM answer.

        The mapping is based on how the OTHER player appears to move on screen
        from the perspective of the moving player's camera.
        """
        mapping = {
            "left": "right",    # Moving left makes other player appear to move right
            "right": "left",    # Moving right makes other player appear to move left
            "forward": "closer",  # Moving forward makes other player appear closer
            "back": "farther"    # Moving back makes other player appear farther
        }
        return mapping.get(movement_direction, "no motion")

    def validate_response(self, response: str, expected: str) -> bool:
        """
        Validate the VLM response against the expected answer.
        """
        return response.strip().lower() == expected.strip().lower()


def test_keyframe_extraction(test_folder: str, num_episodes: int = 10):
    """
    Test the keyframe extraction on the first N episodes.

    This function will:
    1. Find all video pairs in the test folder
    2. Extract keyframes for the first N episodes
    3. Print the results including frame indices and expected answers
    4. Optionally display the actual frames (if opencv is available)
    """
    import re

    test_path = Path(test_folder)
    handler = MinecraftTranslationHandler()

    # Find all video pairs - need to modify the pattern to handle _camera.mp4
    video_pairs = []
    pattern = re.compile(r'(\d+)_(Alpha|Bravo)_instance_(\d+)_camera\.mp4')

    files = {}
    for file in test_path.iterdir():
        # Match video files
        match = pattern.match(file.name)
        if match:
            episode_num, variant, instance_num = match.groups()
            key = (episode_num, instance_num)

            if key not in files:
                files[key] = {}

            if variant == "Alpha":
                files[key]["alpha_video"] = file
                # Also find JSON
                json_file = test_path / f"{episode_num}_Alpha_instance_{instance_num}.json"
                if json_file.exists():
                    files[key]["alpha_json"] = json_file
            else:  # Bravo
                files[key]["bravo_video"] = file
                json_file = test_path / f"{episode_num}_Bravo_instance_{instance_num}.json"
                if json_file.exists():
                    files[key]["bravo_json"] = json_file

    # Create VideoPair objects for complete pairs
    for (episode_num, instance_num), file_dict in files.items():
        required_keys = ["alpha_video", "bravo_video", "alpha_json", "bravo_json"]
        if all(key in file_dict for key in required_keys):
            video_pairs.append(VideoPair(
                episode_num=episode_num,
                instance_num=instance_num,
                alpha_video=file_dict["alpha_video"],
                bravo_video=file_dict["bravo_video"],
                alpha_json=file_dict["alpha_json"],
                bravo_json=file_dict["bravo_json"]
            ))

    # Sort by episode and instance number
    video_pairs.sort(key=lambda p: (p.episode_num, p.instance_num))

    # Limit to first N episodes
    video_pairs = video_pairs[:num_episodes]

    print(f"Testing keyframe extraction on {len(video_pairs)} episodes")
    print("=" * 80)

    results = []
    for i, pair in enumerate(video_pairs, 1):
        print(f"\nEpisode {i}: {pair.episode_num} (instance {pair.instance_num})")
        print("-" * 80)

        queries = handler.extract_keyframes(pair)

        if not queries:
            print("  ⚠ No movement found in this episode")
            continue

        if queries:
            # Print summary once (same for both perspectives)
            meta = queries[0].metadata
            print(f"  Moving bot: {meta['moving_bot'].upper()}")
            print(f"  Sneak frame: {meta['sneak_frame']}")
            print(f"  Movement frame: {meta['movement_frame']}")
            print(f"  Movement direction: {meta['movement_direction']}")
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
                "movement": query_meta['movement_direction'],
                "expected": query.expected_answer,
                "video_path": str(query.video_path)
            })

    return results


def extract_and_display_frames(results: List[dict], output_dir: str = "test_frames"):
    """
    Extract the actual frames and save them for visual inspection.

    Requires opencv-python (cv2) to be installed.
    """
    try:
        import cv2
    except ImportError:
        print("\nError: opencv-python not installed. Install with: pip install opencv-python")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

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

        # Open video
        cap = cv2.VideoCapture(video_path)

        # Extract frame 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
        ret1, frame1 = cap.read()

        # Extract frame 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)
        ret2, frame2 = cap.read()

        cap.release()

        if ret1 and ret2:
            # Save frames
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Minecraft multiplayer keyframe extraction"
    )
    parser.add_argument(
        "--test-folder",
        default="/home/dl3957/Documents/mp_eval_datasets/mc_multiplayer_eval_dev/test",
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
        default="test_frames",
        help="Output directory for extracted frames"
    )

    args = parser.parse_args()

    # Test keyframe extraction
    results = test_keyframe_extraction(args.test_folder, args.num_episodes)

    # Optionally extract actual frames
    if args.extract_frames:
        extract_and_display_frames(results, args.output_dir)
