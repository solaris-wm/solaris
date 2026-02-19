#!/usr/bin/env python3
"""
Handler for turn_to_look (formerly co_observation) dataset.

Compares perspectives from both bots at a later timestamp to check if they
appear to be from nearby perspectives.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk, find_end_of_first_rotation_chunk


class MinecraftTurnToLookHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft "turn to look" evaluation.

    Both bots' perspectives are compared at a later timestamp to determine
    if they appear to be from nearby perspectives.
    """

    DATASET_NAMES = ["turnToLookEval"]
    enable_vlm_thinking = True

    def get_prompt(self) -> str:
        return (
            "You will be shown two Minecraft screenshots. "
            "Do these two screenshots show the same scenery? Be careful and answer based on the content of the screenshots, not just the camera angles."
            "Answer with a single word: \"yes\", \"no\"."
        )

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on the last sneak from any bot.

        Creates a single query per video pair that compares frames from both
        alpha and bravo perspectives at frame2_idx.
        """
        queries = []

        # Load JSON data
        with open(video_pair.alpha_json) as f:
            alpha_data = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo_data = json.load(f)

        # Find the end of first sneak chunk from both bots
        alpha_sneak_frame = find_end_of_first_sneak_chunk(alpha_data)
        bravo_sneak_frame = find_end_of_first_sneak_chunk(bravo_data)

        # Ensure at least one bot has a sneak frame
        if alpha_sneak_frame is None and bravo_sneak_frame is None:
            raise ValueError(f"No sneak frame found in episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Use the LATEST sneak frame as the starting point (frame1)
        sneak_frames = [f for f in [alpha_sneak_frame, bravo_sneak_frame] if f is not None]
        frame1_idx = max(sneak_frames)

        # Find when rotation ends for each bot, then use the latest
        alpha_rotation_end = find_end_of_first_rotation_chunk(alpha_data, frame1_idx, buffer=20)
        bravo_rotation_end = find_end_of_first_rotation_chunk(bravo_data, frame1_idx, buffer=20)
        
        rotation_ends = [f for f in [alpha_rotation_end, bravo_rotation_end] if f is not None]
        if not rotation_ends:
            raise ValueError(f"No rotation found in episode {video_pair.episode_num} instance {video_pair.instance_num}")
        
        # Use the latest rotation end as frame2 (when both bots have finished turning)
        frame2_idx = max(rotation_ends)

        # Validate that frame2_idx exists in both videos
        if frame2_idx >= len(alpha_data) or frame2_idx >= len(bravo_data):
            raise ValueError(f"frame2_idx {frame2_idx} exceeds video length in episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # For co-observation, we expect both bots to be at nearby perspectives
        # This is the default assumption - they should look similar
        expected_answer = "yes"

        # Create a single query that will compare both perspectives
        # We use alpha_video as the primary video_path, but metadata contains both
        # Note: Two-frame query comparing alpha and bravo perspectives at the same timestamp
        # alpha_frame and bravo_frame specify which frames to extract from each video
        # frame1 is kept as reference for generated video offset calculation
        queries.append(KeyframeQuery(
            video_path=video_pair.alpha_video,
            frame_index=frame2_idx,
            expected_answer=expected_answer,
            metadata={
                "variant": "turn_to_look",  # Special marker
                "is_turn_to_look": True,  # Flag for special handling
                "alpha_video": str(video_pair.alpha_video),
                "bravo_video": str(video_pair.bravo_video),
                "alpha_frame": frame2_idx,
                "bravo_frame": frame2_idx,
                "frame1": frame1_idx,
                "episode": video_pair.episode_num,
                "instance": video_pair.instance_num
            }
        ))

        return queries
