#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_structure dataset (structure building task).
"""

import json
from pathlib import Path
from typing import List, Optional
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk, find_last_action_frame


class MinecraftStructureBuildingHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft structure building evaluation.

    Evaluates from the perspective of the non-building bot to see if they
    can observe the structure being built.
    """

    DATASET_NAMES = ["structureEval"]

    def __init__(self, summary_json_path: str):
        """
        Initialize handler with path to structure building summary JSON.
        
        Args:
            summary_json_path: Path to structure_building_summary.json
        """
        self.summary_json_path = summary_json_path
        self.summary_data = self._load_summary()

    def _load_summary(self) -> dict:
        """Load the structure building summary JSON."""
        with open(self.summary_json_path) as f:
            return json.load(f)

    def get_prompt(self) -> str:
        return (
            "Here is a Minecraft screenshot. "
            "Can you tell me whether there is a visible structure built about 6 blocks away from the player? "
            "Answer with a single word from \"yes\", \"no\"."
        )

    def validate_response(self, response: str, expected: str) -> bool:
        """
        Validate the VLM response against the expected structure type.
        
        Maps structure types from JSON format to prompt format:
        - wall_4x1 → yes
        - tower_2x1 → yes
        - wall_2x2 → yes
        
        Args:
            response: VLM response (should be "yes" or "no")
            expected: Expected structure from JSON (e.g., "wall_4x1", "tower_2x1", "wall_2x2")
            
        Returns:
            True if response matches expected structure type, False otherwise
        """
        # Map JSON structure names to prompt answer format
        structure_mapping = {
            "wall_4x1": "yes",
            "tower_2x1": "yes",
            "wall_2x2": "yes"
        }
        
        # Normalize the response
        normalized_response = response.strip().lower()
        
        # Map expected structure to prompt format
        expected_answer = structure_mapping.get(expected.strip().lower(), "no")
        
        # Check if response matches expected answer
        return normalized_response == expected_answer

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes from the non-building bot's perspective.
        
        Args:
            video_pair: Pair of videos and JSON files for alpha and bravo
            
        Returns:
            List of keyframe queries (one per episode, from observer's perspective)
        """
        queries = []

        # Get episode and instance info
        # Strip leading zeros from episode and instance numbers for JSON lookup
        episode_num = str(int(video_pair.episode_num))
        instance_num = str(int(video_pair.instance_num))
        
        episode_key = f"episode_{episode_num}"
        instance_key = f"instance_{instance_num}"

        # Get builder info from summary
        if instance_key not in self.summary_data:
            print(f"  ⚠ Instance {video_pair.instance_num} not found in summary")
            return queries

        if episode_key not in self.summary_data[instance_key]:
            print(f"  ⚠ Episode {video_pair.episode_num} not found in instance {video_pair.instance_num}")
            return queries

        episode_data = self.summary_data[instance_key][episode_key]
        builder = episode_data["builder"]
        structure = episode_data["structure"]
        
        # Determine which bot is observing (not building) and which is building
        if builder == "alpha":
            observer = "bravo"
            observer_video = video_pair.bravo_video
            observer_json = video_pair.bravo_json
            builder_json = video_pair.alpha_json
        else:
            observer = "alpha"
            observer_video = video_pair.alpha_video
            observer_json = video_pair.alpha_json
            builder_json = video_pair.bravo_json

        # Load builder JSON to find sneak frame (sneak is only present in builder's data)
        with open(builder_json) as f:
            builder_data = json.load(f)

        # Load observer JSON to verify it has enough frames
        with open(observer_json) as f:
            observer_data = json.load(f)

        # Find the end of the first sneak chunk from builder's data to determine episode start
        sneak_frame = find_end_of_first_sneak_chunk(builder_data)
        if sneak_frame is None:
            raise ValueError(f"No sneak frame found in builder data for episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Calculate keyframe indices
        frame1_idx = sneak_frame
        
        # Find 20 frames after builder's last action, but clip to frame1 + 240
        last_action = find_last_action_frame(builder_data, frame1_idx, buffer=20)
        if last_action is None:
            raise ValueError(f"No actions found in builder data for episode {video_pair.episode_num} instance {video_pair.instance_num}")
        
        max_frame2 = frame1_idx + 240
        frame2_idx = min(last_action, max_frame2)

        # Check if we have enough frames
        if frame2_idx >= len(observer_data):
            raise ValueError(f"Not enough frames (need {frame2_idx}, have {len(observer_data)}) for episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Create keyframe query only for the observer (non-building bot)
        # Note: Single-frame query, only frame_index is sent to VLM
        # frame1 is kept as reference for generated video offset calculation
        queries.append(KeyframeQuery(
            video_path=observer_video,
            frame_index=frame2_idx,
            expected_answer=structure,
            metadata={
                "variant": observer,
                "builder": builder,
                "structure": structure,
                "alpha_structure": episode_data["alpha_structure"],
                "bravo_structure": episode_data["bravo_structure"],
                "alpha_builds": episode_data["alpha_builds"],
                "bravo_builds": episode_data["bravo_builds"],
                "frame1": frame1_idx,
                "episode": video_pair.episode_num,
                "instance": video_pair.instance_num
            }
        ))

        return queries
