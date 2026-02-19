#!/usr/bin/env python3
"""
Handler for mc_multiplayer_eval_one_looks_away dataset.

One bot looks away and then turns back. This handler creates three queries
in chronological order (all single-frame queries):

1. player_position_during_turn: Bot is mid-turn, other player should be visible
   to the left or right of the screen
   
2. player_invisible_looked_away: Bot has fully turned away, other player should
   NOT be visible on screen
   
3. player_position_turned_back: Bot has turned back to original orientation,
   other player should be back at center
"""

import json
from pathlib import Path
from typing import List
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import (
    find_end_of_first_sneak_chunk,
    find_end_of_first_rotation_chunk,
    find_last_action_frame,
    get_yaw_difference,
)


class MinecraftLooksAwayHandler(EpisodeTypeHandler):
    """
    Handler for Minecraft "looks away" evaluation.

    Bot rotates away and then back. Similar to camera rotation evaluation
    but with longer frame timing.
    """

    DATASET_NAMES = ["oneLooksAwayEval"]

    def get_prompt(self, query_type: str = "player_position_during_turn") -> str:
        # Single frame query: bot has turned away, other player should NOT be visible
        return (
            "Here is a Minecraft screenshot. "
            "Is there another player visible on-screen? "
            "Answer with a single word: \"yes\", \"no\"."
        )


    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes based on sneak and camera rotation.

        Creates three queries in chronological order:
        1. player_position_during_turn: While bot is turning, other player should be visible left/right
        2. player_invisible_looked_away: When bot has fully turned away, other player should NOT be visible
        3. player_position_turned_back: When bot turns back, other player should be at center again
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
            rotating_video = video_pair.alpha_video
            sneak_frame = alpha_sneak_frame
            variant = "alpha"
        elif bravo_sneak_frame is not None:
            rotating_data = bravo_data
            rotating_video = video_pair.bravo_video
            sneak_frame = bravo_sneak_frame
            variant = "bravo"
        else:
            raise ValueError(f"No sneak frame found in episode {video_pair.episode_num} instance {video_pair.instance_num}")

        # Calculate keyframe indices
        frame1_idx = sneak_frame  # Reference frame (before turning)
        
        # Frame when fully turned away: end of first rotation chunk + 10 frame buffer
        looked_away_frame_idx = find_end_of_first_rotation_chunk(rotating_data, frame1_idx, buffer=10)
        if looked_away_frame_idx is None:
            raise ValueError(f"No rotation found for {variant} in episode {video_pair.episode_num} instance {video_pair.instance_num}")
        
        # Frame when turned back: 20 frames after last action
        turned_back_frame_idx = find_last_action_frame(rotating_data, frame1_idx, buffer=20)
        if turned_back_frame_idx is None:
            raise ValueError(f"No actions found for {variant} in episode {video_pair.episode_num} instance {video_pair.instance_num}")




        # Compute delta_yaw for each query
        delta_yaw_looked_away = get_yaw_difference(rotating_data, frame1_idx, looked_away_frame_idx)
        delta_yaw_turned_back = get_yaw_difference(rotating_data, frame1_idx, turned_back_frame_idx)
        rotation_direction = "left" if delta_yaw_looked_away > 0 else "right"

        # Query 1: Fully looked away - player NOT visible
        queries.append(KeyframeQuery(
            video_path=rotating_video,
            frame_index=looked_away_frame_idx,
            expected_answer="no",
            metadata={
                "variant": variant,
                "rotating_bot": variant,
                "query_type": "player_invisible_looked_away",
                "rotation_direction": rotation_direction,
                "delta_yaw": delta_yaw_looked_away,
                "frame1": frame1_idx,
                "episode": video_pair.episode_num,
                "instance": video_pair.instance_num
            }
        ))

        # Query 2: Turned back - player should be back at center
        queries.append(KeyframeQuery(
            video_path=rotating_video,
            frame_index=turned_back_frame_idx,
            expected_answer="yes",
            metadata={
                "variant": variant,
                "rotating_bot": variant,
                "query_type": "player_position_turned_back",
                "rotation_direction": rotation_direction,
                "delta_yaw": delta_yaw_turned_back,
                "frame1": frame1_idx,
                "episode": video_pair.episode_num,
                "instance": video_pair.instance_num
            }
        ))

        return queries
