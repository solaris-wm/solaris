#!/usr/bin/env python3
"""Handler for the co-movement eval, where BOTH bots move simultaneously.

This is translationEval's two-bot counterpart. In translationEval exactly one bot
sneaks and moves while the other stands still. Here both bots sneak and both take
the same movement action at the same time. Because the bots face each other, the
same action always changes their relative position, so every answer is directional
(closer / farther / left / right), and the two cameras can disagree, so each
perspective gets its own expected answer.

Expected answers are computed geometrically from the recorded x/z/yaw rather than
from action labels: the other bot's displacement is projected onto the observer's
forward and right axes and the dominant component wins. On translationEval this
reproduces the action-derived answers on all 64 queries.

Episode structure:

    ~f48   both bots sneak                        (episode start)
    ~f49   chunk 0: both walk forward             (shared approach)
    ~f90   chunk 1: the tested co-movement        <- what we query
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_utils import EpisodeTypeHandler, VideoPair, KeyframeQuery
from handlers.camera_utils import find_end_of_first_sneak_chunk
from handlers.mc_multiplayer_handler_translation import MinecraftTranslationHandler

DIRECTIONS = ("forward", "back", "left", "right")

# Frames after the tested chunk begins, matching translationEval's +40.
QUERY_OFFSET = 40

# Displacement below this in both axes reads as no motion (blocks).
DEADZONE = 0.75


class MinecraftCoMovementHandler(EpisodeTypeHandler):
    """Both bots move at once; each camera is judged on its own."""

    DATASET_NAMES = ["coMovementEval"]

    def get_prompt(self, query_type: str = "co_movement") -> str:
        """translationEval's prompt, so the two evals ask the same question.

        Delegated rather than copied, so the two stay identical.
        """
        return MinecraftTranslationHandler().get_prompt()

    # --- geometry ---------------------------------------------------------

    @staticmethod
    def _camera_axes(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Forward and right unit vectors in (x, z).

        Derived from the data by holding each direction key and comparing the
        travel angle with yaw: forward is yaw+180 deg, right is yaw+90 deg.
        """
        return (-math.sin(yaw), -math.cos(yaw)), (math.cos(yaw), -math.sin(yaw))

    @classmethod
    def _apparent_motion(cls, obs: List[dict], other: List[dict],
                         f1: int, f2: int) -> Tuple[str, dict]:
        """Classify how ``other`` appears to move in ``obs``'s camera."""
        fwd, right = cls._camera_axes(obs[f1]["yaw"])

        def rel(f):
            return other[f]["x"] - obs[f]["x"], other[f]["z"] - obs[f]["z"]

        x1, z1 = rel(f1)
        x2, z2 = rel(f2)
        d_depth = (x2 * fwd[0] + z2 * fwd[1]) - (x1 * fwd[0] + z1 * fwd[1])
        d_lat = (x2 * right[0] + z2 * right[1]) - (x1 * right[0] + z1 * right[1])

        if abs(d_depth) < DEADZONE and abs(d_lat) < DEADZONE:
            answer = "no motion"
        elif abs(d_depth) >= abs(d_lat):
            answer = "closer" if d_depth < 0 else "farther"
        else:
            answer = "right" if d_lat > 0 else "left"

        return answer, {
            "delta_depth": round(d_depth, 3),
            "delta_lateral": round(d_lat, 3),
            "distance_frame1": round(math.hypot(x1, z1), 3),
            "distance_frame2": round(math.hypot(x2, z2), 3),
        }

    # --- keyframes --------------------------------------------------------

    @staticmethod
    def _movement_chunks(data: List[dict]) -> List[Tuple[int, str, int]]:
        out: List[list] = []
        for i, frame in enumerate(data):
            action = frame.get("action", {})
            for d in DIRECTIONS:
                if action.get(d):
                    if not out or out[-1][1] != d or i > out[-1][2] + 1:
                        out.append([i, d, i])
                    else:
                        out[-1][2] = i
        return [tuple(c) for c in out]

    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        with open(video_pair.alpha_json) as f:
            alpha = json.load(f)
        with open(video_pair.bravo_json) as f:
            bravo = json.load(f)

        alpha_sneak = find_end_of_first_sneak_chunk(alpha)
        bravo_sneak = find_end_of_first_sneak_chunk(bravo)
        if alpha_sneak is None or bravo_sneak is None:
            raise ValueError(
                f"co-movement expects both bots to sneak; got alpha={alpha_sneak} "
                f"bravo={bravo_sneak} in episode {video_pair.episode_num} "
                f"instance {video_pair.instance_num}")

        alpha_chunks = self._movement_chunks(alpha)
        bravo_chunks = self._movement_chunks(bravo)
        if len(alpha_chunks) < 2 or len(bravo_chunks) < 2:
            raise ValueError(
                f"expected a shared approach then a tested co-movement; got "
                f"{len(alpha_chunks)}/{len(bravo_chunks)} chunks in episode "
                f"{video_pair.episode_num} instance {video_pair.instance_num}")

        # chunk 0 is the approach both bots share; chunk 1 is the tested move.
        alpha_start, alpha_dir, _ = alpha_chunks[1]
        bravo_start, bravo_dir, _ = bravo_chunks[1]

        # The bots start within a frame or two of each other; take the earlier
        # so the "before" frame precedes both.
        query_frame1 = min(alpha_start, bravo_start)
        usable = min(len(alpha), len(bravo))
        query_frame2 = min(query_frame1 + QUERY_OFFSET, usable - 1)
        if query_frame2 <= query_frame1:
            raise ValueError(
                f"not enough frames after the tested move in episode "
                f"{video_pair.episode_num} instance {video_pair.instance_num}")

        # The episode starts at the sneak, which is what generated videos are
        # rendered from; `frame1` keeps that meaning for offset arithmetic.
        episode_start = min(alpha_sneak, bravo_sneak)

        queries = []
        for variant, obs, other, video in (
            ("alpha", alpha, bravo, video_pair.alpha_video),
            ("bravo", bravo, alpha, video_pair.bravo_video),
        ):
            expected, detail = self._apparent_motion(
                obs, other, query_frame1, query_frame2)
            queries.append(KeyframeQuery(
                video_path=video,
                frame_index=query_frame1,
                second_frame_index=query_frame2,
                expected_answer=expected,
                metadata={
                    "variant": variant,
                    "query_type": "co_movement",
                    "alpha_direction": alpha_dir,
                    "bravo_direction": bravo_dir,
                    "observer_direction": alpha_dir if variant == "alpha" else bravo_dir,
                    "other_direction": bravo_dir if variant == "alpha" else alpha_dir,
                    "tested_chunk_start": query_frame1,
                    "query_frame1": query_frame1,
                    "query_frame2": query_frame2,
                    # Episode start, i.e. generated frame 0 is GT frame1 + 1.
                    "frame1": episode_start,
                    "episode": video_pair.episode_num,
                    "instance": video_pair.instance_num,
                    **detail,
                }
            ))
        return queries

    def validate_response(self, response: str, expected: str) -> bool:
        return response.strip().lower() == expected.strip().lower()
