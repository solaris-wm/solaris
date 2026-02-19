#!/usr/bin/env python3
"""
VLM Evaluation Utilities

This module provides utility functions and data structures for VLM evaluation:
- Data structures for video pairs, queries, and results
- Frame extraction from videos (both ground-truth and generated)
- VLM querying via Gemini API
- Result saving and formatting
"""

import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Model configuration
VLM_MODEL_NAME = "gemini-3-flash-preview"


@dataclass
class VideoPair:
    """Represents a pair of videos (Alpha/Bravo) with the same episode and instance number."""
    episode_num: str
    instance_num: str
    alpha_video: Path
    bravo_video: Path
    alpha_json: Path
    bravo_json: Path

    def __repr__(self):
        return f"VideoPair(episode={self.episode_num}, instance={self.instance_num})"


@dataclass
class KeyframeQuery:
    """Represents a single keyframe query with expected answer.
    
    For single-frame queries, only frame_index is set.
    For two-frame comparison queries (e.g., translation), both frame_index 
    and second_frame_index are set.
    """
    video_path: Path
    frame_index: int  # Primary/first frame index
    expected_answer: str
    second_frame_index: Optional[int] = None  # For two-frame comparison queries
    metadata: Optional[Dict] = None  # Additional context for the query

    def __repr__(self):
        if self.second_frame_index is not None:
            return f"KeyframeQuery(video={self.video_path.name}, frames=[{self.frame_index}, {self.second_frame_index}])"
        return f"KeyframeQuery(video={self.video_path.name}, frame={self.frame_index})"


@dataclass
class EvalResult:
    """Results from evaluating a single keyframe query."""
    query: KeyframeQuery
    vlm_response: str
    is_correct: bool
    metadata: Optional[Dict] = None


class EpisodeTypeHandler(ABC):
    """
    Abstract base class for episode type handlers.

    Each episode type should implement its own handler that defines:
    - How to extract keyframes from video pairs
    - What prompt to use for the VLM
    - How to parse and validate VLM responses
    """

    # List of exact dataset names this handler supports
    # Subclasses should override this with their specific dataset names
    DATASET_NAMES: List[str] = []

    # Whether to enable VLM thinking mode for this handler
    # Subclasses can override this to True if thinking improves accuracy
    enable_vlm_thinking: bool = False

    @abstractmethod
    def get_prompt(self) -> str:
        """
        Return the prompt template to use with the VLM.

        The prompt can use placeholders like {frame_description} if needed.
        """
        pass

    @abstractmethod
    def extract_keyframes(self, video_pair: VideoPair) -> List[KeyframeQuery]:
        """
        Extract keyframes from a video pair and return queries with expected answers.

        This method should:
        1. Read the JSON files for the video pair
        2. Determine which frames to extract based on the actions
        3. Return a list of KeyframeQuery objects with expected answers

        Args:
            video_pair: A VideoPair object containing paths to videos and JSONs

        Returns:
            List of KeyframeQuery objects
        """
        pass

    def validate_response(self, response: str, expected: str) -> bool:
        """
        Validate the VLM response against the expected answer.

        Default implementation does exact string matching.
        Override for custom validation logic.
        """
        return response.strip().lower() == expected.strip().lower()


def extract_frame(video_path: Path, frame_index: int) -> bytes:
    """
    Extract a specific frame from a video as image bytes, resized to 640x360.

    Args:
        video_path: Path to the video file
        frame_index: Zero-indexed frame number to extract

    Returns:
        Image bytes (PNG format), resized to 640x360
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Resize to 640x360 to match generated video quadrant size
        frame_resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.png', frame_resized)
        return buffer.tobytes()
    else:
        raise ValueError(f"Could not extract frame {frame_index} from {video_path}")


def extract_quadrant(frame, quadrant: str):
    """
    Extract a specific quadrant from a side-by-side video frame.

    Args:
        frame: Full video frame (1280x720) for Oasis or (1280x704) for Matrix-Game 2
        quadrant: One of "top-left", "top-right", "bottom-left", "bottom-right"

    Returns:
        Extracted quadrant frame (640x360) for Oasis or (640x352) for Matrix-Game 2
    """
    height, width = frame.shape[:2]

    # Verify expected dimensions
    if width != 1280 or (height != 720 and height != 704):
        raise ValueError(f"Expected frame size 1280x720, got {width}x{height}")

    quad_width = 640
    quad_height = 360 if height == 720 else 352

    if quadrant == "top-left":
        return frame[0:quad_height, 0:quad_width]
    elif quadrant == "top-right":
        return frame[0:quad_height, quad_width:width]
    elif quadrant == "bottom-left":
        return frame[quad_height:height, 0:quad_width]
    elif quadrant == "bottom-right":
        return frame[quad_height:height, quad_width:width]
    else:
        raise ValueError(f"Invalid quadrant: {quadrant}")


def find_generated_video_subdir(generated_base_path: Path, dataset_name: str) -> Optional[Path]:
    """
    Find the generated video subdirectory that matches the dataset name.

    Args:
        generated_base_path: Base path to generated videos
        dataset_name: Dataset name (e.g., "rotationEval", "bothLookAwayEval")

    Returns:
        Path to the subdirectory containing generated videos, or None if not found
    """
    # Map dataset names to the key used in generated video subdirectory names
    # Generated dirs are like: step_0002000_multiplayer_v2_eval_{key}
    dataset_to_subdir_key = {
        "translationEval": "translation",
        "rotationEval": "rotation",
        "bothLookAwayEval": "both_look_away",
        "oneLooksAwayEval": "one_looks_away",
        "turnToLookEval": "turn_to_look",
        "turnToLookOppositeEval": "turn_to_look_opposite",  # Uses same generated videos as turnToLookEval
        "structureEval": "structure",
    }

    subdir_key = dataset_to_subdir_key.get(dataset_name)
    if not subdir_key:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    strippable_suffixes = [] #
    replacement_suffixes = []  #
    
    # Look for subdirectories matching the pattern (e.g., *_eval_translation, *_eval_rotation).
    # Use exact matching to prevent bugs like "turn_to_look" matching "turn_to_look_opposite".
    candidates: List[Path] = []
    for subdir in generated_base_path.iterdir():
        if not subdir.is_dir():
            continue
        _, _, suffix = subdir.name.partition("eval_")
        if not suffix:
            continue
        # Strip known suffixes before matching (with replacement)
        normalized_suffix = suffix
        for strip_suffix, replace_with in zip(strippable_suffixes, replacement_suffixes):
            if normalized_suffix.endswith(strip_suffix):
                normalized_suffix = normalized_suffix[:-len(strip_suffix)] + replace_with
                break
        # Exact match only
        if normalized_suffix == subdir_key:
            candidates.append(subdir)

    if not candidates:
        raise ValueError(
            f"Could not find generated video subdirectory for dataset '{dataset_name}' "
            f"(expected key '{subdir_key}')"
        )

    # If multiple candidates exist, prefer the latest step_XXXX in the directory name.
    def _step_num(p: Path) -> int:
        import re
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=lambda p: (_step_num(p), p.name))


def extract_frame_from_generated(
    generated_video_path: Path,
    frame_index_gt: int,
    frame1_idx_gt: int,
    variant: str
) -> bytes:
    """
    Extract a frame from a generated side-by-side video.

    Args:
        generated_video_path: Path to the generated video_X_side_by_side.mp4
        frame_index_gt: Ground-truth frame index to extract
        frame1_idx_gt: Ground-truth frame1_idx (sneak_frame + SNEAK_FRAME_START_DELAY) - this is frame 0 in generated video
        variant: "alpha" or "bravo" to determine which quadrant to extract

    Returns:
        Image bytes (PNG format) of the extracted quadrant
    """
    import cv2

    # Calculate the frame index in the generated video
    # Generated video frame 0 corresponds to GT frame (frame1_idx + 1)
    generated_frame_idx = frame_index_gt - frame1_idx_gt - 1

    if generated_frame_idx < 0:
        raise ValueError(
            f"Frame index {frame_index_gt} is before the generated video start "
            f"(starts at GT frame {frame1_idx_gt + 1})"
        )

    # Open the generated video
    cap = cv2.VideoCapture(str(generated_video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, generated_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(
            f"Could not extract frame {generated_frame_idx} from {generated_video_path}"
        )

    # Extract the appropriate quadrant
    # Top-right: alpha generated
    # Bottom-right: bravo generated
    quadrant = "top-right" if variant == "alpha" else "bottom-right"
    extracted_frame = extract_quadrant(frame, quadrant)

    # Encode to PNG
    _, buffer = cv2.imencode('.png', extracted_frame)
    return buffer.tobytes()


def query_vlm(prompt: str, image_bytes: bytes, image_bytes_2: Optional[bytes] = None, enable_thinking: bool = False, max_retries: int = 3) -> str:
    """
    Query the VLM (e.g., Gemini) with a prompt and image(s).

    Args:
        prompt: The text prompt for the VLM
        image_bytes: First image data as bytes
        image_bytes_2: Optional second image data as bytes
        enable_thinking: Whether to enable thinking mode (default: False)
        max_retries: Maximum number of retries for rate limit errors (default: 3)

    Returns:
        VLM response as string
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    # Build content parts
    content_parts = [
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png',
        )
    ]

    # Add second image if provided
    if image_bytes_2 is not None:
        content_parts.append(
            types.Part.from_bytes(
                data=image_bytes_2,
                mime_type='image/png',
            )
        )

    # Add the prompt
    content_parts.append(prompt)

    # Configure thinking based on parameter
    if enable_thinking:
        # Use default thinking config (thinking enabled)
        config = types.GenerateContentConfig(
            system_instruction='You are a helpful assistant that evaluates Minecraft screenshots.',
        )
    else:
        # Disable thinking
        config = types.GenerateContentConfig(
            system_instruction='You are a helpful assistant that evaluates Minecraft screenshots.',
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

    # Retry loop with exponential backoff for rate limit errors
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=VLM_MODEL_NAME,
                contents=content_parts,
                config=config,
            )
            
            # Explicitly extract text parts to avoid warning about non-text parts (e.g., thought_signature)
            text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
            if not text_parts:
                warnings.warn(f"No text parts found in VLM response: {response}")
                return ""
            if len(text_parts) > 1:                                                            
                warnings.warn(f"Multiple text parts in VLM response: {text_parts}") 
            return ''.join(text_parts).strip().lower()
            
        except Exception as e:
            last_exception = e
            error_str = str(e)
            
            # Check if this is a rate limit error (429) that we should retry
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 2s, 4s, 8s, ...
                wait_time = 2 ** (attempt + 1)
                print(f"\n  ⚠ Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Not a rate limit error, or we've exhausted retries - re-raise
                raise
    
    # Should not reach here, but just in case
    raise last_exception


def _compute_episode_level_accuracy(results: List[EvalResult]) -> Dict:
    """
    Compute episode-level accuracy metrics.
    
    An episode is considered correct only if ALL queries for that episode are correct.
    
    For datasets with both players (bothLookAwayEval), also computes:
    - Per-player episode accuracy (all 3 queries correct for that player)
    - Full episode accuracy (all 6 queries correct for both players)
    
    Returns:
        Dictionary with episode-level accuracy metrics
    """
    from collections import defaultdict
    
    # Group results by (episode, instance)
    episodes = defaultdict(list)
    for r in results:
        if r.metadata:
            key = (r.metadata.get('episode'), r.metadata.get('instance'))
            episodes[key].append(r)
    
    if not episodes:
        return {}
    
    # Check if this is a "both players" dataset by looking for both alpha and bravo variants
    all_variants = set()
    for r in results:
        if r.metadata and 'variant' in r.metadata:
            all_variants.add(r.metadata['variant'])
    
    is_both_players = 'alpha' in all_variants and 'bravo' in all_variants
    
    # Calculate episode-level accuracy
    total_episodes = len(episodes)
    fully_correct_episodes = 0
    
    # For both-players datasets, also track per-player episode accuracy
    alpha_correct_episodes = 0
    bravo_correct_episodes = 0
    alpha_episode_count = 0
    bravo_episode_count = 0
    
    for (episode, instance), episode_results in episodes.items():
        # Check if all queries in this episode are correct
        all_correct = all(r.is_correct for r in episode_results)
        if all_correct:
            fully_correct_episodes += 1
        
        if is_both_players:
            # Group by variant within this episode
            alpha_results = [r for r in episode_results if r.metadata and r.metadata.get('variant') == 'alpha']
            bravo_results = [r for r in episode_results if r.metadata and r.metadata.get('variant') == 'bravo']
            
            if alpha_results:
                alpha_episode_count += 1
                if all(r.is_correct for r in alpha_results):
                    alpha_correct_episodes += 1
            
            if bravo_results:
                bravo_episode_count += 1
                if all(r.is_correct for r in bravo_results):
                    bravo_correct_episodes += 1
    
    episode_metrics = {
        "total_episodes": total_episodes,
        "fully_correct_episodes": fully_correct_episodes,
        "episode_accuracy": (fully_correct_episodes / total_episodes * 100) if total_episodes > 0 else 0,
    }
    
    if is_both_players:
        episode_metrics["is_both_players_dataset"] = True
        episode_metrics["per_player_episode_accuracy"] = {
            "alpha": {
                "total_episodes": alpha_episode_count,
                "fully_correct_episodes": alpha_correct_episodes,
                "episode_accuracy": (alpha_correct_episodes / alpha_episode_count * 100) if alpha_episode_count > 0 else 0,
            },
            "bravo": {
                "total_episodes": bravo_episode_count,
                "fully_correct_episodes": bravo_correct_episodes,
                "episode_accuracy": (bravo_correct_episodes / bravo_episode_count * 100) if bravo_episode_count > 0 else 0,
            }
        }
    else:
        episode_metrics["is_both_players_dataset"] = False
    
    return episode_metrics


def save_results(results: List[EvalResult], output_path: str, vlm_model_name: str, our_model_name: str, thinking_enabled: bool = False, vlm_errors: Optional[List[Dict]] = None):
    """
    Save evaluation results to a JSON file.

    Args:
        results: List of EvalResult objects (successful VLM queries only)
        output_path: Path to save the JSON file
        vlm_model_name: the VLM judge used for the evaluation
        our_model_name: the name of our video generation model being evaluated, or "ground_truth" for GT videos
        thinking_enabled: Whether thinking mode was enabled for VLM queries
        vlm_errors: Optional list of VLM errors (queries that failed due to API errors)
    """
    if vlm_errors is None:
        vlm_errors = []
    
    # Calculate overall statistics (only from successful queries)
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / total * 100) if total > 0 else 0

    # Calculate breakdown by query type
    query_types = set(r.metadata.get('query_type', 'default') for r in results if r.metadata)
    breakdown_by_query_type = {}

    for qtype in sorted(query_types):
        type_results = [r for r in results if r.metadata and r.metadata.get('query_type', 'default') == qtype]
        type_total = len(type_results)
        type_correct = sum(1 for r in type_results if r.is_correct)

        breakdown_by_query_type[qtype] = {
            "total": type_total,
            "correct": type_correct,
            "accuracy": (type_correct / type_total * 100) if type_total > 0 else 0
        }

    # Calculate episode-level accuracy metrics
    episode_level_accuracy = _compute_episode_level_accuracy(results)

    output_data = {
        "vlm_model_name": vlm_model_name,
        "our_model_name": our_model_name,
        "thinking_enabled": thinking_enabled,
        "total_queries": total,
        "correct": correct,
        "accuracy": accuracy,
        "vlm_errors_count": len(vlm_errors),
        "breakdown_by_query_type": breakdown_by_query_type,
        "episode_level_accuracy": episode_level_accuracy,
        "results": [
            {
                "video": str(r.query.video_path.name),
                "frame_index": r.query.frame_index,
                **({"second_frame_index": r.query.second_frame_index} if r.query.second_frame_index is not None else {}),
                "expected": r.query.expected_answer,
                "response": r.vlm_response,
                "correct": r.is_correct,
                "metadata": r.metadata
            }
            for r in results
        ],
    }
    
    # Include VLM errors if any occurred
    if vlm_errors:
        output_data["vlm_errors"] = [
            {
                "video": str(err["query"].video_path.name),
                "frame_index": err["query"].frame_index,
                **({"second_frame_index": err["query"].second_frame_index} if err["query"].second_frame_index is not None else {}),
                "expected": err["query"].expected_answer,
                "error": err["error"],
                "metadata": err["metadata"],
            }
            for err in vlm_errors
        ]

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
