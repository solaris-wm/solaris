#!/usr/bin/env python3
"""
Visualization helpers for frame extraction and comparison.

This module provides utilities for creating side-by-side comparison images
between ground-truth and generated video frames.
"""

from pathlib import Path
from typing import Optional

from vlm_utils import KeyframeQuery


def add_expected_answer_label(image, expected_answer: str, font_scale: float = 0.6, padding: int = 10):
    """
    Add expected answer text at the bottom of the image.
    
    Args:
        image: CV2 image (numpy array)
        expected_answer: The expected VLM answer to display
        font_scale: Font size scale
        padding: Padding around the text
        
    Returns:
        Image with expected answer label added at the bottom
    """
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # Prepare text with "Expected: " prefix
    text = f"Expected: {expected_answer}"
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # If text is too wide, wrap it
    max_text_width = w - 2 * padding
    lines = []
    if text_width > max_text_width:
        # Simple word wrapping
        words = text.split(' ')
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (test_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if test_width <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    else:
        lines = [text]
    
    # Calculate total height needed for all lines
    line_height = text_height + baseline + 5
    total_text_height = len(lines) * line_height + 2 * padding
    
    # Create a bar at the bottom for the text
    bar = np.zeros((total_text_height, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)  # Dark gray background
    
    # Draw each line of text
    for i, line in enumerate(lines):
        y_pos = padding + (i + 1) * line_height - baseline
        cv2.putText(bar, line, (padding, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    # Stack the original image and the text bar
    result = np.vstack([image, bar])
    return result


def get_side_by_side_output_dir(dataset_name: str, model_name: str, query_type: Optional[str]) -> Path:
    """
    Get the output directory for side-by-side comparison frames.
    
    Structure: frame_extraction_side_by_side/TASK/MODEL_VARIANT/[QUERY_TYPE | default]/
    
    Args:
        dataset_name: Name of the dataset (e.g., "rotationEval")
        model_name: Model name (not used for _real since side-by-side requires generated)
        query_type: Query type from metadata, or None for default
        
    Returns:
        Path to the output directory
    """
    query_folder = query_type if query_type and query_type != "default" else "default"
    return Path("frame_extraction_side_by_side") / dataset_name / model_name / query_folder


def create_side_by_side_comparison(
    gt_frames: dict,
    gen_frames: dict,
    query: KeyframeQuery,
    output_path: Path,
    episode: str,
    instance: str,
    variant: str,
) -> None:
    """
    Create a side-by-side comparison image with GT on left and generated on right.
    
    Adds debugging text labels showing player and frame index.
    If multiple frames, stacks them vertically.
    
    Args:
        gt_frames: Dict of {suffix: bytes} for ground-truth frames
        gen_frames: Dict of {suffix: bytes} for generated frames
        query: The KeyframeQuery for metadata
        output_path: Directory to save the comparison image
        episode: Episode number
        instance: Instance number
        variant: Variant (alpha/bravo/turn_to_look/etc.)
    """
    import cv2
    import numpy as np
    
    meta = query.metadata
    
    def add_label(frame: np.ndarray, label: str) -> np.ndarray:
        """Add a text label to the top-left of a frame."""
        frame = frame.copy()
        # Add semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (len(label) * 10 + 10, 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        # Add text
        cv2.putText(frame, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
    
    def bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
        """Convert PNG bytes to CV2 image."""
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def pad_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple:
        """Pad images with zeros to match dimensions (for hstack/vstack compatibility)."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Pad height to match the taller image
        max_h = max(h1, h2)
        if h1 < max_h:
            pad_top = (max_h - h1) // 2
            pad_bottom = max_h - h1 - pad_top
            img1 = cv2.copyMakeBorder(img1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if h2 < max_h:
            pad_top = (max_h - h2) // 2
            pad_bottom = max_h - h2 - pad_top
            img2 = cv2.copyMakeBorder(img2, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Pad width to match the wider image
        max_w = max(w1, w2)
        if w1 < max_w:
            pad_left = (max_w - w1) // 2
            pad_right = max_w - w1 - pad_left
            img1 = cv2.copyMakeBorder(img1, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if w2 < max_w:
            pad_left = (max_w - w2) // 2
            pad_right = max_w - w2 - pad_left
            img2 = cv2.copyMakeBorder(img2, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return img1, img2
    
    # Determine frame info for labels
    # Note: is_turn_to_look is optional (defaults to False if not present)
    is_turn_to_look = meta.get('is_turn_to_look', False)
    frame1_idx = meta['frame1']  # Required field
    
    # Build list of (gt_frame, gen_frame, label) tuples
    frame_pairs = []
    
    if is_turn_to_look:
        # Turn to look: alpha and bravo perspectives
        # These fields are required for turn_to_look handlers
        alpha_frame_idx = meta['alpha_frame']
        bravo_frame_idx = meta['bravo_frame']
        
        frame_pairs.append((
            gt_frames["alpha_frame"],
            gen_frames["alpha_frame"],
            f"Alpha @ frame {alpha_frame_idx}"
        ))
        frame_pairs.append((
            gt_frames["bravo_frame"],
            gen_frames["bravo_frame"],
            f"Bravo @ frame {bravo_frame_idx}"
        ))
    elif "frame1" in gt_frames and "frame2" in gt_frames:
        # Translation: two frames (frame2 stored in query.second_frame_index)
        frame2_idx = query.second_frame_index
        frame_pairs.append((
            gt_frames["frame1"],
            gen_frames["frame1"],
            f"{variant.capitalize()} @ frame {frame1_idx}"
        ))
        frame_pairs.append((
            gt_frames["frame2"],
            gen_frames["frame2"],
            f"{variant.capitalize()} @ frame {frame2_idx}"
        ))
    else:
        # Single frame
        frame_idx = query.frame_index
        frame_pairs.append((
            gt_frames["frame"],
            gen_frames["frame"],
            f"{variant.capitalize()} @ frame {frame_idx}"
        ))
    
    # Create comparison images for each frame pair
    comparison_rows = []
    for gt_bytes, gen_bytes, label in frame_pairs:
        gt_img = bytes_to_cv2(gt_bytes)
        gen_img = bytes_to_cv2(gen_bytes)
        
        # Add labels
        gt_labeled = add_label(gt_img, f"GT: {label}")
        gen_labeled = add_label(gen_img, f"Gen: {label}")
        
        # Pad to match dimensions (GT may be 360p, generated may be 352p)
        gt_labeled, gen_labeled = pad_to_match(gt_labeled, gen_labeled)
        
        # Combine horizontally (GT on left, Gen on right)
        row = np.hstack([gt_labeled, gen_labeled])
        comparison_rows.append(row)
    
    # Stack rows vertically if multiple frames
    if len(comparison_rows) > 1:
        comparison = np.vstack(comparison_rows)
    else:
        comparison = comparison_rows[0]
    
    # Add expected answer text at the bottom
    expected_answer = query.expected_answer
    if expected_answer:
        comparison = add_expected_answer_label(comparison, expected_answer)
    
    # Save the comparison image
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"ep{episode}_inst{instance}_{variant}_comparison.png"
    cv2.imwrite(str(output_path / filename), comparison)
