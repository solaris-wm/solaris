"""
Tests for attention mask implementations.

These tests verify that:
1. Splash attention masks match the model helper masks
2. Attention patterns are correct for all modes (causal, teacher-forcing)
3. Multiplayer block_size is handled correctly (block_size = spatial_size * num_players)

The tests are parameterized over:
- num_players: 1 (single player) or 2 (multiplayer)
- teacher_forcing: True or False
- Various frame counts and window sizes
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.models.multiplayer.world_model import SolarisMPModel
from src.models.singleplayer.world_model import SolarisSPModel
from src.utils.tpu.splash_attn import CausalBlockMask, TeacherForcingBlockMask

# =============================================================================
# Test configuration defaults
# =============================================================================
DEFAULT_SPATIAL_SIZE = 880  # Patches per frame per player.
DEFAULT_WINDOW_SIZE = 6  # Sliding window size in blocks.
DEFAULT_NUM_FRAMES = 9  # Number of frames.
DEFAULT_NUM_PLAYERS = 2  # Number of players for multiplayer tests.


# =============================================================================
# Helper functions
# =============================================================================


def get_model_class(num_players):
    """Get the appropriate model class based on player count."""
    if num_players == 1:
        return SolarisSPModel
    else:
        return SolarisMPModel


def compute_block_size(spatial_size, num_players):
    """Compute the correct block_size for attention masks.

    CRITICAL: block_size must account for ALL players within a frame.
    A "frame block" contains tokens from all players, so:
        block_size = spatial_size * num_players

    Using just spatial_size (without * num_players) is a common bug that
    causes incorrect attention patterns in multiplayer mode.
    """
    return spatial_size * num_players


def create_splash_causal_mask(total_len, block_size, window_size):
    """Create a causal block mask using splash attention implementation."""
    mask_obj = CausalBlockMask(
        shape=(total_len, total_len), block_size=block_size, window_size=window_size
    )
    q_ids = jnp.arange(total_len)[:, None]
    kv_ids = jnp.arange(total_len)[None, :]
    return np.asarray(mask_obj.mask_function(q_ids, kv_ids))


def create_splash_teacher_forcing_mask(total_len, clean_len, block_size, window_size):
    """Create a teacher-forcing block mask using splash attention implementation."""
    mask_obj = TeacherForcingBlockMask(
        shape=(total_len, total_len),
        block_size=block_size,
        seq_len=clean_len,
        window_size=window_size,
    )
    q_ids = jnp.arange(total_len)[:, None]
    kv_ids = jnp.arange(total_len)[None, :]
    return np.asarray(mask_obj.mask_function(q_ids, kv_ids))


def create_model_causal_mask(num_frames, block_size, window_size, num_players):
    """Create a causal block mask using model helper."""
    model_class = get_model_class(num_players)
    return np.asarray(
        model_class.get_causal_attn_mask(
            num_q_blocks=num_frames,
            num_k_blocks=num_frames,
            q_block_size=block_size,
            k_block_size=block_size,
            sliding_block_size=window_size,
        )
    )


def create_model_teacher_forcing_mask(num_frames, block_size, window_size, num_players):
    """Create a teacher-forcing block mask using model helper."""
    model_class = get_model_class(num_players)
    return np.asarray(
        model_class.get_block_mask_teacher_forcing(
            num_q_blocks=num_frames,
            num_k_blocks=num_frames,
            q_block_size=block_size,
            k_block_size=block_size,
            sliding_block_size=window_size,
        )
    )


def get_token_info(
    token_idx,
    num_frames,
    num_players,
    spatial_size,
    teacher_forcing,
):
    """Get frame, player, and role info for a token index.

    Token layout:
    - Causal mode: tokens ordered as (f, p, s) - frame, player, spatial
    - Teacher forcing: [clean tokens | noisy tokens], each ordered as (f, p, s)
    """
    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = num_players * tokens_per_player_frame
    clean_len = num_frames * tokens_per_frame_block

    if teacher_forcing:
        is_noisy = token_idx >= clean_len
        offset_idx = token_idx - clean_len if is_noisy else token_idx
    else:
        is_noisy = False
        offset_idx = token_idx

    frame = offset_idx // tokens_per_frame_block
    within_frame = offset_idx % tokens_per_frame_block
    player = within_frame // tokens_per_player_frame
    spatial = within_frame % tokens_per_player_frame

    return {
        "frame": frame,
        "player": player,
        "spatial": spatial,
        "is_noisy": is_noisy,
    }


def token_idx_from_info(
    frame,
    player,
    spatial,
    num_frames,
    num_players,
    spatial_size,
    is_noisy=False,
):
    """Compute flat token index from semantic info. Inverse of get_token_info.

    Args:
        frame: Frame index (0-indexed)
        player: Player index (0 for single player, 0 or 1 for multiplayer)
        spatial: Spatial position within player's frame (0 to spatial_size-1)
        num_frames: Total number of frames
        num_players: Number of players (1 or 2)
        spatial_size: Number of spatial positions per player per frame
        is_noisy: If True, returns index in noisy section (for teacher forcing)

    Returns:
        Flat token index into the sequence
    """
    tokens_per_player = spatial_size
    tokens_per_frame = num_players * tokens_per_player
    clean_len = num_frames * tokens_per_frame

    # Compute index within clean/noisy section
    idx = frame * tokens_per_frame + player * tokens_per_player + spatial

    # Add offset for noisy tokens
    if is_noisy:
        idx += clean_len

    return idx


def expected_attention(q_info, k_info, window_size, teacher_forcing):
    """Compute expected attention value based on attention rules.

    Causal mode:
    - Token at frame i can attend to tokens at frames <= i (within window)

    Teacher forcing mode:
    - Clean tokens attend causally to clean tokens (frame-level)
    - Noisy tokens attend to previous clean frames AND same noisy frame only
    - Clean tokens do NOT attend to noisy tokens
    """
    frame_diff = q_info["frame"] - k_info["frame"]
    within_window = frame_diff < window_size

    if not teacher_forcing:
        # Causal: attend to same or earlier frames within window
        return (k_info["frame"] <= q_info["frame"]) and within_window

    # Teacher forcing mode
    q_noisy = q_info["is_noisy"]
    k_noisy = k_info["is_noisy"]

    if not q_noisy and not k_noisy:
        # Clean -> Clean: causal at frame level
        return (k_info["frame"] <= q_info["frame"]) and within_window
    elif q_noisy and not k_noisy:
        # Noisy -> Clean: strictly previous frames only
        return (k_info["frame"] < q_info["frame"]) and within_window
    elif q_noisy and k_noisy:
        # Noisy -> Noisy: same frame only
        return k_info["frame"] == q_info["frame"]
    else:
        # Clean -> Noisy: never allowed
        return False


# =============================================================================
# Test: Splash masks match model helper masks
# =============================================================================


@pytest.mark.parametrize("num_players", [1, 2])
@pytest.mark.parametrize(
    "num_frames,window_size",
    [
        (DEFAULT_NUM_FRAMES, DEFAULT_WINDOW_SIZE),
        (DEFAULT_NUM_FRAMES + 1, 3),  # Sliding window smaller than history
    ],
)
def test_causal_mask_splash_matches_model(num_players, num_frames, window_size):
    """Test that splash causal mask matches model helper for SP and MP."""
    spatial_size = DEFAULT_SPATIAL_SIZE
    block_size = compute_block_size(spatial_size, num_players)
    total_len = num_frames * block_size

    splash_mask = create_splash_causal_mask(total_len, block_size, window_size)
    model_mask = create_model_causal_mask(
        num_frames, block_size, window_size, num_players
    )

    assert (
        splash_mask.shape == model_mask.shape
    ), f"Shape mismatch: splash={splash_mask.shape}, model={model_mask.shape}"
    assert np.array_equal(splash_mask, model_mask), (
        f"Causal mask mismatch for num_players={num_players}, "
        f"num_frames={num_frames}, window_size={window_size}"
    )


@pytest.mark.parametrize("num_players", [1, 2])
@pytest.mark.parametrize(
    "num_frames,window_size",
    [
        (DEFAULT_NUM_FRAMES, DEFAULT_WINDOW_SIZE),
        (DEFAULT_NUM_FRAMES + 1, 3),
    ],
)
def test_teacher_forcing_mask_splash_matches_model(
    num_players, num_frames, window_size
):
    """Test that splash TF mask matches model helper for SP and MP."""
    spatial_size = DEFAULT_SPATIAL_SIZE
    block_size = compute_block_size(spatial_size, num_players)
    clean_len = num_frames * block_size
    total_len = clean_len * 2

    splash_mask = create_splash_teacher_forcing_mask(
        total_len, clean_len, block_size, window_size
    )
    model_mask = create_model_teacher_forcing_mask(
        num_frames, block_size, window_size, num_players
    )

    assert (
        splash_mask.shape == model_mask.shape
    ), f"Shape mismatch: splash={splash_mask.shape}, model={model_mask.shape}"
    assert np.array_equal(splash_mask, model_mask), (
        f"Teacher forcing mask mismatch for num_players={num_players}, "
        f"num_frames={num_frames}, window_size={window_size}"
    )


# =============================================================================
# Test: Attention patterns are correct
# =============================================================================


@pytest.mark.parametrize("num_players", [1, 2])
@pytest.mark.parametrize("teacher_forcing", [False, True])
@pytest.mark.parametrize(
    "num_frames,spatial_size,window_size",
    [
        (3, 4, 6),  # Small, window larger than frames
        (4, 6, 6),  # Medium
        (5, 4, 2),  # Sliding window smaller than history
    ],
)
def test_attention_patterns_are_correct(
    num_players,
    teacher_forcing,
    num_frames,
    spatial_size,
    window_size,
):
    """Verify all attention patterns are correct for all mode combinations.

    This is the most comprehensive test - it checks every (query, key) pair
    against the expected attention rules.
    """
    block_size = compute_block_size(spatial_size, num_players)
    clean_len = num_frames * block_size
    total_len = clean_len * 2 if teacher_forcing else clean_len

    # Create the appropriate mask
    if teacher_forcing:
        mask = create_splash_teacher_forcing_mask(
            total_len, clean_len, block_size, window_size
        )
    else:
        mask = create_splash_causal_mask(total_len, block_size, window_size)

    # Verify every (q, k) pair
    for q_idx in range(total_len):
        q_info = get_token_info(
            q_idx, num_frames, num_players, spatial_size, teacher_forcing
        )

        for k_idx in range(total_len):
            k_info = get_token_info(
                k_idx, num_frames, num_players, spatial_size, teacher_forcing
            )

            expected = expected_attention(q_info, k_info, window_size, teacher_forcing)
            actual = bool(mask[q_idx, k_idx])

            if actual != expected:
                mode = "TF" if teacher_forcing else "causal"
                q_role = "noisy" if q_info["is_noisy"] else "clean"
                k_role = "noisy" if k_info["is_noisy"] else "clean"
                pytest.fail(
                    f"Attention pattern error ({mode}, {num_players}P):\n"
                    f"  q[{q_idx}] = {q_role}_f{q_info['frame']}_p{q_info['player']}\n"
                    f"  k[{k_idx}] = {k_role}_f{k_info['frame']}_p{k_info['player']}\n"
                    f"  expected={expected}, got={actual}"
                )


# =============================================================================
# Test: Multiplayer block_size bug detection
# =============================================================================


@pytest.mark.parametrize("teacher_forcing", [False, True])
def test_incorrect_block_size_produces_different_mask(teacher_forcing):
    """Verify that using wrong block_size (forgetting num_players) produces different results.

    This is a regression test for the bug where block_size was not multiplied
    by num_players in multiplayer mode. Specific incorrect patterns are verified
    in test_cross_player_attention_within_frame_causal and
    test_noisy_should_not_attend_to_same_frame_clean.
    """
    num_frames = DEFAULT_NUM_FRAMES
    spatial_size = DEFAULT_SPATIAL_SIZE
    num_players = DEFAULT_NUM_PLAYERS
    window_size = DEFAULT_WINDOW_SIZE

    correct_block_size = compute_block_size(spatial_size, num_players)
    buggy_block_size = spatial_size  # BUG: missing * num_players!

    clean_len_correct = num_frames * correct_block_size
    clean_len_buggy = num_frames * correct_block_size  # Same input shape
    total_len = clean_len_correct * 2 if teacher_forcing else clean_len_correct

    if teacher_forcing:
        correct_mask = create_splash_teacher_forcing_mask(
            total_len, clean_len_correct, correct_block_size, window_size
        )
        buggy_mask = create_splash_teacher_forcing_mask(
            total_len, clean_len_buggy, buggy_block_size, window_size
        )
    else:
        correct_mask = create_splash_causal_mask(
            total_len, correct_block_size, window_size
        )
        buggy_mask = create_splash_causal_mask(total_len, buggy_block_size, window_size)

    # Masks MUST be different when num_players > 1
    assert not np.array_equal(correct_mask, buggy_mask), (
        f"Correct and buggy masks should differ! "
        f"teacher_forcing={teacher_forcing}, num_players={num_players}"
    )


def test_cross_player_attention_within_frame_causal():
    """Test cross-player attention within the same frame (causal mode).

    With correct block_size: Player 0 can attend to Player 1 within same frame.
    With buggy block_size: Player 0 CANNOT attend to Player 1 (wrong block boundary).
    """
    num_frames = DEFAULT_NUM_FRAMES
    spatial_size = DEFAULT_SPATIAL_SIZE
    num_players = DEFAULT_NUM_PLAYERS
    window_size = DEFAULT_WINDOW_SIZE

    correct_block_size = compute_block_size(spatial_size, num_players)
    buggy_block_size = spatial_size
    total_len = num_frames * correct_block_size

    correct_mask = create_splash_causal_mask(total_len, correct_block_size, window_size)
    buggy_mask = create_splash_causal_mask(total_len, buggy_block_size, window_size)

    # Token indices for frame 0 using token_idx_from_info
    p0_token = token_idx_from_info(
        frame=0,
        player=0,
        spatial=0,
        num_frames=num_frames,
        num_players=num_players,
        spatial_size=spatial_size,
    )
    p1_token = token_idx_from_info(
        frame=0,
        player=1,
        spatial=0,
        num_frames=num_frames,
        num_players=num_players,
        spatial_size=spatial_size,
    )

    # Correct: both players can attend to each other within same frame
    assert (
        correct_mask[p0_token, p1_token] == True
    ), "Correct mask: Player 0 should attend to Player 1 in same frame"
    assert (
        correct_mask[p1_token, p0_token] == True
    ), "Correct mask: Player 1 should attend to Player 0 in same frame"

    # Buggy: Player 0 cannot attend to Player 1 (block boundary in wrong place)
    assert (
        buggy_mask[p0_token, p1_token] == False
    ), "Buggy mask should incorrectly block Player 0 -> Player 1"


def test_noisy_should_not_attend_to_same_frame_clean():
    """Test that noisy tokens do NOT attend to same-frame clean tokens.

    This is the critical teacher-forcing bug case:
    - noisy_f0_p1 -> clean_f0_p0 should be FALSE
    - With buggy block_size, it incorrectly returns TRUE
    """
    num_frames = DEFAULT_NUM_FRAMES
    spatial_size = DEFAULT_SPATIAL_SIZE
    num_players = DEFAULT_NUM_PLAYERS
    window_size = DEFAULT_WINDOW_SIZE

    correct_block_size = compute_block_size(spatial_size, num_players)
    buggy_block_size = spatial_size
    clean_len = num_frames * correct_block_size
    total_len = clean_len * 2

    correct_mask = create_splash_teacher_forcing_mask(
        total_len, clean_len, correct_block_size, window_size
    )
    buggy_mask = create_splash_teacher_forcing_mask(
        total_len, clean_len, buggy_block_size, window_size
    )

    # Helper to get token index
    def tok(frame, player, is_noisy=False):
        return token_idx_from_info(
            frame=frame,
            player=player,
            spatial=0,
            num_frames=num_frames,
            num_players=num_players,
            spatial_size=spatial_size,
            is_noisy=is_noisy,
        )

    # Token indices using token_idx_from_info
    clean_f0_p0 = tok(frame=0, player=0, is_noisy=False)
    clean_f0_p1 = tok(frame=0, player=1, is_noisy=False)
    noisy_f0_p0 = tok(frame=0, player=0, is_noisy=True)
    noisy_f0_p1 = tok(frame=0, player=1, is_noisy=True)

    # Noisy tokens should NOT attend to same-frame clean tokens
    assert (
        correct_mask[noisy_f0_p0, clean_f0_p0] == False
    ), "Noisy f0 p0 should NOT attend to clean f0 p0 (same frame)"
    assert (
        correct_mask[noisy_f0_p1, clean_f0_p0] == False
    ), "Noisy f0 p1 should NOT attend to clean f0 p0 (same frame)"
    assert (
        correct_mask[noisy_f0_p0, clean_f0_p1] == False
    ), "Noisy f0 p0 should NOT attend to clean f0 p1 (same frame)"
    assert (
        correct_mask[noisy_f0_p1, clean_f0_p1] == False
    ), "Noisy f0 p1 should NOT attend to clean f0 p1 (same frame)"

    # Buggy mask incorrectly allows cross-player same-frame attention
    # noisy_f0_p1 -> clean_f0_p0 is the smoking gun
    assert (
        buggy_mask[noisy_f0_p1, clean_f0_p0] == True
    ), "Buggy mask should incorrectly allow noisy_f0_p1 -> clean_f0_p0"

    # Verify noisy CAN attend to PREVIOUS frame clean tokens
    if num_frames > 1:
        noisy_f1_p0 = tok(frame=1, player=0, is_noisy=True)
        assert (
            correct_mask[noisy_f1_p0, clean_f0_p0] == True
        ), "Noisy f1 p0 SHOULD attend to clean f0 p0 (previous frame)"
        assert (
            correct_mask[noisy_f1_p0, clean_f0_p1] == True
        ), "Noisy f1 p0 SHOULD attend to clean f0 p1 (previous frame)"


# =============================================================================
# Test: Edge cases
# =============================================================================


@pytest.mark.parametrize("num_players", [1, 2])
def test_single_frame(num_players):
    """Test masks with only a single frame."""
    num_frames = 1
    spatial_size = DEFAULT_SPATIAL_SIZE
    window_size = DEFAULT_WINDOW_SIZE
    block_size = compute_block_size(spatial_size, num_players)

    # Causal: all tokens in single frame can attend to each other
    total_len = block_size
    mask = create_splash_causal_mask(total_len, block_size, window_size)
    assert np.all(mask), "Single frame causal: all tokens should attend to each other"

    # Teacher forcing: clean attends to clean, noisy attends to noisy only
    total_len_tf = block_size * 2
    mask_tf = create_splash_teacher_forcing_mask(
        total_len_tf, block_size, block_size, window_size
    )

    # Clean-clean quadrant: all True
    assert np.all(
        mask_tf[:block_size, :block_size]
    ), "Single frame TF: clean should fully attend to clean"
    # Clean-noisy quadrant: all False
    assert not np.any(
        mask_tf[:block_size, block_size:]
    ), "Single frame TF: clean should not attend to noisy"
    # Noisy-clean quadrant: all False (same frame, not previous)
    assert not np.any(
        mask_tf[block_size:, :block_size]
    ), "Single frame TF: noisy should not attend to same-frame clean"
    # Noisy-noisy quadrant: all True
    assert np.all(
        mask_tf[block_size:, block_size:]
    ), "Single frame TF: noisy should fully attend to noisy"


@pytest.mark.parametrize("num_players", [1, 2])
def test_window_size_one(num_players):
    """Test masks with window_size=1 (only current frame visible)."""
    num_frames = DEFAULT_NUM_FRAMES
    spatial_size = DEFAULT_SPATIAL_SIZE
    window_size = 1
    block_size = compute_block_size(spatial_size, num_players)
    total_len = num_frames * block_size

    mask = create_splash_causal_mask(total_len, block_size, window_size)

    # Helper to get frame boundaries using token_idx_from_info
    def frame_start(frame):
        return token_idx_from_info(
            frame=frame,
            player=0,
            spatial=0,
            num_frames=num_frames,
            num_players=num_players,
            spatial_size=spatial_size,
        )

    def frame_end(frame):
        # End is start of next frame, or total_len for last frame
        if frame == num_frames - 1:
            return total_len
        return frame_start(frame + 1)

    # Only diagonal blocks should be True (each frame only sees itself)
    for frame in range(num_frames):
        start = frame_start(frame)
        end = frame_end(frame)

        # Diagonal block should be all True
        assert np.all(
            mask[start:end, start:end]
        ), f"Frame {frame} should fully attend to itself"

        # Off-diagonal blocks in same row should be all False
        for other_frame in range(num_frames):
            if other_frame != frame:
                other_start = frame_start(other_frame)
                other_end = frame_end(other_frame)
                assert not np.any(mask[start:end, other_start:other_end]), (
                    f"Frame {frame} should not attend to frame {other_frame} "
                    f"with window_size=1"
                )


# =============================================================================
# Tests for block_size=1 (mouse/keyboard tokens)
# =============================================================================


@pytest.mark.parametrize("num_players", [1, 2])
def test_causal_mask_block_size_one(num_players):
    """Test causal mask with block_size=1 (mouse/keyboard tokens).

    Mouse and keyboard tokens have one token per frame (no spatial dimension),
    so block_size=1 regardless of num_players. The mask should be a simple
    lower-triangular sliding window at the frame level.
    """
    num_frames = DEFAULT_NUM_FRAMES
    window_size = DEFAULT_WINDOW_SIZE
    block_size = 1  # Mouse/keyboard: one token per frame

    model_class = get_model_class(num_players)
    mask = model_class.get_causal_attn_mask(
        num_q_blocks=num_frames,
        num_k_blocks=num_frames,
        q_block_size=block_size,
        k_block_size=block_size,
        sliding_block_size=window_size,
    )

    assert mask.shape == (num_frames, num_frames)

    # Build expected mask: frame i attends to frames [max(0, i-window_size+1), i]
    expected = np.zeros((num_frames, num_frames), dtype=bool)
    for i in range(num_frames):
        for j in range(num_frames):
            if j <= i and j > i - window_size:
                expected[i, j] = True

    assert np.array_equal(
        mask, expected
    ), "block_size=1 causal mask doesn't match expected sliding window pattern"


@pytest.mark.parametrize("num_players", [1, 2])
def test_teacher_forcing_mask_block_size_one(num_players):
    """Test teacher forcing mask with block_size=1 (mouse/keyboard tokens).

    With block_size=1, each frame has exactly one clean and one noisy token.
    Layout: [clean_0, clean_1, ..., clean_{n-1}, noisy_0, noisy_1, ..., noisy_{n-1}]
    """
    num_frames = DEFAULT_NUM_FRAMES
    window_size = DEFAULT_WINDOW_SIZE
    block_size = 1  # Mouse/keyboard: one token per frame

    model_class = get_model_class(num_players)
    mask = model_class.get_block_mask_teacher_forcing(
        num_q_blocks=num_frames,
        num_k_blocks=num_frames,
        q_block_size=block_size,
        k_block_size=block_size,
        sliding_block_size=window_size,
    )

    total_len = 2 * num_frames
    assert mask.shape == (total_len, total_len)

    # Build expected mask
    expected = np.zeros((total_len, total_len), dtype=bool)

    for q_idx in range(total_len):
        q_is_noisy = q_idx >= num_frames
        q_frame = q_idx % num_frames

        for k_idx in range(total_len):
            k_is_noisy = k_idx >= num_frames
            k_frame = k_idx % num_frames

            if not q_is_noisy and not k_is_noisy:
                # Clean -> clean: causal with sliding window
                if k_frame <= q_frame and k_frame > q_frame - window_size:
                    expected[q_idx, k_idx] = True
            elif q_is_noisy and not k_is_noisy:
                # Noisy -> clean: strictly past (no same frame)
                if k_frame < q_frame and k_frame > q_frame - window_size:
                    expected[q_idx, k_idx] = True
            elif q_is_noisy and k_is_noisy:
                # Noisy -> noisy: same frame only
                if k_frame == q_frame:
                    expected[q_idx, k_idx] = True
            # Clean -> noisy: always False

    assert np.array_equal(
        mask, expected
    ), "block_size=1 teacher forcing mask doesn't match expected pattern"
