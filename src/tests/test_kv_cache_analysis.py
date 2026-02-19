"""
Tests for KV cache rolling behavior in Matrix Game models.

This module verifies that the KV cache implementation correctly implements
a sliding window attention during inference, matching the training behavior.

Uses get_token_info pattern to give every token a unique semantic identity
(frame, player, spatial position) for robust verification.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.models.kv_cache import KVCache
from src.utils.tpu.splash_attn import CausalBlockMask

# =============================================================================
# Configuration
# =============================================================================

# Match defaults from test_attention_mask_implementations.py
DEFAULT_SPATIAL_SIZE = 880  # Patches per frame per player
DEFAULT_WINDOW_SIZE = 6  # Number of frames in sliding window
DEFAULT_NUM_FRAMES = 9  # Number of frames
DEFAULT_NUM_PLAYERS = 2  # Number of players for multiplayer tests

# Smaller sizes for dense tests that check every token pair
SMALL_SPATIAL_SIZE = 8  # For exhaustive token-pair tests
SMALL_NUM_FRAMES = 5  # For exhaustive token-pair tests


# =============================================================================
# Helper functions
# =============================================================================


def compute_block_size(spatial_size, num_players):
    """Compute the correct block_size for attention masks.

    CRITICAL: block_size must account for ALL players within a frame.
    A "frame block" contains tokens from all players, so:
        block_size = spatial_size * num_players

    Using just spatial_size (without * num_players) is a common bug that
    causes incorrect attention patterns in multiplayer mode.
    """
    return spatial_size * num_players


def get_token_info(
    token_idx,
    num_frames,
    num_players,
    spatial_size,
):
    """Get frame, player, and spatial position for a token index.

    Token layout: tokens ordered as (f, p, s) - frame, player, spatial
    Total tokens = num_frames * num_players * spatial_size

    Args:
        token_idx: Flat index into the token sequence
        num_frames: Number of frames (used for bounds validation)
        num_players: Number of players (1 for single-player, 2 for multiplayer)
        spatial_size: Number of spatial positions per player per frame

    Returns:
        dict with keys: frame, player, spatial, tokens_per_frame
    """
    tokens_per_player = spatial_size
    tokens_per_frame = num_players * tokens_per_player
    total_tokens = num_frames * tokens_per_frame

    # Validate bounds
    assert (
        0 <= token_idx < total_tokens
    ), f"token_idx {token_idx} out of bounds [0, {total_tokens})"

    frame = token_idx // tokens_per_frame
    within_frame = token_idx % tokens_per_frame
    player = within_frame // tokens_per_player
    spatial = within_frame % tokens_per_player

    return {
        "frame": frame,
        "player": player,
        "spatial": spatial,
        "tokens_per_frame": tokens_per_frame,
    }


def token_idx_from_info(
    frame,
    player,
    spatial,
    num_players,
    spatial_size,
):
    """Compute flat token index from semantic info. Inverse of get_token_info."""
    tokens_per_player = spatial_size
    tokens_per_frame = num_players * tokens_per_player
    return frame * tokens_per_frame + player * tokens_per_player + spatial


def expected_causal_attention(
    q_info,
    k_info,
    window_size,
):
    """Compute expected attention value for causal sliding window.

    Rules:
    - Query at frame i can attend to key at frame j if:
      1. j <= i (causal: can only attend to past/current)
      2. j > i - window_size (sliding window: only recent frames)
    - Within the same frame, all tokens can attend to each other
    """
    q_frame = q_info["frame"]
    k_frame = k_info["frame"]

    # Causal constraint: can only attend to past or current frame
    if k_frame > q_frame:
        return False

    # Sliding window constraint: only attend to recent frames
    if k_frame <= q_frame - window_size:
        return False

    return True


def create_causal_mask(
    num_frames,
    num_players,
    spatial_size,
    window_size,
):
    """Create a causal block mask using splash attention implementation."""
    block_size = compute_block_size(spatial_size, num_players)
    total_len = num_frames * block_size

    mask_obj = CausalBlockMask(
        shape=(total_len, total_len),
        block_size=block_size,
        window_size=window_size,
    )
    q_ids = jnp.arange(total_len)[:, None]
    kv_ids = jnp.arange(total_len)[None, :]
    return np.asarray(mask_obj.mask_function(q_ids, kv_ids))


def get_attended_frames_from_mask(
    mask,
    q_frame,
    num_frames,
    num_players,
    spatial_size,
):
    """Extract which frames a query frame attends to from a mask.

    Uses get_token_info to properly identify frame boundaries.
    """
    block_size = compute_block_size(spatial_size, num_players)
    total_len = num_frames * block_size

    # Verify shape
    assert mask.shape == (
        total_len,
        total_len,
    ), f"Mask shape {mask.shape} doesn't match expected ({total_len}, {total_len})"

    # Get first token of query frame
    q_token = token_idx_from_info(q_frame, 0, 0, num_players, spatial_size)

    # Check which frames are attended
    attended_frames = []
    for k_frame in range(num_frames):
        k_token = token_idx_from_info(k_frame, 0, 0, num_players, spatial_size)
        if mask[q_token, k_token]:
            attended_frames.append(k_frame)

    return attended_frames


def get_frames_in_cache(
    cache,
    window_size,
    num_players,
    spatial_size,
):
    """Extract which frames are currently in the KV cache.

    The cache stores tagged values where cache.k[b, t, h, d] = frame_idx + 1
    for all tokens in that frame. We use get_token_info to find frame boundaries.
    """
    block_size = compute_block_size(spatial_size, num_players)
    kv_len = cache.k.shape[1]
    valid_start = kv_len - int(cache.length)

    # Verify cache shape is consistent with block_size
    assert (
        kv_len % block_size == 0
    ), f"Cache length {kv_len} is not divisible by block_size {block_size}"

    frames_in_cache = []
    for i in range(window_size):
        token_idx = valid_start + i * block_size
        if 0 <= token_idx < kv_len:
            # Read the tagged frame value
            frame_value = int(cache.k[0, token_idx, 0, 0])
            if frame_value > 0:
                frames_in_cache.append(frame_value - 1)  # Convert to 0-indexed

    return frames_in_cache


# =============================================================================
# Test: KV Cache Rolling Behavior
# =============================================================================


class TestKVCacheRollingBehavior:
    """Tests for WanKVCache rolling buffer behavior."""

    @pytest.mark.parametrize("num_players", [1, 2])
    @pytest.mark.parametrize("spatial_size", [1, DEFAULT_SPATIAL_SIZE])
    def test_rolling_buffer_fills_correctly(self, num_players, spatial_size):
        """
        Test that WanKVCache rolling buffer fills and rolls correctly.

        The KV cache is a rolling buffer where:
        - kv_cache_size = self.k.shape[1] (fixed at initialization)
        - update() concatenates new KV pairs, then keeps only the last kv_cache_size
        - length tracks how many positions are valid (up to kv_cache_size)

        spatial_size=1 covers mouse/keyboard tokens (block_size=1 for single player).
        """
        window_size = DEFAULT_WINDOW_SIZE
        num_test_frames = 10

        block_size = compute_block_size(spatial_size, num_players)
        kv_cache_size = window_size * block_size

        batch_size = 1
        num_heads = 4
        head_dim = 16

        # Verify shape calculation
        assert kv_cache_size == window_size * spatial_size * num_players, (
            f"Cache size mismatch: {kv_cache_size} != "
            f"{window_size} * {spatial_size} * {num_players}"
        )

        # Initialize empty cache
        cache = KVCache(
            k=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            v=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            length=0,
        )

        assert cache.length == 0
        assert cache.k.shape == (batch_size, kv_cache_size, num_heads, head_dim)

        # Simulate adding frames one by one
        for frame_idx in range(num_test_frames):
            # Tag each frame's tokens with frame_idx + 1 (so 0 means empty)
            new_k = jnp.ones((batch_size, block_size, num_heads, head_dim)) * (
                frame_idx + 1
            )
            new_v = jnp.ones((batch_size, block_size, num_heads, head_dim)) * (
                frame_idx + 1
            )

            cache = cache.update(new_k, new_v)

            # Verify length is correct
            expected_length = min((frame_idx + 1) * block_size, kv_cache_size)
            assert (
                int(cache.length) == expected_length
            ), f"Frame {frame_idx}: length {cache.length} != expected {expected_length}"

            # Extract frames in cache using helper
            frames_in_cache = get_frames_in_cache(
                cache, window_size, num_players, spatial_size
            )

            # Expected frames: [max(0, frame_idx - window_size + 1), frame_idx]
            expected_frames = list(
                range(max(0, frame_idx - window_size + 1), frame_idx + 1)
            )

            assert frames_in_cache == expected_frames, (
                f"Frame {frame_idx} ({num_players}P): "
                f"cache has {frames_in_cache}, expected {expected_frames}"
            )

    @pytest.mark.parametrize("num_players", [1, 2])
    @pytest.mark.parametrize("spatial_size", [1, DEFAULT_SPATIAL_SIZE])
    def test_buffer_full_at_window_size(self, num_players, spatial_size):
        """Test that buffer becomes full after window_size frames."""
        window_size = DEFAULT_WINDOW_SIZE

        block_size = compute_block_size(spatial_size, num_players)
        kv_cache_size = window_size * block_size

        batch_size = 1
        num_heads = 4
        head_dim = 16

        cache = KVCache(
            k=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            v=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            length=0,
        )

        # Add exactly window_size frames
        for _ in range(window_size):
            new_k = jnp.ones((batch_size, block_size, num_heads, head_dim))
            new_v = jnp.ones((batch_size, block_size, num_heads, head_dim))
            cache = cache.update(new_k, new_v)

        # Buffer should be exactly full
        assert int(cache.length) == kv_cache_size

        # Add one more frame - length should stay at max
        new_k = jnp.ones((batch_size, block_size, num_heads, head_dim))
        new_v = jnp.ones((batch_size, block_size, num_heads, head_dim))
        cache = cache.update(new_k, new_v)

        assert int(cache.length) == kv_cache_size


# =============================================================================
# Test: Inference Masking
# =============================================================================


class TestInferenceMasking:
    """Tests for inference masking behavior."""

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_mask_evolution_as_buffer_fills(self, num_players):
        """
        Test that the validity mask correctly evolves as the buffer fills.

        Mask formula: mask[i] = (i >= kv_len - length)
        - positions [0, threshold) are False (masked)
        - positions [threshold, kv_len) are True (valid)
        where threshold = kv_len - length
        """
        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE

        block_size = compute_block_size(spatial_size, num_players)
        kv_len = window_size * block_size

        for frame_idx in range(8):
            length = min((frame_idx + 1) * block_size, kv_len)
            threshold = kv_len - length

            # Create the actual mask (same formula used in inference)
            mask = jnp.arange(kv_len) >= threshold

            # Verify first True position is at threshold
            if threshold < kv_len:
                first_true_idx = int(jnp.argmax(mask))
                assert (
                    first_true_idx == threshold
                ), f"Frame {frame_idx}: First True at {first_true_idx}, expected {threshold}"

            # Verify all positions before threshold are False
            if threshold > 0:
                assert not mask[
                    :threshold
                ].any(), (
                    f"Frame {frame_idx}: Positions before threshold should be False"
                )

            # Verify all positions from threshold onwards are True
            assert mask[
                threshold:
            ].all(), f"Frame {frame_idx}: Positions from threshold should be True"

            # Verify number of valid positions equals number of tokens for filled frames
            num_valid = int(mask.sum())
            assert (
                num_valid == length
            ), f"Frame {frame_idx}: {num_valid} valid positions, expected {length}"

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_mask_all_true_when_buffer_full(self, num_players):
        """Test that mask is all True when buffer is full."""
        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE

        block_size = compute_block_size(spatial_size, num_players)
        kv_len = window_size * block_size

        # When buffer is full, length == kv_len
        length = kv_len
        threshold = kv_len - length  # = 0

        mask = jnp.arange(kv_len) >= threshold

        assert mask.all(), "Mask should be all True when buffer is full"


# =============================================================================
# Test: Training/Inference Consistency
# =============================================================================


class TestTrainingInferenceConsistency:
    """Tests verifying training and inference attention patterns match.

    These tests verify that the frames attended to during training (via CausalBlockMask)
    match the frames available in the KV cache during inference (via WanKVCache).

    We use get_token_info to give each token a unique semantic identity and verify
    attention patterns at the token level, not just frame level.
    """

    @pytest.mark.parametrize("num_players", [1, 2])
    @pytest.mark.parametrize("spatial_size", [1, DEFAULT_SPATIAL_SIZE])
    def test_sliding_window_frames_match(self, num_players, spatial_size):
        """
        Verify that training sliding window attention matches inference KV cache.

        Training (CausalBlockMask):
        - Query at frame i attends to frames [max(0, i-window_size+1), i]

        Inference (WanKVCache):
        - Rolling buffer holds exactly window_size frames worth of KV pairs
        - When generating frame i, cache contains frames [max(0, i-window_size+1), i]

        spatial_size=1 covers mouse/keyboard tokens (block_size=1 for single player).
        """
        window_size = DEFAULT_WINDOW_SIZE
        num_test_frames = 12

        block_size = compute_block_size(spatial_size, num_players)
        kv_cache_size = window_size * block_size

        # Create training mask
        training_mask = create_causal_mask(
            num_test_frames, num_players, spatial_size, window_size
        )

        # Setup inference KV cache
        batch_size = 1
        num_heads = 1
        head_dim = 1

        cache = KVCache(
            k=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            v=jnp.zeros((batch_size, kv_cache_size, num_heads, head_dim)),
            length=0,
        )

        for q_frame in range(num_test_frames):
            # TRAINING: Extract attended frames from mask
            training_attended = get_attended_frames_from_mask(
                training_mask, q_frame, num_test_frames, num_players, spatial_size
            )

            # INFERENCE: Update cache and extract frames
            new_k = jnp.ones((batch_size, block_size, num_heads, head_dim)) * (
                q_frame + 1
            )
            new_v = jnp.ones((batch_size, block_size, num_heads, head_dim)) * (
                q_frame + 1
            )
            cache = cache.update(new_k, new_v)

            inference_frames = get_frames_in_cache(
                cache, window_size, num_players, spatial_size
            )

            # Verify match
            assert training_attended == inference_frames, (
                f"Frame {q_frame} ({num_players}P): "
                f"Training attends to {training_attended}, "
                f"inference has {inference_frames}"
            )

    @pytest.mark.parametrize("num_players", [1, 2])
    @pytest.mark.parametrize("spatial_size", [1, SMALL_SPATIAL_SIZE])
    def test_all_token_pairs_have_correct_attention(self, num_players, spatial_size):
        """
        Verify attention is correct for ALL token pairs, not just frame representatives.

        This is the most thorough test - it checks every (query_token, key_token) pair
        and verifies the attention value matches expected_causal_attention().

        spatial_size=1 covers mouse/keyboard tokens (block_size=1 for single player).
        """
        window_size = 3
        num_frames = SMALL_NUM_FRAMES

        block_size = compute_block_size(spatial_size, num_players)
        total_tokens = num_frames * block_size

        # Create mask
        mask = create_causal_mask(num_frames, num_players, spatial_size, window_size)

        # Verify shape
        assert mask.shape == (
            total_tokens,
            total_tokens,
        ), f"Mask shape {mask.shape} != expected ({total_tokens}, {total_tokens})"

        # Check every token pair
        errors = []
        for q_idx in range(total_tokens):
            q_info = get_token_info(q_idx, num_frames, num_players, spatial_size)

            for k_idx in range(total_tokens):
                k_info = get_token_info(k_idx, num_frames, num_players, spatial_size)

                expected = expected_causal_attention(q_info, k_info, window_size)
                actual = bool(mask[q_idx, k_idx])

                if actual != expected:
                    errors.append(
                        {
                            "q_idx": q_idx,
                            "k_idx": k_idx,
                            "q_info": q_info,
                            "k_info": k_info,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

        if errors:
            # Report first few errors for debugging
            error_msg = f"Found {len(errors)} attention errors ({num_players}P):\n"
            for e in errors[:5]:
                error_msg += (
                    f"  q[{e['q_idx']}]=f{e['q_info']['frame']}_p{e['q_info']['player']} -> "
                    f"k[{e['k_idx']}]=f{e['k_info']['frame']}_p{e['k_info']['player']}: "
                    f"expected={e['expected']}, got={e['actual']}\n"
                )
            if len(errors) > 5:
                error_msg += f"  ... and {len(errors) - 5} more errors"
            pytest.fail(error_msg)

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_cross_player_attention(self, num_players):
        """
        Verify cross-player attention patterns are correct for all frame relationships.

        This tests three key properties:
        1. Same frame: All players can attend to each other
        2. Past frames: All players can attend to all players in past frames (within window)
        3. Future frames: NO player can attend to ANY player in future frames

        This is a regression test for the block_size bug where using
        block_size=spatial_size instead of block_size=spatial_size*num_players
        incorrectly blocks cross-player attention within the same frame.
        """
        if num_players == 1:
            pytest.skip("Cross-player test only applies to multiplayer")

        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE
        num_frames = 5  # Enough frames to test window boundary

        mask = create_causal_mask(num_frames, num_players, spatial_size, window_size)

        # Sample a few spatial positions to keep test fast
        spatial_samples = list(range(min(3, spatial_size)))

        for q_frame in range(num_frames):
            for k_frame in range(num_frames):
                for q_player in range(num_players):
                    for k_player in range(num_players):
                        for q_spatial in spatial_samples:
                            for k_spatial in spatial_samples:
                                q_idx = token_idx_from_info(
                                    q_frame,
                                    q_player,
                                    q_spatial,
                                    num_players,
                                    spatial_size,
                                )
                                k_idx = token_idx_from_info(
                                    k_frame,
                                    k_player,
                                    k_spatial,
                                    num_players,
                                    spatial_size,
                                )

                                actual = bool(mask[q_idx, k_idx])

                                # Compute expected based on causal sliding window rules
                                if k_frame > q_frame:
                                    # Future frame: NEVER attend
                                    expected = False
                                    case = "future"
                                elif k_frame <= q_frame - window_size:
                                    # Past frame outside window: NEVER attend
                                    expected = False
                                    case = "outside_window"
                                else:
                                    # Same frame or past frame within window: ALWAYS attend
                                    expected = True
                                    case = "same_or_past"

                                assert actual == expected, (
                                    f"Cross-player attention error ({case}):\n"
                                    f"  q: frame={q_frame}, player={q_player}\n"
                                    f"  k: frame={k_frame}, player={k_player}\n"
                                    f"  expected={expected}, got={actual}"
                                )


# =============================================================================
# Test: Cache Sizes
# =============================================================================


class TestCacheSizes:
    """Tests verifying KV cache sizes are correct."""

    def test_single_player_cache_size(self):
        """Test single-player cache size calculation."""
        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE
        num_players = 1

        block_size = compute_block_size(spatial_size, num_players)
        cache_size = window_size * block_size

        # Verify block_size = spatial_size for single player
        assert (
            block_size == spatial_size
        ), f"SP block_size {block_size} != spatial_size {spatial_size}"

        # Verify cache holds exactly window_size frames
        assert (
            cache_size // block_size == window_size
        ), f"Cache holds {cache_size // block_size} frames, expected {window_size}"

    def test_multiplayer_cache_size(self):
        """Test multiplayer cache size calculation."""
        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE
        num_players = 2

        block_size = compute_block_size(spatial_size, num_players)
        cache_size = window_size * block_size

        # Verify block_size = spatial_size * num_players for multiplayer
        assert (
            block_size == spatial_size * num_players
        ), f"MP block_size {block_size} != {spatial_size} * {num_players}"

        # Verify cache holds exactly window_size frames
        assert (
            cache_size // block_size == window_size
        ), f"Cache holds {cache_size // block_size} frames, expected {window_size}"

    @pytest.mark.parametrize("num_players", [1, 2])
    def test_cache_holds_exact_window_frames(self, num_players):
        """Test that cache sizes allow exactly window_size frames."""
        spatial_size = DEFAULT_SPATIAL_SIZE
        window_size = DEFAULT_WINDOW_SIZE

        block_size = compute_block_size(spatial_size, num_players)
        cache_size = window_size * block_size

        # Number of frames that fit in cache
        frames_in_cache = cache_size // block_size

        assert (
            frames_in_cache == window_size
        ), f"{num_players}P: Cache fits {frames_in_cache} frames, expected {window_size}"

        # Verify no remainder (cache perfectly fits window_size frames)
        assert (
            cache_size % block_size == 0
        ), f"{num_players}P: Cache size {cache_size} not divisible by block_size {block_size}"
