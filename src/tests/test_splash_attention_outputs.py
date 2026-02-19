"""
Tests for splash attention output correctness on TPU.

This module tests that splash attention kernels produce numerically correct outputs
by comparing against jax.nn.dot_product_attention with equivalent masks.

Tests are parameterized over:
- num_players: 1 (single player) or 2 (multiplayer)
- Various spatial sizes and window sizes

Note: Tests that require actual splash attention kernels are marked with @requires_tpu
and will be skipped with a warning when TPU is not available.

Mask correctness tests are in test_attention_mask_implementations.py.
"""

import jax
import jax.numpy as jnp
import pytest

from src.tests.conftest import requires_tpu
from src.utils.tpu.splash_attn import (
    CausalBlockMask,
    TeacherForcingBlockMask,
    block_causal_splash_attn,
    full_splash_attn,
    teacher_forcing_block_causal_splash_attn,
)

# =============================================================================
# Configuration
# =============================================================================

# Match defaults from test_attention_mask_implementations.py
DEFAULT_SPATIAL_SIZE = 880  # Patches per frame per player
DEFAULT_WINDOW_SIZE = 6  # Number of frames in sliding window
DEFAULT_NUM_FRAMES = 9  # Number of frames


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def highest_precision():
    """Fixture to set JAX matmul precision to highest and restore after test.

    This is needed because splash attention tests fail without highest precision.
    """
    original_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_default_matmul_precision", "highest")
    yield
    jax.config.update("jax_default_matmul_precision", original_precision)


# =============================================================================
# Helper Functions
# =============================================================================


def print_matching_report(full_output_BTHD, ref_output_BTHD):
    """Print detailed numerical comparison between two attention outputs."""
    print(
        f"Max absolute difference: {jnp.max(jnp.abs(full_output_BTHD - ref_output_BTHD))}"
    )
    print(
        f"Max relative difference: {jnp.max(jnp.abs((full_output_BTHD - ref_output_BTHD) / (ref_output_BTHD + 1e-8)))}"
    )
    print(
        f"Median absolute difference: {jnp.median(jnp.abs(full_output_BTHD - ref_output_BTHD))}"
    )
    print(
        f"Median relative difference: {jnp.median(jnp.abs((full_output_BTHD - ref_output_BTHD) / (ref_output_BTHD + 1e-8)))}"
    )
    print(
        f"p99 relative difference: {jnp.percentile(jnp.abs((full_output_BTHD - ref_output_BTHD) / (ref_output_BTHD + 1e-8)), 99)}"
    )
    print(
        f"Are close (rtol=1e-4, atol=1e-4): {jnp.allclose(full_output_BTHD, ref_output_BTHD, rtol=1e-4, atol=1e-4)}"
    )


# =============================================================================
# Tests: Full (non-masked) splash attention
# =============================================================================


@requires_tpu
def test_splash_attn_fwd_numerical(highest_precision):
    """Test that full splash attention matches jax.nn.dot_product_attention numerically."""
    B, T, H, D = 1, 128, 8, 128

    q_BTHD = jax.random.normal(jax.random.PRNGKey(0), (B, T, H, D))
    k_BTHD = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, D))
    v_BTHD = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, D))

    full_output_BTHD = full_splash_attn(q_BTHD, k_BTHD, v_BTHD)
    ref_output_BTHD = jax.nn.dot_product_attention(q_BTHD, k_BTHD, v_BTHD)

    print("=== Full Splash Attention Forward Numerical Test ===")
    print_matching_report(full_output_BTHD, ref_output_BTHD)

    assert jnp.allclose(
        full_output_BTHD, ref_output_BTHD, rtol=1e-4, atol=1e-4
    ), "Full splash attention output does not match jax.nn.dot_product_attention"


# =============================================================================
# Tests: Causal block splash attention
# =============================================================================


@requires_tpu
def test_causal_block_attn_output(highest_precision):
    """Test that causal block splash attention matches reference with same mask."""
    from src.models.singleplayer.world_model import SolarisSPModel

    num_q_blocks, num_k_blocks = DEFAULT_NUM_FRAMES, DEFAULT_NUM_FRAMES
    q_block_size, k_block_size = DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE
    local_attn_size = DEFAULT_WINDOW_SIZE
    B, T, H, D = 1, num_q_blocks * q_block_size, 8, 128

    # Get mask from SolarisSPModel (reference implementation)
    ref_block_mask = SolarisSPModel.get_causal_attn_mask(
        num_q_blocks=num_q_blocks,
        num_k_blocks=num_k_blocks,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sliding_block_size=local_attn_size,
    )

    # Get mask from CausalBlockMask (splash_attn implementation)
    splash_mask_obj = CausalBlockMask(
        shape=(T, T), block_size=q_block_size, window_size=local_attn_size
    )
    q_ids = jnp.arange(T)[:, None]
    kv_ids = jnp.arange(T)[None, :]
    splash_block_mask = splash_mask_obj.mask_function(q_ids, kv_ids)

    # Exact mask comparison
    masks_match = jnp.array_equal(ref_block_mask, splash_block_mask)
    print("=== Causal Block Attention Mask Test ===")
    print(
        f"  Mask shapes: ref={ref_block_mask.shape}, splash={splash_block_mask.shape}"
    )
    print(f"  Masks exactly equal: {masks_match}")
    if not masks_match:
        diff_count = jnp.sum(ref_block_mask != splash_block_mask)
        print(f"  Number of differences: {diff_count}")
    assert (
        masks_match
    ), "CausalBlockMask does not match SolarisSPModel.get_causal_attn_mask!"

    # Output comparison
    q_BTHD = jax.random.normal(jax.random.PRNGKey(0), (B, T, H, D))
    k_BTHD = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, D))
    v_BTHD = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, D))

    o_ref_BTHD = jax.nn.dot_product_attention(
        q_BTHD, k_BTHD, v_BTHD, mask=ref_block_mask
    )
    o_BTHD = block_causal_splash_attn(
        q_BTHD, k_BTHD, v_BTHD, block_size=q_block_size, window_size=local_attn_size
    )

    print("=== Causal Block Attention Output Comparison ===")
    print_matching_report(o_ref_BTHD, o_BTHD)

    assert jnp.allclose(
        o_ref_BTHD, o_BTHD, rtol=1e-4, atol=1e-4
    ), "Causal block splash attention output does not match reference"


# =============================================================================
# Tests: Teacher forcing block splash attention
# =============================================================================


@requires_tpu
def test_teacher_forcing_block_attn_output(highest_precision):
    """Test that teacher forcing block splash attention matches reference with same mask."""
    from src.models.singleplayer.world_model import SolarisSPModel

    num_q_blocks, num_k_blocks = DEFAULT_NUM_FRAMES, DEFAULT_NUM_FRAMES
    q_block_size, k_block_size = DEFAULT_SPATIAL_SIZE, DEFAULT_SPATIAL_SIZE
    local_attn_size = DEFAULT_WINDOW_SIZE
    seq_len = num_q_blocks * q_block_size  # clean sequence length
    total_seq_len = seq_len * 2  # clean + noisy
    B, H, D = 1, 8, 128

    # Get mask from SolarisSPModel (reference implementation)
    ref_block_mask = SolarisSPModel.get_block_mask_teacher_forcing(
        num_q_blocks=num_q_blocks,
        num_k_blocks=num_k_blocks,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        sliding_block_size=local_attn_size,
    )

    # Get mask from TeacherForcingBlockMask (splash_attn implementation)
    splash_mask_obj = TeacherForcingBlockMask(
        shape=(total_seq_len, total_seq_len),
        block_size=q_block_size,
        seq_len=seq_len,
        window_size=local_attn_size,
    )
    q_ids = jnp.arange(total_seq_len)[:, None]
    kv_ids = jnp.arange(total_seq_len)[None, :]
    splash_block_mask = splash_mask_obj.mask_function(q_ids, kv_ids)

    # Exact mask comparison
    masks_match = jnp.array_equal(ref_block_mask, splash_block_mask)
    print("=== Teacher Forcing Block Attention Mask Test ===")
    print(
        f"  Mask shapes: ref={ref_block_mask.shape}, splash={splash_block_mask.shape}"
    )
    print(f"  Masks exactly equal: {masks_match}")
    if not masks_match:
        diff_count = jnp.sum(ref_block_mask != splash_block_mask)
        print(f"  Number of differences: {diff_count}")
    assert (
        masks_match
    ), "TeacherForcingBlockMask does not match SolarisSPModel.get_block_mask_teacher_forcing!"

    # Output comparison
    q_BTHD = jax.random.normal(jax.random.PRNGKey(0), (B, total_seq_len, H, D))
    k_BTHD = jax.random.normal(jax.random.PRNGKey(1), (B, total_seq_len, H, D))
    v_BTHD = jax.random.normal(jax.random.PRNGKey(2), (B, total_seq_len, H, D))

    o_ref_BTHD = jax.nn.dot_product_attention(
        q_BTHD, k_BTHD, v_BTHD, mask=ref_block_mask
    )
    o_BTHD = teacher_forcing_block_causal_splash_attn(
        q_BTHD, k_BTHD, v_BTHD, block_size=q_block_size, window_size=local_attn_size
    )

    print("=== Teacher Forcing Block Attention Output Comparison ===")
    print_matching_report(o_ref_BTHD, o_BTHD)

    assert jnp.allclose(
        o_ref_BTHD, o_BTHD, rtol=1e-4, atol=1e-4
    ), "Teacher forcing block splash attention output does not match reference"
