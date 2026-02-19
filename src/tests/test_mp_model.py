import sys

import jax.numpy as jnp
import numpy as np

from src.models.multiplayer.world_model import (
    SolarisMPModel as MultiplayerCausalWorldModel,
)


def _expected_frame_block_mask(n_frames, sliding_block_size):
    """Expected (n_frames, n_frames) boolean block mask for local causal attention."""
    i = jnp.arange(n_frames)[:, None]
    j = jnp.arange(n_frames)[None, :]
    return (j >= i - sliding_block_size + 1) & (j <= i)


def test_get_causal_attn_mask_patch_tokens_prints_mask():
    # Keep these small so the printed token mask is readable.
    n_frames = 8
    num_patches = 4
    sliding_block_size = 6

    token_mask = MultiplayerCausalWorldModel.get_causal_attn_mask(
        num_q_blocks=n_frames,
        num_k_blocks=n_frames,
        q_block_size=num_patches,
        k_block_size=num_patches,
        sliding_block_size=sliding_block_size,
    )

    assert token_mask.shape == (n_frames * num_patches, n_frames * num_patches)

    expected_block = _expected_frame_block_mask(n_frames, sliding_block_size)
    m4 = token_mask.reshape(n_frames, num_patches, n_frames, num_patches)

    # The repeated token mask should match the frame-level block mask.
    assert jnp.array_equal(m4[:, 0, :, 0], expected_block)
    assert jnp.all(m4 == expected_block[:, None, :, None])

    print("\n=== patch token mask (int) ===")
    with np.printoptions(threshold=sys.maxsize, linewidth=200):
        print(np.asarray(token_mask, dtype=np.int32))


def test_get_causal_attn_mask_mouse_tokens_prints_mask():
    n_frames = 8
    sliding_block_size = 6

    token_mask = MultiplayerCausalWorldModel.get_causal_attn_mask(
        num_q_blocks=n_frames,
        num_k_blocks=n_frames,
        q_block_size=1,
        k_block_size=1,
        sliding_block_size=sliding_block_size,
    )

    assert token_mask.shape == (n_frames, n_frames)

    expected_block = _expected_frame_block_mask(n_frames, sliding_block_size)
    assert jnp.array_equal(token_mask, expected_block)

    print("\n=== mouse token mask (int) ===")
    with np.printoptions(threshold=sys.maxsize, linewidth=200):
        print(np.asarray(token_mask, dtype=np.int32))


def _expected_teacher_forcing_block_mask(n_blocks, sliding_block_size):
    """
    Expected (2*n_blocks, 2*n_blocks) boolean block mask for teacher forcing.

    Sequence layout is assumed to be concatenated as:
      [clean_0..clean_{n-1}, unclean_0..unclean_{n-1}]
    """
    i = jnp.arange(n_blocks)[:, None]
    j = jnp.arange(n_blocks)[None, :]

    # clean -> clean: causal (with optional sliding window)
    clean_clean = (j >= i - sliding_block_size + 1) & (j <= i)

    # clean -> unclean: disallowed
    clean_unclean = jnp.zeros((n_blocks, n_blocks), dtype=bool)

    # unclean -> clean: strictly past only (no diagonal)
    unclean_clean = (j >= i - sliding_block_size + 1) & (j < i)

    # unclean -> unclean: diagonal only (implemented as sliding_block_size=1)
    unclean_unclean = j == i

    top = jnp.concatenate([clean_clean, clean_unclean], axis=1)
    bottom = jnp.concatenate([unclean_clean, unclean_unclean], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def test_get_block_mask_teacher_forcing_patch_tokens_sliding6():
    # Keep these small so the printed token mask is readable.
    n_blocks = 8
    num_patches = 4
    sliding_block_size = 6

    token_mask = MultiplayerCausalWorldModel.get_block_mask_teacher_forcing(
        num_q_blocks=n_blocks,
        num_k_blocks=n_blocks,
        q_block_size=num_patches,
        k_block_size=num_patches,
        sliding_block_size=sliding_block_size,
    )

    assert token_mask.shape == (2 * n_blocks * num_patches, 2 * n_blocks * num_patches)

    expected_block = _expected_teacher_forcing_block_mask(n_blocks, sliding_block_size)
    m4 = token_mask.reshape(2 * n_blocks, num_patches, 2 * n_blocks, num_patches)

    # The repeated token mask should match the block-level teacher-forcing mask.
    assert jnp.array_equal(m4[:, 0, :, 0], expected_block)
    assert jnp.all(m4 == expected_block[:, None, :, None])

    print("\n=== teacher forcing patch token mask (int) ===")
    with np.printoptions(threshold=sys.maxsize, linewidth=200):
        print(np.asarray(token_mask, dtype=np.int32))


def test_get_block_mask_teacher_forcing_mouse_tokens_sliding6():
    n_blocks = 8
    sliding_block_size = 6

    token_mask = MultiplayerCausalWorldModel.get_block_mask_teacher_forcing(
        num_q_blocks=n_blocks,
        num_k_blocks=n_blocks,
        q_block_size=1,
        k_block_size=1,
        sliding_block_size=sliding_block_size,
    )

    assert token_mask.shape == (2 * n_blocks, 2 * n_blocks)

    expected_block = _expected_teacher_forcing_block_mask(n_blocks, sliding_block_size)
    assert jnp.array_equal(token_mask, expected_block)

    print("\n=== teacher forcing mouse token mask (int) ===")
    with np.printoptions(threshold=sys.maxsize, linewidth=200):
        print(np.asarray(token_mask, dtype=np.int32))
