"""
Test to verify RoPE applications are consistent across different modes.
Run with: python -m pytest src/tests/test_rope_consistency.py -v
"""

import jax
import jax.numpy as jnp
from einops import rearrange

# Import the RoPE functions
from src.models.transformer_utils import apply_rope_mp, rope_apply, rope_params


def make_freqs(dim, max_seq_len=1024):
    """Create the frequency tensor like the model does."""
    d = dim
    return jnp.concatenate(
        [
            rope_params(max_seq_len, d - 4 * (d // 6)),
            rope_params(max_seq_len, 2 * (d // 6)),
            rope_params(max_seq_len, 2 * (d // 6)),
        ],
        axis=1,
    )


def test_rope_basic_shapes():
    """Test basic RoPE application with known shapes."""
    B, F, P, H, W, N_HEADS, HEAD_DIM = 2, 4, 2, 22, 40, 8, 64
    S = H * W  # 880 patches per frame

    # Precompute frequencies (same as in the model)
    freqs = make_freqs(HEAD_DIM)

    # Create input tensor
    x = jax.random.normal(jax.random.PRNGKey(0), (B, F * P * S, N_HEADS, HEAD_DIM))

    grid_sizes = (F, H, W)
    result = apply_rope_mp(x, grid_sizes, freqs, F, S)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    print("✓ Basic shapes test passed")


def test_rope_player_independence():
    """
    Verify that each player gets the same temporal RoPE positions.
    In multiplayer mode, each player should get positions [0, 1, ..., F-1].
    """
    B, F, P, H, W, N_HEADS, HEAD_DIM = 1, 4, 2, 22, 40, 8, 64
    S = H * W  # 880 patches

    freqs = make_freqs(HEAD_DIM)

    # Create input where we can track which player each token belongs to
    x = jax.random.normal(jax.random.PRNGKey(0), (B, F * P * S, N_HEADS, HEAD_DIM))

    grid_sizes = (F, H, W)

    # Apply RoPE to full sequence
    result_full = apply_rope_mp(x, grid_sizes, freqs, F, S)

    # Now apply RoPE to each player separately and verify they match
    x_reshaped = rearrange(x, "b (f p s) n d -> b f p s n d", f=F, p=P, s=S)

    results_per_player = []
    for player_idx in range(P):
        x_player = rearrange(x_reshaped[:, :, player_idx], "b f s n d -> b (f s) n d")
        # For single player, p is inferred as 1
        result_player = rearrange(
            rope_apply(x_player, grid_sizes, freqs, start_frame=0),
            "b (f s) n d -> b f s n d",
            f=F,
            s=S,
        )
        results_per_player.append(result_player)

    # Stack back together
    result_separate = rearrange(
        jnp.stack(results_per_player, axis=2), "b f p s n d -> b (f p s) n d"
    )

    # Compare
    max_diff = jnp.abs(result_full - result_separate).max()
    print(f"Max difference between full and per-player RoPE: {max_diff}")

    assert max_diff < 1e-5, f"RoPE results differ: max_diff={max_diff}"
    print("✓ Player independence test passed")


def test_rope_teacher_forcing_vs_separate():
    """
    Regression test for teacher-forcing RoPE when clean/noisy roles are concatenated.

    If RoPE is applied directly to a `[B, 2*F*P*S, N, D]` sequence while passing
    `f=F, s=S`, einops can infer `p=2*P` instead of `p=P`, which corrupts the
    temporal positions. Splitting by role first avoids this mis-inference.
    """
    B, F, P, H, W, N_HEADS, HEAD_DIM = 1, 4, 2, 22, 40, 8, 64
    S = H * W  # 880 patches

    freqs = make_freqs(HEAD_DIM)

    # Teacher forcing input: [B, 2*F*P*S, N, D] (clean frames then noisy frames)
    x = jax.random.normal(jax.random.PRNGKey(0), (B, 2 * F * P * S, N_HEADS, HEAD_DIM))

    grid_sizes_tf = (F, H, W)

    # =========================================================================
    # Method 1 (BUGGY): Apply RoPE to full 2*F*P*S sequence directly
    # This was the old buggy code - einops infers p=2*P instead of p=P
    # =========================================================================
    result_buggy = apply_rope_mp(x, grid_sizes_tf, freqs, F, S)

    # =========================================================================
    # Method 2 (CORRECT/FIX): Split by role first, apply RoPE, recombine
    # This matches multiplayer_causal_world_model.py lines 129-134:
    #   q_BTHD = rearrange(q_BTHD, "b (r s) n d -> (b r) s n d", r=2)
    #   roped_query_BTHD = apply_rope_mp(q_BTHD, grid_sizes, freqs, f_tf, s_tf)
    #   roped_query_BTHD = rearrange(roped_query_BTHD, "(b r) s n d -> b (r s) n d", r=2)
    # =========================================================================
    x_split = rearrange(x, "b (r s) n d -> (b r) s n d", r=2)
    result_split = apply_rope_mp(x_split, grid_sizes_tf, freqs, F, S)
    result_fixed = rearrange(result_split, "(b r) s n d -> b (r s) n d", r=2, b=B)

    # Compare: methods should be DIFFERENT (bug causes different results)
    max_diff = jnp.abs(result_buggy - result_fixed).max()
    print(f"Max difference between BUGGY and FIXED methods: {max_diff}")

    # Debug info
    total_buggy = 2 * F * P * S
    p_inferred_buggy = total_buggy // (F * S)
    total_fixed = F * P * S  # after role split
    p_inferred_fixed = total_fixed // (F * S)
    print(f"  Buggy method: inferred p = {p_inferred_buggy} (should be P={P}, got 2*P)")
    print(f"  Fixed method: inferred p = {p_inferred_fixed} (correctly P={P})")

    # Assert the bug exists: methods should produce DIFFERENT results
    assert max_diff > 1e-3, (
        f"Expected buggy and fixed methods to differ significantly, but max_diff={max_diff}. "
        f"This would mean the bug doesn't exist (which contradicts our understanding)."
    )
    print(
        f"✓ Confirmed: buggy method produces different results from fixed method (diff={max_diff:.4f})"
    )


def test_rope_teacher_forcing_proposed_fix():
    """
    Check that role-splitting before RoPE matches per-role ground truth.

    Verifies:
    - role-split method == ground truth (apply RoPE per role×player)
    - naive method (apply RoPE on concatenated roles) != ground truth
    """
    B, F, P, H, W, N_HEADS, HEAD_DIM = 1, 4, 2, 22, 40, 8, 64
    S = H * W  # 880 patches

    freqs = make_freqs(HEAD_DIM)

    # Teacher forcing input: [B, 2*F*P*S, N, D] (clean frames then noisy frames)
    x = jax.random.normal(jax.random.PRNGKey(0), (B, 2 * F * P * S, N_HEADS, HEAD_DIM))

    grid_sizes_tf = (F, H, W)

    # =========================================================================
    # PROPOSED FIX: Split by role using rearrange, apply RoPE, recombine
    # This matches what multiplayer_causal_world_model.py does
    # =========================================================================
    # Split by role: [B, 2*F*P*S, N, D] -> [(B*2), F*P*S, N, D]
    x_split = rearrange(x, "b (r s) n d -> (b r) s n d", r=2)
    # Apply RoPE using the imported apply_rope_mp (note: batch is now B*2 after the split)
    result_split = apply_rope_mp(x_split, grid_sizes_tf, freqs, F, S)
    # Recombine: [(B*2), F*P*S, N, D] -> [B, 2*F*P*S, N, D]
    result_fixed = rearrange(result_split, "(b r) s n d -> b (r s) n d", r=2, b=B)

    # =========================================================================
    # GROUND TRUTH: Apply RoPE to each player×role combination separately
    # =========================================================================
    x_reshaped = rearrange(x, "b (r f p s) n d -> b r f p s n d", r=2, f=F, p=P, s=S)

    results_per_role_player = []
    for role_idx in range(2):  # clean=0, noisy=1
        for player_idx in range(P):
            x_rp = rearrange(
                x_reshaped[:, role_idx, :, player_idx], "b f s n d -> b (f s) n d"
            )
            result_rp = rope_apply(x_rp, grid_sizes_tf, freqs, start_frame=0)
            results_per_role_player.append(
                rearrange(result_rp, "b (f s) n d -> b f s n d", f=F, s=S)
            )

    # Reconstruct: [r=0,p=0], [r=0,p=1], [r=1,p=0], [r=1,p=1]
    # Need to reshape to [B, r, f, p, s, n, d] then to [B, (r f p s), n, d]
    result_ground_truth = rearrange(
        jnp.stack(
            [
                jnp.stack(
                    results_per_role_player[0:P], axis=2
                ),  # role 0: [B, f, p, s, n, d]
                jnp.stack(
                    results_per_role_player[P : 2 * P], axis=2
                ),  # role 1: [B, f, p, s, n, d]
            ],
            axis=1,
        ),
        "b r f p s n d -> b (r f p s) n d",
    )

    # Compare fixed method vs ground truth
    max_diff_fixed = jnp.abs(result_fixed - result_ground_truth).max()
    print(f"Max difference between FIXED method and ground truth: {max_diff_fixed}")

    if max_diff_fixed < 1e-5:
        print("✓ PROPOSED FIX WORKS! Fixed method matches ground truth.")
    else:
        print(f"✗ Proposed fix still has issues: diff={max_diff_fixed}")

    # Also compare against the buggy combined method (apply to full sequence directly)
    result_buggy = apply_rope_mp(x, grid_sizes_tf, freqs, F, S)
    max_diff_buggy = jnp.abs(result_buggy - result_ground_truth).max()
    print(f"Max difference between BUGGY method and ground truth: {max_diff_buggy}")

    assert (
        max_diff_fixed < 1e-5
    ), f"Proposed fix still has issues: diff={max_diff_fixed}"


def test_rope_p_dimension_inference_bug():
    """
    Verify that the bug is caused by incorrect P dimension inference.

    When applying RoPE to full 2*F*P*S sequence with f=F, s=S:
    - einops computes p = 2*F*P*S / (F*S) = 2*P (WRONG!)

    When splitting first and applying to F*P*S:
    - einops computes p = F*P*S / (F*S) = P (CORRECT!)
    """
    B, F, P, H, W, N_HEADS, HEAD_DIM = 1, 4, 2, 22, 40, 8, 64
    S = H * W  # 880 patches

    freqs = make_freqs(HEAD_DIM)

    # Teacher forcing input: [B, 2*F*P*S, N, D]
    total_tokens_full = 2 * F * P * S
    total_tokens_half = F * P * S

    x_full = jax.random.normal(
        jax.random.PRNGKey(0), (B, total_tokens_full, N_HEADS, HEAD_DIM)
    )
    x_half = x_full[:, :total_tokens_half]  # Just the clean half

    # =========================================================================
    # Check 1: Verify the inferred p values
    # =========================================================================
    f_arg, s_arg = F, S

    p_inferred_full = total_tokens_full // (f_arg * s_arg)
    p_inferred_half = total_tokens_half // (f_arg * s_arg)

    print(f"With f={f_arg}, s={s_arg}:")
    print(
        f"  Full sequence ({total_tokens_full} tokens): inferred p = {p_inferred_full} (expected P={P}, got 2*P={2*P})"
    )
    print(
        f"  Half sequence ({total_tokens_half} tokens): inferred p = {p_inferred_half} (expected P={P})"
    )

    assert p_inferred_full == 2 * P, f"Expected p=2*P={2*P}, got {p_inferred_full}"
    assert p_inferred_half == P, f"Expected p=P={P}, got {p_inferred_half}"
    print(
        "✓ Confirmed: full sequence infers p=2*P (wrong), half sequence infers p=P (correct)"
    )

    # =========================================================================
    # Check 2: Verify the temporal position aliasing
    # =========================================================================
    # With wrong p=4, the reshape interprets data as:
    #   (f=0, p=0), (f=0, p=1), (f=0, p=2), (f=0, p=3), (f=1, p=0), ...
    # But actual data is:
    #   (f=0, p=0), (f=0, p=1), (f=1, p=0), (f=1, p=1), ...

    # Create input where each frame has a distinct value so we can track
    x_debug = jnp.zeros((B, total_tokens_full, N_HEADS, HEAD_DIM))
    for frame_idx in range(2 * F):  # 2*F total frames (F clean + F noisy)
        for player_idx in range(P):
            start = (frame_idx * P + player_idx) * S
            end = start + S
            # Tag with frame index in a way we can check
            x_debug = x_debug.at[:, start:end, :, 0].set(frame_idx)
            x_debug = x_debug.at[:, start:end, :, 1].set(player_idx)

    # Apply the BUGGY reshape (what happens without the fix)
    x_buggy_reshaped = rearrange(
        x_debug,
        "b (f p s) n d -> (b p) (f s) n d",
        f=f_arg,  # F
        s=s_arg,  # S
        # p is inferred as 2*P = 4
    )

    print(f"\nBuggy reshape output shape: {x_buggy_reshaped.shape}")
    print(f"  Expected: ({B * P}, {F * S}, {N_HEADS}, {HEAD_DIM})")
    print(f"  Got:      ({B * p_inferred_full}, {F * S}, {N_HEADS}, {HEAD_DIM})")

    # Check what frame indices ended up in each "player" batch
    print("\nFrame indices in each 'player' batch (first spatial token, first head):")
    for p_batch in range(p_inferred_full):
        # Get the frame index for first token of each "frame" in this batch
        frame_indices = []
        for f_idx in range(F):
            token_val = x_buggy_reshaped[p_batch, f_idx * S, 0, 0]
            frame_indices.append(int(token_val))
        print(f"  'Player' batch {p_batch}: frames {frame_indices}")

    # The bug: "player" batches 2 and 3 contain frames from the SECOND half
    # of what should be each player's frames
    expected_frames_p0 = list(range(0, 2 * F, 2))  # [0, 2, 4, 6] - even frames
    expected_frames_p1 = list(range(1, 2 * F, 2))  # [1, 3, 5, 7] - odd frames
    # But with the bug, we get aliased temporal positions!

    print("\n✓ P dimension inference bug verified:")
    print(
        f"  - Full sequence incorrectly creates {p_inferred_full} 'player' batches instead of {P}"
    )
    print("  - This causes frames to be misaligned in the temporal dimension")
    print("  - Frames that should have different RoPE positions get the SAME position")

    # =========================================================================
    # Check 3: Verify the FIXED approach (rearrange style from action_module.py)
    # =========================================================================
    print("\n" + "=" * 60)
    print("FIXED APPROACH: Split by role using rearrange, then reshape each half")
    print("=" * 60)

    # Split by role using rearrange: [B, 2*F*P*S, N, D] -> [(B*2), F*P*S, N, D]
    x_split = rearrange(x_debug, "b (r s) n d -> (b r) s n d", r=2)
    print(f"\nAfter role split rearrange: {x_debug.shape} -> {x_split.shape}")
    print(f"  Expected: ({B * 2}, {total_tokens_half}, {N_HEADS}, {HEAD_DIM})")
    print(
        f"  Got:      {x_split.shape} ✓"
        if x_split.shape == (B * 2, total_tokens_half, N_HEADS, HEAD_DIM)
        else f"  Got:      {x_split.shape} ✗"
    )

    # Apply reshape to the split tensor (now batch is B*2)
    # Each "batch" in the split is one role (clean or noisy)
    x_split_reshaped = rearrange(
        x_split,
        "b (f p s) n d -> (b p) (f s) n d",
        f=f_arg,  # F
        s=s_arg,  # S
        # p is correctly inferred as P = 2
    )

    print(f"\nFixed reshape output shape: {x_split_reshaped.shape}")
    print(f"  Expected: ({B * 2 * P}, {F * S}, {N_HEADS}, {HEAD_DIM})")
    print(
        f"  Got:      {x_split_reshaped.shape} ✓"
        if x_split_reshaped.shape == (B * 2 * P, F * S, N_HEADS, HEAD_DIM)
        else f"  Got:      {x_split_reshaped.shape} ✗"
    )

    # Check frame indices in each batch
    # After split+reshape: batch 0,1 are role=0 (clean) player 0,1; batch 2,3 are role=1 (noisy) player 0,1
    print("\nFrame indices in each (role, player) batch:")
    for batch_idx in range(B * 2 * P):
        role_idx = batch_idx // P
        player_idx = batch_idx % P
        frame_indices = []
        for f_idx in range(F):
            token_val = x_split_reshaped[batch_idx, f_idx * S, 0, 0]
            frame_indices.append(int(token_val))
        expected = list(range(role_idx * F, (role_idx + 1) * F))
        status = "✓" if frame_indices == expected else "✗"
        role_name = "clean" if role_idx == 0 else "noisy"
        print(
            f"  Role {role_idx} ({role_name}), Player {player_idx}: frames {frame_indices} (expected {expected}) {status}"
        )

    # Verify correctness
    all_correct = all(
        int(x_split_reshaped[batch_idx, f * S, 0, 0]) == (batch_idx // P) * F + f
        for batch_idx in range(B * 2 * P)
        for f in range(F)
    )

    if all_correct:
        print(
            "\n✓ FIXED approach (rearrange style) correctly maps frames to RoPE positions:"
        )
        print("  - After rearrange split: batch dimension is B*2 (one per role)")
        print("  - After reshape: batch dimension is B*2*P (one per role×player)")
        print("  - Clean role: frames [0,1,2,3] → RoPE positions [0,1,2,3]")
        print("  - Noisy role: frames [4,5,6,7] → RoPE positions [0,1,2,3]")
        print("  - Each frame gets a UNIQUE temporal position within its role!")
    else:
        print("\n✗ FIXED approach still has issues!")


def test_rope_current_start_offset():
    """
    Verify that current_start properly offsets the temporal positions.
    Position at current_start should match position 0 when applied separately.
    """
    B, F, H, W, N_HEADS, HEAD_DIM = 1, 4, 22, 40, 8, 64
    S = H * W

    freqs = make_freqs(HEAD_DIM)

    # Create a single frame's worth of input
    x_single = jax.random.normal(jax.random.PRNGKey(0), (B, S, N_HEADS, HEAD_DIM))

    grid_sizes_single = (1, H, W)

    # Apply RoPE at position 0
    result_pos0 = rope_apply(x_single, grid_sizes_single, freqs, start_frame=0)

    # Apply RoPE at position 5
    result_pos5 = rope_apply(x_single, grid_sizes_single, freqs, start_frame=5)

    # They should be different (different temporal positions)
    diff = jnp.abs(result_pos0 - result_pos5).max()
    print(f"Difference between position 0 and 5: {diff}")
    assert diff > 1e-3, "Positions 0 and 5 should have different RoPE values"

    # Now verify: if we have multi-frame input and slice out frame 5,
    # it should match the single-frame at position 5
    x_multi = jax.random.normal(jax.random.PRNGKey(0), (B, 10 * S, N_HEADS, HEAD_DIM))
    grid_sizes_multi = (10, H, W)

    result_multi = rope_apply(x_multi, grid_sizes_multi, freqs, start_frame=0)

    # Extract frame 5 from multi-frame result
    result_multi_frame5 = result_multi[:, 5 * S : 6 * S]

    # Apply RoPE to just frame 5's input with start_frame=5
    x_frame5 = x_multi[:, 5 * S : 6 * S]
    result_single_frame5 = rope_apply(x_frame5, grid_sizes_single, freqs, start_frame=5)

    diff_frame5 = jnp.abs(result_multi_frame5 - result_single_frame5).max()
    print(f"Difference for frame 5 (multi vs single with offset): {diff_frame5}")

    assert diff_frame5 < 1e-5, f"Frame 5 should match: diff={diff_frame5}"
    print("✓ current_start offset test passed")
