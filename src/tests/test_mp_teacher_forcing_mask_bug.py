"""
Test to confirm the multiplayer teacher forcing attention mask bug.

Bug description:
In multiplayer_causal_world_model.py, when creating splash attention masks:
- Normal causal: uses block_size = s0 * 2 (doubling for 2 players)
- Teacher forcing: uses DEFAULT block_size = 880 (no doubling!)

This means in teacher forcing mode, each player's tokens within a frame are treated
as separate blocks, leading to incorrect attention patterns.

Expected behavior for teacher forcing:
- Clean tokens attend to all clean tokens causally (frame-block level, includes both players)
- Noisy tokens attend to previous clean tokens AND current noisy tokens (same frame-block)
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt


def create_teacher_forcing_mask_correct(
    num_frames,
    num_players,
    spatial_size,
):
    """Create the CORRECT teacher forcing mask accounting for multiplayer.

    Input layout: [clean_f0_p0, clean_f0_p1, clean_f1_p0, clean_f1_p1, ...,
                   noisy_f0_p0, noisy_f0_p1, noisy_f1_p0, noisy_f1_p1, ...]

    Correct behavior:
    - Block = (frame, all players) i.e., block_size = num_players * spatial_size
    - Clean block i attends to clean blocks <= i
    - Noisy block i attends to clean blocks < i AND noisy block i only
    """
    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = num_players * tokens_per_player_frame
    clean_len = num_frames * tokens_per_frame_block
    total_len = clean_len * 2  # clean + noisy

    mask = jnp.zeros((total_len, total_len), dtype=bool)

    # Helper to get block index for a token
    def get_block_idx(token_idx, is_noisy):
        if is_noisy:
            offset_idx = token_idx - clean_len
        else:
            offset_idx = token_idx
        return offset_idx // tokens_per_frame_block

    for q_idx in range(total_len):
        q_is_noisy = q_idx >= clean_len
        q_block = get_block_idx(q_idx, q_is_noisy)

        for k_idx in range(total_len):
            k_is_noisy = k_idx >= clean_len
            k_block = get_block_idx(k_idx, k_is_noisy)

            if not q_is_noisy and not k_is_noisy:
                # Clean attends to clean causally at block level
                mask = mask.at[q_idx, k_idx].set(q_block >= k_block)
            elif q_is_noisy and not k_is_noisy:
                # Noisy attends to PREVIOUS clean blocks (strictly less)
                mask = mask.at[q_idx, k_idx].set(q_block > k_block)
            elif q_is_noisy and k_is_noisy:
                # Noisy attends to SAME noisy block only
                mask = mask.at[q_idx, k_idx].set(q_block == k_block)
            # Clean doesn't attend to noisy (stays False)

    return mask


def create_teacher_forcing_mask_buggy(
    num_frames,
    num_players,
    spatial_size,
):
    """Create the BUGGY mask (current implementation - no multiplayer adjustment).

    Bug: Uses block_size = spatial_size instead of num_players * spatial_size.
    This treats each player's tokens in a frame as a separate block!
    """
    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = num_players * tokens_per_player_frame
    clean_len = num_frames * tokens_per_frame_block
    total_len = clean_len * 2

    # BUGGY: block_size doesn't account for players
    buggy_block_size = (
        tokens_per_player_frame  # This is wrong! Should be tokens_per_frame_block
    )

    mask = jnp.zeros((total_len, total_len), dtype=bool)

    def get_block_idx_buggy(token_idx, is_noisy):
        # BUG: block calculation uses wrong block size
        if is_noisy:
            offset_idx = token_idx - clean_len
        else:
            offset_idx = token_idx
        return offset_idx // buggy_block_size

    num_buggy_blocks = clean_len // buggy_block_size  # More blocks than intended!

    for q_idx in range(total_len):
        q_is_noisy = q_idx >= clean_len
        q_block = get_block_idx_buggy(q_idx, q_is_noisy)

        for k_idx in range(total_len):
            k_is_noisy = k_idx >= clean_len
            k_block = get_block_idx_buggy(k_idx, k_is_noisy)

            if not q_is_noisy and not k_is_noisy:
                mask = mask.at[q_idx, k_idx].set(q_block >= k_block)
            elif q_is_noisy and not k_is_noisy:
                mask = mask.at[q_idx, k_idx].set(q_block > k_block)
            elif q_is_noisy and k_is_noisy:
                mask = mask.at[q_idx, k_idx].set(q_block == k_block)

    return mask


def create_mask_from_splash_implementation(
    num_frames,
    num_players,
    spatial_size,
    use_correct_block_size,
):
    """Create mask using the same logic as TeacherForcingBlockMask in splash_attn.py"""
    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = num_players * tokens_per_player_frame
    clean_len = num_frames * tokens_per_frame_block
    total_len = clean_len * 2

    # This is what's passed to TeacherForcingBlockMask
    if use_correct_block_size:
        block_size = tokens_per_frame_block  # CORRECT: accounts for all players
    else:
        block_size = tokens_per_player_frame  # BUGGY: default 880, no player adjustment

    seq_len = clean_len  # This is passed to TeacherForcingBlockMask

    # Replicate the mask_function from TeacherForcingBlockMask
    q_ids = jnp.arange(total_len)[:, None]
    kv_ids = jnp.arange(total_len)[None, :]

    q_block_abs_id = q_ids // block_size
    kv_block_abs_id = kv_ids // block_size
    num_blocks = seq_len // block_size
    q_is_noisy = q_block_abs_id >= num_blocks
    kv_is_noisy = kv_block_abs_id >= num_blocks
    q_block_id = q_block_abs_id % num_blocks
    kv_block_id = kv_block_abs_id % num_blocks

    mask = (
        (q_is_noisy & kv_is_noisy & (q_block_id == kv_block_id))
        | (q_is_noisy & ~kv_is_noisy & (q_block_id > kv_block_id))
        | (~q_is_noisy & ~kv_is_noisy & (q_block_id >= kv_block_id))
    )

    return mask


def visualize_masks(num_frames=3, num_players=2, spatial_size=4):
    """Visualize the difference between correct and buggy masks."""
    correct_mask = create_teacher_forcing_mask_correct(
        num_frames, num_players, spatial_size
    )
    buggy_mask = create_teacher_forcing_mask_buggy(
        num_frames, num_players, spatial_size
    )

    splash_correct = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, True
    )
    splash_buggy = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, False
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    tokens_per_frame_block = num_players * spatial_size
    clean_len = num_frames * tokens_per_frame_block

    def add_block_lines(ax, block_size, total_len, clean_len):
        # Draw vertical and horizontal lines at block boundaries
        for i in range(0, total_len + 1, block_size):
            ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
        # Draw thick line at clean/noisy boundary
        ax.axhline(y=clean_len - 0.5, color="red", linewidth=2)
        ax.axvline(x=clean_len - 0.5, color="red", linewidth=2)

    # Correct mask
    axes[0, 0].imshow(correct_mask, cmap="Blues", aspect="auto")
    axes[0, 0].set_title(f"Correct Mask\n(block_size={tokens_per_frame_block})")
    add_block_lines(
        axes[0, 0], tokens_per_frame_block, correct_mask.shape[0], clean_len
    )

    # Buggy mask
    axes[0, 1].imshow(buggy_mask, cmap="Reds", aspect="auto")
    axes[0, 1].set_title(f"Buggy Mask\n(block_size={spatial_size})")
    add_block_lines(axes[0, 1], spatial_size, buggy_mask.shape[0], clean_len)

    # Difference
    diff = correct_mask != buggy_mask
    axes[0, 2].imshow(diff, cmap="RdYlGn_r", aspect="auto")
    axes[0, 2].set_title(f"Difference (red = mismatch)\n{jnp.sum(diff)} errors")
    add_block_lines(axes[0, 2], tokens_per_frame_block, diff.shape[0], clean_len)

    # Splash implementation - correct
    axes[1, 0].imshow(splash_correct, cmap="Blues", aspect="auto")
    axes[1, 0].set_title("Splash Impl (correct block_size)")
    add_block_lines(
        axes[1, 0], tokens_per_frame_block, splash_correct.shape[0], clean_len
    )

    # Splash implementation - buggy
    axes[1, 1].imshow(splash_buggy, cmap="Reds", aspect="auto")
    axes[1, 1].set_title("Splash Impl (buggy block_size)")
    add_block_lines(axes[1, 1], spatial_size, splash_buggy.shape[0], clean_len)

    # Verify splash matches our expected implementations
    splash_vs_correct = jnp.array_equal(splash_correct, correct_mask)
    splash_vs_buggy = jnp.array_equal(splash_buggy, buggy_mask)
    axes[1, 2].text(
        0.5,
        0.5,
        f"Verification:\n\n"
        f"Splash correct == Expected correct: {splash_vs_correct}\n"
        f"Splash buggy == Expected buggy: {splash_vs_buggy}",
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.abspath("mp_teacher_forcing_mask_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")

    return correct_mask, buggy_mask


def _find_attention_pattern_errors(num_frames=2, num_players=2, spatial_size=4):
    """
    Helper function to find all attention pattern errors between buggy and correct masks.
    Returns (error_count, errors_list) for use by test functions.
    """
    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = (
        num_players * tokens_per_player_frame
    )  # 8 tokens per frame-block
    clean_len = num_frames * tokens_per_frame_block  # 16 clean tokens

    buggy_mask = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, False
    )
    correct_mask = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, True
    )

    errors = []
    for q_idx in range(2 * clean_len):
        for k_idx in range(2 * clean_len):
            if correct_mask[q_idx, k_idx] != buggy_mask[q_idx, k_idx]:
                q_is_noisy = q_idx >= clean_len
                k_is_noisy = k_idx >= clean_len

                q_frame = (q_idx % clean_len) // tokens_per_frame_block
                q_player = (
                    (q_idx % clean_len) % tokens_per_frame_block
                ) // spatial_size

                k_frame = (k_idx % clean_len) // tokens_per_frame_block
                k_player = (
                    (k_idx % clean_len) % tokens_per_frame_block
                ) // spatial_size

                role_q = "noisy" if q_is_noisy else "clean"
                role_k = "noisy" if k_is_noisy else "clean"

                errors.append(
                    {
                        "q_idx": q_idx,
                        "k_idx": k_idx,
                        "q_role": role_q,
                        "k_role": role_k,
                        "q_frame": q_frame,
                        "k_frame": k_frame,
                        "q_player": q_player,
                        "k_player": k_player,
                        "correct": bool(correct_mask[q_idx, k_idx]),
                        "buggy": bool(buggy_mask[q_idx, k_idx]),
                    }
                )

    return len(errors), errors


def test_specific_attention_patterns():
    """
    Regression test for teacher-forcing attention-mask block sizing.

    Compares a deliberately-wrong mask (`block_size=spatial_size`) against the
    correct mask (`block_size=num_players*spatial_size`) on a few hand-picked
    token pairs. The wrong block sizing produces characteristic errors:

    - noisy -> same-frame clean should be blocked
    - noisy -> same-frame noisy (other player) should be allowed
    - clean cross-player within the same frame should be allowed/symmetric
    """
    num_frames = 2
    num_players = 2
    spatial_size = 4  # Small for easy verification

    tokens_per_player_frame = spatial_size
    tokens_per_frame_block = (
        num_players * tokens_per_player_frame
    )  # 8 tokens per frame-block
    clean_len = num_frames * tokens_per_frame_block  # 16 clean tokens

    # Token layout:
    # Clean: [f0_p0: 0-3, f0_p1: 4-7, f1_p0: 8-11, f1_p1: 12-15]
    # Noisy: [f0_p0: 16-19, f0_p1: 20-23, f1_p0: 24-27, f1_p1: 28-31]

    buggy_mask = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, False
    )
    correct_mask = create_mask_from_splash_implementation(
        num_frames, num_players, spatial_size, True
    )

    print("=" * 60)
    print("Testing specific attention patterns")
    print("=" * 60)
    print(
        f"Setup: {num_frames} frames, {num_players} players, {spatial_size} spatial tokens/player"
    )
    print(f"Tokens per frame-block: {tokens_per_frame_block}")
    print(f"Clean tokens: 0-{clean_len-1}")
    print(f"Noisy tokens: {clean_len}-{2*clean_len-1}")
    print()

    # =========================================================================
    # Test 1: Bug case - noisy attending to same-frame clean (should be blocked)
    # =========================================================================
    # Token 20 (noisy_f0_p1) -> Token 0 (clean_f0_p0)
    # SHOULD BE FALSE (noisy frame 0 should NOT attend to clean frame 0)
    # But buggy mask incorrectly allows this!
    q_token = 20  # noisy, frame 0, player 1
    k_token = 0  # clean, frame 0, player 0

    print(
        f"Test 1 (BUG): Token {q_token} (noisy_f0_p1) -> Token {k_token} (clean_f0_p0)"
    )
    print("  Expected: False (noisy shouldn't attend to same-frame clean)")
    print(f"  Correct mask: {correct_mask[q_token, k_token]}")
    print(f"  Buggy mask:   {buggy_mask[q_token, k_token]}")

    # The buggy mask should incorrectly allow this attention
    assert (
        correct_mask[q_token, k_token] == False
    ), "Correct mask should block this attention"
    assert (
        buggy_mask[q_token, k_token] == True
    ), "Buggy mask should (incorrectly) allow this attention"

    # =========================================================================
    # Test 2: Cross-player clean attention (should be allowed in both)
    # =========================================================================
    # Token 4 (clean_f0_p1) -> Token 0 (clean_f0_p0)
    # SHOULD BE TRUE (clean tokens in same frame can attend to each other)
    q_token = 4  # clean, frame 0, player 1
    k_token = 0  # clean, frame 0, player 0

    print(
        f"\nTest 2 (cross-player clean): Token {q_token} (clean_f0_p1) -> Token {k_token} (clean_f0_p0)"
    )
    print("  Expected: True (same-frame clean tokens should attend to each other)")
    print(f"  Correct mask: {correct_mask[q_token, k_token]}")
    print(f"  Buggy mask:   {buggy_mask[q_token, k_token]}")

    assert (
        correct_mask[q_token, k_token] == True
    ), "Clean tokens in same frame should attend to each other"
    assert (
        buggy_mask[q_token, k_token] == True
    ), "Even buggy mask should allow same-frame clean attention"

    # =========================================================================
    # Test 3: Cross-player noisy attention p1->p0 (BUG: buggy mask blocks this!)
    # =========================================================================
    # Token 20 (noisy_f0_p1) -> Token 16 (noisy_f0_p0)
    # SHOULD BE TRUE (noisy tokens in same frame can attend to each other)
    # But buggy mask incorrectly BLOCKS this because it treats each player as separate block!
    q_token = 20  # noisy, frame 0, player 1
    k_token = 16  # noisy, frame 0, player 0

    print(
        f"\nTest 3 (BUG - cross-player noisy p1->p0): Token {q_token} (noisy_f0_p1) -> Token {k_token} (noisy_f0_p0)"
    )
    print("  Expected: True (same-frame noisy tokens should attend to each other)")
    print(f"  Correct mask: {correct_mask[q_token, k_token]}")
    print(
        f"  Buggy mask:   {buggy_mask[q_token, k_token]} (WRONG - blocks cross-player attention!)"
    )

    assert (
        correct_mask[q_token, k_token] == True
    ), "Noisy tokens in same frame should attend to each other"
    assert (
        buggy_mask[q_token, k_token] == False
    ), "Buggy mask incorrectly blocks cross-player noisy attention"

    # =========================================================================
    # Test 4: Cross-player clean attention p0->p1 (BUG: buggy mask blocks this!)
    # =========================================================================
    # Token 0 (clean_f0_p0) -> Token 4 (clean_f0_p1)
    # SHOULD BE TRUE (clean tokens in same frame can attend to each other)
    # But buggy mask blocks this because p0 is in block 0, p1 is in block 1,
    # and the causal rule (q_block >= k_block) means 0 >= 1 is False!
    # Note: p1->p0 works (block 1 >= block 0), but p0->p1 is blocked (asymmetric bug)
    q_token = 0  # clean, frame 0, player 0
    k_token = 4  # clean, frame 0, player 1

    print(
        f"\nTest 4 (BUG - cross-player clean p0->p1): Token {q_token} (clean_f0_p0) -> Token {k_token} (clean_f0_p1)"
    )
    print("  Expected: True (same-frame clean tokens should attend to each other)")
    print(f"  Correct mask: {correct_mask[q_token, k_token]}")
    print(
        f"  Buggy mask:   {buggy_mask[q_token, k_token]} (WRONG - blocks p0->p1 but allows p1->p0!)"
    )

    assert (
        correct_mask[q_token, k_token] == True
    ), "Clean tokens in same frame should attend to each other"
    assert (
        buggy_mask[q_token, k_token] == False
    ), "Buggy mask incorrectly blocks p0->p1 clean attention"

    # =========================================================================
    # Test 5: Cross-player noisy attention p0->p1 (BUG: buggy mask blocks this!)
    # =========================================================================
    # Token 16 (noisy_f0_p0) -> Token 20 (noisy_f0_p1)
    # SHOULD BE TRUE (noisy tokens in same frame can attend to each other)
    # But buggy mask incorrectly BLOCKS this because it treats each player as separate block!
    q_token = 16  # noisy, frame 0, player 0
    k_token = 20  # noisy, frame 0, player 1

    print(
        f"\nTest 5 (BUG - cross-player noisy p0->p1): Token {q_token} (noisy_f0_p0) -> Token {k_token} (noisy_f0_p1)"
    )
    print("  Expected: True (same-frame noisy tokens should attend to each other)")
    print(f"  Correct mask: {correct_mask[q_token, k_token]}")
    print(
        f"  Buggy mask:   {buggy_mask[q_token, k_token]} (WRONG - blocks cross-player attention!)"
    )

    assert (
        correct_mask[q_token, k_token] == True
    ), "Noisy tokens in same frame should attend to each other"
    assert (
        buggy_mask[q_token, k_token] == False
    ), "Buggy mask incorrectly blocks cross-player noisy attention"

    print()
    print("=" * 60)

    # Find all errors
    print("\nSearching for ALL attention pattern errors...")
    error_count, errors = _find_attention_pattern_errors(
        num_frames, num_players, spatial_size
    )

    print(f"Found {error_count} errors")

    if errors:
        print("\nFirst 10 errors:")
        for err in errors[:10]:
            print(
                f"  [{err['q_idx']}]->[{err['k_idx']}]: "
                f"{err['q_role']}_f{err['q_frame']}_p{err['q_player']} -> "
                f"{err['k_role']}_f{err['k_frame']}_p{err['k_player']} | "
                f"correct={err['correct']}, buggy={err['buggy']}"
            )

        # Categorize errors
        false_positives = [e for e in errors if e["buggy"] and not e["correct"]]
        false_negatives = [e for e in errors if e["correct"] and not e["buggy"]]

        print("\nError breakdown:")
        print(
            f"  False positives (buggy allows, correct doesn't): {len(false_positives)}"
        )
        print(
            f"  False negatives (buggy blocks, correct allows): {len(false_negatives)}"
        )

        if false_positives:
            print(
                "\n  Sample false positives (DANGEROUS - allows attending to wrong tokens):"
            )
            for e in false_positives[:5]:
                print(
                    f"    {e['q_role']}_f{e['q_frame']}_p{e['q_player']} -> "
                    f"{e['k_role']}_f{e['k_frame']}_p{e['k_player']}"
                )

    # Assert the bug exists: there should be errors between buggy and correct masks
    assert error_count > 0, (
        "Expected buggy and correct masks to differ, but found no errors. "
        "This would mean the bug doesn't exist."
    )
    print(f"\n✓ Confirmed: buggy mask has {error_count} attention pattern errors")


def test_bug_confirmation():
    """
    Main test that confirms the teacher forcing attention mask bug exists.

    This test verifies that without the fix (block_size=s_tf * num_players),
    the attention mask has errors that allow noisy tokens to incorrectly
    attend to same-frame clean tokens.

    The fix in multiplayer_causal_world_model.py (line 139) is:
        partial(teacher_forcing_block_causal_splash_attn, block_size=s_tf * 2)
    """
    print("=" * 60)
    print("BUG CONFIRMATION TEST")
    print("=" * 60)
    print()
    print("Testing if teacher forcing mask in multiplayer model is broken")
    print("(without the block_size fix)")
    print()

    error_count, errors = _find_attention_pattern_errors()

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Assert the bug exists
    assert (
        error_count > 0
    ), "Expected to find attention pattern errors confirming the bug, but found none."

    print(f"✓ BUG CONFIRMED: {error_count} attention pattern errors detected")
    print()
    print("The bug was in multiplayer_causal_world_model.py")
    print(
        "  Buggy: partial(teacher_forcing_block_causal_splash_attn)  # default block_size"
    )
    print(
        "  Fixed: partial(teacher_forcing_block_causal_splash_attn, block_size=s_tf * 2)"
    )
