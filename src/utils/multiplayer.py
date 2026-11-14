"""Multiplayer utilities specific to matrix game models.

These utilities handle the reshaping of video, action, and visual-context data
for different multiplayer methods in matrix game models.

Conventions (matrix-game):
    multiplayer_attn:
        video_BPFHWC         ~ (B, P, F, H, W, C)
        actions_*_BPFD       ~ (B, P, F, D_mouse / D_keyboard)
        visual_context_BPFD  ~ (B, P, F, D_v)

    concat_c:
        video_BPFHWC         ~ (B, 1, F, H, W, P*C)         # players stacked in channels
        actions_*_BPFD       ~ (B, 1, F, P*D_mouse/keyboard)
        visual_context_BPFD  ~ (B, 1, F, P*D_v)

This module is the single source of truth for these multiplayer reshapes.
"""

import jax.numpy as jnp
from einops import rearrange


def handle_multiplayer_input(
    cond_concat_BPFHWC,
    actions_mouse_BPFD,
    actions_keyboard_BPFD,
    multiplayer_method,
    visual_context_BPFD,
    video_BPFHWC=None,
):
    """
    Reshape conditioning, video, action, and visual context for multiplayer methods.

    Args:
        cond_concat_BPFHWC: Conditioning data with shape (B, P, F, H, W, C). Always required.
        actions_mouse_BPFD: Mouse actions with shape (B, P, F, D_mouse).
        actions_keyboard_BPFD: Keyboard actions with shape (B, P, F, D_keyboard).
        multiplayer_method: One of 'multiplayer_attn' or 'concat_c'.
        visual_context_BPFD: Visual context of shape (B, P, F, D_v).
        video_BPFHWC: Optional video data with shape (B, P, F, H, W, C). None for eval.

    Returns:
        Tuple of (cond_concat, mouse_actions, keyboard_actions, visual_context, video) with
        shapes:
            multiplayer_attn:
                All tensors unchanged.
            concat_c:
                cond_concat_BPFHWC  -> (B, 1, F, H, W, P*C)
                video_BPFHWC        -> (B, 1, F, H, W, P*C) if provided
                actions_*_BPFD      -> (B, 1, F, P*D_*)
                visual_context_BPFD -> (B, 1, F, P*D_v)
    """
    if multiplayer_method == "multiplayer_attn":
        # No reshaping needed; keep per-player dimensions explicit.
        return (
            cond_concat_BPFHWC,
            actions_mouse_BPFD,
            actions_keyboard_BPFD,
            visual_context_BPFD,
            video_BPFHWC,
        )

    # All concat_* methods operate with an effective player dimension P_eff = 1
    # and push the original player axis into spatial (video) or feature
    # (actions, visual_context) dimensions.
    if multiplayer_method == "concat_c":
        # Concatenate players along channel dimension for video/cond.
        cond_concat_BPFHWC = rearrange(
            cond_concat_BPFHWC, "b p f h w c -> b 1 f h w (p c)"
        )
        if video_BPFHWC is not None:
            video_BPFHWC = rearrange(video_BPFHWC, "b p f h w c -> b 1 f h w (p c)")

    actions_mouse_BPFD = rearrange(actions_mouse_BPFD, "b p f d -> b 1 f (p d)")
    actions_keyboard_BPFD = rearrange(actions_keyboard_BPFD, "b p f d -> b 1 f (p d)")
    visual_context_BPFD = rearrange(visual_context_BPFD, "b p f d -> b 1 f (p d)")

    return (
        cond_concat_BPFHWC,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
        visual_context_BPFD,
        video_BPFHWC,
    )


def handle_multiplayer_output(
    output,
    multiplayer_method,
    num_players=2,
):
    """
    Reverse the reshaping for multiplayer methods to recover original player dimension.

    Works for any tensor whose last axis holds the player-concatenated features and
    whose second axis is the collapsed P=1 effective-player dimension, e.g.
    ``(B, 1, F, H, W, P*C)`` or ``(B, 1, T, P*D)``.

    Args:
        output: Data with P_eff=1 after a concat method, or P players for attn.
        multiplayer_method: One of 'multiplayer_attn' or 'concat_c'
        num_players: Number of players to split into

    Returns:
        Tensor with shape (B, players, ..., per_player_last_dim)
    """
    if multiplayer_method == "concat_c":
        # Split last dim into (num_players, per_player_dim), then move the new
        # player axis to position 1 and drop the old P=1 axis.
        *leading, last_dim = output.shape
        per_player_dim = last_dim // num_players
        output = output.reshape(*leading, num_players, per_player_dim)
        # axes: (b, 1, ..., p, d) -> (b, p, 1, ..., d)
        output = jnp.moveaxis(output, -2, 1)
        # squeeze the old P=1 dim (now at position 2)
        output = output[:, :, 0, ...]

    # For multiplayer_method == "multiplayer_attn", return unchanged
    return output
