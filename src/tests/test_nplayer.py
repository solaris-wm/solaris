"""
CPU tests for N-player support (data loader, eval ids, model and checkpoint loading).

The tests build a tiny synthetic 3-player dataset and a tiny SolarisMPModel, so
they run without pretrained weights, datasets or accelerators.
"""

import json
import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx
from omegaconf import OmegaConf

from src.data.batch_sampler import BatchSamplerMultiplayer, EvalBatchSampler
from src.data.dataset import DatasetMultiplayer, collate_segments_to_batch
from src.data.minecraft import ACTION_KEYS
from src.data.segment import SegmentIdMultiplayer
from src.models.multiplayer.world_model import SolarisMPModel
from src.utils.multiplayer import handle_multiplayer_output
from src.utils.rollout import left_repeat_padding

BOT_NAMES = ["Alpha", "Bravo", "Charlie"]
# Each bot holds one distinct key for the whole episode.
BOT_KEYS = {"Alpha": "forward", "Bravo": "back", "Charlie": "left"}
# Each bot records solid frames of a distinct gray level.
BOT_GRAY = {"Alpha": 40, "Bravo": 128, "Charlie": 216}
# Bravo starts recording 2 frames (100 ms) after Alpha and Charlie.
BOT_TIME_OFFSET_FRAMES = {"Alpha": 0, "Bravo": 2, "Charlie": 0}
FRAME_MS = 50
VIDEO_HW = (48, 64)

EVAL_IDS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_ids"
)


def _action(key):
    action = {
        k: False
        for k in [
            "forward", "back", "left", "right", "jump", "sneak", "sprint", "attack",
            "use", "mine", "place_block", "place_entity", "mount", "dismount",
        ]
    }
    action[key] = True
    action["camera"] = [0.0, 0.0]
    return action


def _write_bot_files(directory, episode, bot, num_frames):
    stem = f"batch_0_{episode:06d}_{bot}_instance_000"
    height, width = VIDEO_HW
    writer = cv2.VideoWriter(
        str(directory / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
    )
    frame = np.full((height, width, 3), BOT_GRAY[bot], dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()
    t0 = 1_000_000 + BOT_TIME_OFFSET_FRAMES[bot] * FRAME_MS
    actions = [
        {"renderTime": t0 + i * FRAME_MS, "action": _action(BOT_KEYS[bot])}
        for i in range(num_frames)
    ]
    with open(directory / f"{stem}.json", "w") as f:
        json.dump(actions, f)


@pytest.fixture(scope="module")
def trio_dir(tmp_path_factory):
    """Two complete 3-bot episodes and one episode without Charlie."""
    directory = tmp_path_factory.mktemp("trio") / "test"
    directory.mkdir()
    for episode in (0, 1):
        for bot in BOT_NAMES:
            _write_bot_files(directory, episode, bot, num_frames=60)
    for bot in ("Alpha", "Bravo"):
        _write_bot_files(directory, 2, bot, num_frames=60)
    return directory


def _dataset(directory, bot_names, dataset_name="trio"):
    return DatasetMultiplayer(
        data_dir=str(directory),
        dataset_name=dataset_name,
        bot_names=bot_names,
        converters=["CameraLinearConverterMatrixGame2"],
        obs_resize=None,
        shuffle_bots=False,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_bots", [2, 3, 5])
def test_segment_id_to_list_round_trip(num_bots):
    starts = [10 * i for i in range(num_bots)]
    stops = [s + 33 for s in starts]
    seg = SegmentIdMultiplayer(7, starts, stops)
    flat = seg.to_list()
    assert seg.num_bots == num_bots
    assert flat == [7] + [v for pair in zip(starts, stops) for v in pair]
    assert SegmentIdMultiplayer(flat[0], flat[1::2], flat[2::2]) == seg


def test_dataset_drops_episode_with_missing_bot(trio_dir):
    ds = _dataset(trio_dir, BOT_NAMES)
    assert ds.num_episodes == 2
    for episode_id in range(ds.num_episodes):
        paths = ds.get_episode_paths(episode_id)
        assert len(paths["video_paths"]) == len(paths["actions_paths"]) == 3
        for bot, video in zip(BOT_NAMES, paths["video_paths"]):
            assert f"_{bot}_" in video
    # The episode without Charlie is valid for a 2-bot dataset.
    assert _dataset(trio_dir, ["Alpha", "Bravo"]).num_episodes == 3


def test_sampler_aligns_bots_by_render_time(trio_dir):
    ds = _dataset(trio_dir, BOT_NAMES)
    num_frames = 9
    sampler = BatchSamplerMultiplayer(
        ds, rank=0, batch_size=4, num_replicas=1, num_frames=num_frames, seed=[0, 1]
    )
    for _ in range(5):
        for seg in sampler.sample():
            assert seg.num_bots == 3
            assert seg.stops == [s + num_frames for s in seg.starts]
            # Bravo started 2 frames later, so its index is 2 frames behind.
            assert seg.starts[0] == seg.starts[1] + 2 == seg.starts[2]


def test_dataset_item_and_batch_shapes(trio_dir):
    ds = _dataset(trio_dir, BOT_NAMES)
    num_frames = 9
    sampler = BatchSamplerMultiplayer(
        ds, rank=0, batch_size=2, num_replicas=1, num_frames=num_frames, seed=[0, 1]
    )
    segs = sampler.sample()
    items = [ds[s] for s in segs]
    height, width = VIDEO_HW
    for item in items:
        assert item.obs.shape == (num_frames, 3, height, width, 3)
        assert item.act.shape == (num_frames, 3, len(ACTION_KEYS))
        for p, bot in enumerate(BOT_NAMES):
            # Bot order follows bot_names when shuffle_bots is False.
            assert np.all(item.act[:, p, ACTION_KEYS.index(BOT_KEYS[bot])] == 1)
            assert abs(item.obs[:, p].mean() - BOT_GRAY[bot]) < 10
    batch = collate_segments_to_batch(num_frames, None, items)
    assert batch.obs.shape == (2, num_frames, 3, height, width, 3)
    assert batch.act.shape == (2, num_frames, 3, len(ACTION_KEYS))
    assert batch.real_lengths.tolist() == [num_frames, num_frames]


@pytest.mark.parametrize("dataset_name,num_bots", [("duet", 2), ("trio", 3)])
def test_eval_sampler_reads_eval_ids_of_any_player_count(trio_dir, dataset_name, num_bots):
    ds = _dataset(trio_dir, BOT_NAMES[:num_bots], dataset_name=dataset_name)
    num_frames = 257
    sampler = EvalBatchSampler(
        ds, rank=0, num_replicas=1, batch_size=1, num_frames=num_frames
    )
    with open(os.path.join(EVAL_IDS_DIR, f"eval_ids_{dataset_name}.json")) as f:
        raw = json.load(f)
    assert len(raw) == len(sampler.examples)
    for entry, seg in zip(raw, sampler.examples):
        assert len(entry) == 1 + 2 * num_bots
        assert seg.episode_id == entry[0]
        # The sampler keeps the start indices and recomputes the stops.
        assert seg.starts == entry[1::2]
        assert seg.stops == [s + num_frames for s in entry[1::2]]


# ---------------------------------------------------------------------------
# Multiplayer reshapes
# ---------------------------------------------------------------------------


def test_handle_multiplayer_output_concat_c_three_players():
    rng = np.random.default_rng(0)
    per_player_video = rng.normal(size=(2, 3, 4, 5, 6, 16)).astype(np.float32)
    concat_video = np.concatenate(
        [per_player_video[:, p : p + 1] for p in range(3)], axis=-1
    )
    assert concat_video.shape == (2, 1, 4, 5, 6, 48)
    out = handle_multiplayer_output(jnp.asarray(concat_video), "concat_c", num_players=3)
    np.testing.assert_array_equal(np.asarray(out), per_player_video)

    per_player_actions = rng.normal(size=(2, 3, 7, 23)).astype(np.float32)
    concat_actions = np.concatenate(
        [per_player_actions[:, p : p + 1] for p in range(3)], axis=-1
    )
    out = handle_multiplayer_output(
        jnp.asarray(concat_actions), "concat_c", num_players=3
    )
    np.testing.assert_array_equal(np.asarray(out), per_player_actions)

    # multiplayer_attn keeps the tensor unchanged.
    out = handle_multiplayer_output(
        jnp.asarray(per_player_video), "multiplayer_attn", num_players=3
    )
    np.testing.assert_array_equal(np.asarray(out), per_player_video)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# The action module asserts 880 tokens per frame, so the latent grid is 44 x 80.
LATENT_HW = (44, 80)
TINY_DIM = 64


def _tiny_model(num_players, multiplayer_method="multiplayer_attn", seed=0):
    action_config = OmegaConf.create(
        dict(
            blocks=list(range(30)),
            enable_keyboard=True,
            enable_mouse=True,
            heads_num=4,
            hidden_size=32,
            img_hidden_size=TINY_DIM,
            keyboard_dim_in=23,
            keyboard_hidden_dim=256,
            mouse_dim_in=2,
            mouse_hidden_dim=256,
            mouse_qk_dim_list=[8, 28, 28],
            patch_size=[1, 2, 2],
            qk_norm=True,
            qkv_bias=False,
            rope_dim_list=[8, 28, 28],
            rope_theta=256,
            vae_time_compression_ratio=4,
            windows_size=3,
            left_action_padding=11,
        )
    )
    # num_layers stays 30 because the forward pass zips the action module over 30 layers.
    return SolarisMPModel(
        dim=TINY_DIM,
        ffn_dim=128,
        num_heads=1,
        num_layers=30,
        local_attn_size=6,
        action_config=action_config,
        cross_attn_norm=False,
        num_players=num_players,
        multiplayer_method=multiplayer_method,
        platform="cpu",
        rngs=nnx.Rngs(seed),
    )


def _model_inputs(batch, num_players, num_latent_frames, seed=0):
    height, width = LATENT_HW
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    shape = (batch, num_players, num_latent_frames, height, width)
    num_raw_frames = 4 * (num_latent_frames - 1) + 1
    return dict(
        x_BPFHWC=jax.random.normal(k[0], shape + (16,)).astype(jnp.bfloat16),
        t_BPT=jnp.full((batch, num_players, num_latent_frames), 500, jnp.int32),
        visual_context=jax.random.normal(k[1], (batch, num_players, 4, 1280)).astype(
            jnp.bfloat16
        ),
        cond_concat=jax.random.normal(k[2], shape + (20,)).astype(jnp.bfloat16),
        mouse_cond=left_repeat_padding(
            jax.random.normal(k[3], (batch, num_players, num_raw_frames, 2)).astype(
                jnp.bfloat16
            ),
            11,
            axis=2,
        ),
        keyboard_cond=left_repeat_padding(
            jnp.zeros((batch, num_players, num_raw_frames, 23), jnp.bfloat16), 11, axis=2
        ),
    )


@pytest.fixture(scope="module")
def tiny_model_3p():
    return _tiny_model(num_players=3)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_mp_model_forward_three_players(tiny_model_3p, bidirectional):
    inputs = _model_inputs(batch=1, num_players=3, num_latent_frames=2)
    out, _, _, _ = tiny_model_3p(**inputs, bidirectional=bidirectional)
    assert out.shape == (1, 3, 2) + LATENT_HW + (16,)
    assert bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))


def test_mp_model_rejects_wrong_player_count(tiny_model_3p):
    inputs = _model_inputs(batch=1, num_players=2, num_latent_frames=2)
    with pytest.raises(AssertionError, match="Expected P=3"):
        tiny_model_3p(**inputs, bidirectional=True)


def test_initialize_kv_cache_scales_with_players(tiny_model_3p):
    height, width = LATENT_HW[0] // 2, LATENT_HW[1] // 2
    cache = tiny_model_3p.initialize_kv_cache(2, height, width, eff_p=3)
    tokens = 6 * height * width * 3
    assert cache.kv_cache.k.shape == (30, 2, tokens, 1, TINY_DIM)
    assert cache.kv_cache_mouse.k.shape[:3] == (30, 3 * 2 * height * width, 6)
    assert cache.kv_cache_keyboard.k.shape[:3] == (30, 3 * 2, 6)


def _abstract_state(num_players):
    def init():
        return nnx.state(_tiny_model(num_players))

    return jax.eval_shape(init)


def test_three_player_params_differ_only_in_player_embed():
    shapes_2p = {
        path: leaf.shape
        for path, leaf in jax.tree_util.tree_leaves_with_path(_abstract_state(2))
    }
    shapes_3p = {
        path: leaf.shape
        for path, leaf in jax.tree_util.tree_leaves_with_path(_abstract_state(3))
    }
    assert shapes_2p.keys() == shapes_3p.keys()
    diff = {p for p in shapes_2p if shapes_2p[p] != shapes_3p[p]}
    assert len(diff) == 1
    (path,) = diff
    assert "player_embed" in jax.tree_util.keystr(path)
    assert shapes_2p[path] == (2, TINY_DIM) and shapes_3p[path] == (3, TINY_DIM)


def test_non_strict_restore_pads_two_player_checkpoint(tmp_path):
    """pretrained_model_strict_load=False loads a 2-player checkpoint into a 3-player model."""
    state_2p = nnx.state(_tiny_model(num_players=2, seed=1))
    checkpointer = ocp.StandardCheckpointer()
    path = tmp_path / "two_player.pt"
    checkpointer.save(path, state_2p)
    checkpointer.wait_until_finished()

    target_3p = nnx.state(_tiny_model(num_players=3, seed=2))
    with pytest.raises(Exception):
        checkpointer.restore(path, target_3p)
    restored = checkpointer.restore(path, target_3p, strict=False)

    embed_2p = np.asarray(state_2p["player_embed"]["embedding"].value)
    embed_3p = np.asarray(restored["player_embed"]["embedding"].value)
    assert embed_3p.shape == (3, TINY_DIM)
    np.testing.assert_array_equal(embed_3p[:2], embed_2p)
    np.testing.assert_array_equal(embed_3p[2], np.zeros(TINY_DIM))
    np.testing.assert_array_equal(
        np.asarray(restored["patch_embedding"]["kernel"].value),
        np.asarray(state_2p["patch_embedding"]["kernel"].value),
    )
