import os
import tempfile

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy as np
import torch
from absl import logging
from einops import rearrange, repeat
from flax import nnx
from torchvision.io import write_video

import src.data.utils as data_utils
import src.utils.sharding as sharding_utils
from src.metrics.compute_metrics import FIDCalculator, calculate_metrics_from_batch
from src.models.utils import jax_to_torch
from src.models.wan_vae import VAE_SCALE
from src.runners.base_runner import BaseRunner
from src.utils.multiplayer import handle_multiplayer_input, handle_multiplayer_output
from src.utils.preprocessing_mp import wan_image_condition_preprocess
from src.utils.rollout import (
    change_tensor_range,
    perform_bidirectional_multiplayer_rollout,
    perform_multiplayer_rollout,
)


def get_model_inputs_for_eval(
    video_BPFHWC,
    clip_model,
    vae_model,
    actions_mouse_BPFD,
    actions_keyboard_BPFD,
    multiplayer_method="multiplayer_attn",
):
    B, P, F, H, W, C = video_BPFHWC.shape
    first_frame_BPFHWC = video_BPFHWC[:, :, 0:1, :, :, :]
    visual_context_BPFD = rearrange(
        jax.lax.stop_gradient(
            clip_model.encode_video(
                rearrange(first_frame_BPFHWC, "b p f h w c -> (b p) c f h w")
            )
        ).astype(jnp.bfloat16),
        "(b p) f d -> b p f d",
        b=B,
    )
    compress = lambda x: rearrange(x, "b p f h w c -> (b p) f h w c")
    uncompress = lambda x: rearrange(x, "(b p) f h w c -> b p f h w c", b=B)
    img_cond_BPFHWC = uncompress(
        jax.lax.stop_gradient(
            vae_model.encode(
                compress(
                    jnp.concatenate(
                        [
                            video_BPFHWC[:, :, :1, :, :, :],
                            jnp.zeros((B, P, F - 1, H, W, C)),
                        ],
                        axis=2,
                    ).astype(jnp.bfloat16)
                ),
                VAE_SCALE,
            )
        )
    )
    mask_cond_BPFHWC = (
        jnp.ones(img_cond_BPFHWC.shape, dtype=jnp.bfloat16).at[:, :, 1:].set(0)
    )
    cond_concat_BPFHWC = jnp.concatenate(
        [mask_cond_BPFHWC[:, :, :, :, :, :4], img_cond_BPFHWC], axis=-1
    ).astype(jnp.bfloat16)

    (
        cond_concat_BPFHWC,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
        visual_context_BPFD,
        _,
    ) = handle_multiplayer_input(
        cond_concat_BPFHWC,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
        multiplayer_method,
        visual_context_BPFD,
        video_BPFHWC=None,
    )

    return (
        cond_concat_BPFHWC,
        visual_context_BPFD,
        actions_mouse_BPFD,
        actions_keyboard_BPFD,
    )


def get_model_inputs(
    video_BPFHWC,
    clip_model,
    vae_model,
):
    B, P, F, H, W, C = video_BPFHWC.shape
    first_frame_BPFHWC = video_BPFHWC[:, :, 0:1, :, :, :]
    visual_context_BPFD = rearrange(
        jax.lax.stop_gradient(
            clip_model.encode_video(
                rearrange(first_frame_BPFHWC, "b p f h w c -> (b p) c f h w")
            )
        ).astype(jnp.bfloat16),
        "(b p) f d -> b p f d",
        b=B,
    )
    compress = lambda x: rearrange(x, "b p f h w c -> (b p) f h w c")
    uncompress = lambda x: rearrange(x, "(b p) f h w c -> b p f h w c", b=B)
    encoded_outputs = uncompress(
        jax.lax.stop_gradient(
            vae_model.cacheless_encode(compress(video_BPFHWC), VAE_SCALE)
        )
    )
    img_cond_BPFHWC = uncompress(
        jax.lax.stop_gradient(
            vae_model.cacheless_encode(
                compress(
                    jnp.concatenate(
                        [
                            video_BPFHWC[:, :, :1, :, :, :],
                            jnp.zeros((B, P, F - 1, H, W, C)),
                        ],
                        axis=2,
                    ).astype(jnp.bfloat16)
                ),
                VAE_SCALE,
            )
        )
    )
    mask_cond_BPFHWC = (
        jnp.ones(img_cond_BPFHWC.shape, dtype=jnp.bfloat16).at[:, :, 1:].set(0)
    )
    cond_concat_BPFHWC = jnp.concatenate(
        [mask_cond_BPFHWC[:, :, :, :, :, :4], img_cond_BPFHWC], axis=-1
    ).astype(jnp.bfloat16)
    return encoded_outputs, cond_concat_BPFHWC, visual_context_BPFD


class BaseMPRunner(BaseRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiplayer_method = self.network_config.params.multiplayer_method

    def _get_curr_batch(self, train_loader_iter):
        batch = self.robust_batch_sample(train_loader_iter)
        batch = batch.to_dict()
        batch = data_utils.torch_pytree_to_numpy(batch)
        batch = self.globalize_batch(batch)
        real_lengths = batch["real_lengths"]
        video_BPFHWC = rearrange(
            batch["obs"].astype(jnp.bfloat16), "b f p h w c -> b p f h w c"
        )
        action_BPFD = rearrange(batch["act"].astype(jnp.bfloat16), "b f p d -> b p f d")

        actions_mouse_BPFD = action_BPFD[:, :, :, -2:]
        actions_keyboard_BPFD = action_BPFD[:, :, :, :-2]
        video_BPFHWC_unprocessed = video_BPFHWC
        video_BPFHWC = wan_image_condition_preprocess(video_BPFHWC, 352, 640).astype(
            jnp.bfloat16
        )
        return (
            video_BPFHWC,
            video_BPFHWC_unprocessed,
            actions_mouse_BPFD,
            actions_keyboard_BPFD,
            real_lengths,
        )

    def _build_eval_rollout_func(self, bidirectional, n_frames_decode, left_action_padding):
        if bidirectional:
            from functools import partial

            return partial(
                perform_bidirectional_multiplayer_rollout,
                left_action_padding=left_action_padding,
                multiplayer_method=self.multiplayer_method,
            )

        def _rollout_func(
            model_graph,
            model_state,
            vae_graph,
            vae_state,
            clip_graph,
            clip_state,
            first_frames_BPHWC,
            mouse_actions_BPTD,
            keyboard_actions_BPTD,
            frames_to_decode,
            mesh,
            num_denoising_steps=None,
        ):
            model = nnx.merge(model_graph, model_state)
            vae_model = nnx.merge(vae_graph, vae_state)
            clip_model = nnx.merge(clip_graph, clip_state)
            repeated_first_frames_BPFHWC = repeat(
                first_frames_BPHWC, "b p h w c -> b p f h w c", f=n_frames_decode
            )
            (
                cond_concat_BPFHWC,
                visual_context_BPFD,
                mouse_actions_BPTD,
                keyboard_actions_BPTD,
            ) = get_model_inputs_for_eval(
                repeated_first_frames_BPFHWC,
                clip_model,
                vae_model,
                mouse_actions_BPTD,
                keyboard_actions_BPTD,
                self.multiplayer_method,
            )
            rngs = jax.random.PRNGKey(0)

            final_frame_BPFHWC, _, _ = perform_multiplayer_rollout(
                model,
                cond_concat_BPFHWC,
                visual_context_BPFD,
                mouse_actions_BPTD,
                keyboard_actions_BPTD,
                rngs,
                mesh,
                left_action_padding,
                num_denoising_steps=num_denoising_steps,
            )
            final_frame_BPFHWC = handle_multiplayer_output(
                final_frame_BPFHWC, self.multiplayer_method, num_players=2
            )
            B = final_frame_BPFHWC.shape[0]
            decoded_final_frame_BPFHWC = rearrange(
                vae_model.decode(
                    rearrange(
                        final_frame_BPFHWC.astype(jnp.bfloat16),
                        "b p f h w c -> (b p) f h w c",
                    ),
                    scale=VAE_SCALE,
                ),
                "(b p) f h w c -> b p f h w c",
                b=B,
            )
            decoded_final_frame_BPFHWC = change_tensor_range(
                decoded_final_frame_BPFHWC, [-1, 1], [0, 255], dtype=jnp.uint8
            )
            return decoded_final_frame_BPFHWC

        return jax.jit(
            _rollout_func,
            static_argnames=(
                "model_graph",
                "vae_graph",
                "clip_graph",
                "mesh",
                "num_denoising_steps",
            ),
        )

    def run_eval(
        self,
        model,
        num_denoising_steps,
        eval_dataloader_info,
        eval_dir_name,
    ):
        local_num_batches = eval_dataloader_info["local_num_batches"]
        eval_num_frames = eval_dataloader_info["dataloader"].batch_sampler.num_frames
        eval_iter = iter(eval_dataloader_info["dataloader"])

        if str(self.eval_save_dir).startswith("gs://"):
            evaluation_output_directory = tempfile.mkdtemp()
        else:
            evaluation_output_directory = os.path.join(
                self.eval_save_dir, eval_dir_name
            )
            os.makedirs(evaluation_output_directory, exist_ok=True)

        model.eval()
        eval_graph, eval_state = nnx.split(model)
        vae_graph, vae_state = nnx.split(self.vae_model)
        clip_graph, clip_state = nnx.split(self.clip_model)

        bidirectional = getattr(self, "bidirectional", False)
        rollout_func = self._build_eval_rollout_func(
            bidirectional, eval_num_frames, self.left_action_padding
        )
        num_sources = 1 if self.multiplayer_method is None else 2
        fid_calculator = FIDCalculator(num_sources=num_sources)

        video_offset = 0
        for batch_idx in range(local_num_batches):
            # Plain next(): robust_batch_sample retries can advance the iterator on
            # only some hosts and desynchronize multi-host collectives.
            try:
                batch = next(eval_iter)
            except StopIteration as e:
                raise RuntimeError(
                    f"Eval dataloader exhausted at batch {batch_idx + 1}/"
                    f"{local_num_batches} (iterator shorter than local_num_batches?)"
                ) from e
            batch = batch.to_dict()
            batch = data_utils.torch_pytree_to_numpy(batch)
            batch = self.globalize_batch(batch)
            real_lengths = batch["real_lengths"]
            video_b = rearrange(
                batch["obs"].astype(jnp.bfloat16), "b f p h w c -> b p f h w c"
            )
            action_BPFD = rearrange(
                batch["act"].astype(jnp.bfloat16), "b f p d -> b p f d"
            )
            actions_mouse_BPFD = action_BPFD[:, :, :, -2:]
            actions_keyboard_BPFD = action_BPFD[:, :, :, :-2]

            logging.info(
                "Evaluating batch %d/%d",
                batch_idx + 1,
                local_num_batches,
            )

            self.evaluate_mp(
                bidirectional,
                eval_state,
                eval_graph,
                vae_state,
                vae_graph,
                clip_state,
                clip_graph,
                video_b,
                actions_mouse_BPFD,
                actions_keyboard_BPFD,
                real_lengths,
                eval_dir=evaluation_output_directory,
                mesh=self.mesh,
                left_action_padding=self.left_action_padding,
                num_denoising_steps=num_denoising_steps,
                rollout_func=rollout_func,
                shared_fid_calculator=fid_calculator,
                video_offset=video_offset,
            )
            video_offset += int(video_b.shape[0])

        metric_curve = {"fid": np.array(fid_calculator.get_fid_curve_jax())}
        for k, v in metric_curve.items():
            logging.info("test_%s: %s", k, v.mean().item())
        return metric_curve

    def evaluate_mp(
        self,
        bidirectional,
        model_state,
        model_graph,
        vae_state,
        vae_graph,
        clip_state,
        clip_graph,
        video,
        mouse_actions,
        keyboard_actions,
        real_lengths,
        eval_dir,
        mesh,
        left_action_padding,
        num_denoising_steps=None,
        rollout_func=None,
        shared_fid_calculator=None,
        video_offset=0,
    ):
        num_eval_frames = video.shape[2]
        processed_video_BPFHWC = wan_image_condition_preprocess(video, 352, 640)
        video = change_tensor_range(
            processed_video_BPFHWC, [-1, 1], [0, 255], dtype=jnp.uint8
        )
        first_frame_BPHWC = processed_video_BPFHWC[:, :, 0, :, :, :].astype(
            jnp.bfloat16
        )

        if rollout_func is None:
            rollout_func = self._build_eval_rollout_func(
                bidirectional, num_eval_frames, left_action_padding
            )

        D = len(jax.devices())

        first_frame_DBPHWC = rearrange(
            first_frame_BPHWC, "(d b) p h w c -> d b p h w c", d=D
        )
        mouse_actions_DBPFD = rearrange(
            mouse_actions, "(device b) p f d -> device b p f d", device=D
        )
        keyboard_actions_DBPFD = rearrange(
            keyboard_actions, "(device b) p f d -> device b p f d", device=D
        )

        local_batch_size = first_frame_DBPHWC.shape[1]
        rollouts = []
        for i in range(local_batch_size):
            rollout = rollout_func(
                model_graph,
                model_state,
                vae_graph,
                vae_state,
                clip_graph,
                clip_state,
                first_frame_DBPHWC[:, i, :, :, :],
                mouse_actions_DBPFD[:, i, :, :],
                keyboard_actions_DBPFD[:, i, :, :],
                num_eval_frames,
                mesh=mesh,
                num_denoising_steps=num_denoising_steps,
            )
            rollouts.append(rollout)
        res_DBPFHWC = jnp.stack(rollouts, axis=1)
        rollout_frames_BPFHWC = rearrange(
            res_DBPFHWC, "d b p f h w c -> (d b) p f h w c", d=D
        )

        if eval_dir is not None:
            rollout_local_slice = sharding_utils.get_local_slice_from_fsarray(
                rollout_frames_BPFHWC
            )
            gt_local_slice = sharding_utils.get_local_slice_from_fsarray(video)

            torch_rollout_local_slice = jax_to_torch(rollout_local_slice)
            torch_gt_local_slice = jax_to_torch(gt_local_slice)
            torch_side_by_side_video_BPFHWC = torch.concatenate(
                [torch_gt_local_slice, torch_rollout_local_slice], axis=4
            )
            torch_side_by_side_video_BFHWC = rearrange(
                torch_side_by_side_video_BPFHWC, "b p f h w c -> b f (p h) w c"
            )

            num_processes = jax.process_count()
            process_index = jax.process_index()
            global_i = video_offset + process_index
            for i in range(torch_side_by_side_video_BFHWC.shape[0]):
                video_int8 = torch_side_by_side_video_BFHWC[i]
                write_video(
                    f"{eval_dir}/video_{global_i}_side_by_side.mp4", video_int8, fps=20
                )
                global_i += num_processes

        metrics = ["fid"]
        n_prompt_frames = 1

        if shared_fid_calculator is not None:
            fid_calculator = shared_fid_calculator
        else:
            num_sources = 1 if self.multiplayer_method is None else 2
            fid_calculator = FIDCalculator(num_sources=num_sources)

        pred_frames = rearrange(rollout_frames_BPFHWC, "b p f h w c -> b f p h w c")[
            :, n_prompt_frames:
        ]
        gt_frames = rearrange(video, "b p f h w c -> b f p h w c")
        gt_frames = gt_frames[:, n_prompt_frames:]
        calculate_metrics_from_batch(
            preds=pred_frames,
            targets=gt_frames,
            metrics_to_cal=metrics,
            real_lengths=real_lengths - n_prompt_frames,
            fid_calculator=fid_calculator,
        )

        if shared_fid_calculator is not None:
            return {}

        all_metrics = {"fid": np.array(fid_calculator.get_fid_curve_jax())}
        return all_metrics
