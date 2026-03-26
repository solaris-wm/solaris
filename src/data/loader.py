import functools
import math

import jax
import torch

from .batch_sampler import BatchSampler, BatchSamplerMultiplayer, EvalBatchSampler
from .dataset import DatasetMultiplayer, collate_segments_to_batch
from .utils import anything_to_seed


def build_data_loader(
    dataset,
    batch_size,
    num_workers,
    num_frames,
    seed_data,
    eval,
    eval_num_samples=None,
    eval_pseudo_process_index=None,
    eval_pseudo_process_count=None,
    seed_offset=None,
):
    """Build a torch DataLoader.

    In eval mode, ``batch_size`` is the global per-step minibatch
    (``device.eval_minibatch_size``); ``eval_num_samples`` is how many eval
    segments to run in total (``device.eval_num_samples`` / runner param).
    """

    batch_size = batch_size
    local_batch_size = batch_size // jax.process_count()

    if eval:
        assert eval_num_samples is not None, "eval_num_samples is required in eval mode"
        assert batch_size > 0, "batch_size must be positive in eval mode"
        assert batch_size <= eval_num_samples, (
            f"global batch_size ({batch_size}) must be <= eval_num_samples "
            f"({eval_num_samples}); lower device.eval_minibatch_size (or raise "
            f"eval_num_samples)"
        )
        assert batch_size % jax.process_count() == 0, (
            f"batch_size ({batch_size}) must be divisible by jax.process_count() "
            f"({jax.process_count()})"
        )
        if (
            eval_pseudo_process_index is not None
            or eval_pseudo_process_count is not None
        ):
            assert (
                not jax.distributed.is_initialized()
            ), "jax.distributed should be disabled with eval_pseudo_process_index and eval_pseudo_process_count!"
            assert (
                batch_size == local_batch_size
            ), "batch_size should be equal to local_batch_size when using eval_pseudo_process_index and eval_pseudo_process_count!"
        rank = (
            eval_pseudo_process_index
            if eval_pseudo_process_index is not None
            else jax.process_index()
        )
        num_replicas = (
            eval_pseudo_process_count
            if eval_pseudo_process_count is not None
            else jax.process_count()
        )
        sampler = EvalBatchSampler(
            dataset,
            rank=rank,
            num_replicas=num_replicas,
            batch_size=local_batch_size,
            num_frames=num_frames,
            num_global_samples=eval_num_samples,
        )
        num_batches = len(sampler)
        pad_batch_to = calculate_last_batch_padding(eval_num_samples, batch_size)
    else:
        if isinstance(dataset, DatasetMultiplayer):
            sampler = BatchSamplerMultiplayer(
                dataset,
                rank=jax.process_index(),
                num_replicas=jax.process_count(),
                batch_size=local_batch_size,
                num_frames=num_frames,
            )
        else:

            sampler = BatchSampler(
                dataset,
                rank=jax.process_index(),
                num_replicas=jax.process_count(),
                batch_size=local_batch_size,
                num_frames=num_frames,
                seed=(
                    [anything_to_seed("sampler"), seed_data] + [seed_offset]
                    if seed_offset is not None
                    else []
                ),
            )

        if eval_num_samples is not None:
            num_batches = math.ceil(
                eval_num_samples / jax.process_count() / local_batch_size
            )
            pad_batch_to = calculate_last_batch_padding(eval_num_samples, batch_size)
        else:
            num_batches = None
            pad_batch_to = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=functools.partial(
            collate_segments_to_batch, sampler.num_frames, pad_batch_to
        ),
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader, num_batches


def calculate_last_batch_padding(num_global_samples, batch_size):
    last_batch_samples = num_global_samples % batch_size
    num_devices = jax.device_count()
    per_device = math.ceil(last_batch_samples / num_devices)
    return per_device * jax.local_device_count()
