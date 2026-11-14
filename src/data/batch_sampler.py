import bisect
import json
import logging
import math
import os

import numpy as np
import torch

from .dataset import DatasetMultiplayer
from .segment import SegmentId, SegmentIdMultiplayer


class BatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def sample(self):
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        short_episode_ids = np.where(self.dataset.lengths < self.num_frames)[0]
        episodes_partition = episodes_partition[
            ~np.isin(episodes_partition, short_episode_ids)
        ]

        episode_ids = self.rng.choice(
            episodes_partition, size=self.batch_size, replace=True
        )
        starts = self.rng.integers(
            low=0,
            high=self.dataset.lengths[episode_ids] - self.num_frames + 1,
        )
        stops = starts + self.num_frames

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]


class BatchSamplerMultiplayer(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        self._seed = seed
        self.reset_rng()
        self._episode_infos = {}

    def __len__(self):
        raise NotImplementedError(
            "BatchSampler does not have a fixed length. Use __iter__ instead."
        )

    def __iter__(self):
        while True:
            yield self.sample()

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def get_episode_info(self, episode_id):
        if episode_id in self._episode_infos:
            return self._episode_infos[episode_id]
        episode_paths = self.dataset.get_episode_paths(episode_id)

        all_actions = []
        for ap in episode_paths["actions_paths"]:
            with open(self.dataset.directory / ap, "r") as f:
                bot_actions = json.load(f)
            if len(bot_actions) < 1:
                self._episode_infos[episode_id] = None
                logging.warning(
                    f"Episode {episode_id} {episode_paths} has less than 1 action, resampling"
                )
                return self._episode_infos[episode_id]
            all_actions.append(bot_actions)

        # Overlapping time window across all bots
        start_time = max(actions[0]["renderTime"] for actions in all_actions)
        end_time = min(actions[-1]["renderTime"] for actions in all_actions)

        # Extract render times for each bot
        all_times = [[a["renderTime"] for a in actions] for actions in all_actions]

        # Ensure times are strictly increasing for all bots
        for i, times in enumerate(all_times):
            if not all(t_next > t_prev for t_prev, t_next in zip(times, times[1:])):
                self._episode_infos[episode_id] = None
                logging.warning(
                    f"Episode {episode_id} {episode_paths} bot{i + 1} has non-increasing times, resampling"
                )
                return self._episode_infos[episode_id]

        # Compute valid index ranges for each bot
        start_indices = [bisect.bisect_left(times, start_time) for times in all_times]
        end_indices = [
            bisect.bisect_right(times, end_time) - 1 - self.num_frames
            for times in all_times
        ]

        for i, (si, ei) in enumerate(zip(start_indices, end_indices)):
            if ei - si < 1 or ei < 0 or si >= len(all_actions[i]):
                self._episode_infos[episode_id] = None
                logging.warning(
                    f"Episode {episode_id} {episode_paths} doesn't have enough actions for bot{i + 1}. "
                    f"start_idx: {si}, end_idx: {ei}, num_actions: {len(all_actions[i])}, dataset_dir: {self.dataset.directory}, resampling"
                )
                return self._episode_infos[episode_id]

        start_time = min(
            all_actions[i][si]["renderTime"] for i, si in enumerate(start_indices)
        )
        end_time = min(
            all_actions[i][ei]["renderTime"] for i, ei in enumerate(end_indices)
        )

        self._episode_infos[episode_id] = {
            "start_time": start_time,
            "end_time": end_time,
            "all_times": all_times,
        }
        return self._episode_infos[episode_id]

    def sample(self):
        num_episodes = self.dataset.num_episodes

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)

        segment_ids = []
        while len(segment_ids) < self.batch_size:
            episode_ids = self.rng.choice(
                episodes_partition,
                size=self.batch_size - len(segment_ids),
                replace=True,
            )
            for episode_id in episode_ids:
                episode_info = self.get_episode_info(episode_id)
                if episode_info is None:
                    continue
                start_time = self.rng.uniform(
                    episode_info["start_time"], episode_info["end_time"]
                )

                all_times = episode_info["all_times"]
                starts = [bisect.bisect_left(times, start_time) for times in all_times]
                stops = [s + self.num_frames for s in starts]
                segment_ids.append(SegmentIdMultiplayer(episode_id, starts, stops))
                if len(segment_ids) >= self.batch_size:
                    break

        return segment_ids


class EvalBatchSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        dataset,
        rank,
        batch_size,
        num_replicas,
        num_frames,
        num_global_samples=None,  # global number of samples
        seed=[0],
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = num_replicas
        self.batch_size = batch_size
        self.num_frames = num_frames
        logging.info(f"Eval num_frames: {num_frames}")
        self._seed = seed
        self.reset_rng()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        # hardcoded full episodes for worldmem_demo
        if "worldmem_demo" in str(dataset.directory):
            self.ids = [
                [0, 0, 1024],
                [1, 0, 1024],
                [2, 0, 1024],
                [3, 0, 1024],
                [4, 0, 1024],
                [5, 0, 1024],
            ]
        else:
            with open(
                os.path.join(
                    base_dir, "eval_ids", f"eval_ids_{dataset.dataset_name}.json"
                ),
                "r",
            ) as f:
                self.ids = json.load(f)

        assert num_frames <= 1024, "num_frames must be at most 1024"
        num_global_samples = (
            min(num_global_samples, len(self.ids))
            if num_global_samples is not None
            else len(self.ids)
        )
        if isinstance(dataset, DatasetMultiplayer):
            # eval_ids entry format: [episode_id, bot1_start, bot1_stop, bot2_start, bot2_stop, ...]
            self.examples = []
            for entry in self.ids:
                episode_id = entry[0]
                pairs = entry[1:]
                starts = [pairs[i * 2] for i in range(len(pairs) // 2)]
                self.examples.append(
                    SegmentIdMultiplayer(
                        episode_id,
                        starts,
                        [s + self.num_frames for s in starts],
                    )
                )
        else:
            # For other datasets, replace the endpoint with start + self.num_frames
            # Note this could potentially lead to an error later on when calling Dataset.__getitem__(...) if out of bounds
            self.examples = [
                SegmentId(episode_id, start, start + self.num_frames)
                for (episode_id, start, _) in self.ids
            ]
        self.examples = self.examples[:num_global_samples]
        self.examples = self.examples[self.rank :: self.world_size]
        self.num_batches = math.ceil(len(self.examples) / self.batch_size)

    def reset_rng(self):
        # Reset the random number generator to the initial seed
        self.rng = np.random.default_rng(self._seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.examples))
            yield self.examples[start:end]
