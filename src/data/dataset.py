import json
import logging
import random
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from torch.utils.data import Dataset as TorchDataset

from . import minecraft
from .batch import Batch
from .segment import Segment


class InputConverter:
    def convert(self, act):
        return act


class CameraLinearConverterMatrixGame2(InputConverter):
    def convert(self, act):
        act[:, 23] = minecraft.compress_mouse_linear(act[:, 23])
        act[:, 24] = minecraft.compress_mouse_linear(act[:, 24])
        return act


input_converters = {
    "CameraLinearConverterMatrixGame2": CameraLinearConverterMatrixGame2(),
}


class VideoReadError(Exception):
    pass


class Dataset(TorchDataset):

    def __init__(
        self,
        data_dir,
        dataset_name,
        converters,
        obs_resize=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()

        with open(self.directory / "episodes_info.json", "r") as json_file:
            self.episodes_info = json.load(json_file)
        self._num_episodes = self.episodes_info["num_episodes"]
        self._lengths = np.array(
            [ep["length"] for ep in self.episodes_info["episodes"]]
        )
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]

    @property
    def num_episodes(self):
        return self._num_episodes

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def lengths(self):
        return self._lengths

    def __getitem__(self, segment_id):
        episode_info = self.episodes_info["episodes"][segment_id.episode_id]
        video_path = self.directory / episode_info["video_path"]
        actions_path = self.directory / episode_info["actions_path"]
        try:
            decord_video = VideoReader(str(video_path), ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"Error reading video {video_path}: {e}") from e

        try:
            act = self.read_act_slice(actions_path, segment_id.start, segment_id.stop)
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e

        try:
            obs_decord = minecraft.read_obs_slice_decord(
                decord_video,
                segment_id.start,
                segment_id.stop,
                self._obs_resize,
            )

        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        for converter in self._converters:
            act = converter.convert(act)
        segment = Segment(obs_decord, act)

        return segment

    def read_act_slice(
        self,
        path,
        start,
        stop,
    ):
        full_actions_json = read_actions_json(path)
        return minecraft.read_act_slice_vpt(
            full_actions_json,
            start,
            stop,
        )


class DatasetMultiplayer(TorchDataset):
    def __init__(
        self,
        data_dir,
        dataset_name,
        bot_names,
        converters,
        obs_resize=None,
        shuffle_bots=True,
        shuffle_bot_seed=None,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.directory = Path(data_dir).expanduser()
        self.bot_names = list(bot_names)
        self._obs_resize = obs_resize
        self._converters = [input_converters[c] for c in converters]
        self._shuffle_bots = shuffle_bots
        self._shuffle_bot_seed = shuffle_bot_seed
        # Create a separate Random instance for shuffling
        self._rng = random.Random(shuffle_bot_seed)
        self._episodes = {}

        anchor_name = self.bot_names[0]
        mp4_files = sorted(
            [p for p in self.directory.glob(f"*_{anchor_name}_*.mp4")],
            key=lambda p: p.name,
        )
        episode_id = 0

        for anchor_video_path in mp4_files:
            anchor_name_str = anchor_video_path.name

            video_paths = [anchor_video_path]
            actions_path = anchor_video_path.with_suffix(".json")
            if actions_path.name.endswith("_camera.json"):
                actions_path = actions_path.with_name(
                    actions_path.name.replace("_camera.json", ".json")
                )
            actions_paths = [actions_path]

            # Find counterpart files for each additional bot
            valid = actions_path.exists()
            for other_name in self.bot_names[1:]:
                counterpart_name = anchor_name_str.replace(
                    f"_{anchor_name}_", f"_{other_name}_", 1
                )
                other_video_path = self.directory / counterpart_name
                other_actions_path = other_video_path.with_suffix(".json")
                if other_actions_path.name.endswith("_camera.json"):
                    other_actions_path = other_actions_path.with_name(
                        other_actions_path.name.replace("_camera.json", ".json")
                    )
                if not (other_video_path.exists() and other_actions_path.exists()):
                    valid = False
                    break
                video_paths.append(other_video_path)
                actions_paths.append(other_actions_path)

            if valid:
                self._episodes[episode_id] = {
                    "video_paths": [p.name for p in video_paths],
                    "actions_paths": [p.name for p in actions_paths],
                }
                episode_id += 1

        self._num_episodes = len(self._episodes)
        logging.info(
            f"Dataset {self.dataset_name} loaded {self._num_episodes} episodes. Data directory: {self.directory}"
        )

    def action_dim(self):
        return len(minecraft.ACTION_KEYS)

    @property
    def num_episodes(self):
        return self._num_episodes

    def get_episode_paths(self, episode_id):
        return self._episodes[episode_id]

    def __getitem__(self, segment_id):
        episode_paths = self.get_episode_paths(segment_id.episode_id)
        n_bots = len(self.bot_names)

        decord_videos = []
        for vp in episode_paths["video_paths"]:
            video_path = self.directory / vp
            try:
                decord_videos.append(VideoReader(str(video_path), ctx=cpu(0)))
            except Exception as e:
                raise ValueError(f"Error reading video {video_path}: {e}") from e

        try:
            all_actions = []
            for i, ap in enumerate(episode_paths["actions_paths"]):
                with open(self.directory / ap, "r") as f:
                    bot_actions = json.load(f)
                bot_actions = bot_actions[segment_id.starts[i] : segment_id.stops[i]]
                all_actions.append(
                    [{**a, "bot": self.bot_names[i]} for a in bot_actions]
                )
        except Exception as e:
            raise ValueError(
                f"Error reading episode actions {segment_id.episode_id}: {e}"
            ) from e

        try:
            all_obs = [
                minecraft.read_obs_slice_decord(
                    decord_videos[i],
                    segment_id.starts[i],
                    segment_id.stops[i],
                    self._obs_resize,
                )
                for i in range(n_bots)
            ]
        except Exception as e:
            raise VideoReadError(f"Error reading segment slice: {e}") from e

        all_act = [minecraft.convert_act_slice_mineflayer(a) for a in all_actions]
        for converter in self._converters:
            all_act = [converter.convert(a) for a in all_act]

        # Randomly shuffle bot order if enabled
        if self._shuffle_bots:
            indices = list(range(n_bots))
            self._rng.shuffle(indices)
            all_obs = [all_obs[i] for i in indices]
            all_act = [all_act[i] for i in indices]

        obs = np.stack(all_obs, axis=1)
        act = np.stack(all_act, axis=1)
        segment = Segment(obs, act)

        return segment


def read_actions_json(actions_path):
    try:
        with open(actions_path) as json_file:
            json_lines = json_file.readlines()
    except Exception as e_utf:
        try:
            with open(actions_path, encoding="windows-1252") as json_file:
                json_lines = json_file.readlines()
        except Exception as e_win:
            raise ValueError(
                f"Error reading file {actions_path} in utf-8 and windows-1252 encodings: {e_utf} / {e_win}"
            ) from e_win

    json_data = "[" + ",".join(json_lines) + "]"
    json_data = json.loads(json_data)
    return json_data


def collate_segments_to_batch(sequence_length, pad_batch_to, segments):
    # Filter out None segments
    valid_segments = [s for s in segments if s]
    if not valid_segments:
        raise ValueError("No valid segments to collate.")

    # Find max length for obs and act
    # Pad obs and act to max length
    obs_padded = []
    act_padded = []
    for s in valid_segments:
        obs = s.obs
        act = s.act

        # Calculate pad widths to pad with zeros on the right along the first dimension
        # Also collect the real (unpadded) lengths for each segment
        # real_lengths will be used to indicate where padding starts for each segment
        # (i.e., the original length of each segment)
        # This array will be of size B (number of valid segments)
        obs_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (obs.ndim - 1)
        act_pad_width = [(0, sequence_length - len(s))] + [(0, 0)] * (act.ndim - 1)
        obs_padded.append(np.pad(obs, obs_pad_width, mode="constant"))
        act_padded.append(np.pad(act, act_pad_width, mode="constant"))

    real_lengths = np.array([len(s) for s in valid_segments], dtype=np.int32)
    obs_batch = np.stack(obs_padded)
    act_batch = np.stack(act_padded)
    if pad_batch_to is not None and pad_batch_to > len(valid_segments):
        obs_batch = np.pad(
            obs_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (obs_batch.ndim - 1),
            mode="constant",
        )
        act_batch = np.pad(
            act_batch,
            [(0, pad_batch_to - len(valid_segments))] + [(0, 0)] * (act_batch.ndim - 1),
            mode="constant",
        )
        real_lengths = np.pad(
            real_lengths, (0, pad_batch_to - len(valid_segments)), mode="constant"
        )

    return Batch(obs_batch, act_batch, real_lengths)
