from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np


@dataclass
class SegmentId:
    episode_id: Union[int, str]
    start: int
    stop: int

    def to_list(self) -> list[int]:
        return [self.episode_id, self.start, self.stop]


@dataclass
class SegmentIdMultiplayer:
    episode_id: Union[int, str]
    starts: list[int]
    stops: list[int]

    @property
    def num_bots(self) -> int:
        return len(self.starts)

    def to_list(self) -> list[int]:
        result = [self.episode_id]
        for s, e in zip(self.starts, self.stops):
            result.extend([s, e])
        return result


@dataclass
class Segment:
    obs: np.ndarray
    act: np.ndarray

    def __len__(self) -> int:
        return self.obs.shape[0]
