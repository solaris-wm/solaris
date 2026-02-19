import json
import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import CameraLinearConverterMatrixGame2
from src.data.minecraft import ACTION_KEYS

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "dataset_converters"


def _load_array(name):
    data = json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=np.float32)
    assert arr.ndim == 2
    assert arr.shape[1] == len(ACTION_KEYS)
    return arr


class TestCameraLinearConverterMatrixGame2(unittest.TestCase):
    def test_camera_linear_matrix_game2_fixture(self):
        inp = _load_array("camera_linear_matrix_game2.input.json")
        expected = _load_array("camera_linear_matrix_game2.expected.json")

        out = CameraLinearConverterMatrixGame2().convert(inp.copy())

        self.assertEqual(out.shape, expected.shape)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1e-6)
