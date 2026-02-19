import json
import unittest
from pathlib import Path

import numpy as np

from src.data.minecraft import ACTION_KEYS, convert_act_slice_mineflayer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "mineflayer_action_slices"


def _load_fixture(name):
    path = FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expected_for_input_fixture(input_name):
    expected_name = input_name.replace(".json", ".expected.json")
    expected = _load_fixture(expected_name)
    arr = np.asarray(expected, dtype=np.float32)
    assert arr.shape[1] == len(ACTION_KEYS)
    return arr


class TestConvertActSliceMineflayer(unittest.TestCase):
    def test_all_off_fixture(self):
        input_name = "all_off.json"
        actions = _load_fixture(input_name)
        out = convert_act_slice_mineflayer(actions)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (1, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)

        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

        # Explicitly verify unsupported keys remain zero (inventory/ESC/swapHands/pickItem/drop)
        for key in ("inventory", "ESC", "swapHands", "pickItem", "drop"):
            self.assertEqual(out[0, ACTION_KEYS.index(key)], 0.0)

    def test_each_boolean_on_fixture(self):
        input_name = "each_boolean_on.json"
        actions = _load_fixture(input_name)
        out = convert_act_slice_mineflayer(actions)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (len(actions), len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)

        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_camera_ranges_fixture(self):
        input_name = "camera_ranges.json"
        actions = _load_fixture(input_name)
        out = convert_act_slice_mineflayer(actions)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (len(actions), len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)

        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)
