import json
import unittest
from pathlib import Path

import numpy as np

from src.data.minecraft import ACTION_KEYS, read_act_slice_vpt

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "vpt_action_slices"

# For each fixture we intentionally provide the *full* jsonl file as a list and
# choose explicit slice [start:stop] indices to validate.
SLICE_RANGES = {
    "all_off.jsonl": (0, 1),
    "each_binary_on.jsonl": (0, 33),
    "camera_ranges.jsonl": (0, 29),
    "attack_stuck.jsonl": (0, 4),
    # Non-zero starts to ensure read_act_slice_vpt correctly handles "attack stuck"
    # state based on entries *before* the slice range.
    "attack_stuck_offset_unstuck_before.jsonl": (2, 4),
    "attack_stuck_offset_unstuck_during.jsonl": (1, 5),
    # Non-zero start to ensure read_act_slice_vpt correctly handles "last hotbar"
    # state based on entries *before* the slice range.
    "hotbar_offset_set_before_slice.jsonl": (2, 4),
}


def _load_jsonl_fixture(name):
    path = FIXTURES_DIR / name
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _expected_for_input_fixture(input_name):
    expected_name = input_name.replace(".jsonl", ".expected.json")
    expected = json.loads((FIXTURES_DIR / expected_name).read_text(encoding="utf-8"))
    arr = np.asarray(expected, dtype=np.float32)
    assert arr.shape[1] == len(ACTION_KEYS)
    return arr


def _assert_allclose_with_mismatch_report(
    actual,
    expected,
    *,
    rtol,
    atol,
    max_mismatches=30,
):
    """
    Wrapper around numpy's assert_allclose that, on failure, shows the exact
    mismatching indices and values (helpful when numpy prints truncated arrays).
    """
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        return
    except AssertionError as e:
        is_close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
        mismatch_idx = np.argwhere(~is_close)

        lines = [
            str(e).rstrip(),
            "",
            f"actual.dtype={actual.dtype} expected.dtype={expected.dtype}",
            "First mismatches:",
        ]
        if mismatch_idx.size == 0:
            # Should be rare (e.g. dtype/shape mismatch) but keep message robust.
            lines.append(
                "  (no mismatching indices found via isclose; see error above)"
            )
        else:
            for n, idx in enumerate(mismatch_idx[:max_mismatches]):
                idx_tup = tuple(int(i) for i in idx.tolist())
                a = float(actual[tuple(idx)])
                b = float(expected[tuple(idx)])
                abs_diff = abs(a - b)
                rel_diff = abs_diff / abs(b) if b != 0 else float("inf")

                extra = ""
                if (
                    actual.ndim == 2
                    and len(idx_tup) == 2
                    and idx_tup[1] < len(ACTION_KEYS)
                ):
                    extra = f" key={ACTION_KEYS[idx_tup[1]]}"

                lines.append(
                    f"  {n+1:>2}. idx={idx_tup}{extra} actual={a:.6f} expected={b:.6f} "
                    f"abs_diff={abs_diff:.6f} rel_diff={rel_diff:.6f}"
                )

            remaining = mismatch_idx.shape[0] - min(
                mismatch_idx.shape[0], max_mismatches
            )
            if remaining > 0:
                lines.append(f"  ... and {remaining} more mismatches")

        raise AssertionError("\n".join(lines)) from e


class TestReadActSliceVPT(unittest.TestCase):
    def test_all_off_fixture(self):
        input_name = "all_off.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[input_name]
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_each_binary_on_fixture(self):
        input_name = "each_binary_on.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[input_name]
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

        # Sanity check: every non-camera action key is exercised at least once.
        exercised = set()
        for t in range(out.shape[0]):
            for k in ACTION_KEYS:
                if k.startswith("camera"):
                    continue
                if out[t, ACTION_KEYS.index(k)] == 1.0:
                    exercised.add(k)
        missing = {k for k in ACTION_KEYS if not k.startswith("camera")} - exercised
        self.assertEqual(missing, set())

    def test_camera_ranges_fixture(self):
        input_name = "camera_ranges.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[input_name]
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        _assert_allclose_with_mismatch_report(out, expected, rtol=1e-3, atol=1e-3)

    def test_attack_stuck_fixture(self):
        input_name = "attack_stuck.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[input_name]
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_attack_stuck_nonzero_start_unstuck_before_slice(self):
        # Regression test: the fixture contains episode-start stuck attack, then becomes
        # unstuck *before* the slice begins. The slice begins at an index where
        # mouse.newButtons == [0] (a normal press), which should NOT be misinterpreted
        # as "attack stuck" just because the slice starts there.
        input_name = "attack_stuck_offset_unstuck_before.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[input_name]
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_attack_stuck_nonzero_start_unstuck_during_slice(self):
        # Regression test: slice starts after t=0, but "attack stuck" was detected at
        # true episode start (t=0). The action should remain removed until it becomes
        # unstuck *during* the slice.
        input_name = "attack_stuck_offset_unstuck_during.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[
            input_name
        ]  # non-zero start; unstuck occurs at t=2 within this slice
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)

    def test_hotbar_nonzero_start_inherits_last_hotbar_before_slice(self):
        # Regression test: hotbar selection is stateful. If the last hotbar was set
        # before the slice begins, read_act_slice_vpt must inherit that state rather
        # than assuming the slice starts from hotbar=0.
        input_name = "hotbar_offset_set_before_slice.jsonl"
        actions = _load_jsonl_fixture(input_name)
        start, stop = SLICE_RANGES[
            input_name
        ]  # non-zero start; last_hotbar set before this slice
        out = read_act_slice_vpt(actions, start=start, stop=stop)
        expected = _expected_for_input_fixture(input_name)

        self.assertEqual(out.shape, (stop - start, len(ACTION_KEYS)))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, rtol=0, atol=0)
