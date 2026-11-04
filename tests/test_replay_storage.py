"""Compact replay snapshots preserve sampling and physical ring-buffer writes."""

from copy import deepcopy
import io
import unittest

import numpy as np
import torch

from solvercell.learner import Replay


class ReplayStorageTests(unittest.TestCase):
    def replay(self, count=0):
        replay = Replay(capacity=7, obs_dim=4, action_dim=2, cost_dim=5)
        for index in range(count):
            self.add(replay, index)
        return replay

    def add(self, replay, index):
        replay.add(obs=np.arange(4, dtype=np.float32) + index,
                   action=np.array([index % 2, (index + 1) % 2], np.float32),
                   cost=np.arange(5, dtype=np.float32) / 7 - index,
                   reward=index / 3, next_obs=np.arange(4, dtype=np.float32) - index,
                   done=index % 3 == 0)

    def assert_replay_equal(self, left, right):
        self.assertEqual((left.capacity, left.size, left.pos),
                         (right.capacity, right.size, right.pos))
        self.assertEqual(left.data.keys(), right.data.keys())
        for key in left.data:
            np.testing.assert_array_equal(left.data[key], right.data[key])
            self.assertEqual(left.data[key].dtype, right.data[key].dtype)

    def roundtrip(self, state):
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=False)

    def test_empty_partial_full_and_wrapped_roundtrip_next_samples_and_writes(self):
        for count in (0, 4, 7, 12):
            for mode in ("compact", "full", "legacy"):
                with self.subTest(count=count, mode=mode):
                    original = self.replay(count)
                    state = (deepcopy(original.__dict__) if mode == "legacy" else
                             original.state_dict(compact=mode == "compact"))
                    rows = original.size if mode == "compact" else original.capacity
                    for value in state["data"].values():
                        self.assertEqual(len(value), rows)
                        if mode != "legacy":
                            self.assertIsInstance(value, torch.Tensor)
                            self.assertEqual(value.dtype, torch.float32)
                    restored = self.replay(3)
                    restored.load_state_dict(self.roundtrip(state))
                    self.assert_replay_equal(original, restored)
                    left_rng, right_rng = np.random.default_rng(271), np.random.default_rng(271)
                    # Cross the first wrap and a second wrap after restoring.
                    for index in range(count, count + 16):
                        if original.size:
                            left = original.sample(13, left_rng)
                            right = restored.sample(13, right_rng)
                            for key in left:
                                np.testing.assert_array_equal(left[key], right[key])
                            self.assertEqual(left_rng.bit_generator.state, right_rng.bit_generator.state)
                        self.add(original, index)
                        self.add(restored, index)
                        self.assert_replay_equal(original, restored)

    def test_snapshot_and_restore_do_not_alias_or_advance_rng(self):
        original = self.replay(4)
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        state = original.state_dict()
        expected = deepcopy(state)
        self.add(original, 42)
        original.data["obs"][0] = -100
        for key in state["data"]:
            np.testing.assert_array_equal(state["data"][key], expected["data"][key])
        restored = self.replay()
        restored.load_state_dict(state)
        state["data"]["obs"][0] = -200
        np.testing.assert_array_equal(restored.data["obs"][:4], expected["data"]["obs"])
        current_rng = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_rng[0])
        np.testing.assert_array_equal(numpy_rng[1], current_rng[1])
        self.assertEqual(numpy_rng[2:], current_rng[2:])
        torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_legacy_full_snapshot_preserves_unused_rows(self):
        original = self.replay(3)
        # Legacy payloads stored all physical rows, including unused storage.
        original.data["cost"][5] = 123
        restored = self.replay()
        restored.load_state_dict(self.roundtrip(original.__dict__))
        self.assert_replay_equal(original, restored)
        restored.load_state_dict(self.roundtrip(original.state_dict(compact=False)))
        self.assert_replay_equal(original, restored)
        restored.load_state_dict(original.state_dict())
        np.testing.assert_array_equal(restored.data["cost"][3:], 0)

    def test_invalid_payload_rejected_before_mutation(self):
        valid = self.replay(4).state_dict()
        mutations = [
            lambda state: state.update(format_version=2),
            lambda state: state.update(format_version=True),
            lambda state: state.update(format_version=None),
            lambda state: state.pop("format_version"),
            lambda state: state.update(compact=1),
            lambda state: state.update(capacity=8),
            lambda state: state.update(size=8),
            lambda state: state.update(size=-1),
            lambda state: state.update(pos=7),
            lambda state: state.update(pos=0),
            lambda state: state.update(size=True),
            lambda state: state.update(size=4.0),
            lambda state: state.pop("data"),
            lambda state: state["data"].pop("done"),
            lambda state: state["data"].update(extra=np.zeros(4, np.float32)),
            lambda state: state["data"].update(obs=np.zeros((4, 5), np.float32)),
            lambda state: state["data"].update(cost=np.zeros((3, 5), np.float32)),
            lambda state: state["data"].update(reward=np.zeros((4, 1), np.float32)),
            lambda state: state["data"].update(done=np.zeros(4, np.float64)),
            lambda state: state["data"].update(done=[0] * 4),
            lambda state: state["data"].update(done=torch.zeros(4, dtype=torch.float64)),
            lambda state: state["data"].update(done=torch.zeros(4, device="meta")),
        ]
        restored = self.replay(9)
        before = deepcopy(restored)
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                state = deepcopy(valid)
                mutate(state)
                with self.assertRaises(ValueError):
                    restored.load_state_dict(state)
                self.assert_replay_equal(before, restored)
        for dimensions in ((5, 2, 5), (4, 3, 5), (4, 2, 6)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                Replay(7, *dimensions).load_state_dict(valid)


if __name__ == "__main__":
    unittest.main()
