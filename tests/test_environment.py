"""Source parity and rollout-isolation tests for the benchmark port."""

import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from solvercell.environment import InterventionEnv


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "lsflow"


def upstream_numpy_factory():
    """Execute the original NumPy classes, replacing unused visualization/
    Torch wrappers only. No copied formulas are used for the parity oracle.
    """
    namespace = {
        "np": np,
        "gym": SimpleNamespace(Env=object),
        "torch": SimpleNamespace(is_tensor=lambda value: False),
        "num_to_chr": lambda value: str(value),
        "visualize_graph": lambda *args, **kwargs: None,
    }
    for filename, names in (
        ("environment/base_envs.py", {"BaseRMAB", "MultiStateRMAB"}),
        ("environment/multi_action.py", {"MultiActionRMAB"}),
        ("environment/rmab_instances.py", {"multistate_rmab", "get_rmab_sigmoid"}),
    ):
        source = UPSTREAM / filename
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.parse(source.read_text())
        nodes = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    # Actual TorchMultiStateRMAB calls MultiStateRMAB.__init__ with supplied
    # init_state/state_r then converts arrays to tensors; conversion draws no
    # NumPy random numbers. TorchMultiActionRMAB is constructed after the
    # NumPy instance is complete and cannot affect its parameter arrays.
    namespace["TorchMultiStateRMAB"] = namespace["MultiStateRMAB"]
    namespace["TorchMultiActionRMAB"] = lambda *args: None
    return namespace["get_rmab_sigmoid"]


class EnvironmentTest(unittest.TestCase):
    def test_instance_matches_pinned_upstream(self):
        factory = upstream_numpy_factory()
        old_state = np.random.get_state()
        try:
            for seed in (0, 1, 7):
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    np.random.seed(seed)
                    original, _ = factory(40, 20, 10, 20)
                port = InterventionEnv(instance_seed=seed)
                for name in ("weights_p", "theta_p", "a_vals", "b_vals", "x0_vals",
                             "good_arms", "bad_arms", "good_actions", "bad_actions"):
                    np.testing.assert_array_equal(getattr(port, name), getattr(original, name))
                np.testing.assert_array_equal(port.state_r, original.rmab.state_r)
                np.testing.assert_array_equal(port.base_transitions, original.rmab.transitions)
                np.testing.assert_array_equal(port.wrapper_init_state, original.init_state)
        finally:
            np.random.set_state(old_state)

    def test_transition_matches_original_with_common_legacy_random_draws(self):
        factory = upstream_numpy_factory()
        old_state = np.random.get_state()
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                np.random.seed(0)
                original, _ = factory(40, 20, 10, 20)
            port = InterventionEnv(instance_seed=0)
            np.random.seed(1234)
            port.rng = np.random.RandomState(1234)  # test-only common draws
            np.testing.assert_array_equal(port.reset(), original.reset())
            actions_rng = np.random.default_rng(4)
            for t in range(20):
                action = port.random_action(actions_rng)
                actual = port.step(action)
                expected = original.step(action)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1], float(expected[1]))
                self.assertEqual(actual[2], expected[2])
                self.assertAlmostEqual(actual[3]["next_state_expected"],
                                       expected[4]["next_state_expected"], places=11)
        finally:
            np.random.set_state(old_state)

    def test_global_rng_is_not_changed_and_seed_repeats(self):
        before = np.random.get_state()
        env = InterventionEnv(instance_seed=7)
        env.reset(seed=12)
        actions = [env.random_action(np.random.default_rng(i)) for i in range(3)]
        result = [env.step(a) for a in actions]
        env.reset(seed=12)
        repeated = [env.step(a) for a in actions]
        for left, right in zip(result, repeated):
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:], right[1:])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_eval_stream_and_instance_roundtrip(self):
        train = InterventionEnv()
        reference = InterventionEnv()
        train.reset(seed=11)
        reference.reset(seed=11)
        evaluation = train.clone(seed=99)
        evaluation.reset()
        for _ in range(5):
            evaluation.step(evaluation.random_action())
        action = np.zeros(20)
        np.testing.assert_array_equal(train.step(action)[0], reference.step(action)[0])
        roundtrip = InterventionEnv.from_instance(json.loads(json.dumps(train.instance_dict())))
        self.assertEqual(train.instance_hash, roundtrip.instance_hash)
        self.assertEqual(train.instance_hash, evaluation.instance_hash)
        self.assertFalse(np.shares_memory(train.weights_p, evaluation.weights_p))

    def test_expected_reward_manual_and_boundaries(self):
        env = InterventionEnv()
        env.state = (np.arange(env.n_arms) % 4).astype(np.float32)
        action = np.zeros(env.n_actions)
        action[:4] = 1
        value = 0.0
        for j, state in enumerate(env.state.astype(int)):
            score = sum(env.weights_p[j, k] * action[k] for k in range(env.n_actions))
            prob = (env.a_vals[j] / (1 + np.exp(env.x0_vals[j] - env.b_vals[j] * score))
                    - env.a_vals[j] / (1 + np.exp(env.x0_vals[j]))) * 0.98 + 0.01
            value += prob * env.state_r[j, min(state + 1, 3)]
            value += (1 - prob) * env.state_r[j, max(state - 1, 0)]
        self.assertAlmostEqual(env.expected_reward(action), value, places=12)
        np.testing.assert_allclose(env.transition_probabilities(np.zeros(20)), 0.01)

    def test_valid_actions_done_and_greedy(self):
        env = InterventionEnv(horizon=3)
        env.reset(seed=0)
        for action in (np.ones(20), np.full(20, 0.1), np.zeros(40), np.full(20, np.nan)):
            with self.assertRaises(ValueError):
                env.step(action)
        greedy = env.greedy_action()
        self.assertLessEqual(greedy.sum(), env.budget)
        self.assertTrue(np.all((greedy == 0) | (greedy == 1)))
        self.assertGreaterEqual(env.expected_reward(greedy), env.expected_reward(np.zeros(20)))
        for t in range(3):
            observation, reward, done, _ = env.step(env.random_action())
            self.assertEqual(done, t == 2)
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.all((observation >= 0) & (observation <= 3)))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(20))

    def test_preserves_upstream_failure_at_40_actions(self):
        with self.assertRaisesRegex(ValueError, "Upstream weight construction"):
            InterventionEnv(n_actions=40)


if __name__ == "__main__":
    unittest.main()
