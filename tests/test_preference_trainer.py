"""Small end-to-end checks of preference artifacts and continuation state."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from solvercell.environment import InterventionEnv
from solvercell.preference_config import PreferenceConfig, debug_preference_config
from solvercell.preference_data import PreferenceTask
from solvercell.preference_learner import PreferenceLearner
from solvercell.preference_trainer import evaluate_preference, modes_for, train_preference


METHODS = ("preference_diffusion", "preference_sphere")


def tiny_config(method="preference_diffusion", **kwargs):
    values = dict(method=method, horizon=4, hidden=16, replay_capacity=9,
                  validation_preferences=(.5,), heldout_preferences=(.25,),
                  eval_episodes=1, final_eval_episodes=1)
    values.update(kwargs)
    return replace(debug_preference_config(PreferenceConfig()), **values).validate()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_journal(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class PreferenceTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, left, right, *, ignore=(), path="root"):
        if isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                if key not in ignore:
                    self.assert_nested_equal(left[key], right[key], ignore=ignore,
                                             path=f"{path}.{key}")
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right), path)
            self.assertEqual(len(left), len(right), path)
            for i, (a, b) in enumerate(zip(left, right)):
                self.assert_nested_equal(a, b, ignore=ignore, path=f"{path}[{i}]")
        elif torch.is_tensor(left):
            self.assertEqual(left.dtype, right.dtype, path)
            self.assertEqual(left.shape, right.shape, path)
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, np.ndarray):
            self.assertEqual(left.dtype, right.dtype, path)
            np.testing.assert_array_equal(left, right, err_msg=path)
        else:
            self.assertEqual(left, right, path)

    def assert_finite(self, value):
        if isinstance(value, dict):
            for item in value.values():
                self.assert_finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.assert_finite(item)
        elif torch.is_tensor(value) and value.is_floating_point():
            self.assertTrue(torch.isfinite(value).all())
        elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            self.assertTrue(np.isfinite(value).all())
        elif isinstance(value, (float, np.floating)):
            self.assertTrue(np.isfinite(value))

    def run_training(self, cfg, output, **kwargs):
        with redirect_stdout(io.StringIO()):
            return train_preference(cfg, output, **kwargs)

    def test_debug_completes_both_methods_with_full_artifacts(self):
        with TemporaryDirectory() as directory:
            for method in METHODS:
                with self.subTest(method=method):
                    cfg = tiny_config(method)
                    output = Path(directory) / method
                    result = self.run_training(cfg, output)
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["train_steps"], 4)
                    self.assertEqual(result["updates"], 4)
                    self.assertEqual(result["env_steps"], 12)
                    self.assertEqual(result["complete_episodes"], 3)
                    self.assertEqual(result["validation_points"], 3)
                    self.assertEqual(result["heldout_rows"], len(modes_for(cfg)))
                    for name in ("config.json", "source.json", "worker.json", "instance.json",
                                 "effective_algorithm.json", "behavioral_baselines.json",
                                 "metrics.json", "status.json", "checkpoint.pt"):
                        self.assertTrue((output / name).is_file(), name)
                    self.assertEqual(read_json(output / "config.json"), cfg.to_dict())
                    self.assertEqual(read_json(output / "status.json")["status"], "complete")
                    self.assertEqual(read_json(output / "metrics.json"), result)
                    self.assertEqual(read_json(output / "worker.json")["source_sha256"],
                                     result["source_sha256"])
                    checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
                    self.assertEqual(checkpoint["source_sha256"], result["source_sha256"])
                    self.assertEqual(checkpoint["learner"]["updates"], 4)
                    self.assertEqual(checkpoint["learner"]["guide_updates"],
                                     4 if method == "preference_diffusion" else 0)
                    self.assertEqual(checkpoint["replay"]["size"], cfg.replay_capacity)
                    self.assertEqual(checkpoint["replay"]["pos"], 12 % cfg.replay_capacity)
                    self.assert_finite(checkpoint)
                    metrics = read_journal(output / "metrics.jsonl")
                    self.assertEqual([r["train_steps"] for r in metrics], [1, 2, 3, 4])
                    evaluations = read_journal(output / "evaluations.jsonl")
                    expected = [(split, step, mode)
                                for split, steps in (("validation", (0, 2, 4)), ("heldout", (4,)))
                                for step in steps for mode in modes_for(cfg)]
                    self.assertEqual([(r["split"], r["train_steps"], r["mode"])
                                      for r in evaluations], expected)
                    for row in evaluations:
                        self.assertEqual(row["episode_lengths"], [cfg.horizon])
                        self.assertEqual(row["feasibility_rate"], 1.)
                        self.assertLessEqual(row["mean_action_count"], cfg.budget)
                        expected_weight = .25 if row["split"] == "heldout" else .5
                        self.assertEqual(row["low_response_weight"], expected_weight)
                        components = np.asarray(row["component_returns"])
                        np.testing.assert_allclose(components @ row["preference"],
                                                   row["scalarized_returns"], rtol=0, atol=1e-6)
                        np.testing.assert_allclose(.5 * components.sum(-1), row["original_returns"],
                                                   rtol=0, atol=1e-5)
                    episodes = read_journal(output / "episodes.jsonl")
                    self.assertEqual([r["env_steps"] for r in episodes], [4, 8, 12])
                    self.assertTrue(all(r["length"] == cfg.horizon for r in episodes))
                    self.assert_finite([result, metrics, evaluations, episodes])

    def test_evaluation_isolates_environment_all_training_rng_and_parameters(self):
        for method in METHODS:
            with self.subTest(method=method):
                cfg = tiny_config(method)
                env = InterventionEnv(horizon=cfg.horizon, seed=85)
                task = PreferenceTask(env)
                learner = PreferenceLearner(cfg, task)
                state = env.reset()
                next_state, _, _, _ = env.step(np.zeros(env.n_actions, np.float32))
                # Populate optimizers and all fitting generators before isolation checks.
                batch = {"obs": np.repeat(task.encode(state, 4)[None], 2, axis=0),
                         "action": np.zeros((2, env.n_actions), np.float32),
                         "reward_components": np.repeat(task.components(next_state)[None], 2, axis=0),
                         "next_obs": np.repeat(task.encode(next_state, 3)[None], 2, axis=0),
                         "done": np.zeros(2, np.float32)}
                learner.update(batch)

                def snapshot():
                    return deepcopy({"python": random.getstate(), "numpy": np.random.get_state(),
                                     "torch": torch.get_rng_state(),
                                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                                     "env_rng": env.rng.bit_generator.state, "env_state": env.state,
                                     "env_t": env.t, "instance": env.instance_dict(),
                                     "learner": learner.state_dict()})

                for mode in (*modes_for(cfg), "random_prior", "myopic_primitive"):
                    before = snapshot()
                    result = evaluate_preference(task, learner, .25, mode, 1, 100000)
                    after = snapshot()
                    self.assert_nested_equal(before, after, ignore={"counters"})
                    for key, count in before["learner"]["counters"].items():
                        self.assertGreaterEqual(after["learner"]["counters"][key], count)
                    repeated = evaluate_preference(task, learner, .25, mode, 1, 100000)
                    self.assert_nested_equal(result, repeated, ignore={"eval_seconds"})
                    self.assertEqual(result["episode_lengths"], [cfg.horizon])

    def test_interrupted_resume_is_exact_including_ring_rng_and_journals(self):
        with TemporaryDirectory() as directory:
            for method in METHODS:
                with self.subTest(method=method):
                    cfg = tiny_config(method)
                    full = Path(directory) / (method + "_full")
                    resumed = Path(directory) / (method + "_resumed")
                    full_result = self.run_training(cfg, full)
                    stopped = self.run_training(cfg, resumed, stop_after_updates=2)
                    self.assertEqual(stopped, {"status": "stopped", "train_steps": 2})
                    self.assertFalse((resumed / "metrics.json").exists())
                    self.assertEqual(read_json(resumed / "status.json")["status"], "stopped")
                    checkpoint = torch.load(resumed / "checkpoint.pt", weights_only=False)
                    self.assertEqual(checkpoint["env_steps"], 10)
                    self.assertEqual(checkpoint["env_t"], 2)  # Mid-episode restoration is required.
                    self.assertEqual(checkpoint["replay"]["pos"], 1)  # Ring has already wrapped.
                    # A crash can leave journal rows newer than the last durable checkpoint.
                    for name in ("metrics.jsonl", "evaluations.jsonl", "episodes.jsonl"):
                        with (resumed / name).open("a") as stream:
                            stream.write(json.dumps({"env_steps": 999, "stale": True}) + "\n")
                    with (resumed / "evaluations.jsonl").open("a") as stream:
                        stream.write(json.dumps({"env_steps": 10, "split": "heldout", "stale": True}) + "\n")
                    resumed_result = self.run_training(cfg, resumed, resume=True)
                    self.assert_nested_equal(full_result, resumed_result, ignore={"wall_seconds"})
                    full_checkpoint = torch.load(full / "checkpoint.pt", weights_only=False)
                    resumed_checkpoint = torch.load(resumed / "checkpoint.pt", weights_only=False)
                    # This traverses every model, optimizer, generator, replay field,
                    # current episode state, counters, and global RNG checkpoint.
                    self.assert_nested_equal(full_checkpoint, resumed_checkpoint, ignore={"wall_seconds"})
                    for name in ("metrics.jsonl", "evaluations.jsonl", "episodes.jsonl"):
                        self.assert_nested_equal(read_journal(full / name), read_journal(resumed / name),
                                                 ignore={"wall_seconds", "eval_seconds"}, path=name)
                    with self.assertRaisesRegex(ValueError, "completed"):
                        self.run_training(cfg, resumed, resume=True)

    def test_config_rejects_fitting_heldout_overlap(self):
        cfg = tiny_config()
        with self.assertRaisesRegex(ValueError, "Heldout preferences"):
            replace(cfg, heldout_preferences=(.25, .5)).validate()
        with self.assertRaisesRegex(ValueError, "Validation preferences"):
            replace(cfg, validation_preferences=(.25,)).validate()
        with self.assertRaisesRegex(ValueError, "seed ranges overlap"):
            replace(cfg, final_seed_start=cfg.eval_seed_start).validate()


if __name__ == "__main__":
    unittest.main()
