"""End-to-end checks for actor diagnostics on offset critic-only log steps."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import torch

from solvercell.config import Config, debug_config
from solvercell.learner import Learner
from solvercell.trainer import train


class ActorLoggingTest(unittest.TestCase):
    def config(self):
        return replace(debug_config(Config()), hidden=16, horizon=2, warmup_steps=2,
                       train_steps=4, log_interval=2, actor_interval=2,
                       eval_interval=4, eval_episodes=1, final_eval_episodes=1,
                       checkpoint_interval=1, replay_capacity=16).validate()

    def read_rows(self, output):
        return [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]

    def assert_actor_rows(self, rows, actor_info):
        self.assertEqual([row["updates"] for row in rows], [2, 4])
        for row, actor_update in zip(rows, (1, 3)):
            self.assertEqual(row["actor_metrics_update"], actor_update)
            self.assertEqual(row["actor_metrics_age_updates"], 1)
            self.assertEqual(row["actor_updates"], (actor_update + 1) // 2)
            for key in ("actor_loss", "actor_grad_norm", "endpoint_transport", "old_point_transport",
                        "margin_matched_point_transport", "candidate_unique_actions", "weight_ess",
                        "projection_seconds", "velocity_mse", "noise_mse"):
                self.assertEqual(row[key], actor_info[actor_update][key])
            # Critic diagnostics must belong to the current update, not to the
            # cached actor update whose diagnostics happen to be in this row.
            self.assertEqual(row["critic_loss"], actor_info[row["updates"]]["critic_loss"])

    def test_even_log_steps_keep_latest_actor_diagnostics(self):
        captured = {}
        original_update = Learner.update

        def capture_update(learner, batch):
            info = original_update(learner, batch)
            captured[learner.updates] = info.copy()
            return info

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            with mock.patch.object(Learner, "update", capture_update):
                result = train(self.config(), output)
            self.assertEqual(result["status"], "complete")
            self.assert_actor_rows(self.read_rows(output), captured)
            saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["actor_metrics_cache"]["update"], 3)
            self.assertEqual(saved["actor_metrics_cache"]["metrics"]["actor_loss"], captured[3]["actor_loss"])

    def test_resume_restores_actor_cache_before_next_critic_only_log(self):
        captured = {}
        original_update = Learner.update

        def capture_and_stop(learner, batch):
            info = original_update(learner, batch)
            captured[learner.updates] = info.copy()
            # Invoke the runner's safe-stop handler, without signalling the
            # test process or altering the learner's mathematical update.
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return info

        def capture_update(learner, batch):
            info = original_update(learner, batch)
            captured[learner.updates] = info.copy()
            return info

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            with mock.patch.object(Learner, "update", capture_and_stop):
                stopped = train(self.config(), output)
            self.assertEqual(stopped, {"status": "stopped", "train_steps": 1})
            self.assertFalse((output / "metrics.jsonl").exists())
            saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["actor_metrics_cache"]["update"], 1)
            self.assertEqual(saved["actor_metrics_cache"]["metrics"]["actor_loss"], captured[1]["actor_loss"])
            self.assertNotIn("critic_loss", saved["actor_metrics_cache"]["metrics"])
            with mock.patch.object(Learner, "update", capture_update):
                result = train(self.config(), output, resume=True)
            self.assertEqual(result["status"], "complete")
            self.assert_actor_rows(self.read_rows(output), captured)


if __name__ == "__main__":
    unittest.main()
