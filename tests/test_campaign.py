"""Controller contract tests use a real subprocess with a tiny fake trainer."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from aggregate import seed_groups, summarize_run, write_outputs
from solvercell.config import Config
from tools.campaign import (REPO, audit_complete, fingerprint, read_json, run_campaign,
                            source_sha256)


FAKE_TRAINER = '''import argparse, json, os, pathlib, socket, sys, time
p = argparse.ArgumentParser()
p.add_argument('--config'); p.add_argument('--output'); p.add_argument('--debug', action='store_true')
a = p.parse_args()
c = json.loads(pathlib.Path(a.config).read_text())
o = pathlib.Path(a.output); o.mkdir(parents=True, exist_ok=True)
def save(name, value):
    (o / name).write_text(json.dumps(value))
with pathlib.Path('starts.log').open('a') as f:
    f.write(str(c['seed']) + '\\n')
save('config.json', c)
save('worker.json', {'pid': os.getpid(), 'comm': 'llmagent', 'host': socket.gethostname(),
                     'source_sha256':os.environ['SOLVERCELL_SOURCE_SHA256'],
                     'start_ticks': pathlib.Path('/proc/self/stat').read_text().split()[21]})
save('status.json', {'status': 'running', 'train_steps': 0})
if c['seed'] == 7:
    time.sleep(1.5)
if c['seed'] == 9:
    save('status.json', {'status': 'failed', 'error': 'deliberate test failure'})
    sys.exit(3)
m = dict(status='complete', method=c['method'], seed=c['seed'],
         train_steps=c['train_steps'], warmup_steps=c['warmup_steps'],
         env_steps=c['train_steps']+c['warmup_steps'], updates=c['train_steps'],
         final_validation_return=3., last_window_return=3., validation_auc=2.,
         heldout_return_mean=4., heldout_return_std=.1, wall_seconds=.01,
         source_sha256=os.environ['SOLVERCELL_SOURCE_SHA256'], parameter_counts={'actor': 1})
save('metrics.json', m)
save('source.json', {'source_sha256': m['source_sha256']})
(o/'metrics.jsonl').write_text(json.dumps({'train_steps':c['train_steps'], 'loss':1.})+'\\n')
validation = [dict(train_steps=step, return_mean=float(step+1), split='validation', episodes=c['eval_episodes'],
                   seed_start=c['eval_seed_start']) for step in [0, *range(c['eval_interval'], c['train_steps'], c['eval_interval']), c['train_steps']]]
heldout = dict(train_steps=c['train_steps'], return_mean=4., split='heldout', episodes=c['final_eval_episodes'], seed_start=c['final_seed_start'])
(o/'evaluations.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in [*validation, heldout]))
(o/'checkpoint.pt').write_bytes(b'fake checkpoint; this test checks controller accounting')
save('status.json', {'status':'complete', 'train_steps':c['train_steps']})
'''


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "run.py").write_text(FAKE_TRAINER)
        (self.root / "configs").mkdir()
        (self.root / "campaign").mkdir()
        self.manifest_path = self.root / "campaign" / "manifest.json"

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, seeds):
        jobs = []
        for seed in seeds:
            config = Config(seed=seed, train_steps=2, warmup_steps=2, batch_size=2,
                            eval_interval=1, log_interval=1).to_dict()
            config_path = self.root / "configs" / f"s{seed}.json"
            config_path.write_text(json.dumps(config))
            jobs.append({"id": f"s{seed}", "config": str(config_path.relative_to(self.root)),
                         "config_sha256": fingerprint(config), "output": f"runs/s{seed}"})
        cpus = sorted(os.sched_getaffinity(0))
        manifest = {"campaign_id": "fake-test", "project_root": str(self.root),
                    "max_parallel": 2, "device": "cpu", "python": sys.executable,
                    "cpu_sets": [[cpus[0]], [cpus[-1]]], "jobs": jobs}
        self.manifest_path.write_text(json.dumps(manifest))
        return manifest

    def test_failures_do_not_stop_independent_arms_or_auto_retry(self):
        manifest = self.manifest([9, 0])
        self.assertEqual(run_campaign(self.manifest_path, poll_seconds=0.02), 1)
        state = read_json(self.manifest_path.parent / "controller_state.json")
        self.assertEqual(state["jobs"]["s9"]["status"], "failed")
        self.assertEqual(state["jobs"]["s0"]["status"], "complete")
        self.assertEqual(read_json(self.root / "runs/s9/exit.json")["returncode"], 3)
        before = (self.root / "starts.log").read_text()
        self.assertEqual(run_campaign(self.manifest_path, poll_seconds=0.02), 1)
        self.assertEqual((self.root / "starts.log").read_text(), before)
        rows = [summarize_run(job, self.root, source_sha256(self.root), controller_entry=state["jobs"][job["id"]])
                for job in manifest["jobs"]]
        summary = write_outputs(rows, self.root / "report", {"campaign_id": "test"})
        self.assertEqual(summary["status_counts"], {"complete": 1, "failed": 1})
        self.assertEqual(len(summary["runs"]), 2)
        self.assertIsNone(summary["seed_groups"][0]["heldout_return_mean"]["bootstrap_ci95"])
        for name in ("summary.json", "runs.csv", "runs.tex"):
            self.assertTrue((self.root / "report" / name).is_file())

    def test_completed_skip_requires_config_and_source_evidence(self):
        self.manifest([0])
        self.assertEqual(run_campaign(self.manifest_path, poll_seconds=0.02), 0)
        output = self.root / "runs/s0"
        config = read_json(self.root / "configs/s0.json")
        self.assertTrue(audit_complete(output, config, source_sha256(self.root))[0])
        changed = dict(config, temperature=0.2)
        (output / "config.json").write_text(json.dumps(changed))
        valid, reasons = audit_complete(output, config, source_sha256(self.root))
        self.assertFalse(valid)
        self.assertIn("resolved config mismatch", reasons)
        before = (self.root / "starts.log").read_text()
        self.assertEqual(run_campaign(self.manifest_path, poll_seconds=0.02), 1)
        self.assertEqual((self.root / "starts.log").read_text(), before)
        self.assertFalse(audit_complete(output, changed, "wrong-source")[0])

    def test_concurrent_controller_is_refused(self):
        self.manifest([0])
        with (self.manifest_path.parent / "campaign.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                run_campaign(self.manifest_path, poll_seconds=0.02)
        self.assertFalse((self.root / "starts.log").exists())

    def test_resume_observes_surviving_worker_without_duplicate_launch(self):
        self.manifest([7])
        with (self.root / "controller.log").open("w") as log:
            controller = subprocess.Popen(
                [sys.executable, str(REPO / "tools/campaign.py"), "run", "--manifest", str(self.manifest_path),
                 "--poll-seconds", "0.02"], stdout=log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 10
            while not (self.root / "runs/s7/launcher.json").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            try:
                self.assertTrue((self.root / "runs/s7/launcher.json").exists())
            finally:
                controller.terminate()  # Only this test's own controller; worker stays alive.
                controller.wait(timeout=5)
        self.assertEqual(run_campaign(self.manifest_path, poll_seconds=0.02), 0)
        self.assertEqual((self.root / "starts.log").read_text().splitlines(), ["7"])

    def test_config_drift_is_rejected_before_a_worker_starts(self):
        self.manifest([0])
        config_path = self.root / "configs/s0.json"
        config = read_json(config_path)
        config["temperature"] = 0.4
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "config SHA256 mismatch"):
            run_campaign(self.manifest_path, poll_seconds=0.02)
        self.assertFalse((self.root / "starts.log").exists())

    def test_intervals_resample_training_seeds_and_keep_configs_separate(self):
        rows = []
        for seed, temperature in ((0, 1.0), (1, 1.0), (2, 0.3)):
            rows.append({"id": str(seed), "status": "complete", "config": Config(seed=seed, temperature=temperature).to_dict(),
                         "source_sha256": "same", "seed": seed, "last_window_return": float(seed),
                         "validation_auc": float(seed), "heldout_return_mean": float(seed)})
        groups = seed_groups(rows)
        self.assertEqual(len(groups), 2)
        replicated = next(group for group in groups if group["n_training_seeds"] == 2)
        singleton = next(group for group in groups if group["n_training_seeds"] == 1)
        self.assertEqual(replicated["heldout_return_mean"]["ci_unit"], "training_seed")
        self.assertEqual(replicated["heldout_return_mean"]["mean"], 0.5)
        self.assertIsNone(singleton["heldout_return_mean"]["bootstrap_ci95"])


if __name__ == "__main__":
    unittest.main()
