#!/usr/bin/env python3
"""Run a frozen, finite campaign without retrying failed or partial experiments.

Paths in jobs are relative to project_root (default: this repository), not the
manifest directory. Keep each campaign manifest in its own directory: its lock
and durable controller_state.json are placed beside it. Workers have independent
supervisors, so restarting the controller observes an existing worker instead
of launching it again. The controller never terminates somebody else's process.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from solvercell.config import debug_config, load_config


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_sha256(root):
    """Exactly match solvercell.storage.source_manifest, without importing torch."""
    root = Path(root)
    hashes = {}
    for pattern in ("solvercell/*.py", "tools/*.py", "run.py", "configs.py", "aggregate.py", "vendor/lsflow/**/*.py"):
        for path in sorted(root.glob(pattern)):
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprint(hashes)


def process_identity(pid):
    try:
        # comm may contain spaces: parse only the fields following its final ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": int(pid), "start_ticks": fields[19], "host": socket.gethostname(),
                "process_state": fields[0]}
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def is_alive(identity):
    if not identity or identity.get("host") != socket.gethostname():
        return False
    actual = process_identity(identity.get("pid", -1))
    return bool(actual and actual["start_ticks"] == str(identity.get("start_ticks"))
                and actual["process_state"] != "Z")


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def resolved_config(path, debug=False):
    config = load_config(path)
    return (debug_config(config) if debug else config).to_dict()


def audit_complete(output, expected_config, expected_source, expected_updates=None):
    """Return (valid, reasons). A metrics file alone never establishes completion."""
    output = Path(output)
    reasons = []
    required = ("config.json", "status.json", "worker.json", "source.json", "exit.json", "metrics.json",
                "metrics.jsonl", "evaluations.jsonl", "checkpoint.pt")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        return False, ["missing artifacts: " + ", ".join(missing)]
    try:
        config = read_json(output / "config.json")
        metrics = read_json(output / "metrics.json")
        status = read_json(output / "status.json")
        exit_info = read_json(output / "exit.json")
        identity = read_json(output / "worker.json")
        recorded_source = read_json(output / "source.json")
        if config != expected_config:
            reasons.append("resolved config mismatch")
        if status.get("status") != "complete" or metrics.get("status") != "complete":
            reasons.append("completion status mismatch")
        if exit_info.get("returncode") != 0:
            reasons.append("worker did not exit zero")
        if metrics.get("source_sha256") != expected_source:
            reasons.append("source SHA256 mismatch")
        if recorded_source.get("source_sha256") != expected_source or identity.get("source_sha256") != expected_source:
            reasons.append("source or worker manifest mismatch")
        if identity.get("comm") != "llmagent" or not all(identity.get(key) for key in ("pid", "start_ticks", "host")):
            reasons.append("missing or unexpected worker identity")
        if metrics.get("method") != expected_config["method"] or metrics.get("seed") != expected_config["seed"]:
            reasons.append("method or seed mismatch")
        expectations = {
            "train_steps": expected_config["train_steps"],
            "warmup_steps": expected_config["warmup_steps"],
            "env_steps": expected_config["train_steps"] + expected_config["warmup_steps"],
            "updates": expected_config["train_steps"] if expected_updates is None else expected_updates,
        }
        for key, expected in expectations.items():
            if metrics.get(key) != expected:
                reasons.append(f"{key}: expected {expected}, got {metrics.get(key)}")
        for key in ("final_validation_return", "last_window_return", "validation_auc",
                    "heldout_return_mean", "heldout_return_std", "wall_seconds"):
            if not isinstance(metrics.get(key), (float, int)) or not math.isfinite(metrics[key]):
                reasons.append(f"missing or nonfinite {key}")
        if not finite_tree(metrics):
            reasons.append("nonfinite summary metrics")
        jsonl_rows = {}
        for name in ("metrics.jsonl", "evaluations.jsonl"):
            lines = (output / name).read_text().splitlines()
            rows = [json.loads(line) for line in lines if line.strip()]
            jsonl_rows[name] = rows
            if not rows:
                reasons.append(f"empty {name}")
            elif not all(finite_tree(row) for row in rows):
                reasons.append(f"nonfinite {name}")
        evaluations = jsonl_rows["evaluations.jsonl"]
        validation = [row for row in evaluations if row.get("split") == "validation"]
        heldout = [row for row in evaluations if row.get("split") == "heldout"]
        final_step = expected_config["train_steps"]
        expected_steps = [0, *range(expected_config["eval_interval"], final_step, expected_config["eval_interval"]), final_step]
        if [row.get("train_steps") for row in validation] != expected_steps:
            reasons.append("validation curve does not cover the expected training steps")
        if any(row.get("episodes") != expected_config["eval_episodes"] or
               row.get("seed_start") != expected_config["eval_seed_start"] for row in validation):
            reasons.append("validation episode budget or seeds mismatch")
        if len(heldout) != 1 or heldout[0].get("train_steps") != final_step or \
                heldout[0].get("episodes") != expected_config["final_eval_episodes"] or \
                heldout[0].get("seed_start") != expected_config["final_seed_start"]:
            reasons.append("held-out evaluation budget, steps, or seeds mismatch")
        if (output / "checkpoint.pt").stat().st_size == 0:
            reasons.append("empty checkpoint")
    except (ValueError, KeyError, OSError, TypeError) as error:
        reasons.append(f"artifact read failed: {error}")
    return not reasons, reasons


def gpu_available():
    """Conservative gate: all compute applications must have left the GPU(s)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10, check=False,
        )
        if result.returncode:
            return False, "nvidia-smi failed; GPU availability is unverified"
        occupied = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return not occupied, "compute processes present: " + ", ".join(occupied) if occupied else "idle"
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def worker_main(args):
    """Independent supervisor records the actual runner exit after controller loss."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    returncode = 125
    error = None
    try:
        if source_sha256(args.root) != args.source:
            raise RuntimeError("source changed before worker launch")
        if fingerprint(resolved_config(args.config, args.debug)) != args.config_sha:
            raise RuntimeError("config changed before worker launch")
        cpus = [int(cpu) for cpu in args.cpus.split(",") if cpu]
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1", SOLVERCELL_SOURCE_SHA256=args.source)
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        command = [args.python, str(Path(args.root) / "run.py"), "--config", args.config,
                   "--output", str(output)]
        if args.debug:
            command.append("--debug")
        process = subprocess.Popen(command, cwd=args.root, env=env,
                                   preexec_fn=(lambda: os.sched_setaffinity(0, cpus)) if cpus else None)
        atomic_json(output / "launcher.json", {
            "supervisor": process_identity(os.getpid()), "worker": process_identity(process.pid),
            "command": command, "cpu_affinity": cpus, "source_sha256": args.source,
            "started_unix": started,
        })
        returncode = process.wait()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        print(error, file=sys.stderr, flush=True)
    atomic_json(output / "exit.json", {"returncode": returncode, "started_unix": started,
                                       "finished_unix": time.time(), "error": error,
                                       "source_sha256": args.source})
    return returncode


def run_campaign(manifest_path, root_override=None, poll_seconds=5):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    root = Path(root_override or manifest.get("project_root", REPO)).resolve()
    jobs = manifest["jobs"]
    ids = [job["id"] for job in jobs]
    outputs = [str((root / job["output"]).resolve()) for job in jobs]
    if len(set(ids)) != len(ids) or len(set(outputs)) != len(outputs):
        raise ValueError("job IDs and output directories must be unique")
    maximum = int(manifest.get("max_parallel", 1))
    if maximum < 1:
        raise ValueError("max_parallel must be positive")
    device = manifest.get("device", "cpu")
    cpu_sets = manifest.get("cpu_sets", [])
    if cpu_sets and len(cpu_sets) < maximum:
        raise ValueError("provide one cpu_set for every parallel slot")
    allowed_cpus = os.sched_getaffinity(0)
    for cpus in cpu_sets:
        if not cpus or not set(cpus) <= allowed_cpus:
            raise ValueError(f"unavailable CPU affinity: {cpus}")
    debug = bool(manifest.get("debug", False))
    configs = {job["id"]: resolved_config(root / job["config"], debug) for job in jobs}
    for job in jobs:
        config = configs[job["id"]]
        if config["device"] != device:
            raise ValueError(f"{job['id']}: config device differs from manifest")
        if job.get("config_sha256") and job["config_sha256"] != fingerprint(config):
            raise ValueError(f"{job['id']}: expected config SHA256 mismatch")
    current_source = source_sha256(root)
    if manifest.get("expected_source_sha256", current_source) != current_source:
        raise RuntimeError("expected source SHA256 mismatch")

    lock_path = manifest_path.parent / "campaign.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError(f"campaign controller already running: {lock_path}") from error
    try:
        state_path = manifest_path.parent / "controller_state.json"
        manifest_hash = fingerprint(manifest)
        if state_path.exists():
            state = read_json(state_path)
            if state["manifest_sha256"] != manifest_hash or state["source_sha256"] != current_source:
                raise RuntimeError("resume refused: manifest or source changed")
        else:
            state = {"campaign_id": manifest["campaign_id"], "manifest_sha256": manifest_hash,
                     "source_sha256": current_source, "created_unix": time.time(), "jobs": {}}
        state.update(controller=process_identity(os.getpid()), status="running", updated_unix=time.time())
        process_handles = {}

        def save():
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        for job, output_string in zip(jobs, outputs):
            output = Path(output_string)
            entry = state["jobs"].setdefault(job["id"], {"status": "queued", "output": output_string,
                                                       "config_sha256": fingerprint(configs[job["id"]])})
            if entry["config_sha256"] != fingerprint(configs[job["id"]]):
                raise RuntimeError(f"resume refused: config changed for {job['id']}")
            if output.exists() and any(output.iterdir()):
                valid, reasons = audit_complete(output, configs[job["id"]], current_source,
                                                job.get("expected_updates"))
                if valid:
                    entry.update(status="complete", audit_errors=[])
                elif is_alive(entry.get("supervisor")):
                    entry["status"] = "running"
                else:
                    launcher = read_json(output / "launcher.json") if (output / "launcher.json").exists() else {}
                    if is_alive(launcher.get("supervisor")) or is_alive(launcher.get("worker")):
                        entry.update(status="running", supervisor=launcher.get("supervisor"),
                                     recovered_worker=launcher.get("worker"))
                    else:
                        entry.update(status="failed" if (output / "exit.json").exists() else "partial",
                                     audit_errors=reasons)
            elif entry["status"] != "queued":
                entry.update(status="partial", audit_errors=["previously launched output is missing"])
        save()
        print(json.dumps({"campaign": manifest["campaign_id"], "status": "running", "source": current_source}), flush=True)

        while True:
            for job in jobs:
                entry = state["jobs"][job["id"]]
                if entry["status"] != "running":
                    continue
                handle = process_handles.get(job["id"])
                if handle is not None:
                    handle.poll()  # Reap exited supervisors; zombies are not live workers.
                if is_alive(entry.get("supervisor")) or is_alive(entry.get("recovered_worker")):
                    continue
                valid, reasons = audit_complete(entry["output"], configs[job["id"]], current_source,
                                                job.get("expected_updates"))
                entry.update(status="complete" if valid else "failed", audit_errors=reasons)
                print(json.dumps({"job": job["id"], "status": entry["status"], "errors": reasons}), flush=True)

            running = [item for item in state["jobs"].values() if item["status"] == "running"]
            queued = [job for job in jobs if state["jobs"][job["id"]]["status"] == "queued"]
            if not running and not queued:
                statuses = [item["status"] for item in state["jobs"].values()]
                state["status"] = "complete" if all(status == "complete" for status in statuses) else "finished_with_failures"
                save()
                return 0 if state["status"] == "complete" else 1

            if source_sha256(root) != current_source:
                raise RuntimeError("source changed during campaign; queued jobs will not launch")
            can_launch, gate = (True, "CPU campaign") if device == "cpu" else gpu_available()
            state["resource_gate"] = gate
            used_slots = {item.get("slot") for item in running}
            for job in queued:
                if len(running) >= maximum or not can_launch:
                    break
                if resolved_config(root / job["config"], debug) != configs[job["id"]]:
                    raise RuntimeError(f"config changed before launch: {job['id']}")
                slot = next(index for index in range(maximum) if index not in used_slots)
                output = (root / job["output"]).resolve()
                output.mkdir(parents=True, exist_ok=True)
                cpus = cpu_sets[slot] if cpu_sets else sorted(allowed_cpus)
                python = str(manifest.get("python", sys.executable))
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root),
                           "--config", str((root / job["config"]).resolve()), "--output", str(output),
                           "--source", current_source, "--device", device, "--cpus", ",".join(map(str, cpus)),
                           "--python", python, "--config-sha", fingerprint(configs[job["id"]])]
                if debug:
                    command.append("--debug")
                with (output / "worker.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                process_handles[job["id"]] = process
                entry = state["jobs"][job["id"]]
                entry.update(status="running", slot=slot, supervisor=process_identity(process.pid),
                             started_unix=time.time())
                used_slots.add(slot)
                running.append(entry)
                save()
                print(json.dumps({"job": job["id"], "status": "running", "pid": process.pid, "slot": slot}), flush=True)
                # Recheck GPU occupancy before another GPU job. CPU jobs can fill all slots.
                if device != "cpu":
                    can_launch = False
            save()
            time.sleep(poll_seconds)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    controller = subparsers.add_parser("run")
    controller.add_argument("--manifest", required=True)
    controller.add_argument("--root")
    controller.add_argument("--poll-seconds", type=float, default=5)
    worker = subparsers.add_parser("worker")
    for name in ("root", "config", "output", "source", "device", "cpus", "python", "config-sha"):
        worker.add_argument("--" + name, required=True)
    worker.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "worker":
            return worker_main(args)
        if args.poll_seconds <= 0:
            raise ValueError("poll-seconds must be positive")
        return run_campaign(args.manifest, args.root, args.poll_seconds)
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"campaign error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
