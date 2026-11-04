"""Auditable training/evaluation runner for the first intervention study."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import signal
import sys
import time

import numpy as np
import torch

from .config import Config, is_native_method
from .environment import InterventionEnv
from .geometry import decode
from .learner import ALGORITHMS, Replay
from .storage import atomic_json, append_json, atomic_checkpoint, source_manifest, worker_identity


# These diagnostics are emitted only when the learner optimizes its actor.
# Carry them across critic-only logging steps without carrying stale critic
# metrics. The update index below uses completed, one-based learner updates.
_ACTOR_DIAGNOSTIC_KEYS = frozenset({
    "policy_loss", "native_log_alpha", "endpoint_transport", "old_point_transport",
    "margin_matched_point_transport", "candidate_unique_actions", "weight_ess",
    "projection_seconds", "velocity_mse", "noise_mse",
})


def evaluate(env, learner, episodes, seed_start, mode="policy"):
    """Paired rollout seeds; no validation RNG state leaks into training."""
    returns, counts, actions_seen = [], [], set()
    start = time.perf_counter()
    eval_env = env.clone()
    device = learner.device if learner is not None else torch.device("cpu")
    for seed in range(seed_start, seed_start + episodes):
        obs = eval_env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1000000)
        generator = torch.Generator(device=device).manual_seed(seed + 2000000)
        score = 0.0
        for _ in range(env.horizon):
            if mode == "random":
                action = eval_env.random_action(rng=rng)
            elif mode == "greedy":
                action = eval_env.greedy_action()
            else:
                action, _ = learner.act(obs, generator=generator, stochastic=False)
            obs, reward, done, info = eval_env.step(action)
            score += reward
            counts.append(info["action_count"])
            actions_seen.add(np.packbits(action.astype(np.uint8)).tobytes())
            if done:
                break
        returns.append(score)
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            "reward_per_step": float(np.mean(returns) / env.horizon), "episode_returns": returns,
            "episodes": episodes, "seed_start": seed_start, "mean_action_count": float(np.mean(counts)),
            "unique_actions": len(actions_seen), "feasibility_rate": 1.0,
            "eval_seconds": time.perf_counter() - start, "mode": mode}


def train(cfg: Config, output: Path, resume=False):
    cfg.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    if resume and (not config_path.exists() or json.loads(config_path.read_text()) != cfg.to_dict()):
        raise ValueError("Resume requires an identical resolved configuration")
    if resume and (output / "metrics.json").exists():
        raise ValueError("A completed run must not be resumed or re-evaluated in place")
    torch.set_num_threads(cfg.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source = source_manifest()
    checkpoint = None
    if resume:
        # Replay and RNG payloads stay on CPU; model/optimizer loaders move
        # their own tensors to the configured device.
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["source_sha256"] != source["source_sha256"]:
            raise ValueError("Cannot resume with changed scientific source")
    env = InterventionEnv(cfg.n_arms, cfg.n_actions, cfg.budget, cfg.horizon,
                          instance_seed=cfg.instance_seed, seed=cfg.seed + 100)
    if checkpoint is not None and checkpoint["instance_hash"] != env.instance_hash:
        raise ValueError("Cannot resume with a different benchmark instance")
    identity = worker_identity()
    identity.update({"argv": sys.argv, "device": cfg.device, "source_sha256": source["source_sha256"],
                     "method": cfg.method, "run_id": cfg.run_id(), "config": cfg.to_dict()})
    atomic_json(config_path, cfg.to_dict())
    atomic_json(output / "source.json", source)
    atomic_json(output / "worker.json", identity)
    atomic_json(output / "instance.json", env.instance_dict())
    native = is_native_method(cfg.method)
    if cfg.method == "lsflow_upstream":
        from .upstream import NativeLSFlow
        learner = NativeLSFlow(cfg, env)
    elif cfg.method == "lsflow_tensor":
        from .fast_native import TensorNativeLSFlow
        learner = TensorNativeLSFlow(cfg, env)
    elif native:
        from .native_diffusion import NativeDiffusion
        learner = NativeDiffusion(cfg, env)
    else:
        learner = ALGORITHMS[cfg.method](cfg, env)
    effective_algorithm = {"method": cfg.method, "parameter_counts": learner.parameter_counts(),
                           "decoder": "primitive_positive_topk", "latent_dim": cfg.n_actions,
                           "critic_input": "binary_action", "reward_scale": cfg.reward_scale,
                           "observation_scale": 1 / 3, "actor_update_interval": cfg.actor_interval,
                           "point_target": "old_cost_projected_to_same_margin_cell",
                           "cell_target": ("independent_gaussian_projected_to_margin_cell" if cfg.method == "cell_independent_diffusion"
                                           else "new_initial_noise_projected_to_margin_cell")}
    if native:
        effective_algorithm.update({"decoder": "upstream_weights_transpose_then_positive_topk",
                                    "latent_dim": cfg.n_arms, "critic_input": "clean_cost_center",
                                    "reward_scale": 1.0, "observation_scale": 1.0,
                                    "actor_update_interval": 1,
                                    "native_config": asdict(learner.native_config),
                                    "screen_overrides": learner.screen_overrides})
        if hasattr(learner, "bridge_spec"):
            effective_algorithm["actor_replacement"] = learner.bridge_spec
    atomic_json(output / "algorithm.json", effective_algorithm)
    cost_dim = cfg.n_arms if native else cfg.n_actions
    replay = Replay(cfg.replay_capacity, cfg.n_arms, cfg.n_actions, cost_dim)
    replay_rng = np.random.default_rng(cfg.seed + 101)
    behavior_rng = np.random.default_rng(cfg.seed + 102)
    validation = []
    env_steps = train_steps = episode_count = 0
    episode_return = 0.0
    obs = env.reset()
    start = time.perf_counter()
    elapsed_previous = 0.0
    stopped = [False]
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    pretrain_done = False
    last_actor_info = {}
    last_actor_update = None

    if resume:
        learner.load_state_dict(checkpoint["learner"])
        replay.load_state_dict(checkpoint["replay"])
        env.state, env.t = checkpoint["env_state"], checkpoint["env_t"]
        env.rng.bit_generator.state = checkpoint["env_rng"]
        replay_rng.bit_generator.state = checkpoint["replay_rng"]
        behavior_rng.bit_generator.state = checkpoint["behavior_rng"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        obs = env.observation()
        env_steps, train_steps = checkpoint["env_steps"], checkpoint["train_steps"]
        episode_count, episode_return = checkpoint["episode_count"], checkpoint["episode_return"]
        pretrain_done, validation = checkpoint["pretrain_done"], checkpoint["validation"]
        actor_cache = checkpoint.get("actor_metrics_cache", {})
        last_actor_info = dict(actor_cache.get("metrics", {}))
        last_actor_update = actor_cache.get("update")
        elapsed_previous = checkpoint["wall_seconds"]
        # Discard only this run's journal entries newer than the last complete checkpoint.
        for filename in ("metrics.jsonl", "evaluations.jsonl", "episodes.jsonl"):
            path = output / filename
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows
                                        if row.get("env_steps", 0) <= env_steps and row.get("split") != "heldout"))

    def wall():
        return elapsed_previous + time.perf_counter() - start

    def status(state, **extra):
        atomic_json(output / "status.json", {"status": state, "env_steps": env_steps,
                    "train_steps": train_steps, "updates": learner.updates,
                    "actor_updates": learner.actor_updates, "wall_seconds": wall(),
                    "pid": os.getpid(), "source_sha256": source["source_sha256"],
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})

    def save():
        atomic_checkpoint(output / "checkpoint.pt", {
            "schema_version": 2, "config": cfg.to_dict(), "source_sha256": source["source_sha256"],
            "learner": learner.state_dict(), "replay": replay.state_dict(),
            "env_state": env.state, "env_t": env.t, "env_rng": env.rng.bit_generator.state,
            "replay_rng": replay_rng.bit_generator.state, "behavior_rng": behavior_rng.bit_generator.state,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device.startswith("cuda") else None,
            "env_steps": env_steps, "train_steps": train_steps, "episode_count": episode_count,
            "episode_return": episode_return, "pretrain_done": pretrain_done,
            "actor_metrics_cache": {"update": last_actor_update, "metrics": last_actor_info},
            "validation": validation, "wall_seconds": wall(), "instance_hash": env.instance_hash})

    def validate():
        result = evaluate(env, learner, cfg.eval_episodes, cfg.eval_seed_start)
        result.update({"env_steps": env_steps, "train_steps": train_steps, "split": "validation"})
        validation.append(result)
        append_json(output / "evaluations.jsonl", result)
        print(json.dumps({"event": "evaluation", "method": cfg.method, "train_steps": train_steps,
                          "return_mean": result["return_mean"], "wall_seconds": wall()}), flush=True)

    try:
        status("running", phase="initial_evaluation")
        if not resume:
            floors = {mode: evaluate(env, None, cfg.eval_episodes, cfg.eval_seed_start, mode=mode)
                      for mode in ("random", "greedy")}
            atomic_json(output / "behavioral_baselines.json", floors)
            validate()
        last_heartbeat = time.perf_counter()
        last_info = {}
        while train_steps < cfg.train_steps and not stopped[0]:
            warmup = env_steps < cfg.warmup_steps
            if native:
                action, cost = learner.act(obs, stochastic=True)
            elif warmup or behavior_rng.random() < cfg.exploration_epsilon:
                # Uniform random objective; same sampler for every matched method.
                cost = behavior_rng.standard_normal(cfg.n_actions).astype(np.float32)
                action = decode(torch.from_numpy(cost), cfg.budget).numpy()
            else:
                action, cost = learner.act(obs, stochastic=True)
            nxt, reward, done, _ = env.step(action)
            replay.add(obs, action, cost, reward, nxt, float(done))
            env_steps += 1
            episode_return += reward
            obs = nxt
            if done:
                episode_count += 1
                if native:
                    learner.native.policy_version += 1
                append_json(output / "episodes.jsonl", {"env_steps": env_steps, "episode": episode_count,
                            "return": episode_return, "length": env.horizon, "phase": "warmup" if warmup else "train"})
                episode_return = 0.0
                obs = env.reset()
            if warmup:
                if env_steps == cfg.warmup_steps and hasattr(learner, "pretrain"):
                    learner.pretrain(lambda: replay.sample(cfg.batch_size, replay_rng))
                    pretrain_done = True
                if time.perf_counter() - last_heartbeat >= 10:
                    status("running", phase="warmup")
                    last_heartbeat = time.perf_counter()
                continue
            last_info = learner.update(replay.sample(cfg.batch_size, replay_rng))
            train_steps += 1
            if "actor_loss" in last_info:
                last_actor_info = {key: value for key, value in last_info.items()
                                   if key.startswith("actor_") or key in _ACTOR_DIAGNOSTIC_KEYS}
                last_actor_update = learner.updates
            if train_steps % cfg.log_interval == 0:
                log_info = {**last_actor_info, **last_info}
                if last_actor_update is not None:
                    log_info.update({"actor_metrics_update": last_actor_update,
                                     "actor_metrics_age_updates": learner.updates - last_actor_update})
                append_json(output / "metrics.jsonl", {"env_steps": env_steps, "train_steps": train_steps,
                            "updates": learner.updates, "actor_updates": learner.actor_updates,
                            "wall_seconds": wall(), **log_info})
            if train_steps % cfg.eval_interval == 0 or train_steps == cfg.train_steps:
                status("running", phase="validation")
                validate()
            if train_steps % cfg.checkpoint_interval == 0 or train_steps == cfg.train_steps:
                save()
            if time.perf_counter() - last_heartbeat >= 10:
                status("running", phase="train", last_metrics=last_info)
                last_heartbeat = time.perf_counter()
        if stopped[0]:
            save()
            status("stopped", phase="safe_checkpoint")
            return {"status": "stopped", "train_steps": train_steps}
        if learner.updates != cfg.train_steps or env_steps != cfg.train_steps + cfg.warmup_steps:
            raise RuntimeError("Incomplete update/environment accounting")
        status("running", phase="heldout_evaluation")
        heldout = evaluate(env, learner, cfg.final_eval_episodes, cfg.final_seed_start)
        heldout.update({"env_steps": env_steps, "train_steps": train_steps, "split": "heldout"})
        append_json(output / "evaluations.jsonl", heldout)
        xs = np.asarray([v["train_steps"] for v in validation], dtype=float)
        ys = np.asarray([v["return_mean"] for v in validation], dtype=float)
        window = max(1, int(np.ceil(0.2 * cfg.train_steps / cfg.eval_interval)))
        for name, value in learner.state_dict().items():
            if isinstance(value, dict):
                for tensor in value.values():
                    if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise FloatingPointError(f"Nonfinite final state: {name}")
        save()
        # Confirm the saved artifact itself is loadable and agrees with the counter.
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["train_steps"] != train_steps or saved["learner"]["updates"] != learner.updates:
            raise RuntimeError("Checkpoint accounting mismatch")
        result = {"status": "complete", "method": cfg.method, "seed": cfg.seed,
                  "train_steps": train_steps, "warmup_steps": cfg.warmup_steps,
                  "env_steps": env_steps, "updates": learner.updates, "actor_updates": learner.actor_updates,
                  "final_validation_return": float(ys[-1]), "last_window_return": float(ys[-window:].mean()),
                  "validation_auc": float(np.trapz(ys, xs) / cfg.train_steps),
                  "heldout_return_mean": heldout["return_mean"], "heldout_return_std": heldout["return_std"],
                  "heldout_episodes": cfg.final_eval_episodes, "wall_seconds": wall(),
                  "train_updates_per_wall_second": train_steps / wall(),
                  "parameter_counts": learner.parameter_counts(), "source_sha256": source["source_sha256"],
                  "instance_hash": env.instance_hash, "decoder_calls": learner.decoder_calls,
                  "decoder_calls_scope": ("native executed actions including evaluation" if native else "matched policy candidates including evaluation; excludes trainer random actions and endpoint checks"),
                  "projection_seconds": learner.projection_seconds,
                  "feasibility_rate": heldout["feasibility_rate"],
                  "evaluation_count": len(validation), "seed_scope": "single_training_seed",
                  "sampling_steps": cfg.sampling_steps,
                  "network_evaluations_per_sample": cfg.sampling_steps * (2 if cfg.method in {"sphere_actioncritic", "lsflow_upstream", "lsflow_tensor", "native_quotient_sphere"} else 1)}
        atomic_json(output / "metrics.json", result)
        status("complete", phase="finished")
        print(json.dumps({"event": "complete", **result}), flush=True)
        return result
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
