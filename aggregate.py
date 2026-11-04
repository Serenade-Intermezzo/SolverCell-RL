#!/usr/bin/env python3
"""Audit every campaign arm and export summaries; incomplete arms are retained.

Example: python aggregate.py --manifest campaigns/screen/manifest.json
Outputs summary.json, runs.csv, runs.tex, and curves.png under --output.
Confidence intervals, when possible, resample training seeds, never evaluation
episodes. Single-seed screens deliberately have no training-stability interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from tools.campaign import (REPO, atomic_json, audit_complete, fingerprint, is_alive, read_json,
                            resolved_config, source_sha256)


def read_jsonl(path):
    rows, errors = [], []
    if not path.exists():
        return rows, errors
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            errors.append(f"{path.name}:{number}: malformed/incomplete JSON")
    return rows, errors


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def summarize_run(job, root, expected_source, debug=False, controller_entry=None):
    output = (root / job["output"]).resolve()
    row = {"id": job["id"], "output": str(output), "status": "queued", "audit_errors": []}
    try:
        if job.get("config"):
            config = resolved_config(root / job["config"], debug)
        else:
            config = read_json(output / "config.json")
        row.update(config=config, method=config["method"], seed=config["seed"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        row.update(status="invalid", audit_errors=[f"config unreadable: {error}"])
        return row
    for filename, key in (("metrics.json", "metrics"), ("status.json", "progress"), ("exit.json", "exit")):
        if (output / filename).exists():
            try:
                row[key] = read_json(output / filename)
            except (OSError, ValueError) as error:
                row["audit_errors"].append(f"{filename}: {error}")
    curve, curve_errors = read_jsonl(output / "evaluations.jsonl")
    row["curve"] = [point for point in curve if point.get("split", "validation") == "validation"]
    row["audit_errors"].extend(curve_errors)
    if not output.exists() or not any(output.iterdir()):
        return row
    valid, errors = audit_complete(output, config, expected_source, job.get("expected_updates"))
    if valid and not row["audit_errors"]:
        row["status"] = "complete"
    else:
        progress = row.get("progress", {}).get("status")
        controller_status = (controller_entry or {}).get("status")
        identities = [(controller_entry or {}).get("supervisor")]
        for filename in ("worker.json", "launcher.json"):
            if (output / filename).exists():
                try:
                    identity = read_json(output / filename)
                    identities.extend([identity, identity.get("supervisor"), identity.get("worker")])
                except (OSError, ValueError):
                    pass
        live = any(is_alive(identity) for identity in identities)
        if live:
            row["status"] = "running"
        elif progress == "complete":
            row["status"] = "invalid_complete"
        elif row.get("exit", {}).get("returncode", 0) != 0 or progress == "failed" or controller_status == "failed":
            row["status"] = "failed"
        elif progress == "running" and controller_status in (None, "running"):
            row["status"] = "running_unverified"
            row["audit_errors"].append("running artifact found, but live worker identity was not verified on this host")
        else:
            row["status"] = controller_status if controller_status in ("partial", "queued") else "partial"
        row["audit_errors"].extend(errors)
    row.update({key: row.get("metrics", {}).get(key) for key in (
        "last_window_return", "validation_auc", "final_validation_return", "heldout_return_mean",
        "heldout_return_std", "wall_seconds", "train_steps", "env_steps", "updates", "source_sha256")})
    if row["status"] != "complete":
        for key in ("train_steps", "env_steps", "updates", "wall_seconds"):
            row[key] = row.get("progress", {}).get(key, row.get(key))
    return sanitize(row)


def quantile(sorted_values, probability):
    index = (len(sorted_values) - 1) * probability
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def seed_groups(rows):
    groups = {}
    operational = {"seed", "device", "threads", "reference_profile", "log_interval", "checkpoint_interval"}
    for row in rows:
        if row["status"] != "complete":
            continue
        scientific_config = {key: value for key, value in row["config"].items() if key not in operational}
        # Different source revisions cannot be treated as matched replications.
        key = fingerprint({"config": scientific_config, "source": row["source_sha256"]})
        group = groups.setdefault(key, {"scientific_config": scientific_config,
                                       "source_sha256": row["source_sha256"], "runs": []})
        group["runs"].append(row)
    summaries = []
    for key, group in groups.items():
        seeds = [row["seed"] for row in group["runs"]]
        summary = {"group_id": key[:12], "scientific_config": group["scientific_config"],
                   "source_sha256": group["source_sha256"], "seeds": sorted(seeds),
                   "runs": [row["id"] for row in group["runs"]], "n_training_seeds": len(set(seeds))}
        duplicate_seeds = len(seeds) != len(set(seeds))
        summary["scope"] = "duplicate_seeds_no_inference" if duplicate_seeds else (
            "single_training_seed_no_stability_claim" if len(seeds) == 1 else "matched_training_seed_replication")
        for metric in ("last_window_return", "validation_auc", "heldout_return_mean"):
            values = [row[metric] for row in group["runs"]]
            stats = {"mean": statistics.mean(values), "std_across_training_seeds": None, "bootstrap_ci95": None}
            if len(values) > 1 and not duplicate_seeds:
                stats["std_across_training_seeds"] = statistics.stdev(values)
                rng = random.Random(29417)
                means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
                stats["bootstrap_ci95"] = [quantile(means, 0.025), quantile(means, 0.975)]
                stats["ci_unit"] = "training_seed"
                stats["bootstrap_replicates"] = 10000
                stats["small_seed_count_caution"] = len(values) < 5
            summary[metric] = stats
        summaries.append(summary)
    return summaries


def latex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_outputs(rows, destination, metadata):
    destination.mkdir(parents=True, exist_ok=True)
    complete = sorted((row for row in rows if row["status"] == "complete"),
                      key=lambda row: (-row["last_window_return"], -row["validation_auc"]))
    for rank, row in enumerate(complete, 1):
        row["rank"] = rank
    ordered = complete + [row for row in rows if row["status"] != "complete"]
    summary = {**metadata, "ranking": "complete arms only, descending last-window return then validation AUC",
               "inference": "training-seed variation only; evaluation episode std is not seed stability",
               "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
               "runs": ordered, "seed_groups": seed_groups(rows)}
    columns = ["id", "status", "rank", "method", "seed", "train_steps", "env_steps", "updates",
               "last_window_return", "validation_auc", "heldout_return_mean", "heldout_return_std",
               "wall_seconds", "source_sha256", "output", "audit_errors"]
    with (destination / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in ordered:
            flat = {key: row.get(key) for key in columns}
            flat["audit_errors"] = "; ".join(row["audit_errors"])
            writer.writerow(flat)
    lines = [r"\begin{tabular}{llrrrr}", r"Arm & Status & Seed & Last window & AUC & Held-out \\", r"\hline"]
    for row in ordered:
        values = [latex_escape(row["id"]), latex_escape(row["status"]), str(row.get("seed", "--"))]
        for key in ("last_window_return", "validation_auc", "heldout_return_mean"):
            value = row.get(key) if row["status"] == "complete" else None
            values.append("--" if value is None else f"{value:.3f}")
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\end{tabular}")
    (destination / "runs.tex").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(10, 6))
        plotted = 0
        for row in ordered:
            points = [(point.get("train_steps", point.get("env_steps")), point.get("return_mean"))
                      for point in row.get("curve", [])]
            points = [(x, y) for x, y in points if isinstance(x, (int, float)) and isinstance(y, (int, float))
                      and math.isfinite(x) and math.isfinite(y)]
            if not points:
                continue
            axis.plot(*zip(*points), label=f"{row['id']} [{row['status']}]",
                      linestyle="-" if row["status"] == "complete" else "--", linewidth=1.7)
            plotted += 1
        axis.set(xlabel="Training steps after warm-up", ylabel="Validation episode return",
                 title="All campaign arms; incomplete curves remain visible")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=7, loc="best")
        else:
            axis.text(0.5, 0.5, "No validation evaluations yet", transform=axis.transAxes, ha="center")
        fig.tight_layout()
        fig.savefig(destination / "curves.png", dpi=180)
        plt.close(fig)
        summary["curves_plot"] = "curves.png"
    except ImportError as error:
        summary["curves_plot"] = None
        summary["plot_unavailable_reason"] = str(error)
    atomic_json(destination / "summary.json", sanitize(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--runs-root")
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--output", default="reports/aggregate")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    state = {}
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest = read_json(manifest_path)
        root = Path(manifest.get("project_root", root)).resolve()
        if (manifest_path.parent / "controller_state.json").exists():
            state = read_json(manifest_path.parent / "controller_state.json")
        expected_source = state.get("source_sha256", manifest.get("expected_source_sha256", source_sha256(root)))
        jobs = manifest["jobs"]
        debug = manifest.get("debug", False)
        campaign_id = manifest["campaign_id"]
    else:
        run_root = Path(args.runs_root).resolve()
        directories = {path.parent for name in ("config.json", "worker.log", "status.json", "metrics.json", "exit.json", "checkpoint.pt")
                       for path in run_root.rglob(name)}
        jobs = [{"id": str(path.relative_to(run_root)), "output": str(path)} for path in sorted(directories)]
        expected_source = None
        debug = False
        campaign_id = run_root.name
    rows = []
    for job in jobs:
        source_hash = expected_source
        if source_hash is None:
            source_file = root / job["output"] / "source.json"
            source_hash = read_json(source_file).get("source_sha256") if source_file.exists() else source_sha256(root)
        rows.append(summarize_run(job, root, source_hash, debug, state.get("jobs", {}).get(job["id"])))
    summary = write_outputs(rows, Path(args.output).resolve(), {"campaign_id": campaign_id,
                                                              "source_sha256": expected_source})
    print(json.dumps({"output": str(Path(args.output).resolve()), "status_counts": summary["status_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
