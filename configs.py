"""One fixed list defines the initial controlled screen; no adaptive auto-sweep."""
from dataclasses import replace
import json
from pathlib import Path

from solvercell.config import Config

BASE = Config()
CONFIGS = {
    "cell_m010": BASE,
    "point": replace(BASE, method="point_diffusion"),
    "cell_independent": replace(BASE, method="cell_independent_diffusion"),
    "sphere_matched": replace(BASE, method="sphere_actioncritic"),
    "cell_flow": replace(BASE, method="cell_flow", diffusion_scale=0.0),
    "cell_m005": replace(BASE, margin=0.05),
    "cell_m020": replace(BASE, margin=0.2),
    # Original algorithm/arm-cost decoder, with explicitly shortened screen budgets.
    "lsflow_native": replace(BASE, method="lsflow_upstream", candidates=12,
                              bootstrap_candidates=16, sampling_steps=12),
}


def write_configs(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    folder = root / "experiments" / "intervention_screen_20261104"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, config in CONFIGS.items():
        config.validate()
        path = folder / f"{name}.json"
        path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
        jobs.append({"id": name, "config": str(path.relative_to(root)),
                     "output": f"outputs/intervention_screen_20261104/{name}"})
    manifest = {"campaign_id": "intervention_screen_20261104", "max_parallel": 2,
                "device": "cpu", "cpu_sets": [[24], [25]], "jobs": jobs}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


if __name__ == "__main__":
    print(write_configs())
