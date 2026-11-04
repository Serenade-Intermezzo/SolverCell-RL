import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_json(path, data):
    with open(path, "a") as f:
        f.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def source_manifest():
    hashes = {}
    for pattern in ("solvercell/*.py", "tools/*.py", "run.py", "configs.py", "aggregate.py", "vendor/lsflow/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = None
    return {"git_commit": commit, "files": hashes,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "upstream_commit": "7e50d03e7129974a11061c9ce500bc2dc5593246"}


def worker_identity():
    try:
        ctypes.CDLL(None).prctl(15, b"llmagent", 0, 0, 0)
    except (AttributeError, OSError):
        pass
    return {"pid": os.getpid(), "comm": Path("/proc/self/comm").read_text().strip(),
            "start_ticks": Path("/proc/self/stat").read_text().split()[21],
            "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def atomic_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
