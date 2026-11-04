"""Bounded CPU-only preference pilot controller, using immutable source/configs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvercell.preference_config import load_preference_config
from solvercell.storage import atomic_json, source_manifest, worker_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = Path(args.manifest).resolve()
    lock = (path.parent / 'campaign.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads(path.read_text())
    if manifest['expected_source_sha256'] != source_manifest()['source_sha256']:
        raise RuntimeError('Frozen scientific source does not match manifest')
    jobs = manifest['jobs']
    if not 0 < len(jobs) <= 2 or manifest['device'] != 'cpu':
        raise ValueError('This bounded pilot supports at most two CPU workers')
    state_path = path.parent / 'controller_state.json'
    if state_path.exists():
        raise FileExistsError('Do not restart or retry a recorded preference pilot in place')
    allowed = os.sched_getaffinity(0)
    cpus = manifest['cpu_sets']
    if len(cpus) < len(jobs) or any(not row or not set(row) <= allowed for row in cpus[:len(jobs)]):
        raise ValueError('Provide an available CPU set for each worker')
    outputs = [root / job['output'] for job in jobs]
    if len({str(p.resolve()) for p in outputs}) != len(jobs) or any(p.exists() for p in outputs):
        raise FileExistsError('Output paths must be distinct and fresh')
    configs = [load_preference_config(root / job['config']) for job in jobs]
    for cfg, job in zip(configs, jobs):
        sha = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
        if sha != job['config_sha256'] or cfg.device != 'cpu':
            raise ValueError('Config identity or device mismatch')
    state = {'campaign_id': manifest['campaign_id'], 'status': 'running', 'controller': worker_identity(),
             'source_sha256': manifest['expected_source_sha256'], 'jobs': {}}
    processes, logs = {}, []
    try:
        for index, (job, output) in enumerate(zip(jobs, outputs)):
            output.mkdir(parents=True, exist_ok=False)
            logfile = (output / 'stdout.log').open('a')
            logs.append(logfile)
            command = [sys.executable, '-m', 'tools.run_preference', '--config', job['config'], '--output', job['output']]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1',
                       SOLVERCELL_SOURCE_SHA256=manifest['expected_source_sha256'])
            binding = set(cpus[index])
            process = subprocess.Popen(command, cwd=root, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=lambda binding=binding: os.sched_setaffinity(0, binding))
            processes[job['id']] = process
            row = {'status': 'running', 'pid': process.pid, 'command': command, 'cpus': sorted(binding),
                   'output': job['output'], 'config_sha256': job['config_sha256'], 'started_unix': time.time()}
            state['jobs'][job['id']] = row
            atomic_json(output / 'launcher.json', {**row, 'source_sha256': state['source_sha256']})
        atomic_json(state_path, state)
        while any(p.poll() is None for p in processes.values()):
            for name, process in processes.items():
                if process.poll() is not None:
                    state['jobs'][name].update(status='exited', returncode=process.returncode)
            state['updated_unix'] = time.time()
            atomic_json(state_path, state)
            time.sleep(2)
        okay = True
        for name, process in processes.items():
            output = root / state['jobs'][name]['output']
            complete = process.returncode == 0 and (output / 'metrics.json').exists()
            if complete:
                complete = json.loads((output / 'metrics.json').read_text())['status'] == 'complete'
            state['jobs'][name].update(status='complete' if complete else 'failed', returncode=process.returncode)
            okay &= complete
        state.update(status='complete' if okay else 'failed', finished_unix=time.time())
        atomic_json(state_path, state)
        return 0 if okay else 1
    finally:
        for handle in logs:
            handle.close()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
