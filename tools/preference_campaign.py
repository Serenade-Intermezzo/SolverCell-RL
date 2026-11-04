"""CPU-only preference campaigns: two bounded lanes, immutable source/configs."""
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
    if not jobs or manifest['device'] != 'cpu':
        raise ValueError('A nonempty CPU campaign is required')
    state_path = path.parent / 'controller_state.json'
    if state_path.exists():
        raise FileExistsError('Do not restart or retry a recorded preference pilot in place')
    allowed = os.sched_getaffinity(0)
    cpus = manifest['cpu_sets']
    parallel = manifest.get('max_parallel', 2)
    if not 1 <= parallel <= 2 or len(cpus) != parallel:
        raise ValueError('Use one or two CPU lanes')
    if any(not row or not set(row) <= allowed for row in cpus):
        raise ValueError('Provide available CPU sets')
    if len(set().union(*(set(row) for row in cpus))) != sum(len(row) for row in cpus):
        raise ValueError('CPU lanes must not overlap')
    bindings = [job.get('cpu_set', cpus[index % parallel]) for index, job in enumerate(jobs)]
    if any(binding not in cpus for binding in bindings):
        raise ValueError('Job CPU assignment must match a lane')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('Job ids must be unique')
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
    processes, logs, pending = {}, [], list(range(len(jobs)))
    for index, job in enumerate(jobs):
        state['jobs'][job['id']] = {'status': 'queued', 'output': job['output'],
                                    'cpus': sorted(bindings[index]), 'config_sha256': job['config_sha256']}
    try:
        atomic_json(state_path, state)
        while pending or processes:
            for name, process in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                output = root / state['jobs'][name]['output']
                valid = code == 0 and (output / 'metrics.json').exists() and (output / 'exit.json').exists()
                if valid:
                    valid = (json.loads((output / 'metrics.json').read_text())['status'] == 'complete'
                             and json.loads((output / 'exit.json').read_text())['returncode'] == 0)
                state['jobs'][name].update(status='complete' if valid else 'failed',
                                          returncode=code, finished_unix=time.time())
                del processes[name]
            busy = [state['jobs'][name]['cpus'] for name in processes]
            for index in list(pending):
                binding = sorted(bindings[index])
                if len(processes) >= parallel or binding in busy:
                    continue
                job, output = jobs[index], outputs[index]
                output.mkdir(parents=True, exist_ok=False)
                logfile = (output / 'stdout.log').open('a')
                logs.append(logfile)
                command = [sys.executable, '-m', 'tools.run_preference', '--config', job['config'], '--output', job['output']]
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1',
                           SOLVERCELL_SOURCE_SHA256=manifest['expected_source_sha256'])
                process = subprocess.Popen(command, cwd=root, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                    start_new_session=True, preexec_fn=lambda binding=set(binding): os.sched_setaffinity(0, binding))
                processes[job['id']] = process
                row = {'status': 'running', 'pid': process.pid, 'command': command, 'cpus': binding,
                       'output': job['output'], 'config_sha256': job['config_sha256'], 'started_unix': time.time()}
                state['jobs'][job['id']] = row
                atomic_json(output / 'launcher.json', {**row, 'source_sha256': state['source_sha256']})
                pending.remove(index)
                busy.append(binding)
            state['updated_unix'] = time.time()
            atomic_json(state_path, state)
            if processes:
                time.sleep(2)
        okay = all(row['status'] == 'complete' for row in state['jobs'].values())
        state.update(status='complete' if okay else 'failed', finished_unix=time.time())
        atomic_json(state_path, state)
        return 0 if okay else 1
    except Exception as error:
        state.update(status='controller_failed', error=repr(error), failed_unix=time.time())
        atomic_json(state_path, state)
        raise
    finally:
        for handle in logs:
            handle.close()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
