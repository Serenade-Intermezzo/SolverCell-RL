"""Single controller for the G3 screen: frozen configs, no overwrites, no silent retries, exit evidence kept."""
import itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
HERE = Path(__file__).resolve().parent; OUT = Path(sys.argv[1]); REPO = sys.argv[2]; CPUS = [int(c) for c in sys.argv[3].split(',')]
LANES = queue.Queue()
for c in CPUS: LANES.put(c)
OUT.mkdir(parents=True, exist_ok=False); (OUT / 'configs').mkdir()
TASKS = ('intervention', 'routing_published', 'scheduling', 'routing', 'assignment'); runs = []
for task, inst in itertools.product(TASKS, range(3)):
    name = f'struct_{task}_i{inst}'; cfg = dict(kind='struct', task=task, instance=inst, n_solver=200, n_mip=20, n_topb=32, pm_draws=2, pm_k=32, forward_limit=120.); p = OUT / 'configs' / f'{name}.json'; p.write_text(json.dumps(cfg, indent=1)); runs.append((name, p))
for task, inst, h in itertools.product(TASKS, range(3), (0, 10)):
    name = f'q1_{task}_i{inst}_h{h}'; cfg = dict(kind='q1', task=task, instance=inst, h=h, n_actions=150, rollouts=16); p = OUT / 'configs' / f'{name}.json'; p.write_text(json.dumps(cfg, indent=1)); runs.append((name, p))
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(HERE / 'pylib'), SOLVERCELL_REPO=REPO)
def one(r):
    name, p = r; t = time.time(); cpu = LANES.get()
    with open(OUT / f'{name}.log', 'w') as f:
        code = subprocess.call(['taskset', '-c', str(cpu), str(HERE / 'llmagent'), '-W', 'ignore', str(HERE / 'screen.py'), '--config', str(p), '--output', str(OUT / name)], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    LANES.put(cpu)
    with open(OUT / 'exits.jsonl', 'a') as f: f.write(json.dumps(dict(run=name, exit=code, sec=time.time() - t, cpu=cpu)) + '\n')
    return code
with ThreadPoolExecutor(len(CPUS)) as ex: codes = list(ex.map(one, runs))
print('done', sum(c == 0 for c in codes), 'of', len(codes), 'failed', [r[0] for r, c in zip(runs, codes) if c])
