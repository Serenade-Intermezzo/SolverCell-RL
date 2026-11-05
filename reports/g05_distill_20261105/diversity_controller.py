"""Single controller for the diversity diagnostic: frozen configs, no overwrites, no silent retries."""
import hashlib, itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
HERE = Path(__file__).resolve().parent; OUT = Path(sys.argv[1]); MODELS = sys.argv[2]; CPUS = [int(c) for c in sys.argv[3].split(',')]
LANES = queue.Queue()
for c in CPUS: LANES.put(c)
OUT.mkdir(parents=True, exist_ok=False); (OUT / 'configs').mkdir()
runs = []
for inst, beta in itertools.product(range(6), (2, 5, 10)):
    name = f'i{inst}_b{beta}'; p = OUT / 'configs' / f'{name}.json'; p.write_text(json.dumps(dict(instance=inst, beta=beta, models=MODELS, n=4096), indent=1)); runs.append((name, p))
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
def one(r):
    name, p = r; t = time.time(); cpu = LANES.get()
    with open(OUT / f'{name}.log', 'w') as f:
        code = subprocess.call(['taskset', '-c', str(cpu), str(HERE / 'llmagent'), str(HERE / 'diversity.py'), '--config', str(p), '--output', str(OUT / name)], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    LANES.put(cpu)
    with open(OUT / 'exits.jsonl', 'a') as f: f.write(json.dumps(dict(run=name, exit=code, sec=time.time() - t, cpu=cpu)) + '\n')
    return code
with ThreadPoolExecutor(len(CPUS)) as ex: codes = list(ex.map(one, runs))
print('done', sum(c == 0 for c in codes), 'of', len(codes), 'failed', [r[0] for r, c in zip(runs, codes) if c])
