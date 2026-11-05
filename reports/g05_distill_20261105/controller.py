"""Single controller for the G0.5 campaign: frozen configs, no overwrites, no silent retries, exit evidence kept."""
import hashlib, itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
HERE = Path(__file__).resolve().parent; OUT = Path(sys.argv[1]); CPUS = [int(c) for c in sys.argv[2].split(',')]; WORKERS = len(CPUS)
LANES = queue.Queue()
for c in CPUS: LANES.put(c)
OUT.mkdir(parents=True, exist_ok=False); (HERE / 'configs').mkdir(exist_ok=True)
BASE = dict(K=12, rounds=8, n_out=256, n_eval=512, fit_steps=400, batch=64, lr=1e-3, ref_chunks=32)
runs = []
for inst, beta, arm in itertools.product(range(6), (2, 5, 10), ('structured', 'ef', 'plain')):
    name = f'i{inst}_b{beta}_{arm}'; cfg = dict(BASE, arm=arm, instance=inst, beta=beta, fit_seed=inst)
    p = HERE / 'configs' / f'{name}.json'; p.write_text(json.dumps(cfg, indent=1)); runs.append((name, p))
src = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in ('g05.py', 'routing_core.py'))).hexdigest()
manifest = dict(source=src, base=BASE, runs=[r[0] for r in runs], workers=WORKERS, started=time.time()); (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=1))
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
def one(r):
    name, p = r; t = time.time(); cpu = LANES.get()
    with open(OUT / f'{name}.log', 'w') as f:
        code = subprocess.call(['taskset', '-c', str(cpu), str(HERE / 'llmagent'), str(HERE / 'g05.py'), '--config', str(p), '--output', str(OUT / name)], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    LANES.put(cpu)
    with open(OUT / 'exits.jsonl', 'a') as f: f.write(json.dumps(dict(run=name, exit=code, sec=time.time() - t, cpu=cpu)) + '\n')
    return code
with ThreadPoolExecutor(WORKERS) as ex: codes = list(ex.map(one, runs))
print('done', sum(c == 0 for c in codes), 'of', len(codes), 'failed', [r[0] for r, c in zip(runs, codes) if c])
