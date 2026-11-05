"""Single controller for G0.6 Stage A: phase 1, the preregistered w5 rule, then phase 2. Frozen configs, no overwrites, no silent retries."""
import json, os, signal, statistics as st, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
HERE = Path(__file__).resolve().parent; OUT = Path(sys.argv[1]); LIB = sys.argv[2]; CPUS = [int(c) for c in sys.argv[3].split(',')]
INST = range(100, 106); LANES = queue.Queue()
for c in CPUS: LANES.put(c)
OUT.mkdir(parents=True, exist_ok=False); (OUT / 'configs').mkdir()
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=LIB)
P1 = dict(phase=1, rlimit=16e9, n_rho=8192, sweeps=200, n_a4=4096, w5_grid=[.5, 1, 2])
PA = dict(phase='a1', rlimit=64e9, phase1=str(OUT), a1_time=600, a1_mem=64e9)
P2 = dict(phase=2, rlimit=16e9, phase1=str(OUT), n_rho=8192, sweeps=200, n_a2=300, n_a3=1024, n_a7=4096, n_a7_rho=16384, n_exact=16384, n_a5=2048, n_l=512)
ACTIVE = set()
def watchdog():
    while True:
        avail = int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) * 1024
        if avail < 20e9:
            print('watchdog: low memory, terminating own workers', avail, flush=True)
            for pr in list(ACTIVE): pr.terminate()
        time.sleep(5)
threading.Thread(target=watchdog, daemon=True).start()
def one(r):
    name, cfg = r; p = OUT / 'configs' / f'{name}.json'; p.write_text(json.dumps(cfg, indent=1)); t = time.time(); cpu = LANES.get()
    with open(OUT / f'{name}.log', 'w') as f:
        pr = subprocess.Popen(['taskset', '-c', str(cpu), str(HERE / 'llmagent'), str(HERE / 'stage_a.py'), '--config', str(p), '--output', str(OUT / name)], stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE); ACTIVE.add(pr); code = pr.wait(); ACTIVE.discard(pr)
    LANES.put(cpu)
    with open(OUT / 'exits.jsonl', 'a') as f: f.write(json.dumps(dict(run=name, exit=code, sec=time.time() - t, cpu=cpu)) + '\n')
    return code
with ThreadPoolExecutor(len(CPUS)) as ex: codes = list(ex.map(one, [(f'p1_i{i}', dict(P1, instance=i, w5_grid=[.5, 1, 2])) for i in INST]))
if any(codes): sys.exit(f'phase 1 failed {codes}')
R = [json.loads((OUT / f'p1_i{i}' / 'result.json').read_text()) for i in INST]; means = {w: st.mean(r['a4'][str(w)]['5']['r2'] for r in R) for w in P1['w5_grid']}
ok = [w for w in P1['w5_grid'] if means[w] < .9]; w5 = ok[0] if ok else P1['w5_grid'][-1]
(OUT / 'w5.json').write_text(json.dumps(dict(mean_r2=means, passed=bool(ok), w5=w5), indent=1)); print('w5', w5, means, flush=True)
for i in INST: one((f'a1_i{i}', dict(PA, instance=i, w5=w5)))
with ThreadPoolExecutor(len(CPUS)) as ex: codes = list(ex.map(one, [(f'p2_i{i}', dict(P2, instance=i, w5=w5)) for i in INST]))
print('done', codes)
