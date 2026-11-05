"""Summarise the diversity diagnostic with the criteria fixed in docs/G05_DISTILLATION_DESIGN.md. Decides no gate."""
import json, math, statistics as st, sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
INST, BETAS = range(6), [2, 5, 10]
R = {(i, b): json.loads((ROOT / f'i{i}_b{b}' / 'result.json').read_text()) for i in INST for b in BETAS}
exits = [json.loads(l) for l in (ROOT / 'exits.jsonl').read_text().splitlines()]
assert len(exits) == 18 and all(e['exit'] == 0 for e in exits) and len({r['source'] for r in R.values()}) == 1
METHODS = ['uniform', 'shift', 'loglinear_fp', 'shift_res12', 'ef_exact', 'ef', 'structured', 'plain']


def ms(x):
    return dict(mean=st.mean(x), se=st.stdev(x) / math.sqrt(len(x)), values=x)


def h_low(v):
    return v['exact_h'] if v['exact_h'] is not None else (v['h2'] if v['h2'] is not None else v['h2_floor'])


per = {}
for b in BETAS:
    per[b] = {}
    for m in METHODS:
        vs = [R[i, b][m] for i in INST]
        per[b][m] = {k: ms([v[k] for v in vs]) for k in ['gain', 'j_low', 'j_up', 'h_up', 'unique_frac', 'top1', 'hamming', 'h_district_set']}
        per[b][m]['h_low'] = ms([h_low(v) for v in vs])
        per[b][m]['collisions'] = [v['collisions'] for v in vs]
        per[b][m]['concentrated'] = sum(v['h2'] is not None and v['h2'] < math.log(100) for v in vs)
verdict = {}
for b in BETAS:
    verdict[b] = {}
    for x in ['shift_res12', 'ef_exact', 'loglinear_fp']:
        better = [i for i in INST if R[i, b]['structured']['j_low'] > R[i, b][x]['j_up']]
        worse = [i for i in INST if R[i, b]['structured']['j_up'] < R[i, b][x]['j_low']]
        verdict[b][x] = dict(better=better, worse=worse, unclear=[i for i in INST if i not in better and i not in worse])
target = {b: dict(value=ms([R[i, b]['target_j_is']['value'] for i in INST]), ess=[R[i, b]['target_j_is']['ess'] for i in INST]) for b in BETAS}
repro = {b: max(abs(R[i, b]['structured']['gain'] - R[i, b]['stored_direct']['structured']) for i in INST) for b in BETAS}
OUT.write_text(json.dumps(dict(source=next(iter(R.values()))['source'], per=per, verdict=verdict, target=target, repro=repro, instances={f'i{i}_b{b}': {m: {k: R[i, b][m][k] for k in ['gain', 'j_low', 'j_up', 'h2', 'h_up', 'top1', 'unique_frac', 'hamming']} for m in METHODS} for i in INST for b in BETAS}), indent=1))
for b in BETAS:
    print('beta', b, 'target_is %.1f' % target[b]['value']['mean'], 'ess', [round(e, 1) for e in target[b]['ess']], 'repro max |gain diff| %.2f' % repro[b])
    for m in METHODS:
        p = per[b][m]
        print(f"  {m:13s} gain {p['gain']['mean']:6.2f} J [{p['j_low']['mean']:7.1f},{p['j_up']['mean']:7.1f}] H [{p['h_low']['mean']:5.1f},{p['h_up']['mean']:5.1f}] uniq {p['unique_frac']['mean']:.2f} top1 {p['top1']['mean']:.3f} ham {p['hamming']['mean']:.2f} Hdist {p['h_district_set']['mean']:.2f} conc {p['concentrated']}/6")
    print('  verdict', verdict[b])
    print('  per-inst struct J', [(round(R[i, b]['structured']['j_low'], 1), round(R[i, b]['structured']['j_up'], 1)) for i in INST])
    print('  per-inst sr12   J', [(round(R[i, b]['shift_res12']['j_low'], 1), round(R[i, b]['shift_res12']['j_up'], 1)) for i in INST])
    print('  per-inst ef_ex  J', [round(R[i, b]['ef_exact']['j_low'], 1) for i in INST])
    print('  per-inst fp     J', [round(R[i, b]['loglinear_fp']['j_low'], 1) for i in INST])
    print('  per-inst target', [round(R[i, b]['target_j_is']['value'], 1) for i in INST])
