"""Apply the preregistered G0.5 gates to a results directory and write summary.json. Sections under 'posthoc' did not decide anything."""
import json, math, statistics as st, sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
INST, BETAS, ARMS = range(6), [2, 5, 10], ['structured', 'ef', 'plain']
R = {f'i{i}_b{b}_{a}': json.loads((ROOT / f'i{i}_b{b}_{a}' / 'result.json').read_text()) for i in INST for b in BETAS for a in ARMS}
exits = [json.loads(l) for l in (ROOT / 'exits.jsonl').read_text().splitlines()]
assert len(exits) == 54 and all(e['exit'] == 0 for e in exits)
assert all(r['status'] == 'completed' for r in R.values()) and len({r['source'] for r in R.values()}) == 1


def gain(i, b, a, k):
    return R[f'i{i}_b{b}_{a}'][k]['gain']


def ms(x):
    return dict(mean=st.mean(x), se=st.stdev(x) / math.sqrt(len(x)), median=st.median(x), positive=sum(v > 0 for v in x), values=x)


def diff(b, f, inst=INST):
    return [f(i, b) for i in inst]


da = lambda i, b: gain(i, b, 'structured', 'direct') - gain(i, b, 'structured', 'shift_res12')
db = lambda i, b: gain(i, b, 'structured', 'direct') - gain(i, b, 'ef', 'direct')
a = {b: ms(diff(b, da)) for b in BETAS}
g_b = {b: ms(diff(b, db)) for b in BETAS}
gate_a = sum(a[b]['mean'] >= .5 for b in BETAS) >= 2 and all(a[b]['mean'] >= -.25 for b in BETAS)
gate_b = sum(g_b[b]['mean'] >= .5 for b in BETAS) >= 2
means = {b: {**{f'{x}.direct': ms([gain(i, b, x, 'direct') for i in INST]) for x in ARMS},
             **{f'{x}.prop_res12': ms([gain(i, b, x, 'prop_res12') for i in INST]) for x in ARMS},
             **{k: ms([gain(i, b, 'structured', k) for i in INST]) for k in ['shift', 'shift_res12', 'loglinear_fp', 'ref']},
             'ef.exact_direct': ms([gain(i, b, 'ef', 'ef_exact_direct') for i in INST])} for b in BETAS}
curves = {b: {x: [st.mean(R[f'i{i}_b{b}_{x}']['rounds'][j]['direct']['gain'] for i in INST) for j in range(8)] for x in ARMS} for b in BETAS}
posthoc = dict(
    a_without_instance_0={b: ms(diff(b, da, range(1, 6))) for b in BETAS},
    b_without_instance_0={b: ms(diff(b, db, range(1, 6))) for b in BETAS},
    ref_ess={b: [R[f'i{i}_b{b}_structured']['ref']['ess'] for i in INST] for b in BETAS},
    ref_gain={b: [gain(i, b, 'structured', 'ref') for i in INST] for b in BETAS},
    prop12_minus_shift_res12={b: ms([gain(i, b, 'structured', 'prop_res12') - gain(i, b, 'structured', 'shift_res12') for i in INST]) for b in BETAS},
)
summary = dict(source=next(iter(R.values()))['source'], runs=len(R), gate_a=gate_a, gate_b=gate_b, a=a, b=g_b, means=means, curves=curves, posthoc=posthoc)
OUT.write_text(json.dumps(summary, indent=1))
print('gate_a', gate_a, 'gate_b', gate_b)
for b in BETAS:
    print(b, 'a %.2f±%.2f med %.2f pos %d/6 | b %.2f±%.2f med %.2f pos %d/6 | noI0 a %.2f b %.2f | prop12-sr12 %.2f±%.2f' % (
        a[b]['mean'], a[b]['se'], a[b]['median'], a[b]['positive'], g_b[b]['mean'], g_b[b]['se'], g_b[b]['median'], g_b[b]['positive'],
        posthoc['a_without_instance_0'][b]['mean'], posthoc['b_without_instance_0'][b]['mean'],
        posthoc['prop12_minus_shift_res12'][b]['mean'], posthoc['prop12_minus_shift_res12'][b]['se']))
