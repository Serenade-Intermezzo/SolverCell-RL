"""Apply the preregistered Stage A gates and write summary.json."""
import json, statistics as st, sys
from pathlib import Path
ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); INST = range(100, 106)
ex = [json.loads(l) for l in (ROOT / 'exits.jsonl').read_text().splitlines()]; assert len(ex) == 18 and all(e['exit'] == 0 for e in ex)
P1 = {i: json.loads((ROOT / f'p1_i{i}' / 'result.json').read_text()) for i in INST}; P2 = {i: json.loads((ROOT / f'p2_i{i}' / 'result.json').read_text()) for i in INST}; A1 = {i: json.loads((ROOT / f'a1_i{i}' / 'result.json').read_text()) for i in INST}
assert len({r['source'] for r in [*P1.values(), *P2.values(), *A1.values()]}) == 1
w5info = json.loads((ROOT / 'w5.json').read_text()); w5 = w5info['w5']; B = ['2', '5', '10']
G = {}
hard = [A1[i]['a1']['hard'] for i in INST]
G['A1'] = dict(passed=sum(hard) >= 5, hard=hard, per=[dict(forward=A1[i]['a1']['sec'], completed=A1[i]['a1']['completed'], mean=A1[i]['a1'].get('exact_mean_sec'), peak=A1[i]['a1']['peak'], width=max(A1[i]['a1']['widths']), bytes=A1[i]['a1']['max_bytes'], log_a=A1[i]['a1'].get('log_count'), arms_done=A1[i]['a1'].get('arms_done')) for i in INST])
G['A2'] = dict(passed=sum(P2[i]['a2']['p95'] <= .1 for i in INST) >= 5, per=[P2[i]['a2'] for i in INST])
a3 = [P2[i]['a3']['n_ok'] >= 2 and P2[i]['a3']['pareto_points'] >= 5 for i in INST]
G['A3'] = dict(passed=sum(a3) >= 5, per=[dict(n_ok=P2[i]['a3']['n_ok'], pareto=P2[i]['a3']['pareto_points'], pairs=P2[i]['a3']['pairs']) for i in INST])
G['A4'] = dict(passed=w5info['passed'], w5=w5, mean_r2=w5info['mean_r2'], by_beta={b: st.mean(P1[i]['a4'][str(w5)][b]['r2'] for i in INST) for b in B}, group_positive=[P1[i]['group_positive'] for i in INST], scales=[P1[i]['scales'] for i in INST])
a5 = {b: st.mean(P2[i]['a5'][b]['mcmc12']['gap_low'] for i in INST) for b in B}
G['A5'] = dict(passed=any(v >= 1 for v in a5.values()), gap_low_mean=a5, per={b: {f'mcmc{s}': dict(gain=st.mean(P2[i]['a5'][b][f'mcmc{s}']['gain'] for i in INST), gap_low=st.mean(P2[i]['a5'][b][f'mcmc{s}']['gap_low'] for i in INST), h_up=st.mean(P2[i]['a5'][b][f'mcmc{s}']['h_up'] for i in INST)) for s in (0, 12, 48, 192)} | dict(j_star=st.mean(P2[i]['a5'][b]['j_star'] for i in INST), target_gain=st.mean(P2[i]['a5'][b]['target_gain'] for i in INST), target_group=st.mean(P2[i]['a5'][b]['target_group'] for i in INST)) for b in B})
G['A6'] = dict(passed=sum(P2[i]['a6']['5']['mae'] <= .05 for i in INST) >= 5, per=[P2[i]['a6'] for i in INST])
a7 = [all(v['bound'] >= v['exact'] - .2 and v['slack'] <= 2 for v in P2[i]['a7'].values()) and P2[i]['a7_rho']['linf'] <= .02 for i in INST]
G['A7'] = dict(passed=sum(a7) >= 5, per=[dict(bound=P2[i]['a7'], rho=P2[i]['a7_rho']) for i in INST])
s = dict(source=P1[100]['source'], w5=w5, gates={k: v['passed'] for k, v in G.items()}, detail=G, l_mcmc={b: {k: st.mean(P2[i]['l_mcmc'][b][k] for i in INST) for k in ('map', 'mcmc12', 'long')} for b in B})
OUT.write_text(json.dumps(s, indent=1)); print(s['gates'], 'w5', w5)
print('A1', hard, [(p['forward'], p['completed'], p['mean'], p['peak'], p['width'], p['log_a'], p['arms_done']) for p in G['A1']['per']])
print('A2', [(round(p['p50'] * 1e3, 1), round(p['p95'] * 1e3, 1), p['feasible']) for p in G['A2']['per']])
print('A3', [(p['n_ok'], p['pareto']) for p in G['A3']['per']]); print(' pairs i100', G['A3']['per'][0]['pairs'])
print('A4', G['A4']['mean_r2'], G['A4']['by_beta'], [round(x, 4) for x in G['A4']['group_positive']], [round(p[4], 3) for p in G['A4']['scales']])
print('A5 gap', a5); [print(' ', b, {k: (round(v['gain'], 2), round(v['gap_low'], 2), round(v['h_up'], 1)) for k, v in x.items() if k.startswith('mcmc')}, 'J*', round(x['j_star'], 2), 'tgt', round(x['target_gain'], 2), round(x['target_group'], 2)) for b, x in G['A5']['per'].items()]
print('A6', [round(p['5']['mae'], 3) for p in G['A6']['per']], 'rel', [round(p['5']['mae'] / p['5']['mean_marginal'], 2) for p in G['A6']['per']], 'arm', [round(p['5']['arm_mae'], 3) for p in G['A6']['per']])
print('A7', [([round(v['slack'], 2) for v in p['bound'].values()], round(p['rho']['linf'], 3)) for p in G['A7']['per']])
print('L mcmc', s['l_mcmc'])
print('sec', [round(P1[i]['sec']) for i in INST], [round(P2[i]['sec']) for i in INST])
