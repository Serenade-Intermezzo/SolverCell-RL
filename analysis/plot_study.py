"""Render complete learning curves from the two frozen campaigns."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
fig,axes=plt.subplots(1,2,figsize=(15,5.8))
first=root/'outputs/intervention_screen_20261104'
second=root/'outputs/temperature_screen_20261104'
colors={'cell':'#2171b5','point':'#d95f02','independent':'#7570b3'}
for run in sorted(first.iterdir()):
 if not (run/'metrics.json').exists():continue
 es=[json.loads(l) for l in (run/'evaluations.jsonl').read_text().splitlines()];es=[v for v in es if v['split']=='validation']
 axes[0].plot([v['train_steps'] for v in es],[v['return_mean'] for v in es],label=run.name,linewidth=1.6)
for run in sorted(second.iterdir()):
 if not (run/'metrics.json').exists():continue
 es=[json.loads(l) for l in (run/'evaluations.jsonl').read_text().splitlines()];es=[v for v in es if v['split']=='validation']
 method=run.name.split('_t')[0];temp=run.name.split('_t')[1]
 axes[1].plot([v['train_steps'] for v in es],[v['return_mean'] for v in es],color=colors[method],linestyle='-' if temp=='010' else '--',label=f'{method}, T={"0.10" if temp=="010" else "0.25"}',linewidth=1.8)
floors=json.loads((first/'cell_m010/behavioral_baselines.json').read_text())
for ax in axes:
 ax.axhline(floors['random']['return_mean'],color='#666666',linewidth=1,linestyle=':',label='random feasible')
 ax.set(xlabel='Training updates after 1,000-step warm-up',ylabel='Validation episode return (H=20)')
 ax.grid(alpha=.22);ax.legend(fontsize=8,loc='best')
axes[0].axhline(floors['greedy']['return_mean'],color='#333333',linewidth=1,linestyle='-.',label='model-informed myopic')
axes[0].legend(fontsize=8,loc='best')
axes[0].set_title('Initial screen: eight arms, T=1 for matched methods')
axes[1].set_title('Controlled temperature follow-up: same margin 0.1')
fig.suptitle('Dynamic Intervention (N=40, A=20, B=10): one training seed / one instance',fontsize=12)
fig.tight_layout()
out=root/'reports/study_20261104';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
print(out/'learning_curves.png')
