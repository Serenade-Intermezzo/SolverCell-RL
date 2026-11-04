"""Render the completed native-structure study without seed/episode ambiguity."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
out=root/'reports/native_revision_study_20261104';out.mkdir(parents=True,exist_ok=True)
def runs(campaign):
 m=json.loads((root/'experiments'/campaign/'manifest.json').read_text());rows=[]
 for job in m['jobs']:
  p=root/job['output'];cfg=json.loads((p/'config.json').read_text());metrics=json.loads((p/'metrics.json').read_text());ex=json.loads((p/'exit.json').read_text())
  assert metrics['status']=='complete' and ex['returncode']==0
  evaluations=[json.loads(l) for l in (p/'evaluations.jsonl').read_text().splitlines()];vs=[v for v in evaluations if v['split']=='validation']
  rows.append({'id':job['id'],'config':cfg,'metrics':metrics,'x':np.array([v['train_steps'] for v in vs]),'y':np.array([v['return_mean'] for v in vs])})
 return rows
core=runs('native_revision_20261104');follow=runs('native_margin_20261104')
fig,axes=plt.subplots(2,2,figsize=(14,10))
for row in core:axes[0,0].plot(row['x'],row['y'],label=row['id'],lw=1.6)
axes[0,0].set_title('A. Native training structure: seed 0');axes[0,0].legend(fontsize=7)
methods={'native_quotient_point_diffusion':('Point','#d95f02'),'native_quotient_cell_diffusion':('Cell','#2171b5'),'native_quotient_independent_diffusion':('Independent','#7570b3')}
for method,(label,color) in methods.items():
 rows=[r for r in core if r['config']['method']==method]+[r for r in follow if r['config']['method']==method and r['config']['seed']==0 and r['config']['margin']!=.1]
 rows.sort(key=lambda r:r['config']['margin'])
 axes[0,1].plot([r['config']['margin'] for r in rows],[r['metrics']['last_window_return'] for r in rows],marker='o',label=label,color=color)
axes[0,1].set(title='B. Shared margin tuning: seed 0',xlabel='Unnormalized projection margin',ylabel='Last-window validation return');axes[0,1].legend()
for method,label,color in [('native_quotient_point_diffusion','Point diffusion','#d95f02'),('native_quotient_sphere','Spherical flow','#333333')]:
 rows=sorted([r for r in follow if r['config']['method']==method and r['config']['margin']==.1],key=lambda r:r['config']['seed'])
 assert [r['config']['seed'] for r in rows]==[0,1,2]
 ys=np.stack([r['y'] for r in rows]);x=rows[0]['x'];mean=ys.mean(0);std=ys.std(0,ddof=1)
 axes[1,0].plot(x,mean,label=label,color=color);axes[1,0].fill_between(x,mean-std,mean+std,color=color,alpha=.15)
 axes[1,1].plot([r['config']['seed'] for r in rows],[r['metrics']['last_window_return'] for r in rows],marker='o',label=label,color=color)
axes[1,0].set_title('C. Fixed margin 0.1: mean +/- training-seed SD (n=3)');axes[1,0].legend()
axes[1,1].set(title='D. Paired training seeds: final-window return',xlabel='Training seed',ylabel='Last-window validation return',xticks=[0,1,2]);axes[1,1].legend()
for ax in (axes[0,0],axes[1,0]):ax.set(xlabel='Training updates after warm-up',ylabel='Validation episode return (H=20)')
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('Dynamic Intervention: native-structure quotient policies',fontsize=14);fig.tight_layout()
fig.savefig(out/'learning_and_replication.png',dpi=180);fig.savefig(out/'learning_and_replication.pdf');plt.close(fig)
print(out/'learning_and_replication.png')
