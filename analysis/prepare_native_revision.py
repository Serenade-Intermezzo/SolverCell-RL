"""Generate a new immutable native-structure experiment manifest."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from solvercell.config import Config
from tools.campaign import fingerprint,source_sha256
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--campaign-id',default='native_revision_20261104');args=p.parse_args()
folder=root/'experiments'/args.campaign_id
if folder.exists():raise FileExistsError(folder)
folder.mkdir(parents=True)
base=replace(Config(),hidden=256,reward_scale=1.0,actor_interval=1,exploration_epsilon=0.,
             candidates=12,bootstrap_candidates=16,reference_profile='native_structure_revision',
             bridge_sigma=.5,margin=.1)
methods={'native_reference':'lsflow_tensor','full_point':'native_full_point_diffusion',
         'quotient_sphere':'native_quotient_sphere','quotient_point':'native_quotient_point_diffusion',
         'quotient_cell':'native_quotient_cell_diffusion','quotient_independent':'native_quotient_independent_diffusion'}
jobs=[]
for name,method in methods.items():
 cfg=replace(base,method=method).validate();path=folder/(name+'.json');path.write_text(json.dumps(cfg.to_dict(),indent=2,sort_keys=True)+'\n')
 jobs.append({'id':name,'config':str(path.relative_to(root)),'config_sha256':fingerprint(cfg.to_dict()),'output':f'outputs/{args.campaign_id}/{name}'})
manifest={'campaign_id':args.campaign_id,'expected_source_sha256':source_sha256(root),'max_parallel':4,
          'device':'cpu','cpu_sets':[[24],[25],[26],[27]],'jobs':jobs}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(folder)
