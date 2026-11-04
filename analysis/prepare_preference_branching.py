"""Freeze fixed-checkpoint branching experiments, without launching workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json, source_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-id',required=True)
    p.add_argument('--debug',action='store_true')
    args=p.parse_args()
    if Path(args.campaign_id).name!=args.campaign_id or args.campaign_id in {'.','..'}:
        raise ValueError('Use a new single-directory campaign name')
    directory=ROOT/'experiments'/args.campaign_id
    directory.mkdir(parents=True,exist_ok=False)
    parents={}
    for seed in (0,1,2):
        for method in ('sphere','diffusion'):
            path=(Path('outputs/preference_tuning_20261104')/('sphere_t010' if method=='sphere' else 'diff_b020_t010')
                  if seed==0 else Path('outputs/preference_replication_20261104')/f'{method}_seed{seed}')
            files={name:hashlib.sha256((ROOT/path/name).read_bytes()).hexdigest()
                   for name in ('checkpoint.pt','config.json','source.json')}
            parents[f'{method}_seed{seed}']={'output':str(path),'files_sha256':files}
    topology={f'tree_m{m}_k{k}':{'roots':m,'children':8//m,'branch_index':k}
              for m in (2,4) for k in (2,4,6)}
    independent=[1,3,4,5,6,7,8]
    manifest={'schema':'preference_branching_v3','campaign_id':args.campaign_id,'debug':args.debug,
        'expected_source_sha256':source_manifest()['source_sha256'],
        'parent_source_sha256':'33fcff51897f7485ddce4a303d089526d0cada8e165a4c380e7c9509db6680b3',
        'parents':parents,'topologies':topology,'independent_candidates':independent,
        'screen_preferences':[.5] if args.debug else [0.,.5,1.],
        'final_preferences':[.25] if args.debug else [.125,.375,.625,.875],
        'episodes':2 if args.debug else 32,'screen_seed_start':700000,'final_seed_start':800000,
        'timing_cycles':2 if args.debug else 32,'variance_prefixes':4 if args.debug else 32,
        'variance_suffixes':4 if args.debug else 32,'support_candidates':8 if args.debug else 128,
        'continuation_repetitions':2 if args.debug else 8,
        'selection_rule':{'mean_utility_fraction':.98,'minimum_per_preference_fraction':.95,
            'maximum_latency_fraction':1.05,'minimum_nfe_saving_fraction':.10,
            'rank':'lower NFE, higher mean utility, lower median randomized inference latency'},
        'design':'docs/PREFERENCE_BRANCHING_V3.md','scope':'read-only inference; zero training updates',
        'runner_script_sha256':hashlib.sha256((ROOT/'analysis/run_preference_branching.py').read_bytes()).hexdigest(),
        'max_parallel':2,'device':'cpu','cpu_sets':[[24],[25]]}
    atomic_json(directory/'manifest.json',manifest)
    print(directory/'manifest.json')


if __name__=='__main__':main()
