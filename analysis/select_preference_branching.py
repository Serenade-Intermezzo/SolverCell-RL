"""Lock one branching topology using seed0 validation and randomized timing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solvercell.storage import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--screen',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Selection is immutable')
    manifest=json.loads(args.manifest.read_text());screen=json.loads((args.screen/'summary.json').read_text())
    assert screen['status']=='complete' and screen['stage']=='screen' and screen['seed']==0
    assert json.loads((args.screen/'exit.json').read_text())['returncode']==0
    assert screen['source_sha256']==manifest['expected_source_sha256']
    parent=next(row for row in screen['parents'] if row['model']=='diffusion_seed0')
    rows=[row for row in screen['evaluations'] if row['model']=='diffusion_seed0']
    weights=manifest['screen_preferences'];rule=manifest['selection_rule']
    refs={w:next(r for r in rows if r['mode']=='is8' and r['low_response_weight']==w) for w in weights}
    reference_mean=np.mean([row['return_mean'] for row in refs.values()])
    timing=parent['timing'];candidates={}
    for name,config in manifest['topologies'].items():
        selected={w:next(r for r in rows if r['mode']==name and r['low_response_weight']==w) for w in weights}
        value=float(np.mean([row['return_mean'] for row in selected.values()]))
        nfe=config['roots']*config['branch_index']+8*(8-config['branch_index'])
        utility_ratio=value/reference_mean
        minimum=min(selected[w]['return_mean']/refs[w]['return_mean'] for w in weights)
        latency=timing[name]['median_seconds']/timing['is8']['median_seconds']
        paired={str(w):{'mean':float(np.mean(np.asarray(selected[w]['scalarized_returns'])-refs[w]['scalarized_returns'])),
            'episode_mean_se':float(np.std(np.asarray(selected[w]['scalarized_returns'])-refs[w]['scalarized_returns'],ddof=1)/np.sqrt(manifest['episodes']))} for w in weights}
        eligible=(utility_ratio>=rule['mean_utility_fraction'] and minimum>=rule['minimum_per_preference_fraction']
                  and latency<=rule['maximum_latency_fraction'] and 1-nfe/64>=rule['minimum_nfe_saving_fraction'])
        candidates[name]={'topology':config,'mean_utility':value,'utility_fraction_vs_is8':utility_ratio,
            'minimum_preference_fraction':minimum,'median_latency_fraction_vs_is8':latency,
            'actor_nfe_per_decision':nfe,'eligible':bool(eligible),'paired_validation':paired}
    eligible=[name for name,row in candidates.items() if row['eligible']]
    if eligible:
        chosen=min(eligible,key=lambda name:(candidates[name]['actor_nfe_per_decision'],
            -candidates[name]['mean_utility'],candidates[name]['median_latency_fraction_vs_is8']))
    else:chosen=max(candidates,key=lambda name:(candidates[name]['mean_utility'],-candidates[name]['actor_nfe_per_decision']))
    result={'source_sha256':manifest['expected_source_sha256'],'scope':'seed0 validation-only engineering selection; no final episodes used',
        'promotion_eligible':bool(eligible),'evaluated_topology':chosen,
        'reason':'eligible lowest-NFE topology' if eligible else 'no promotion; best utility topology evaluated diagnostically',
        'candidates':candidates,'selection_rule':rule,'heldout_used':False}
    atomic_json(args.output,result);print(json.dumps({'topology':chosen,'promotion_eligible':bool(eligible),'heldout_used':False}))


if __name__=='__main__':main()
