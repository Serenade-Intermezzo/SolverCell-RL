"""Explicit v4 factory: calibrated objectives and a declared clean terminal kernel.

Legacy PreferenceConfig defaults intentionally retain the historical policy law.
This factory requires a frozen training-side calibration; old guides are never
loaded into a newly declared chain.
"""
from pathlib import Path
import hashlib
import json
from .preference_config import PreferenceConfig


def repaired_preference_config(calibration_path,*,method='preference_diffusion',seed=0,prediction='epsilon',**overrides):
    path=Path(calibration_path);calibration=json.loads(path.read_text())
    if calibration['status']!='complete' or not calibration['tradeoff_gate']['passed']:
        raise ValueError('Controllability gate must pass before repaired task construction')
    if calibration['calibration_seed_start']!=1200000:
        raise ValueError('Unexpected calibration split')
    scales=tuple(calibration['frozen_component_scales'])
    base=dict(method=method,seed=seed,component_scales=scales,diffusion_final_noise=False,
              diffusion_prediction=prediction,reward_scale=1.,temperature=.1,guide_temperature=.1,
              profile='preference_repair_v4')
    base.update(overrides)
    return PreferenceConfig(**base).validate(),{'calibration_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'objective':'requested preference dot raw components / frozen component scales',
        'raw_components_unchanged':True,'legacy_objective':False}
