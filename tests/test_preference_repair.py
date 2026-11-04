import copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from solvercell.environment import InterventionEnv
from solvercell.preference_data import PreferenceTask
from solvercell.preference_config import PreferenceConfig
from solvercell.preference_learner import PreferenceLearner
from solvercell.preference_policy import PreferenceActor
from solvercell.preference_trainer import evaluate_preference


class RepairTests(unittest.TestCase):
    def test_last_noise_only_preserves_prior_prefix_and_rng(self):
        torch.manual_seed(71);a=PreferenceActor(3,8,hidden=16,final_noise=True);b=PreferenceActor(3,8,hidden=16,final_noise=False);b.load_state_dict(a.state_dict())
        o=torch.zeros(64,3);p=torch.full((64,2),.5);g=torch.Generator().manual_seed(29);h=torch.Generator().manual_seed(29)
        x=a.sample_path(o,p,g);y=b.sample_path(o,p,h)
        self.assertTrue(torch.equal(x[:,:-1],y[:,:-1]));self.assertTrue(torch.equal(g.get_state(),h.get_state()))
        mean,std=b.transition_parameters(o,p,y[:,-2],7);self.assertEqual(float(std),0);self.assertTrue(torch.equal(y[:,-1],mean))
        self.assertFalse(torch.equal(x[:,-1],y[:,-1]))
        with self.assertRaises(ValueError):b.transition_log_prob(o,p,y[:,-2],y[:,-1],7)
        repeated=b.suffix(o,p,y[0,-2].repeat(64,1),7,g);self.assertTrue(torch.equal(repeated[0].expand_as(repeated),repeated))

    def test_x0_and_epsilon_equivalent_kernel_when_predictions_equivalent(self):
        a=PreferenceActor(3,4,hidden=16);b=PreferenceActor(3,4,hidden=16,prediction_type='x0')
        o=torch.zeros(2,3);p=torch.ones(2,2)/2;x=torch.randn(2,4);clean=torch.randn(2,4)
        for index in range(8):
            abar=a.alpha_bars[7-index];eps=(x-abar.sqrt()*clean)/(1-abar).sqrt()
            with patch.object(a,'forward',return_value=eps),patch.object(b,'forward',return_value=clean):
                m,s=a.transition_parameters(o,p,x,index);n,t=b.transition_parameters(o,p,x,index)
                self.assertTrue(torch.allclose(m,n,atol=1e-6));self.assertEqual(s,t)
        loss,diag=b.loss(o,p,clean,generator=torch.Generator().manual_seed(3));loss.backward();self.assertIn('endpoint_mse',diag)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None))

    def test_raw_replay_scaled_td_and_raw_evaluation_are_separate(self):
        cfg=PreferenceConfig(component_scales=(2.,4.),horizon=2,hidden=16);task=PreferenceTask(InterventionEnv(horizon=2));learner=PreferenceLearner(cfg,task)
        batch={'obs':torch.zeros(2,41),'next_obs':torch.zeros(2,41),'action':torch.zeros(2,20),
               'reward_components':torch.tensor([[4.,8.],[10.,20.]]),'done':torch.ones(2)}
        original=batch['reward_components'].clone();target=learner.critic_targets(batch,torch.ones(2,2)/2)
        self.assertTrue(torch.equal(batch['reward_components'],original));self.assertTrue(torch.allclose(target,original/torch.tensor([2.,4.])*.025))
        r=evaluate_preference(task,learner,.5,'direct',2,1500000);raw=np.array(r['component_returns'])
        self.assertTrue(np.allclose(raw.mean(1),r['original_returns']));self.assertTrue(np.allclose((raw/[2,4])@np.array([.5,.5]),r['scalarized_returns']))
        self.assertEqual(r['component_scales'],[2.,4.])

    def test_old_checkpoint_defaults_and_changed_chain_guard(self):
        task=PreferenceTask(InterventionEnv());a=PreferenceLearner(PreferenceConfig(hidden=16),task);old=copy.deepcopy(a.state_dict())
        for key in ('diffusion_final_noise','diffusion_prediction','component_scales'):old['spec'].pop(key)
        b=PreferenceLearner(PreferenceConfig(hidden=16),task);b.load_state_dict(old)
        for cfg in (PreferenceConfig(hidden=16,diffusion_final_noise=False),PreferenceConfig(hidden=16,diffusion_prediction='x0'),PreferenceConfig(hidden=16,component_scales=(2.,3.))):
            with self.assertRaises(ValueError):PreferenceLearner(cfg,task).load_state_dict(old)

    def test_actual_suffix_labels_use_the_repaired_chain(self):
        cfg=PreferenceConfig(hidden=16,diffusion_final_noise=False,guide_suffixes=8)
        learner=PreferenceLearner(cfg,PreferenceTask(InterventionEnv()))
        obs=torch.zeros(4,41);prefs=torch.ones(4,2)/2
        prefix,index,labels,_=learner.guide_training_targets(obs,prefs,generator=torch.Generator().manual_seed(43),path_indices=torch.full((4,),7))
        terminal=learner.target_actor.suffix(obs,prefs,prefix,7,torch.Generator().manual_seed(44))
        baseline=(learner._twin_vector(obs,torch.zeros(4,20),prefs)*prefs).sum(-1)
        utility=(learner._twin_vector(obs,learner._decode(terminal),prefs)*prefs).sum(-1)
        self.assertTrue(torch.allclose(labels,(utility-baseline)/cfg.guide_temperature,atol=1e-6))

    def test_scales_and_mode_validation(self):
        for scales in ((1.,0.),(1.,float('nan')),(1.,),(-1.,2.)):
            with self.assertRaises(ValueError):PreferenceConfig(component_scales=scales).validate()
        with self.assertRaises(ValueError):PreferenceConfig(diffusion_final_noise=1).validate()
        with self.assertRaises(ValueError):PreferenceActor(3,4,prediction_type='unknown')


if __name__=='__main__':unittest.main()
