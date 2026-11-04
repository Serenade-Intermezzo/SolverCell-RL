"""Distribution, grouping and actual compute checks for prefix sharing."""
import unittest
import torch

from solvercell.preference_branching import branch_weight_sample, BranchingPreferencePolicy
from solvercell.preference_policy import PreferenceActor, terminal_weight_sample
import test_preference_learner as fixtures


class PreferenceBranchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_one_child_exactly_equals_original_importance_sampler(self):
        torch.manual_seed(32)
        actor=PreferenceActor(3, 2, hidden=16, steps=8)
        obs=torch.zeros(2,3);prefs=torch.tensor([[.2,.8],[.8,.2]])
        score=lambda x:x[:,0]-x[:,1]
        for index in (0,4,6,8):
            a,stats=branch_weight_sample(actor,obs,prefs,score,roots=4,children=1,
                branch_index=index,generator=torch.Generator().manual_seed(3))
            b,reference=terminal_weight_sample(actor,obs,prefs,score,4,torch.Generator().manual_seed(3))
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(stats['nfe'],reference['nfe'])
            self.assertAlmostEqual(stats['log_normalizer_mean'],reference['log_normalizer_mean'])

    def test_shared_prefixes_have_fresh_suffixes_and_exact_forward_cost(self):
        actor=PreferenceActor(3,2,hidden=16,steps=8)
        obs=torch.zeros(2,3);prefs=torch.ones(2,2)/2
        prefix_inputs=[];rows=[]
        original=actor.suffix
        def suffix(o,p,x,index,generator=None):
            prefix_inputs.append(x.clone())
            self.assertEqual(index,6)
            return original(o,p,x,index,generator)
        actor.suffix=suffix
        hook=actor.register_forward_hook(lambda module,args,out: rows.append(len(args[0])))
        terminal=[]
        def score(x):terminal.append(x.clone());return x[:,0]
        _,stats=branch_weight_sample(actor,obs,prefs,score,roots=4,children=2,
            branch_index=6,generator=torch.Generator().manual_seed(23))
        hook.remove()
        self.assertEqual(rows,[8]*6+[16]*2)
        self.assertEqual(sum(rows),stats['nfe'])
        self.assertEqual(stats['nfe'],80)
        self.assertTrue(torch.equal(prefix_inputs[0][::2],prefix_inputs[0][1::2]))
        self.assertFalse(torch.equal(terminal[0][::2],terminal[0][1::2]))
        self.assertEqual(stats['terminal_evaluations'],16)

    def test_unequal_counts_do_not_reweight_root_population(self):
        class FixedRoots:
            kind,obs_dim,act_dim,steps='diffusion',3,2,2
            def _initial(self,obs,generator):return torch.eye(2).repeat(len(obs)//2,1)
            def suffix(self,obs,prefs,x,index,generator):return x
        actor=FixedRoots();obs=torch.zeros(10000,3);prefs=torch.ones(10000,2)/2
        out,stats=branch_weight_sample(actor,obs,prefs,lambda x:x.new_zeros(len(x)),
            roots=2,branch_counts=(1,3),branch_index=0,generator=torch.Generator().manual_seed(6))
        self.assertLess(abs(float(out[:,1].mean())-.5),.02)
        self.assertAlmostEqual(stats['log_normalizer_mean'],0.,places=12)
        self.assertEqual(stats['terminal_evaluations'],40000)

    def test_invalid_support_and_deterministic_flow_are_rejected(self):
        obs=torch.zeros(1,3);prefs=torch.ones(1,2)/2
        actor=PreferenceActor(3,2,hidden=16,steps=8)
        with self.assertRaises(ValueError):
            branch_weight_sample(actor,obs,prefs,lambda x:x.new_full((len(x),),-torch.inf),roots=2,branch_index=4)
        with self.assertRaises(ValueError):
            branch_weight_sample(actor,obs,prefs,lambda x:x[:,0],roots=2,branch_counts=(1,0),branch_index=4)
        sphere=PreferenceActor(3,2,hidden=16,steps=8,kind='sphere')
        with self.assertRaises(ValueError):
            branch_weight_sample(sphere,obs,prefs,lambda x:x[:,0],roots=2,branch_index=4)

    def test_single_observation_shape_and_terminal_fork_degeneracy(self):
        actor=PreferenceActor(3,2,hidden=16,steps=8)
        seen=[]
        def score(x):seen.append(x.clone());return x[:,0]
        out,stats=branch_weight_sample(actor,torch.zeros(3),torch.ones(2)/2,score,
            roots=3,children=2,branch_index=8,generator=torch.Generator().manual_seed(11))
        self.assertEqual(tuple(out.shape),(1,2))
        self.assertTrue(torch.equal(seen[0][::2],seen[0][1::2]))
        self.assertEqual(stats['nfe'],24)
        self.assertEqual(stats['terminal_evaluations'],6)

    def test_leaf_gaussian_marginals_and_sibling_covariance(self):
        actor=PreferenceActor(3,2,hidden=8,steps=8)
        for parameter in actor.parameters():parameter.data.zero_()
        obs=torch.zeros(3000,3);prefs=torch.ones(3000,2)/2
        captured=[]
        def score(x):captured.append(x.clone());return x.new_zeros(len(x))
        branch_weight_sample(actor,obs,prefs,score,roots=2,children=2,branch_index=4,
                             generator=torch.Generator().manual_seed(19))
        raw=captured[0].reshape(3000,2,2,2).double()
        variance=1.
        prefix_variance=None
        for index in range(actor.steps):
            level=actor.steps-1-index
            variance=variance/float(actor.alphas[level])+float(actor.betas[level])
            if index==3:prefix_variance=variance
        expected_covariance=prefix_variance/float(actor.alphas[:4].prod())
        self.assertLess(abs(float(raw.mean())),.05*variance**.5)
        self.assertLess(abs(float(raw.var())-variance),.07*variance)
        covariance=float((raw[:,:,0]*raw[:,:,1]).mean())
        self.assertLess(abs(covariance-expected_covariance),.07*expected_covariance)

    def test_adapter_preserves_rng_and_matches_all_actual_query_counts(self):
        learner=fixtures.PreferenceLearner(fixtures.config(sampling_steps=8),fixtures.TinyTask())
        adapter=BranchingPreferencePolicy(learner,{'branch':{'roots':4,'children':2,'branch_index':6}})
        states={name:getattr(learner,name).get_state().clone() for name in
                ('generator','preference_generator','guide_generator','behavior_generator')}
        rows=[];qs=[];guide=[]
        hooks=[learner.target_actor.register_forward_hook(lambda m,a,o:rows.append(len(a[0]))),
               learner.tq1.register_forward_hook(lambda m,a,o:qs.append(len(a[0]))),
               learner.tq2.register_forward_hook(lambda m,a,o:qs.append(len(a[0]))),
               learner.guide.register_forward_hook(lambda m,a,o:guide.append(len(a[0])))]
        action,cost,stats=adapter.act([.2,.3,1],[.5,.5],'branch',torch.Generator().manual_seed(8))
        for hook in hooks:hook.remove()
        self.assertEqual(sum(rows),40);self.assertEqual(sum(qs),18);self.assertEqual(len(guide),0)
        self.assertEqual(stats['nfe'],40);self.assertEqual(stats['q_calls'],18);self.assertEqual(stats['decoder_calls'],9)
        for name,state in states.items():self.assertTrue(torch.equal(state,getattr(learner,name).get_state()))

    def test_default_adapter_rng_is_learner_owned(self):
        learner=fixtures.PreferenceLearner(fixtures.config(sampling_steps=8),fixtures.TinyTask())
        adapter=BranchingPreferencePolicy(learner,{'branch':{'roots':4,'children':2,'branch_index':6}})
        global_before=torch.get_rng_state().clone()
        own_before=learner.behavior_generator.get_state().clone()
        adapter.act([.2,.3,1],[.5,.5],'branch')
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        self.assertFalse(torch.equal(own_before,learner.behavior_generator.get_state()))


if __name__=='__main__':unittest.main()
