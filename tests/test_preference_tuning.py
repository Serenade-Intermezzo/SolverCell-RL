"""Mechanism tests for schedule tuning and gauge-centered guide fitting."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from solvercell.preference_config import PreferenceConfig
from solvercell.preference_learner import PreferenceLearner, centered_guide_loss
from solvercell.preference_policy import PreferenceActor
from test_preference_learner import TinyTask, batch, config
import test_preference_learner as fixtures


class PreferenceTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_schedule_exact_and_stronger_terminal_noise(self):
        torch.manual_seed(31)
        before = torch.get_rng_state()
        old = PreferenceActor(3, 2, hidden=16, steps=8)
        torch.set_rng_state(before)
        explicit = PreferenceActor(3, 2, hidden=16, steps=8, beta_end=.2)
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], explicit.state_dict()[key]))
        obs, prefs = torch.zeros(4, 3), torch.tensor([[.5, .5]]).expand(4, -1)
        self.assertTrue(torch.equal(old.sample_path(obs, prefs, torch.Generator().manual_seed(8)),
                                    explicit.sample_path(obs, prefs, torch.Generator().manual_seed(8))))
        stronger = PreferenceActor(3, 2, hidden=16, steps=8, beta_end=.6)
        self.assertLess(float(stronger.alpha_bars[-1]), float(old.alpha_bars[-1]))
        # Unit endpoints have squared norm1; the Gaussian prior has E norm^2=D.
        for actor in (old, stronger):
            self.assertTrue(torch.all(actor.betas > 0) and torch.all(actor.betas < 1))
        self.assertLess(float(stronger.alpha_bars[-1]) * (8 - 1), float(old.alpha_bars[-1]) * (8 - 1))
        for end in (0., 1., float('nan')):
            with self.assertRaises(ValueError):
                PreferenceActor(3, 2, beta_end=end)

    def test_centered_loss_has_gauge_invariance_and_correct_scale(self):
        prediction = torch.tensor([1., 2., 4., -1., 3., 7.], requires_grad=True)
        labels = torch.tensor([2., 4., 5., -2., 2., 8.])
        loss = centered_guide_loss(prediction, labels, 3)
        manual = ((prediction.view(2, 3) - prediction.view(2, 3).mean(1, keepdim=True)
                   - labels.view(2, 3) + labels.view(2, 3).mean(1, keepdim=True)) ** 2).mean()
        torch.testing.assert_close(loss, manual)
        offsets = torch.tensor([100., 100., 100., -20., -20., -20.])
        torch.testing.assert_close(loss, centered_guide_loss(prediction + offsets, labels - offsets, 3))
        loss.backward()
        torch.testing.assert_close(prediction.grad.view(2, 3).sum(1), torch.zeros(2), atol=1e-6, rtol=0)
        self.assertLess(float(centered_guide_loss(labels + offsets, labels, 3)), 1e-10)
        with self.assertRaises(ValueError):
            centered_guide_loss(prediction, labels, 0)

    def test_group_conditions_and_guide_objective_do_not_change_base_learning(self):
        cfg = config(guide_interval=1, guide_batch_size=4, guide_prefixes=2,
                     guide_objective='grouped_absolute')
        absolute, centered = PreferenceLearner(cfg, TinyTask()), PreferenceLearner(
            replace(cfg, guide_objective='centered'), TinyTask())
        comparator = fixtures.PreferenceLearnerTest()
        observed = []
        def capture(obs, prefs, generator=None, path_indices=None):
            torch.testing.assert_close(obs[::2], obs[1::2])
            torch.testing.assert_close(prefs[::2], prefs[1::2])
            self.assertTrue(torch.equal(path_indices[::2], path_indices[1::2]))
            result = original(obs, prefs, generator=generator, path_indices=path_indices)
            observed.append(result)
            self.assertFalse(torch.equal(result[0][::2], result[0][1::2]))
            return result
        original = centered.guide_training_targets
        with patch.object(centered, 'guide_training_targets', side_effect=capture):
            for _ in range(3):
                absolute.update(batch())
                centered.update(batch())
        self.assertEqual(len(observed), 3)
        for name in ('actor', 'target_actor', 'q1', 'q2', 'tq1', 'tq2', 'actor_optim', 'q_optim'):
            comparator.equal(getattr(absolute, name).state_dict(), getattr(centered, name).state_dict())
        for name in ('generator', 'preference_generator', 'guide_generator', 'behavior_generator'):
            self.assertTrue(torch.equal(getattr(absolute, name).get_state(), getattr(centered, name).get_state()))
        self.assertEqual(absolute.counters, centered.counters)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(absolute.guide.parameters(), centered.guide.parameters())))

    def test_group_validation_and_centered_checkpoint_recovery(self):
        with self.assertRaises(ValueError):
            replace(PreferenceConfig(), guide_objective='centered', guide_batch_size=2).validate()
        cfg = config(guide_objective='centered', guide_prefixes=2)
        first, second = PreferenceLearner(cfg, TinyTask()), PreferenceLearner(cfg, TinyTask())
        first.update(batch())
        import copy
        second.load_state_dict(copy.deepcopy(first.state_dict()))
        comparator = fixtures.PreferenceLearnerTest()
        comparator.equal(first.update(batch()), second.update(batch()))
        comparator.equal(first.state_dict(), second.state_dict())

    def test_legacy_default_checkpoint_loads_but_cannot_silently_change_schedule(self):
        import copy
        cfg = config()
        original = PreferenceLearner(cfg, TinyTask())
        original.update(batch())
        old = copy.deepcopy(original.state_dict())
        old['format_version'] = 1
        for key in ('diffusion_beta_end', 'guide_objective', 'guide_prefixes'):
            del old['spec'][key]
        restored = PreferenceLearner(cfg, TinyTask())
        restored.load_state_dict(old)
        fixtures.PreferenceLearnerTest().equal(original.state_dict(), restored.state_dict())
        changed = PreferenceLearner(replace(cfg, diffusion_beta_end=.6), TinyTask())
        with self.assertRaises(ValueError):
            changed.load_state_dict(old)


if __name__ == '__main__':
    unittest.main()
