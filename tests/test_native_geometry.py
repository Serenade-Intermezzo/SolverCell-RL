"""Independent constrained-QP and original-decoder checks for native cells."""

import unittest

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from solvercell.environment import InterventionEnv
from solvercell.geometry import decode
from solvercell.native_geometry import NativeCellProjector


def explicit_primitive_constraints(weights, action, budget, margin):
    """Build the original primitive-coordinate inequalities, without quotienting."""
    selected = np.flatnonzero(action)
    unselected = np.flatnonzero(1 - action)
    matrix = [weights[:, i] for i in selected]
    lower = [margin] * len(selected)
    if len(selected) < budget:
        matrix.extend(-weights[:, j] for j in unselected)
        lower.extend([margin] * len(unselected))
    else:
        for i in selected:
            for j in unselected:
                if not np.array_equal(weights[:, i], weights[:, j]):
                    matrix.append(weights[:, i] - weights[:, j])
                    lower.append(2 * margin)
    return np.asarray(matrix), np.asarray(lower)


class NativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def small_weights(self):
        # Different column norms ensure that a plain unweighted score-space
        # projection would fail the ambient-distance QP comparison.
        return torch.tensor([[2., 0., 2., 0., 0.],
                             [0., 3., 0., 3., 0.],
                             [0., 0., 0., 0., 1.],
                             [0., 0., 0., 0., 0.]], dtype=torch.float64)

    def test_matches_independent_ambient_quadratic_program(self):
        weights = self.small_weights()
        generator = torch.Generator().manual_seed(681)
        for budget in (1, 2, 3, 4):
            projector = NativeCellProjector(weights, budget)
            original = torch.randn(32, 4, dtype=torch.float64, generator=generator)
            noise = torch.randn(32, 4, dtype=torch.float64, generator=generator)
            actions = projector.decode(original)
            projected, stats = projector.project(noise, actions, 0.17, original)
            self.assertEqual(stats["fallback_rows"], 0)
            for index in range(len(noise)):
                z = noise[index].numpy()
                matrix, lower = explicit_primitive_constraints(weights.numpy(), actions[index].numpy(), budget, 0.17)
                qp = minimize(lambda x: 0.5 * np.square(x - z).sum(), z.copy(),
                              jac=lambda x: x - z, constraints=[LinearConstraint(matrix, lower, np.inf)],
                              method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
                with self.subTest(budget=budget, row=index):
                    self.assertTrue(qp.success, qp.message)
                    np.testing.assert_allclose(projected[index].numpy(), qp.x, atol=2e-7, rtol=1e-7)
                    self.assertTrue(np.all(matrix @ projected[index].numpy() >= lower - 1e-10))

    def test_original_benchmark_batched_decode_kernel_and_normalization(self):
        weights = torch.tensor(InterventionEnv().weights_p, dtype=torch.float32)
        projector = NativeCellProjector(weights, 10)
        self.assertEqual(projector.n_groups, 8)
        torch.testing.assert_close(projector.squared_norms, torch.full((8,), 27., dtype=torch.float64))
        generator = torch.Generator().manual_seed(720)
        original = torch.nn.functional.normalize(torch.randn(4, 64, 40, generator=generator), dim=-1)
        noise = torch.nn.functional.normalize(torch.randn(4, 64, 40, generator=generator), dim=-1)
        actions = projector.decode(original)
        # GEMM can round duplicate columns differently. Group broadcast makes
        # their ties exact; whichever duplicate gets selected has the same
        # original environment transition effect.
        raw_gemm_actions = decode(original @ weights, 10)
        torch.testing.assert_close(actions @ weights.T, raw_gemm_actions @ weights.T, atol=0, rtol=0)
        projected, stats = projector.project(noise, actions, 0.1, original)
        self.assertEqual(stats["fallback_rows"], 0)
        torch.testing.assert_close(projector.decode(projected), actions)
        normalized = torch.nn.functional.normalize(projected, dim=-1)
        torch.testing.assert_close(projector.decode(normalized), actions)
        # Projection changes only the row space of W and leaves 32 kernel
        # coordinates (including within-support contrasts) unchanged.
        basis = projector.unique_columns
        kernel = torch.eye(40) - basis @ torch.diag(projector.squared_norms.float().reciprocal()) @ basis.T
        torch.testing.assert_close(projected @ kernel, noise @ kernel, atol=2e-7, rtol=2e-6)
        again, _ = projector.project(projected, actions, 0.1, original)
        torch.testing.assert_close(again, projected, atol=3e-7, rtol=2e-6)
        repaired_original, _ = projector.project(original, actions, 0.1, original)
        self.assertTrue(torch.all((projected - noise).square().sum(-1) <=
                                  (repaired_original - noise).square().sum(-1) + 2e-6))

    def test_partial_tie_uses_canonical_prefix_and_multiple_ties_fall_back(self):
        weights = self.small_weights()
        projector = NativeCellProjector(weights, 2)
        # All group scores equal one. Stable top-k picks columns 0 and 1,
        # splitting two different duplicate groups: no strict-margin cell exists.
        original = torch.tensor([[0.5, 1 / 3, 1., 9.], [2., -1., -1., 4.]], dtype=torch.float64)
        actions = projector.decode(original)
        torch.testing.assert_close(actions[0], torch.tensor([1., 1., 0., 0., 0.], dtype=torch.float64))
        noise = torch.tensor([[-2., 1., 0.5, -3.], [-1., 1., 0.5, -5.]], dtype=torch.float64)
        projected, stats = projector.project(noise, actions, 0.2, original)
        self.assertEqual(stats["degenerate_tie_rows"], 1)
        self.assertEqual(stats["fallback_rows"], 1)
        self.assertEqual(stats["fallback_fraction"], 0.5)
        torch.testing.assert_close(projected[0], original[0], atol=0, rtol=0)
        torch.testing.assert_close(projector.decode(projected), actions)

    def test_single_partial_group_is_positive_and_separated_from_other_groups(self):
        weights = self.small_weights()
        projector = NativeCellProjector(weights, 3)
        original = torch.tensor([2., 1., -1., 0.], dtype=torch.float64)
        action = projector.decode(original)
        projected, stats = projector.project(torch.tensor([-2., -1., 2., 7.], dtype=torch.float64), action, 0.2, original)
        self.assertEqual(stats["fallback_rows"], 0)
        scores = projected @ weights
        self.assertGreaterEqual(float(scores[1]), 0.2 - 1e-12)
        self.assertGreaterEqual(float(scores[0] - scores[1]), 0.4 - 1e-12)
        self.assertGreaterEqual(float(scores[1] - scores[4]), 0.4 - 1e-12)
        self.assertEqual(float(projected[-1]), 7.)

    def test_invalid_fallback_and_unsupported_weights_are_rejected(self):
        weights = self.small_weights()
        nonorthogonal = weights.clone()
        nonorthogonal[0, 4] = 1
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            NativeCellProjector(nonorthogonal, 2)
        projector = NativeCellProjector(weights, 2)
        original = torch.ones(4, dtype=torch.float64)
        action = projector.decode(original)
        with self.assertRaisesRegex(ValueError, "fallback_costs"):
            projector.project(torch.zeros_like(original), action, 0.1, -original)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            projector.project(original, action, 0., original)


if __name__ == "__main__":
    unittest.main()
