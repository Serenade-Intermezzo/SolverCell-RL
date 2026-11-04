"""Independent checks against an enumerated action-space quadratic program."""

from __future__ import annotations

import itertools
import unittest

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from solvercell.geometry import decode, project_cell


def feasible_actions(dimension: int, budget: int) -> np.ndarray:
    return np.asarray(
        [bits for bits in itertools.product((0.0, 1.0), repeat=dimension) if sum(bits) <= budget]
    )


class GeometryTests(unittest.TestCase):
    def test_decoder_matches_exhaustive_linear_optimization(self):
        rng = np.random.default_rng(913)
        for dimension, budget in ((1, 0), (1, 1), (5, 2), (5, 5)):
            candidates = feasible_actions(dimension, budget)
            costs = rng.normal(size=(17, dimension))
            decoded = decode(torch.tensor(costs), budget).numpy()
            self.assertTrue(np.all(decoded.sum(axis=-1) <= budget))
            np.testing.assert_allclose((decoded * costs).sum(-1), (costs @ candidates.T).max(-1))

    def test_decoder_has_stable_ties_and_omits_nonpositive_scores(self):
        costs = torch.tensor([[3.0, 3.0, 3.0, -1.0], [0.0, -1.0, 2.0, 0.0]])
        expected = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(decode(costs, 2), expected)

    def test_projection_matches_full_action_space_qp(self):
        # The reference enumerates every feasible competitor and imposes the
        # original Hamming-margin inequalities.  It does not reuse the threshold
        # reduction, clipping formula, or bisection used by project_cell.
        rng = np.random.default_rng(271)
        for dimension, budget in ((1, 1), (4, 2), (4, 4)):
            candidates = feasible_actions(dimension, budget)
            for action in candidates:
                matrix = action[None] - candidates
                nontrivial = np.any(matrix != 0, axis=1)
                matrix = matrix[nontrivial]
                margin = 0.23
                rhs = margin * np.abs(matrix).sum(-1)
                for _ in range(3):
                    noise = rng.normal(size=dimension) * 2
                    projected = project_cell(
                        torch.tensor(noise), torch.tensor(action), budget, margin
                    ).numpy()
                    qp = minimize(
                        fun=lambda x: 0.5 * np.square(x - noise).sum(),
                        x0=margin * (2 * action - 1),
                        jac=lambda x: x - noise,
                        constraints=[LinearConstraint(matrix, rhs, np.inf)],
                        method="SLSQP",
                        options={"ftol": 1e-12, "maxiter": 500},
                    )
                    with self.subTest(dimension=dimension, budget=budget, action=action):
                        self.assertTrue(qp.success, qp.message)
                        self.assertTrue(np.all(matrix @ projected >= rhs - 1e-10))
                        np.testing.assert_allclose(projected, qp.x, atol=2e-7, rtol=1e-7)
                        np.testing.assert_array_equal(
                            decode(torch.tensor(projected), budget).numpy(), action
                        )

    def test_batched_projection_is_idempotent_and_preserves_decoder(self):
        generator = torch.Generator().manual_seed(41)
        noise = torch.randn(3, 7, 40, generator=generator, dtype=torch.float64)
        # Include full-budget, partial-budget, and empty target actions.
        actions = decode(torch.randn(3, 7, 40, generator=generator, dtype=torch.float64), 10)
        actions[0, 0] = 0
        actions[1, 0, 2:] = 0
        projected = project_cell(noise, actions, 10, 0.1)
        self.assertEqual(projected.shape, noise.shape)
        torch.testing.assert_close(decode(projected, 10), actions)
        torch.testing.assert_close(project_cell(projected, actions, 10, 0.1), projected)
        self.assertTrue(torch.isfinite(projected).all())

    def test_zero_budget_has_no_cell_constraints(self):
        noise = torch.tensor([[-3.0, 4.0, 2.0]], dtype=torch.float64)
        action = torch.zeros_like(noise)
        torch.testing.assert_close(project_cell(noise, action, 0, 2.0), noise)

    def test_zero_margin_boundary_and_zero_threshold_boundary(self):
        action = torch.tensor([1.0, 0.0], dtype=torch.float64)
        # Zero margin includes ties; the projection may have a different decode.
        torch.testing.assert_close(
            project_cell(torch.tensor([-1.0, -3.0], dtype=torch.float64), action, 1, 0),
            torch.tensor([0.0, -3.0], dtype=torch.float64),
        )
        # The optimum is theta=0, and negative unselected values stay unchanged.
        torch.testing.assert_close(
            project_cell(torch.tensor([-3.0, -10.0], dtype=torch.float64), action, 1, 0.2),
            torch.tensor([0.2, -10.0], dtype=torch.float64),
        )

    def test_dtype_and_input_validation(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            noise = torch.tensor([2.0, -4.0, 1.0], dtype=dtype)
            action = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
            projected = project_cell(noise, action, 1, 0.25)
            self.assertEqual(projected.dtype, dtype)
            torch.testing.assert_close(decode(projected, 1), action)
        with self.assertRaises(ValueError):
            project_cell(torch.ones(3), torch.ones(3), 2, 0.1)
        with self.assertRaises(ValueError):
            project_cell(torch.ones(3), torch.zeros(3), 2, -0.1)
        with self.assertRaises(ValueError):
            decode(torch.tensor([float("nan")]), 1)
        with self.assertRaises(ValueError):
            decode(torch.ones(3), 4)


if __name__ == "__main__":
    unittest.main()
