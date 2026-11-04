"""Tie-aware cells for the original LSFlow decoder ``topk_positive(c @ W)``.

The released intervention instance has eight distinct, mutually orthogonal
columns in its 40-by-20 weight matrix.  Working in those eight group scores gives
an exact Euclidean projection in the original 40-dimensional coefficient space:
the score-coordinate metric has weights 1 / ||W_group||^2, and the kernel part
of the input coefficient is unchanged.  This is not a spherical projection.

Identical columns have unavoidable ties. Scores are computed once per unique
column and broadcast, making ties exact even when float32 GEMM would accumulate
duplicate columns differently. This changes only numerical tie choices among
identical columns, whose environment effects are identical. Their stable-index
action prefix is preserved, with no positive margin between identical columns.
Some measure-zero actions split two distinct tied groups simultaneously. Their
strict-margin region is empty; such rows explicitly fall back to their verified
original coefficient. Unsupported nonorthogonal weight matrices are rejected.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .geometry import decode


class NativeCellProjector:
    """Exact orthogonal-group cell projection with an explicit safe fallback.

    ``project(noise, actions, margin, fallback_costs)`` returns ``(costs, stats)``.
    Both cost tensors have final dimension N; actions have final dimension A.
    Arbitrary matching batch prefixes are allowed. Stats contain numeric scalar
    counts/rates suitable for JSONL logging. All returned rows are checked with
    the original stable positive-top-k decoder before return. Fallback rows
    preserve their action but carry no margin or nearest-point guarantee.
    """

    def __init__(self, weights: Tensor, budget: int):
        weights = torch.as_tensor(weights)
        if not weights.is_floating_point():
            weights = weights.float()
        if weights.ndim != 2 or min(weights.shape) < 1 or not torch.isfinite(weights).all():
            raise ValueError("weights must be a finite nonempty [cost_dim, action_dim] matrix")
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= weights.shape[1]:
            raise ValueError("budget must be an integer between zero and action_dim")
        self.weights = weights.detach().clone()
        self.budget = budget
        self.cost_dim, self.action_dim = weights.shape
        unique_rows, inverse = torch.unique(self.weights.T, dim=0, sorted=True, return_inverse=True)
        # Check in float64; orthogonality is required, not approximated away.
        gram = unique_rows.double() @ unique_rows.double().T
        squared_norms = gram.diag()
        if torch.any(squared_norms <= 0):
            raise ValueError("zero weight columns are unsupported")
        if torch.count_nonzero(gram - torch.diag(squared_norms)):
            raise ValueError("unique weight columns must be mutually orthogonal")
        self.unique_columns = unique_rows.T.contiguous()
        self.group_index = inverse
        self.n_groups = len(unique_rows)
        self.group_sizes = torch.bincount(inverse, minlength=self.n_groups)
        self.squared_norms = squared_norms
        self.basis = self.unique_columns / squared_norms.sqrt().to(weights.dtype)
        self.membership = torch.nn.functional.one_hot(inverse, num_classes=self.n_groups)
        self.group_positions = (self.membership.cumsum(dim=0) - 1)[
            torch.arange(self.action_dim, device=weights.device), inverse
        ]

    @torch.no_grad()
    def decode(self, costs: Tensor) -> Tensor:
        if not costs.is_floating_point() or costs.shape[-1] != self.cost_dim:
            raise ValueError("costs must be floating-point with final dimension cost_dim")
        if costs.device != self.weights.device:
            raise ValueError("costs and weights must share a device")
        group_scores = costs @ self.unique_columns.to(costs.dtype)
        return decode(group_scores[..., self.group_index], self.budget)

    @torch.no_grad()
    def project(self, noise: Tensor, actions: Tensor, margin: float, fallback_costs: Tensor):
        if not isinstance(noise, Tensor) or not noise.is_floating_point() or noise.ndim < 1:
            raise ValueError("noise must be a floating-point tensor")
        if noise.shape[-1] != self.cost_dim or noise.device != self.weights.device:
            raise ValueError("noise has the wrong cost dimension or device")
        if not isinstance(actions, Tensor) or actions.shape != (*noise.shape[:-1], self.action_dim):
            raise ValueError("actions must match the batch prefix and action dimension")
        if not isinstance(fallback_costs, Tensor) or fallback_costs.shape != noise.shape:
            raise ValueError("fallback_costs must have the same shape as noise")
        if actions.device != noise.device or fallback_costs.device != noise.device:
            raise ValueError("all inputs must share a device")
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError("margin must be finite and strictly positive")
        if not torch.isfinite(noise).all() or not torch.isfinite(fallback_costs).all():
            raise ValueError("noise and fallback_costs must be finite")
        if not torch.all((actions == 0) | (actions == 1)) or torch.any(actions.sum(-1) > self.budget):
            raise ValueError("actions must be binary and satisfy the budget")
        # Verify the actual fallback in the output precision, before any work.
        fallback = fallback_costs.to(noise.dtype)
        if not torch.equal(self.decode(fallback).bool(), actions.bool()):
            raise ValueError("fallback_costs must decode to the requested actions")

        shape = noise.shape
        work_dtype = torch.float64 if noise.dtype == torch.float64 else torch.float32
        values = noise.reshape(-1, self.cost_dim).to(work_dtype)
        target = actions.reshape(-1, self.action_dim).to(work_dtype)
        rows = len(values)
        if rows == 0:
            raise ValueError("empty batches are unsupported")
        if self.budget == 0:
            return noise.clone(), {"rows": rows, "projected_rows": rows, "fallback_rows": 0,
                                   "fallback_fraction": 0.0, "degenerate_tie_rows": 0,
                                   "numerical_fallback_rows": 0, "projection_distance_sq_mean": 0.0}

        columns = self.unique_columns.to(work_dtype)
        metric = self.squared_norms.to(work_dtype).reciprocal()
        scores = values @ columns
        group_counts = target @ self.membership.to(work_dtype)
        full = group_counts == self.group_sizes
        empty = group_counts == 0
        partial = ~full & ~empty
        partial_count = partial.sum(-1)
        cardinality = target.sum(-1)
        canonical = self.group_positions < group_counts[:, self.group_index]
        if not torch.equal(canonical, target.bool()):
            raise ValueError("target action violates stable-index ties inside an identical-column group")
        if torch.any((cardinality < self.budget) & (partial_count != 0)):
            raise ValueError("an under-budget decoder action cannot split an identical-column group")

        # Non-full-budget actions have independent sign constraints per group.
        projected_scores = torch.where(full, scores.clamp_min(margin), scores.clamp_max(-margin))
        degenerate = partial_count > 1
        coupled = (cardinality == self.budget) & ~degenerate
        if torch.any(coupled):
            q = scores[coupled]
            selected_groups = full[coupled]
            empty_groups = empty[coupled]
            partial_groups = partial[coupled]
            has_partial = partial_groups.any(-1, keepdim=True)
            # A partial group's score itself is the threshold t >= margin;
            # otherwise t is the separating threshold constrained to t >= 0.
            offset = margin * (1 + has_partial.to(work_dtype))
            lower_bound = margin * has_partial.to(work_dtype)
            partial_center = (q * partial_groups).sum(-1, keepdim=True)
            partial_metric = (metric * partial_groups).sum(-1, keepdim=True)

            def derivative(threshold):
                above = torch.where(selected_groups, (threshold + offset - q).clamp_min(0), 0)
                below = torch.where(empty_groups, (q - threshold + offset).clamp_min(0), 0)
                return ((above - below) * metric).sum(-1, keepdim=True) + \
                    partial_metric * (threshold - partial_center)

            lower = lower_bound.clone()
            upper = torch.maximum(q.amax(-1, keepdim=True) + offset, lower_bound)
            interior = derivative(lower_bound) < 0
            for _ in range(52 if work_dtype == torch.float64 else 32):
                middle = lower * 0.5 + upper * 0.5
                below = derivative(middle) < 0
                lower = torch.where(below, middle, lower)
                upper = torch.where(below, upper, middle)
            threshold = torch.where(interior, lower * 0.5 + upper * 0.5, lower_bound)
            coupled_scores = torch.where(selected_groups, torch.maximum(q, threshold + offset),
                                         torch.minimum(q, threshold - offset))
            coupled_scores = torch.where(partial_groups, threshold, coupled_scores)
            projected_scores[coupled] = coupled_scores

        # Orthogonal right inverse gives the minimum-norm correction. It never
        # discards or substitutes the noise's unconstrained kernel component.
        projected = (values + ((projected_scores - scores) * metric) @ columns.T).to(noise.dtype)
        finite = torch.isfinite(projected).all(-1)
        # Decode only finite candidate rows; every failed row uses the verified
        # original coefficient. The final check below guards against any bug in
        # that bookkeeping as well as precision-induced boundary changes.
        recovered = torch.zeros_like(target, dtype=torch.bool)
        if torch.any(finite):
            recovered[finite] = self.decode(projected[finite]).bool()
        wrong_action = ~torch.all(recovered == target.bool(), dim=-1)
        use_fallback = degenerate | ~finite | wrong_action
        projected[use_fallback] = fallback.reshape(-1, self.cost_dim)[use_fallback]
        result = projected.reshape(shape)
        if not torch.equal(self.decode(result).bool(), actions.bool()):
            raise RuntimeError("native projection failed its final decoder-preservation check")
        stats = {
            "rows": rows,
            "projected_rows": int((~use_fallback).sum()),
            "fallback_rows": int(use_fallback.sum()),
            "fallback_fraction": float(use_fallback.float().mean()),
            "degenerate_tie_rows": int(degenerate.sum()),
            "numerical_fallback_rows": int((use_fallback & ~degenerate).sum()),
            "projection_distance_sq_mean": float((result.to(work_dtype) - noise.to(work_dtype)).square().sum(-1).mean()),
        }
        return result, stats
