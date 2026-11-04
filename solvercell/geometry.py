"""Solver cells for binary, positive-score selection with an upper budget.

The decoder maximizes ``costs @ action`` over binary actions of cardinality at
most ``budget``.  Zero scores are excluded and score ties are resolved by the
original coordinate order.  Projection uses the *closed* robust cell

    (action - other) @ costs >= margin * ||action - other||_1

for every other feasible action.  With a strictly positive margin, the target
action is the unique optimum.  All geometry here is Euclidean, not spherical.
"""

from __future__ import annotations

import numbers

import torch
from torch import Tensor


def _validate_costs(costs: Tensor, budget: int) -> None:
    if not isinstance(costs, Tensor) or not costs.is_floating_point():
        raise TypeError("costs must be a floating-point torch.Tensor")
    if costs.ndim < 1 or costs.shape[-1] == 0:
        raise ValueError("costs must have a nonempty final action dimension")
    if isinstance(budget, bool) or not isinstance(budget, numbers.Integral):
        raise TypeError("budget must be an integer")
    if not 0 <= budget <= costs.shape[-1]:
        raise ValueError("budget must be between zero and the action dimension")
    if not torch.isfinite(costs).all():
        raise ValueError("costs must be finite")


def decode(costs: Tensor, budget: int) -> Tensor:
    """Select up to ``budget`` positive scores along the final dimension.

    Returns a binary floating tensor with the same shape, dtype, and device as
    ``costs``.  At equal scores, the smaller coordinate index is selected first.
    In particular, this is not an exact-k decoder: nonpositive scores are never
    selected, even when the budget is not exhausted.
    """
    _validate_costs(costs, budget)
    result = torch.zeros_like(costs)
    if budget == 0:
        return result
    indices = torch.argsort(costs, dim=-1, descending=True, stable=True)[..., :budget]
    positive = costs.gather(-1, indices) > 0
    return result.scatter(-1, indices, positive.to(costs.dtype))


def project_cell(noise: Tensor, actions: Tensor, budget: int, margin: float) -> Tensor:
    """Euclidean nearest-point projection onto robust up-to-k decoder cells.

    ``noise`` and the binary target ``actions`` have identical shapes.  For a
    target with fewer than k selected coordinates, the constraints are simply
    selected scores >= margin and unselected scores <= -margin.  At cardinality
    k, they are selected scores >= margin and every selected/unselected gap >=
    2 * margin.  The latter constraints admit a common threshold theta >= 0:

        selected >= theta + margin; unselected <= theta - margin.

    For fixed theta the closest point is coordinatewise clipping.  The resulting
    squared distance is convex in theta, and its derivative is monotone.  A
    batched scalar bisection solves it to floating-point precision.  This gives
    the nearest point in the entire cell, not a projection onto one chosen ray.

    Float16/bfloat16 inputs are computed in float32 and cast back.  Rounding can
    reduce the requested numerical margin; use float32 or float64 when checking
    tight margins.  At margin zero the cell includes decoder tie boundaries, so
    decoding the projection need not recover the requested action.  When k=0
    there is only one possible action, and its cell is the whole space.
    """
    _validate_costs(noise, budget)
    if not isinstance(actions, Tensor) or actions.shape != noise.shape:
        raise ValueError("actions must be a tensor with the same shape as noise")
    if actions.device != noise.device:
        raise ValueError("actions and noise must be on the same device")
    if not torch.all((actions == 0) | (actions == 1)):
        raise ValueError("actions must be binary")
    if not isinstance(margin, numbers.Real) or not 0 <= margin < float("inf"):
        raise ValueError("margin must be a finite nonnegative real number")
    selected = actions.bool()
    counts = selected.sum(dim=-1)
    if torch.any(counts > budget):
        raise ValueError("target action exceeds the budget")
    if budget == 0:
        return noise.clone()

    work_dtype = torch.float64 if noise.dtype == torch.float64 else torch.float32
    work = noise.to(work_dtype)
    result = torch.where(selected, work.clamp_min(margin), work.clamp_max(-margin))

    # Only full-budget actions require a coupled projection.  The rest already
    # have the independent half-space projection computed above.
    full = counts == budget
    if not torch.any(full):
        return result.to(noise.dtype)
    flat_work = work.reshape(-1, work.shape[-1])
    flat_selected = selected.reshape_as(flat_work)
    flat_full = full.reshape(-1)
    values = flat_work[flat_full]
    mask = flat_selected[flat_full]

    def half_derivative(theta: Tensor) -> Tensor:
        positive = torch.where(mask, (theta + margin - values).clamp_min(0), 0)
        negative = torch.where(~mask, (values - theta + margin).clamp_min(0), 0)
        return (positive - negative).sum(dim=-1, keepdim=True)

    lower = torch.zeros_like(values[:, :1])
    upper = (values.amax(dim=-1, keepdim=True) + margin).clamp_min(0)
    interior = half_derivative(lower) < 0
    iterations = 48 if work_dtype == torch.float64 else 32
    for _ in range(iterations):
        middle = lower * 0.5 + upper * 0.5
        below_root = half_derivative(middle) < 0
        lower = torch.where(below_root, middle, lower)
        upper = torch.where(below_root, upper, middle)
    theta = torch.where(interior, lower * 0.5 + upper * 0.5, 0)
    projected = torch.where(
        mask,
        torch.maximum(values, theta + margin),
        torch.minimum(values, theta - margin),
    )
    flat_result = result.reshape_as(flat_work).clone()
    flat_result[flat_full] = projected
    return flat_result.reshape_as(noise).to(noise.dtype)
