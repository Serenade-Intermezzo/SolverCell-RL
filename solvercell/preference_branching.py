"""Shared-prefix importance pools for the ACTUAL stochastic diffusion policy.

Every leaf marginal is the original conditional base policy. Leaves are
correlated within a root; self-normalized selection is a finite-pool
approximation, not an exact sample from the utility tilt. Consistency requires
the number of independent roots to increase, not merely more children of one
root. This module changes no trained actor, critic or Bellman target.
"""
import math
import torch

from .preference_policy import _condition, _particle_inputs, _index, _log_values, _probabilities, _draw_cost


@torch.no_grad()
def branch_weight_sample(actor, obs, prefs, terminal_logweight, *, roots, branch_index,
                         children=2, branch_counts=None, generator=None):
    """Generate M independent prefixes and R_j conditional suffixes per prefix.

    Static branch counts have one entry per root. The weight of a leaf is
    exp(U)/R_j, so unequal branch allocations do not reweight the root law.
    Counts must be chosen independently of the suffixes being averaged.
    No guide, early selection, drift approximation or solver gradient is used.
    """
    if actor.kind != 'diffusion':
        raise ValueError('Shared suffix branching requires the stochastic diffusion chain')
    obs, prefs = _condition(obs, prefs, actor.obs_dim)
    robs, rprefs, batch = _particle_inputs(actor, obs, prefs, roots)
    index = _index(branch_index, actor.steps)
    if branch_counts is None:
        if type(children) is not int or children < 1:
            raise ValueError('children must be a positive integer')
        counts = (children,) * roots
    else:
        counts = tuple(branch_counts)
        if len(counts) != roots or any(type(value) is not int or value < 1 for value in counts):
            raise ValueError('One positive integer branch count is required per root')
    leaves = sum(counts)
    prefix = actor._initial(robs, generator)
    for step in range(index):
        prefix = actor.transition(robs, rprefs, prefix, step, generator)
    uniform = len(set(counts)) == 1
    if uniform:
        suffix = prefix.repeat_interleave(counts[0], dim=0)
    else:
        repeats = torch.tensor(counts, device=obs.device, dtype=torch.long).repeat(batch)
        suffix = prefix.repeat_interleave(repeats, dim=0)
    lobs = obs.repeat_interleave(leaves, 0)
    lprefs = prefs.repeat_interleave(leaves, 0)
    raw = actor.suffix(lobs, lprefs, suffix, index, generator)
    log_weights = _log_values(terminal_logweight(raw), batch*leaves,
                             'terminal_logweight', True).reshape(batch, leaves)
    if not uniform:
        inverse_count = torch.repeat_interleave(
            torch.tensor(counts, device=obs.device, dtype=torch.float64).log(),
            torch.tensor(counts, device=obs.device, dtype=torch.long))
        log_weights = log_weights - inverse_count[None]
    probabilities, ess = _probabilities(log_weights)
    selected = _draw_cost(raw.reshape(batch, leaves, actor.act_dim), probabilities, generator)
    prefix_nfe = batch * roots * index
    suffix_nfe = batch * leaves * (actor.steps-index)
    return selected, {
        'nfe': float(prefix_nfe+suffix_nfe), 'actor_nfe': float(prefix_nfe+suffix_nfe),
        'prefix_nfe': float(prefix_nfe), 'suffix_nfe': float(suffix_nfe),
        'independent_roots': float(batch*roots), 'leaf_candidates': float(batch*leaves),
        'terminal_evaluations': float(batch*leaves), 'terminal_draws': float(batch),
        'guide_calls': 0., 'resampling_events': 0.,
        'ess_mean': float(ess.mean()), 'ess_min': float(ess.min()),
        'ess_final': float(ess.mean()), 'ess_final_fraction': float((ess/leaves).mean()),
        'log_normalizer_mean': float((torch.logsumexp(log_weights, -1)
                                      -math.log(leaves if uniform else roots)).mean()),
    }


class BranchingPreferencePolicy:
    """Evaluation adapter. Existing modes delegate to the unchanged learner."""
    def __init__(self, learner, topologies):
        self.learner = learner
        self.device = learner.device
        self.topologies = topologies

    @torch.no_grad()
    def act(self, obs, pref, mode='direct', generator=None):
        learner = self.learner
        if mode not in self.topologies:
            return learner.act(obs, pref, mode=mode, generator=generator)
        cfg = self.topologies[mode]
        generator = learner.behavior_generator if generator is None else generator
        obs = learner._tensor(obs)[None]
        prefs = learner._preferences(pref, 1)
        before = learner.counters.copy()
        baseline = (learner._twin_vector(obs, obs.new_zeros((1, learner.action_dim)), prefs)*prefs).sum(-1)
        def terminal(raw):
            count = len(raw)
            utility = (learner._twin_vector(obs.expand(count, -1), learner._decode(raw),
                        prefs.expand(count, -1))*prefs).sum(-1)
            return (utility-baseline)/learner.cfg.guide_temperature
        costs, diagnostics = branch_weight_sample(learner.target_actor, obs, prefs, terminal,
            roots=cfg['roots'], children=cfg['children'], branch_index=cfg['branch_index'],
            generator=generator)
        learner.counters['nfe'] += int(diagnostics['nfe'])
        learner.counters['actor_sampling_nfe'] += int(diagnostics['nfe'])
        actions = learner._decode(costs)
        stats = {**diagnostics, **{key: learner.counters[key]-before[key] for key in learner.COUNTER_KEYS}}
        stats['mode'] = mode
        return actions[0].cpu().numpy().copy(), costs[0].cpu().numpy().copy(), stats
