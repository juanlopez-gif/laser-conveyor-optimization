"""Baseline methods: the online dispatching policies and exhaustive search (tiny n)."""

from __future__ import annotations

import itertools
import math
import time

from optimization.evaluate import evaluate_sequence
from optimization.solvers import Solution, register_method


def _policy_method(name: str):
    def method(instance, objective, time_limit, seed):
        return Solution(policy=name)
    method.__name__ = f"policy_{name}"
    return method


for _name in ("fifo", "nearest_job", "nearest_head", "round_robin", "weighted_score"):
    register_method(f"policy:{_name}")(_policy_method(_name))


@register_method("exhaustive")
def exhaustive(instance, objective, time_limit, seed) -> Solution:
    """Enumerate every permutation (only for n <= 8). Optimal over the decoder Phi."""
    if instance.n > 8:
        return Solution(info={"skipped": "n > 8"})
    start, best, best_seq, count = time.time(), math.inf, None, 0
    for perm in itertools.permutations(instance.ids):
        if time.time() - start > time_limit:
            return Solution(sequence=best_seq, info={"evaluated": count, "timeout": True})
        ev = evaluate_sequence(instance, list(perm), objective)
        count += 1
        if ev.objective < best:
            best, best_seq = ev.objective, list(perm)
    return Solution(sequence=best_seq, info={"evaluated": count, "value_search_dt": best},
                    proven_optimal=False)   # optimal for Phi at search dt, not for P
