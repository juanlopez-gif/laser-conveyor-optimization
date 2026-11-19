"""Belt order is not dominant (docs/methods/restriction_gap.md, version 0.5.2)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from restriction_gap import LOADS, respects_belt_order  # noqa: E402

from optimization.evaluate import evaluate_sequence  # noqa: E402
from optimization.instances import generate_instance, make_instance  # noqa: E402


def test_respects_belt_order():
    inst = make_instance([(0.0, 1, 2.0), (1.0, 1, 2.0), (2.0, 2, 2.0)])
    assert respects_belt_order(("P001", "P003", "P002"), inst)
    assert respects_belt_order(("P003", "P001", "P002"), inst)
    assert not respects_belt_order(("P002", "P001", "P003"), inst)


def test_counterexample_belt_order_not_optimal():
    """Burst instance (n = 5, seed 1): the best order serves a later part of a conveyor
    first and beats EVERY belt-order permutation by more than 1 %."""
    inst = generate_instance(5, 1, "stochastic", overrides=LOADS["burst"])
    best_belt = min(evaluate_sequence(inst, list(p), "avg_flow_time").objective
                    for p in itertools.permutations(inst.ids) if respects_belt_order(p, inst))
    order = ["P001", "P003", "P005", "P004", "P002"]
    assert not respects_belt_order(tuple(order), inst)
    ev = evaluate_sequence(inst, order, "avg_flow_time", check=True)
    assert ev.feasible and not ev.violations
    assert ev.objective < best_belt * 0.99
