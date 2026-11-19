"""Shared optimization foundation: instances, fixed-sequence decoder, evaluator."""

from __future__ import annotations

from optimization.evaluate import evaluate_policy, evaluate_sequence
from optimization.instances import Instance, generate_instance, make_instance


def test_instance_roundtrip(tmp_path):
    inst = generate_instance(8, seed=3)
    inst.save(tmp_path / "i.json")
    assert Instance.load(tmp_path / "i.json") == inst
    assert inst.ids == [f"P{i:03d}" for i in range(1, 9)]


def test_fixed_sequence_replays_the_given_order():
    inst = make_instance([(0.0, 1, 2.5), (0.0, 2, 2.5), (0.0, 3, 2.5)])
    for order in (["P003", "P001", "P002"], ["P002", "P003", "P001"]):
        ev = evaluate_sequence(inst, order, dt=0.01, check=True)
        assert ev.feasible and ev.sequence == order and ev.violations == []


def test_policy_and_sequence_agree():
    inst = generate_instance(10, seed=1)
    pol = evaluate_policy(inst, "nearest_job")
    seq = evaluate_sequence(inst, pol.sequence, dt=0.01, check=True)
    assert pol.feasible and seq.feasible
    assert abs(seq.objective - pol.objective) < 0.05


def test_deadlock_sequence_is_infeasible():
    # Two parts on the same belt 40 cm apart (> 35 cm laser zone): processing the
    # upstream one first is impossible because the downstream one blocks the belt.
    inst = make_instance([(0.0, 1, 2.5), (40.0 / 3.0, 1, 2.5)])
    ev = evaluate_sequence(inst, ["P002", "P001"], dt=0.01)
    assert not ev.feasible and "blocked" in ev.reason
    assert evaluate_sequence(inst, ["P001", "P002"], dt=0.01).feasible
