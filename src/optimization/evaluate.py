"""Exact evaluation of a solution on an instance by simulation.

The simulator is the reference implementation of the formal model P (non-idling
conveyors, blocking, interception). Every solver in ``optimization/solvers`` hands
its answer to these functions, so all methods are compared on the same ground:

* :func:`evaluate_sequence` — a processing order (decoder Phi of the formulation);
* :func:`evaluate_policy`   — an online policy run on the instance.

Both return an :class:`Evaluation` with feasibility, the KPI summary, the value of
the requested objective and the formulation check (``schedule_check``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from optimization.instances import Instance, instance_config
from optimization.objective_functions import OBJECTIVES, composite_objective
from optimization.schedule_check import check_schedule
from policies import get_policy
from policies.fixed_sequence import InfeasibleSequence
from simulation.engine import Simulation, SimulationResult

#: dt used inside search loops (fast) and for final validation (accurate)
SEARCH_DT = 0.05
VALIDATION_DT = 0.01


@dataclass
class Evaluation:
    feasible: bool
    objective: float
    summary: dict = field(default_factory=dict)
    sequence: list[str] = field(default_factory=list)     # realized processing order
    violations: list[str] = field(default_factory=list)
    reason: str = ""


def objective_value(summary: dict, objective: str | dict) -> float:
    """``objective`` is a single name (see OBJECTIVES) or a weights dict (composite)."""
    if isinstance(objective, dict):
        return composite_objective(summary, objective)
    return OBJECTIVES[objective](summary)


def _finish(result: SimulationResult, inst: Instance, objective, check: bool) -> Evaluation:
    s = result.summary
    order = [p.id for p in sorted((p for p in result.parts if p.start_processing_time is not None),
                                  key=lambda p: p.start_processing_time)]
    if s["parts_completed"] != inst.n:
        return Evaluation(False, math.inf, s, order, reason="not all parts processed (time limit)")
    violations = check_schedule(result) if check else []
    return Evaluation(not violations, objective_value(s, objective) if not violations else math.inf,
                      s, order, violations)


def evaluate_sequence(inst: Instance, sequence: list[str], objective: str | dict = "makespan",
                      dt: float = SEARCH_DT, check: bool = False,
                      max_time: float | None = None) -> Evaluation:
    """Simulate the processing order ``sequence`` (must be a permutation of ``inst.ids``)."""
    if sorted(sequence) != sorted(inst.ids):
        raise ValueError("sequence must be a permutation of the instance part ids")
    horizon = max_time or (max(j.time for j in inst.jobs) + 60.0 * inst.n + 300.0)
    cfg = instance_config(inst, "fixed_sequence", {"sequence": list(sequence)}, dt,
                          [f"simulation.max_time={horizon}"])
    try:
        result = Simulation(cfg, record_trace=False).run()
    except InfeasibleSequence as exc:
        return Evaluation(False, math.inf, sequence=list(sequence), reason=str(exc))
    return _finish(result, inst, objective, check)


def evaluate_policy(inst: Instance, policy: str, params: dict | None = None,
                    objective: str | dict = "makespan", dt: float = VALIDATION_DT,
                    check: bool = True) -> Evaluation:
    """Run an online policy on the instance."""
    cfg = instance_config(inst, policy, params, dt)
    result = Simulation(cfg, policy=get_policy(policy, params or cfg.policy_params),
                        record_trace=False).run()
    return _finish(result, inst, objective, check)
