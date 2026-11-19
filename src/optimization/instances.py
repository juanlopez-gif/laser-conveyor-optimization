"""Static (offline) instances of the problem and their conversion to a simulator config.

An :class:`Instance` is the data of the formal model: the list of parts with
arrival time r_j, conveyor k(j) and processing time p_j, plus optional config
overrides (speeds, geometry...). Part ids are assigned exactly as the simulator
does (``P001, P002, ...`` in order of (arrival time, conveyor)), so a sequence of
ids found by a solver can be replayed with the ``fixed_sequence`` policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from simulation.arrivals import generate_arrivals
from simulation.config import Config, load_config


@dataclass(frozen=True)
class Job:
    id: str
    time: float          # r_j  [s]
    conveyor: int        # k(j) (1-based)
    processing_time: float  # p_j [s]


@dataclass(frozen=True)
class Instance:
    name: str
    jobs: tuple[Job, ...]
    overrides: dict = field(default_factory=dict)   # dotted config overrides, e.g. {"laser.speed_x": 8}

    @property
    def n(self) -> int:
        return len(self.jobs)

    @property
    def ids(self) -> list[str]:
        return [j.id for j in self.jobs]

    def to_dict(self) -> dict:
        return {"name": self.name, "overrides": self.overrides,
                "jobs": [j.__dict__ for j in self.jobs]}

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        return cls(d["name"], tuple(Job(**j) for j in d["jobs"]), dict(d.get("overrides", {})))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def make_instance(jobs: list[tuple[float, int, float]], name: str = "custom",
                  overrides: dict | None = None) -> Instance:
    """Build an instance from ``(time, conveyor, processing_time)`` tuples."""
    ordered = sorted(jobs, key=lambda j: (j[0], j[1]))
    return Instance(name, tuple(Job(f"P{i + 1:03d}", float(t), int(k), float(p))
                                for i, (t, k, p) in enumerate(ordered)), dict(overrides or {}))


def generate_instance(n_parts: int, seed: int, mode: str = "stochastic",
                      overrides: dict | None = None, name: str | None = None) -> Instance:
    """First ``n_parts`` arrivals of the configured arrival process (default config)."""
    sets = [f"arrivals.mode={mode}", f"simulation.seed={seed}",
            "simulation.arrival_horizon=100000"]
    sets += [f"{k}={v}" for k, v in (overrides or {}).items()]
    cfg = load_config(set_items=sets)
    recs = generate_arrivals(cfg)[:n_parts]
    if len(recs) < n_parts:
        raise ValueError("arrival process produced fewer parts than requested")
    return make_instance([(r.time, r.conveyor_index + 1, r.processing_time) for r in recs],
                         name or f"{mode}_n{n_parts}_s{seed}", overrides)


def instance_config(inst: Instance, policy: str = "nearest_job", policy_params: dict | None = None,
                    dt: float | None = None, extra_sets: list[str] | None = None) -> Config:
    """Simulator config that reproduces ``inst`` exactly (explicit arrivals, drain)."""
    arrivals = [{"time": j.time, "conveyor": j.conveyor, "processing_time": j.processing_time}
                for j in inst.jobs]
    sets = ["arrivals.mode=explicit", "simulation.arrival_horizon=0", "simulation.drain=true",
            f"policy.name={policy}"]
    if dt is not None:
        sets.append(f"simulation.dt={dt}")
    sets += [f"{k}={json.dumps(v) if isinstance(v, (list, dict)) else v}"
             for k, v in inst.overrides.items()]
    sets += list(extra_sets or [])
    overrides = {"arrivals": {"explicit": arrivals}}
    if policy_params is not None:
        overrides["policy"] = {"params": {policy: policy_params}}
    return load_config(overrides=overrides, set_items=sets)
