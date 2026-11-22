"""Configuration loading.

``config/default_config.yaml`` is the single source of truth for every
parameter. A user configuration file only lists the keys it wants to change;
it is deep-merged on top of the defaults. Dotted overrides
(``"laser.speed_x=8"``) are applied last, which is what the CLI ``--set``
option uses.

The raw dictionary is converted into small typed dataclasses so the rest of
the code never touches untyped dicts or hard-coded numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"


# --------------------------------------------------------------------------- #
# Typed views of the configuration                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    """Longitudinal layout of every conveyor (all conveyors share it)."""

    entry_length: float
    active_length: float
    exit_length: float

    @property
    def total_length(self) -> float:
        return self.entry_length + self.active_length + self.exit_length

    @property
    def active_start(self) -> float:
        """X where the laser zone begins (parts become READY here)."""
        return self.entry_length

    @property
    def active_end(self) -> float:
        """X where the laser zone ends."""
        return self.entry_length + self.active_length


@dataclass(frozen=True)
class Workspace:
    """Rectangle the laser head can physically reach."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float, tol: float = 1e-9) -> bool:
        return (self.x_min - tol <= x <= self.x_max + tol
                and self.y_min - tol <= y <= self.y_max + tol)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (min(max(x, self.x_min), self.x_max),
                min(max(y, self.y_min), self.y_max))


@dataclass(frozen=True)
class ConveyorSpec:
    id: int          # 1-based, as shown to humans
    y: float
    speed: float


@dataclass(frozen=True)
class LaserSpec:
    speed_x: float
    speed_y: float
    motion_model: str
    workspace: Workspace
    home_x: float
    home_y: float
    #: none | when_idle | always — dispatch parts still in the entry zone (model P)
    #: instead of READY parts only (P^sim). See default_config.yaml.
    prepositioning: str = "none"


@dataclass(frozen=True)
class ArrivalSpec:
    mode: str
    intervals: tuple[float, ...]          # deterministic
    first_arrival: tuple[float, ...]      # deterministic
    distribution: str                     # stochastic: exponential | burst
    mean_interarrival: tuple[float, ...]  # stochastic
    explicit: tuple[dict, ...]            # explicit list of arrivals
    #: bursty arrivals (0.7.10, ``distribution: burst``): an on/off Poisson source with
    #: the same mean rate; fraction of time the source is on and mean on-period [s]
    burst_on_fraction: tuple[float, ...] = ()
    burst_on_mean: tuple[float, ...] = ()


@dataclass(frozen=True)
class ProcessingSpec:
    distribution: str
    time: float
    uniform_low: float
    uniform_high: float
    normal_mean: float
    normal_std: float
    normal_min: float


@dataclass(frozen=True)
class SimulationSpec:
    seed: int
    dt: float
    arrival_horizon: float
    drain: bool
    max_time: float
    trace_interval: float
    min_part_spacing: float
    unprocessed_at_zone_end: str
    conveyor_stop_policy: str
    due_date_allowance: float
    #: information model (0.7.3): zone | belts | full, resolved from 'auto'
    #: (simulation/observation.py)
    observation: str = "belts"
    #: length of the entry-zone stretch, upstream of the laser zone, that the camera
    #: sees at level ``belts`` [cm] (0.7.10); null in the YAML = the whole entry zone
    visible_length: float = 0.0


@dataclass(frozen=True)
class Config:
    """Complete, validated configuration of one simulation run."""

    simulation: SimulationSpec
    geometry: Geometry
    conveyors: tuple[ConveyorSpec, ...]
    laser: LaserSpec
    arrivals: ArrivalSpec
    processing: ProcessingSpec
    policy_name: str
    policy_params: dict[str, Any]
    objective_weights: dict[str, float]
    output: dict[str, Any]
    raw: dict[str, Any] = field(repr=False, compare=False)
    #: blocking penalty Pi = per_event * N_B + per_second * T_B  (formal_formulation §1.4)
    blocking_penalty_per_event: float = 0.0
    blocking_penalty_per_second: float = 0.0

    @property
    def n_conveyors(self) -> int:
        return len(self.conveyors)


# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` recursively updated with ``override``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_dotted(raw: dict, assignments: list[str] | None) -> dict:
    """Apply ``["a.b.c=value", ...]`` overrides. Values are parsed as YAML."""
    out = copy.deepcopy(raw)
    for item in assignments or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def _per_conveyor(value: Any, n: int, name: str) -> tuple[float, ...]:
    """Broadcast a scalar to N conveyors or validate a list of length N."""
    if isinstance(value, (int, float)):
        return tuple(float(value) for _ in range(n))
    values = tuple(float(v) for v in value)
    if len(values) != n:
        raise ValueError(
            f"'{name}' has {len(values)} values but conveyors.count = {n}. "
            "Give one value per conveyor or a single scalar.")
    return values


PREPOSITIONING_MODES = ("none", "when_idle", "always")


def _prepositioning_mode(value: Any) -> str:
    """YAML booleans are accepted: false -> none, true -> always."""
    if value is None or value is False:
        return "none"
    if value is True:
        return "always"
    mode = str(value)
    if mode not in PREPOSITIONING_MODES:
        raise ValueError(f"laser.prepositioning must be one of {PREPOSITIONING_MODES}")
    return mode


def load_raw(path: str | Path | None = None,
             overrides: dict | None = None,
             set_items: list[str] | None = None) -> dict:
    """Load defaults, merge an optional user file, a dict and dotted overrides."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, encoding="utf-8") as fh:
            raw = deep_merge(raw, yaml.safe_load(fh) or {})
    if overrides:
        raw = deep_merge(raw, overrides)
    return apply_dotted(raw, set_items)


def load_config(path: str | Path | None = None,
                overrides: dict | None = None,
                set_items: list[str] | None = None) -> Config:
    """Public entry point: build a validated :class:`Config`."""
    return build_config(load_raw(path, overrides, set_items))


def build_config(raw: dict) -> Config:
    """Convert a raw (already merged) dictionary into a :class:`Config`."""
    s, g, c, l, a, p = (raw["simulation"], raw["geometry"], raw["conveyors"],
                        raw["laser"], raw["arrivals"], raw["processing"])

    sim = SimulationSpec(
        seed=int(s["seed"]), dt=float(s["dt"]),
        arrival_horizon=float(s["arrival_horizon"]), drain=bool(s["drain"]),
        max_time=float(s["max_time"]), trace_interval=float(s["trace_interval"]),
        min_part_spacing=float(s["min_part_spacing"]),
        unprocessed_at_zone_end=str(s["unprocessed_at_zone_end"]),
        conveyor_stop_policy=str(s["conveyor_stop_policy"]),
        due_date_allowance=float(s["due_date_allowance"]))
    if sim.dt <= 0:
        raise ValueError("simulation.dt must be > 0")
    if sim.unprocessed_at_zone_end not in ("block", "pass"):
        raise ValueError("simulation.unprocessed_at_zone_end must be 'block' or 'pass'")

    geom = Geometry(float(g["entry_length"]), float(g["active_length"]),
                    float(g["exit_length"]))

    n = int(c["count"])
    if n < 1:
        raise ValueError("conveyors.count must be >= 1")
    ys = c.get("y_positions")
    ys = (tuple(float(c["spacing"]) * i for i in range(n)) if ys is None
          else _per_conveyor(ys, n, "conveyors.y_positions"))
    speeds = _per_conveyor(c["speed"], n, "conveyors.speed")
    conveyors = tuple(ConveyorSpec(i + 1, ys[i], speeds[i]) for i in range(n))

    ws_raw = l.get("workspace") or {}
    ws = Workspace(
        x_min=geom.active_start if ws_raw.get("x_min") is None else float(ws_raw["x_min"]),
        x_max=geom.active_end if ws_raw.get("x_max") is None else float(ws_raw["x_max"]),
        y_min=min(ys) if ws_raw.get("y_min") is None else float(ws_raw["y_min"]),
        y_max=max(ys) if ws_raw.get("y_max") is None else float(ws_raw["y_max"]))
    for spec in conveyors:
        if not (ws.y_min - 1e-9 <= spec.y <= ws.y_max + 1e-9):
            raise ValueError(f"Conveyor {spec.id} (y={spec.y}) is outside the laser workspace")
    home = l.get("home") or {}
    laser = LaserSpec(
        speed_x=float(l["speed_x"]), speed_y=float(l["speed_y"]),
        motion_model=str(l["motion_model"]), workspace=ws,
        home_x=(ws.x_min + ws.x_max) / 2 if home.get("x") is None else float(home["x"]),
        home_y=(ws.y_min + ws.y_max) / 2 if home.get("y") is None else float(home["y"]),
        prepositioning=_prepositioning_mode(l.get("prepositioning", "none")))
    if laser.motion_model not in ("simultaneous", "sequential"):
        raise ValueError("laser.motion_model must be 'simultaneous' or 'sequential'")
    from dataclasses import replace as _replace
    from .observation import resolve_observation, resolve_visible_length
    sim = _replace(sim, observation=resolve_observation(s.get("observation", "auto"),
                                                        laser.prepositioning),
                   visible_length=resolve_visible_length(s.get("visible_length"),
                                                         geom.entry_length))

    det, sto = a.get("deterministic", {}), a.get("stochastic", {})
    burst = sto.get("burst") or {}
    mode = str(a["mode"])
    # Only the lists of the ACTIVE arrival mode must match conveyors.count.
    is_det, is_sto = mode == "deterministic", mode == "stochastic"
    arrivals = ArrivalSpec(
        mode=mode,
        intervals=_per_conveyor(det.get("intervals", 0.0), n,
                                "arrivals.deterministic.intervals") if is_det else (),
        first_arrival=_per_conveyor(det.get("first_arrival", 0.0), n,
                                    "arrivals.deterministic.first_arrival") if is_det else (),
        distribution=str(sto.get("distribution", "exponential")),
        mean_interarrival=_per_conveyor(sto.get("mean_interarrival", 1.0), n,
                                        "arrivals.stochastic.mean_interarrival") if is_sto else (),
        explicit=tuple(a.get("explicit") or ()),
        burst_on_fraction=_per_conveyor(burst.get("on_fraction", 1.0), n,
                                        "arrivals.stochastic.burst.on_fraction") if is_sto else (),
        burst_on_mean=_per_conveyor(burst.get("on_mean", 20.0), n,
                                    "arrivals.stochastic.burst.on_mean") if is_sto else ())
    if arrivals.mode not in ("deterministic", "stochastic", "explicit"):
        raise ValueError("arrivals.mode must be deterministic | stochastic | explicit")
    if is_sto:
        if arrivals.distribution not in ("exponential", "burst"):
            raise ValueError("arrivals.stochastic.distribution must be exponential | burst")
        if any(not 0.0 < f <= 1.0 for f in arrivals.burst_on_fraction):
            raise ValueError("arrivals.stochastic.burst.on_fraction must be in (0, 1]")
        if any(m <= 0.0 for m in arrivals.burst_on_mean):
            raise ValueError("arrivals.stochastic.burst.on_mean must be > 0")

    proc = ProcessingSpec(
        distribution=str(p["distribution"]), time=float(p["time"]),
        uniform_low=float(p["uniform"]["low"]), uniform_high=float(p["uniform"]["high"]),
        normal_mean=float(p["normal"]["mean"]), normal_std=float(p["normal"]["std"]),
        normal_min=float(p["normal"]["min"]))

    pol = raw["policy"]
    bp = raw["objective"].get("blocking_penalty") or {}
    per_event, per_second = float(bp.get("per_event", 0.0)), float(bp.get("per_second", 0.0))
    if per_event < 0 or per_second < 0:
        raise ValueError("objective.blocking_penalty values must be >= 0")
    return Config(
        simulation=sim, geometry=geom, conveyors=conveyors, laser=laser,
        arrivals=arrivals, processing=proc, policy_name=str(pol["name"]),
        policy_params=dict((pol.get("params") or {}).get(str(pol["name"]), {}) or {}),
        objective_weights={k: float(v) for k, v in raw["objective"]["weights"].items()},
        output=dict(raw.get("output", {})), raw=raw,
        blocking_penalty_per_event=per_event, blocking_penalty_per_second=per_second)
