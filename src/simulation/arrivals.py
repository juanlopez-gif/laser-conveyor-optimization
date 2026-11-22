"""Part generation: arrival times and processing times.

Reproducibility: every conveyor owns two independent random streams (arrivals
and processing times) seeded from the master seed with a string key, e.g.
``"42/arrivals/1"``. Python's ``random.Random`` hashes string seeds
deterministically, so the same seed always produces the same experiment, and
two policies run with the same seed see exactly the same parts (common random
numbers — useful for fair comparisons).

Stochastic arrivals (``arrivals.mode: stochastic``) are a renewal process per conveyor:

* ``distribution: exponential`` — Poisson arrivals with mean inter-arrival time
  ``mean_interarrival`` (every version up to 0.7.9).
* ``distribution: burst`` (0.7.10) — an **on/off Poisson source** (interrupted Poisson
  process) with the **same mean rate**: the source alternates exponentially distributed
  on-periods of mean ``burst.on_mean`` seconds, during which parts arrive as a Poisson
  process of rate ``1 / (mean_interarrival * on_fraction)``, and off-periods of mean
  ``on_mean * (1 - on_fraction) / on_fraction`` seconds without arrivals. The source is
  on a fraction ``on_fraction`` of the time, so the long-run rate is
  ``1 / mean_interarrival`` exactly as with ``exponential``; ``on_fraction: 1`` is the
  Poisson process. The inter-arrival times of this process are hyper-exponential
  (two-phase), i.e. the "batch Poisson or hyper-exponential" model asked for in
  ``docs/plans/papers.md`` with one burstiness parameter (``on_fraction``) and one
  time scale (``on_mean``). The on/off switching uses its own stream
  (``"<seed>/bursts/<k>"``), so ``exponential`` instances are unchanged by this
  addition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class ArrivalRecord:
    time: float
    conveyor_index: int        # 0-based
    processing_time: float


class ProcessingTimeSampler:
    def __init__(self, config: Config, rng: random.Random) -> None:
        self.spec = config.processing
        self.rng = rng

    def sample(self) -> float:
        s = self.spec
        if s.distribution == "constant":
            return s.time
        if s.distribution == "uniform":
            return self.rng.uniform(s.uniform_low, s.uniform_high)
        if s.distribution == "normal":
            return max(s.normal_min, self.rng.gauss(s.normal_mean, s.normal_std))
        raise ValueError(f"Unknown processing distribution '{s.distribution}'")


def generate_arrivals(config: Config) -> list[ArrivalRecord]:
    """Pre-compute every arrival up to ``arrival_horizon`` (sorted by time).

    Pre-computing (instead of sampling on the fly) makes the instance explicit:
    the very same list can later be handed to an offline optimiser.
    """
    a, horizon, seed = config.arrivals, config.simulation.arrival_horizon, config.simulation.seed
    records: list[ArrivalRecord] = []

    if a.mode == "explicit":
        for item in a.explicit:
            pt = item.get("processing_time", config.processing.time)
            records.append(ArrivalRecord(float(item["time"]), int(item["conveyor"]) - 1, float(pt)))
        return sorted(records, key=lambda r: (r.time, r.conveyor_index))

    for i in range(config.n_conveyors):
        arr_rng = random.Random(f"{seed}/arrivals/{i + 1}")
        sampler = ProcessingTimeSampler(config, random.Random(f"{seed}/processing/{i + 1}"))
        if a.mode == "deterministic":
            if a.intervals[i] <= 0:
                continue
            t = a.first_arrival[i]
            while t <= horizon + 1e-9:
                records.append(ArrivalRecord(t, i, sampler.sample()))
                t += a.intervals[i]
        elif a.mode == "stochastic":
            if a.distribution == "exponential" or a.burst_on_fraction[i] >= 1.0:
                t = arr_rng.expovariate(1.0 / a.mean_interarrival[i])
                while t <= horizon:
                    records.append(ArrivalRecord(t, i, sampler.sample()))
                    t += arr_rng.expovariate(1.0 / a.mean_interarrival[i])
            elif a.distribution == "burst":
                for t in _burst_times(a.mean_interarrival[i], a.burst_on_fraction[i],
                                      a.burst_on_mean[i], horizon, arr_rng,
                                      random.Random(f"{seed}/bursts/{i + 1}")):
                    records.append(ArrivalRecord(t, i, sampler.sample()))
            else:
                raise ValueError(f"Unknown inter-arrival distribution '{a.distribution}'")
    return sorted(records, key=lambda r: (r.time, r.conveyor_index))


def _burst_times(mean_interarrival: float, on_fraction: float, on_mean: float,
                 horizon: float, arr_rng: random.Random, sw_rng: random.Random) -> list[float]:
    """Arrival instants of one on/off Poisson source in ``(0, horizon]`` (see the
    module docstring). ``arr_rng`` draws the inter-arrival times inside the on-periods,
    ``sw_rng`` the period lengths and the initial state (on with probability
    ``on_fraction``: the stationary distribution of the two-state chain)."""
    rate_on = 1.0 / (mean_interarrival * on_fraction)
    off_mean = on_mean * (1.0 - on_fraction) / on_fraction
    times: list[float] = []
    t, on = 0.0, sw_rng.random() < on_fraction
    while t <= horizon:
        if on:
            end = t + sw_rng.expovariate(1.0 / on_mean)
            ta = t + arr_rng.expovariate(rate_on)
            while ta < end and ta <= horizon:
                times.append(ta)
                ta += arr_rng.expovariate(rate_on)
            t = end
        else:
            t += sw_rng.expovariate(1.0 / off_mean)
        on = not on
    return times
