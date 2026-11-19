"""Part generation: arrival times and processing times.

Reproducibility: every conveyor owns two independent random streams (arrivals
and processing times) seeded from the master seed with a string key, e.g.
``"42/arrivals/1"``. Python's ``random.Random`` hashes string seeds
deterministically, so the same seed always produces the same experiment, and
two policies run with the same seed see exactly the same parts (common random
numbers — useful for fair comparisons).
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
            if a.distribution != "exponential":
                raise ValueError(f"Unknown inter-arrival distribution '{a.distribution}'")
            t = arr_rng.expovariate(1.0 / a.mean_interarrival[i])
            while t <= horizon:
                records.append(ArrivalRecord(t, i, sampler.sample()))
                t += arr_rng.expovariate(1.0 / a.mean_interarrival[i])
    return sorted(records, key=lambda r: (r.time, r.conveyor_index))
