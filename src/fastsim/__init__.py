"""Event-driven fast simulator (strong-RL plan, phase 1). See :mod:`fastsim.core`."""

from .core import FastSim
from .policies import POLICIES, nearest_head, nearest_job

__all__ = ["FastSim", "POLICIES", "nearest_job", "nearest_head"]
