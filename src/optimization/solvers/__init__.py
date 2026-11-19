"""Solver plug-ins for the benchmark.

Each module in this package registers one or more methods with
``@register_method("name")``. A method is a callable

    method(instance: Instance, objective: str | dict, time_limit: float, seed: int) -> Solution

Modules are discovered automatically, exactly like policies, so solver
developers never edit shared files. A module whose optional dependency is
missing (e.g. ortools) is skipped with a warning instead of breaking the
others.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from dataclasses import dataclass, field
from typing import Callable

METHODS: dict[str, Callable] = {}


@dataclass
class Solution:
    """What a method returns."""

    sequence: list[str] | None = None      # offline methods: processing order to replay
    policy: str | None = None              # online methods: policy name to run instead
    policy_params: dict = field(default_factory=dict)
    lower_bound: float | None = None       # proven bound on the objective, if any
    proven_optimal: bool = False           # the method certifies optimality (for its model)
    info: dict = field(default_factory=dict)


def register_method(name: str):
    def decorator(fn: Callable) -> Callable:
        if name in METHODS and METHODS[name] is not fn:
            raise ValueError(f"method '{name}' registered twice")
        METHODS[name] = fn
        return fn
    return decorator


def discover() -> dict[str, Callable]:
    for module in pkgutil.iter_modules(__path__):
        try:
            importlib.import_module(f"{__name__}.{module.name}")
        except ImportError as exc:          # optional dependency missing
            warnings.warn(f"solver module '{module.name}' skipped: {exc}")
    return METHODS
