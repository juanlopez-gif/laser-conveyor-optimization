"""Scheduling policies with a self-registering plug-in mechanism.

Every module in this package is imported automatically, so dropping a new
file with a ``@register_policy("name")`` class is enough to make it
available from the configuration and the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .base_policy import BasePolicy, DecisionContext

POLICY_REGISTRY: dict[str, type[BasePolicy]] = {}


def register_policy(name: str):
    """Class decorator that adds a policy to :data:`POLICY_REGISTRY`."""
    def decorator(cls: type[BasePolicy]) -> type[BasePolicy]:
        if name in POLICY_REGISTRY and POLICY_REGISTRY[name] is not cls:
            raise ValueError(f"Policy '{name}' registered twice")
        cls.name = name
        POLICY_REGISTRY[name] = cls
        return cls
    return decorator


def _discover() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name not in ("base_policy",):
            importlib.import_module(f"{__name__}.{module.name}")


def available_policies() -> list[str]:
    return sorted(POLICY_REGISTRY)


def get_policy(name: str, params: dict[str, Any] | None = None) -> BasePolicy:
    """Instantiate a registered policy by name."""
    try:
        cls = POLICY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy '{name}'. Available: {available_policies()}") from exc
    return cls(**(params or {}))


_discover()

__all__ = ["BasePolicy", "DecisionContext", "register_policy", "get_policy",
           "available_policies", "POLICY_REGISTRY"]
