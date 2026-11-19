"""Shared fixtures. ``pytest.ini`` puts ``src/`` on the import path."""

from __future__ import annotations

import pytest

from simulation.config import load_config


def make_config(**sets):
    """Default config + dotted overrides given as keyword args (dots -> '__')."""
    items = [f"{k.replace('__', '.')}={v}" for k, v in sets.items()]
    return load_config(set_items=items)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def explicit_config():
    """Factory: config whose arrivals are an explicit list (fully controlled instance)."""
    def factory(arrivals, **sets):
        cfg_sets = {"arrivals__mode": "explicit", "simulation__arrival_horizon": 0.0}
        cfg_sets.update(sets)
        items = [f"{k.replace('__', '.')}={v}" for k, v in cfg_sets.items()]
        return load_config(overrides={"arrivals": {"explicit": arrivals}}, set_items=items)
    return factory
