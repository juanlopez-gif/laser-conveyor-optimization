"""Conveyor stopping rules while the laser processes a part.

The default rule (requirement of V1) stops ONLY the conveyor that carries the
part being processed. The rule is a small strategy object so other policies
(stop everything, stop neighbours, ...) can be studied without touching the
engine: add a class, register it in ``STOP_POLICIES`` and select it with
``simulation.conveyor_stop_policy`` in the config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .conveyor import Conveyor
from .part import Part


class ConveyorStopPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def conveyors_to_stop(self, part: Part, conveyors: Sequence[Conveyor]) -> list[Conveyor]:
        """Conveyors that must be stopped while ``part`` is processed."""


class ProcessingConveyorOnly(ConveyorStopPolicy):
    name = "processing_conveyor_only"

    def conveyors_to_stop(self, part, conveyors):
        return [c for c in conveyors if c.id == part.conveyor_id]


class AllConveyors(ConveyorStopPolicy):
    name = "all_conveyors"

    def conveyors_to_stop(self, part, conveyors):
        return list(conveyors)


STOP_POLICIES: dict[str, type[ConveyorStopPolicy]] = {
    ProcessingConveyorOnly.name: ProcessingConveyorOnly,
    AllConveyors.name: AllConveyors,
}


def get_stop_policy(name: str) -> ConveyorStopPolicy:
    try:
        return STOP_POLICIES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown conveyor stop policy '{name}'. "
                         f"Available: {sorted(STOP_POLICIES)}") from exc
