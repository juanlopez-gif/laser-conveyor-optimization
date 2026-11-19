"""Event log: human-readable record of everything that happens in a run."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class EventKind(str, Enum):
    SIM_START = "SIM_START"
    SIM_END = "SIM_END"
    ARRIVAL = "ARRIVAL"                  # part generated (enters the upstream queue)
    ENTER = "ENTER"                      # part placed on the belt
    READY = "READY"                      # part entered the laser zone
    ASSIGN = "ASSIGN"                    # policy chose a part
    UNASSIGN = "UNASSIGN"                # assignment dropped (part became unreachable)
    LASER_ARRIVED = "LASER_ARRIVED"
    PROCESS_START = "PROCESS_START"
    PROCESS_END = "PROCESS_END"
    CONVEYOR_STOP = "CONVEYOR_STOP"
    CONVEYOR_START = "CONVEYOR_START"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Event:
    time: float
    kind: EventKind
    message: str
    part_id: str | None = None
    conveyor_id: int | None = None

    def line(self) -> str:
        return f"{self.time:8.2f} | {self.message}"


class EventLog:
    """Append-only list of :class:`Event` with export helpers."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def log(self, time: float, kind: EventKind, message: str,
            part_id: str | None = None, conveyor_id: int | None = None) -> None:
        self.events.append(Event(round(time, 6), kind, message, part_id, conveyor_id))

    def __len__(self) -> int:
        return len(self.events)

    def lines(self) -> list[str]:
        return [e.line() for e in self.events]

    def of_kind(self, kind: EventKind) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def save_text(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.lines()) + "\n", encoding="utf-8")

    def save_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["time", "kind", "message",
                                                    "part_id", "conveyor_id"])
            writer.writeheader()
            for e in self.events:
                row = asdict(e)
                row["kind"] = e.kind.value
                writer.writerow(row)
