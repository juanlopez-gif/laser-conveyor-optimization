"""Parts (jobs) that travel on the conveyors and must be processed by the laser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PartStatus(str, Enum):
    """Life cycle of a part.

    WAITING     arrived, queued upstream of the belt entry (belt stopped / no room)
    MOVING      on the belt, inside the entry zone (laser cannot reach it yet)
    READY       inside the laser zone, not yet processed (may be moving or stopped)
    PROCESSING  the laser is working on it (its conveyor is stopped)
    COMPLETED   processed, still on the belt travelling to the exit
    EXITED      left the conveyor (processed or, in ``pass`` mode, unprocessed)
    """

    WAITING = "WAITING"
    MOVING = "MOVING"
    READY = "READY"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    EXITED = "EXITED"


#: One-letter codes used in the compact visualization trace.
STATUS_CODE = {
    PartStatus.WAITING: "W", PartStatus.MOVING: "M", PartStatus.READY: "R",
    PartStatus.PROCESSING: "P", PartStatus.COMPLETED: "C", PartStatus.EXITED: "E",
}


@dataclass
class Part:
    """A single job. Times are absolute simulation times in seconds."""

    id: str
    conveyor_id: int
    arrival_time: float
    processing_time: float
    due_date: float
    x: float = 0.0
    y: float = 0.0
    status: PartStatus = PartStatus.WAITING
    assigned: bool = False                 # the laser is currently heading to it
    entry_time: float | None = None        # entered the belt (x = 0)
    ready_time: float | None = None        # entered the laser zone
    start_processing_time: float | None = None
    completion_time: float | None = None
    exit_time: float | None = None

    @property
    def number(self) -> int:
        """Numeric part of the id (``P007`` -> 7)."""
        return int(self.id[1:])

    @property
    def is_processed(self) -> bool:
        return self.completion_time is not None

    @property
    def needs_processing(self) -> bool:
        """True while the part still requires (and has not started) laser work."""
        return self.start_processing_time is None

    @property
    def waiting_time(self) -> float | None:
        """Time from arrival until the laser starts working on it."""
        if self.start_processing_time is None:
            return None
        return self.start_processing_time - self.arrival_time

    @property
    def flow_time(self) -> float | None:
        """Time from arrival until processing is completed."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time

    @property
    def lateness(self) -> float | None:
        if self.completion_time is None:
            return None
        return self.completion_time - self.due_date

    def as_record(self) -> dict:
        """Flat dict for CSV export."""
        return {
            "id": self.id, "conveyor_id": self.conveyor_id, "status": self.status.value,
            "arrival_time": self.arrival_time, "processing_time": self.processing_time,
            "due_date": self.due_date, "entry_time": self.entry_time,
            "ready_time": self.ready_time, "start_processing_time": self.start_processing_time,
            "completion_time": self.completion_time, "exit_time": self.exit_time,
            "waiting_time": self.waiting_time, "flow_time": self.flow_time,
            "lateness": self.lateness, "processed": self.is_processed,
            "final_x": round(self.x, 4), "y": self.y,
        }
