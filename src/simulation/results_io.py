"""Saving simulation outputs (CSV / JSON / text log / visualization trace)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .engine import SimulationResult

VIS_DATA_FILE = PROJECT_ROOT / "visualization" / "data" / "trace.js"


def _json_default(obj):
    if isinstance(obj, float) and obj != obj:   # NaN
        return None
    return str(obj)


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_result(result: SimulationResult, out_dir: str | Path,
                save_trace: bool = True, update_visualization: bool = False) -> Path:
    """Write every output file of a run into ``out_dir``. Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "summary.json").write_text(json.dumps(result.summary, indent=2, default=_json_default),
                                      encoding="utf-8")
    write_dict_csv(out / "summary.csv", [result.summary])
    write_dict_csv(out / "parts.csv", [p.as_record() for p in result.parts])
    result.events.save_text(out / "events.log")
    result.events.save_csv(out / "events.csv")

    n = result.config.n_conveyors
    with open(out / "queue_lengths.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time"] + [f"conveyor_{i + 1}" for i in range(n)])
        for t, q in result.queue_series:
            writer.writerow([t] + q)

    (out / "config_used.json").write_text(json.dumps(result.config.raw, indent=2), encoding="utf-8")

    if result.trace is not None and save_trace:
        (out / "trace.json").write_text(json.dumps(result.trace, separators=(",", ":")),
                                        encoding="utf-8")
    if result.trace is not None and update_visualization:
        write_visualization_traces({result.policy_name: result.trace})
    return out


def write_visualization_traces(traces: dict[str, dict], path: Path = VIS_DATA_FILE) -> Path:
    """Write ``{policy_name: trace}`` as a JS file (works from file://, no server needed).

    The first key is the policy the player shows when the page opens.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.SIM_TRACES = " + json.dumps(traces, separators=(",", ":")) + ";\n",
                    encoding="utf-8")
    return path
