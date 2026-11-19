"""Run ONE simulation (headless) and save its results.

Examples
--------
    python run_simulation.py
    python run_simulation.py --policy nearest_job
    python run_simulation.py --config config/examples/stochastic.yaml --seed 7
    python run_simulation.py --set laser.speed_x=10 --set conveyors.speed=3

Outputs go to ``results/<name>/`` (summary.json/csv, parts.csv, events.log,
events.csv, queue_lengths.csv, trace.json) and, unless ``--no-viz`` is given,
``visualization/data/trace.js`` is refreshed so that opening
``visualization/index.html`` replays this run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from simulation.config import load_config  # noqa: E402
from simulation.engine import Simulation  # noqa: E402
from simulation.results_io import save_result  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file merged on top of config/default_config.yaml")
    ap.add_argument("--policy", help="override policy.name")
    ap.add_argument("--seed", type=int, help="override simulation.seed")
    ap.add_argument("--mode", choices=["deterministic", "stochastic"], help="override arrivals.mode")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, e.g. --set laser.speed_x=8 (repeatable)")
    ap.add_argument("--name", help="results sub-folder name (default: <policy>_seed<seed>)")
    ap.add_argument("--no-viz", action="store_true", help="do not update visualization/data/trace.js")
    ap.add_argument("--quiet", action="store_true", help="do not print the event log head")
    args = ap.parse_args(argv)

    sets = list(args.set)
    if args.policy:
        sets.append(f"policy.name={args.policy}")
    if args.seed is not None:
        sets.append(f"simulation.seed={args.seed}")
    if args.mode:
        sets.append(f"arrivals.mode={args.mode}")
    cfg = load_config(args.config, set_items=sets)

    result = Simulation(cfg, record_trace=True).run()
    name = args.name or f"{cfg.policy_name}_{cfg.arrivals.mode}_seed{cfg.simulation.seed}"
    out = save_result(result, ROOT / cfg.output.get("results_dir", "results") / name,
                      save_trace=bool(cfg.output.get("save_trace", True)),
                      update_visualization=not args.no_viz
                      and bool(cfg.output.get("update_visualization_data", True)))

    if not args.quiet:
        print("--- first events " + "-" * 50)
        for line in result.events.lines()[:25]:
            print(line)
        print("...")
    s = result.summary
    print("--- summary " + "-" * 55)
    keys = ["policy", "total_simulation_time", "parts_generated", "parts_completed",
            "parts_exited", "parts_exited_unprocessed", "throughput_per_min",
            "avg_waiting_time", "max_waiting_time", "avg_flow_time", "makespan",
            "total_laser_travel_distance", "laser_utilization",
            "laser_processing_utilization", "laser_movement_utilization",
            "laser_idle_time", "total_conveyor_stop_time", "objective"]
    keys += [k for k in s if k.startswith("conveyor_") and (k.endswith("stopped_time")
                                                            or k.endswith("avg_queue_length"))]
    for k in keys:
        v = s.get(k)
        print(f"{k:38s} {v:.3f}" if isinstance(v, float) else f"{k:38s} {v}")
    print(f"\nResults saved to {out}")
    if not args.no_viz:
        print("Open visualization/index.html in a browser to replay this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
