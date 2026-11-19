"""Shortcut: ``python run_experiments.py --runs 1000`` (see experiments/run_experiment.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from run_experiment import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
