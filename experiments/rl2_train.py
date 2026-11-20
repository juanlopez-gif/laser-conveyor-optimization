"""Train the strong-RL dispatcher by expert iteration (docs/methods/rl2.md §6).

Thin wrapper around :mod:`rl2.expert_iteration` that puts ``src/`` on the path at
import time, so the spawned worker processes (Windows) can import the package.

    python experiments/rl2_train.py --tag rl2_gen_s --iters 6 --workers 8
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl2.expert_iteration import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
