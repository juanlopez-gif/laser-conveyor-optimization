"""PPO baseline over the token policy (docs/methods/rl2_ppo.md).

Thin wrapper around :mod:`rl2.ppo` that puts ``src/`` on the path at import time, so
the spawned worker processes can import the package.

    python experiments/rl2_ppo.py --tag rl2_ppo_s0 --seed 0 --episodes 40000 --workers 40
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl2.ppo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
