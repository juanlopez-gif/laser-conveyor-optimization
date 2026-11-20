"""Strong RL (docs/plans/strong_rl_plan.md): set-of-tokens features and a
Transformer pointer dispatcher. Needs torch (requirements-rl.txt)."""

from rl2.features import Observation, encode
from rl2.model import SIZES, Batch, DispatchNet, ModelConfig, NetPolicy, collate

__all__ = ["Observation", "encode", "SIZES", "Batch", "DispatchNet", "ModelConfig",
           "NetPolicy", "collate"]
