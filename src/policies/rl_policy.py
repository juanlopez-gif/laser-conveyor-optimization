"""Policy ``rl``: a trained reinforcement-learning agent (docs/methods/rl.md).

The agent chooses a conveyor; the policy serves the head of that conveyor (same
action space as :mod:`rl.env`). Parameters (``policy.params.rl``):

model:          path of the saved MaskablePPO model (default ``models/rl06_wi_s0.zip``,
                the 0.6.0 DAgger agent trained with pre-positioning ``when_idle`` and
                selected on validation seeds 2000+; docs/methods/rl.md §6. Until 0.5.2
                the default was ``models/rl_bc.zip``)
deterministic:  greedy action (default True)

It is ``parametric`` (needs a trained model and the optional RL dependencies), so it
is not listed among the stand-alone policies and the visualization/experiments do
not run it unless it is asked for explicitly.
"""

from __future__ import annotations

from pathlib import Path

from simulation.config import PROJECT_ROOT

from . import register_policy
from .base_policy import BasePolicy

DEFAULT_MODEL = PROJECT_ROOT / "models" / "rl06_wi_s0.zip"   # 0.6.0 (was rl_bc.zip)
_MODEL_CACHE: dict[str, object] = {}


def load_model(path: str | Path):
    key = str(Path(path).resolve())
    if key not in _MODEL_CACHE:
        from sb3_contrib import MaskablePPO           # optional dependency
        _MODEL_CACHE[key] = MaskablePPO.load(key, device="cpu")
    return _MODEL_CACHE[key]


@register_policy("rl")
class RLPolicy(BasePolicy):
    parametric = True

    def __init__(self, model: str | None = None, deterministic: bool = True, **params):
        super().__init__(model=model, deterministic=deterministic, **params)
        self.model_path = Path(model) if model else DEFAULT_MODEL
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        self.deterministic = bool(deterministic)
        self._model = None

    def select(self, candidates, ctx):
        from rl.features import action_mask, encode, part_for_action
        sim = ctx.simulation
        n = len(ctx.conveyors)
        mask = action_mask(candidates, n)
        if mask.sum() == 1:                       # forced decision (as in the env)
            return part_for_action(candidates, int(mask.argmax()), n)
        if self._model is None:
            self._model = load_model(self.model_path)
        action, _ = self._model.predict(encode(sim, candidates), action_masks=mask,
                                        deterministic=self.deterministic)
        return part_for_action(candidates, int(action), n)
