"""Set-Transformer dispatcher with a pointer policy and a value head (strong-RL plan,
phase 2).

* Each token type (part, conveyor, laser, context) has its own linear embedding to
  width ``d``, plus a learned type vector.
* A Transformer encoder (pre-norm, ReLU feed-forward, **no positional encoding**) mixes all tokens of a
  state. Self-attention without positions is permutation-equivariant, so shuffling
  part or conveyor tokens shuffles the part outputs the same way and leaves pooled
  quantities unchanged (tested in ``tests/test_rl2_model.py``).
* **Policy:** a pointer over the part tokens. The logit of part i is an MLP of
  ``[h_i, g]`` where ``g`` is the masked mean of all token outputs. Non-candidates get
  ``-inf``, so any candidate part can be chosen, not only belt heads.
* **Value:** an MLP of ``[g, h_ctx, h_laser]`` predicting the normalized cost-to-go.
  The scale is kept by :class:`RunningScale` (phase 3 trains it).

Sizes (plan §3): ``S`` 4 × 128 × 4 heads (~0.8 M parameters, CPU), ``M`` 8 × 256 × 8,
``L`` 12 × 512 × 8. Same code; the size is a config.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from rl2.features import CONV_F, CTX_F, LASER_F, PART_F, Observation

TYPE_CTX, TYPE_LASER, TYPE_CONV, TYPE_PART = range(4)


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    d: int = 128
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    activation: str = "relu"

    @staticmethod
    def size(name: str) -> "ModelConfig":
        return SIZES[name.upper()]


SIZES = {
    "S": ModelConfig(4, 128, 4),
    "M": ModelConfig(8, 256, 8),
    "L": ModelConfig(12, 512, 8),
}


# ---------------------------------------------------------------- batching
@dataclass
class Batch:
    """Padded tensors for B states. Token order: ctx, laser, conveyors, parts."""

    part: torch.Tensor        # (B, P, PART_F)
    part_valid: torch.Tensor  # (B, P) bool
    cand: torch.Tensor        # (B, P) bool
    conv: torch.Tensor        # (B, N, CONV_F)
    conv_valid: torch.Tensor  # (B, N) bool
    laser: torch.Tensor       # (B, LASER_F)
    ctx: torch.Tensor         # (B, CTX_F)

    def to(self, device) -> "Batch":
        return Batch(*(getattr(self, f).to(device, non_blocking=True) for f in
                       ("part", "part_valid", "cand", "conv", "conv_valid", "laser", "ctx")))


def collate(obs: list[Observation]) -> Batch:
    B = len(obs)
    P = max(1, max(o.part.shape[0] for o in obs))
    N = max(1, max(o.conv.shape[0] for o in obs))
    part = np.zeros((B, P, PART_F), np.float32)
    pv = np.zeros((B, P), bool)
    cand = np.zeros((B, P), bool)
    conv = np.zeros((B, N, CONV_F), np.float32)
    cv = np.zeros((B, N), bool)
    for i, o in enumerate(obs):
        p, n = o.part.shape[0], o.conv.shape[0]
        part[i, :p] = o.part
        pv[i, :p] = True
        cand[i, :p] = o.cand
        conv[i, :n] = o.conv
        cv[i, :n] = True
    return Batch(torch.from_numpy(part), torch.from_numpy(pv), torch.from_numpy(cand),
                 torch.from_numpy(conv), torch.from_numpy(cv),
                 torch.from_numpy(np.stack([o.laser for o in obs])),
                 torch.from_numpy(np.stack([o.ctx for o in obs])))


# ------------------------------------------------------------------- model
def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class DispatchNet(nn.Module):
    def __init__(self, cfg: ModelConfig = SIZES["S"]):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.emb_part = nn.Linear(PART_F, d)
        self.emb_conv = nn.Linear(CONV_F, d)
        self.emb_laser = nn.Linear(LASER_F, d)
        self.emb_ctx = nn.Linear(CTX_F, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff_mult * d, cfg.dropout,
                                           activation=cfg.activation, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.policy = _mlp(2 * d, d, 1)
        self.value = _mlp(3 * d, d, 1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, P) with -inf on non-candidates and padding,
        value (B,))."""
        B, P = b.part_valid.shape
        N = b.conv_valid.shape[1]
        te = self.type_emb
        x = torch.cat([
            (self.emb_ctx(b.ctx) + te[TYPE_CTX]).unsqueeze(1),
            (self.emb_laser(b.laser) + te[TYPE_LASER]).unsqueeze(1),
            self.emb_conv(b.conv) + te[TYPE_CONV],
            self.emb_part(b.part) + te[TYPE_PART],
        ], dim=1)
        valid = torch.cat([torch.ones(B, 2, dtype=torch.bool, device=b.part.device),
                           b.conv_valid, b.part_valid], 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=~valid))
        w = valid.unsqueeze(-1).to(h.dtype)
        g = (h * w).sum(1) / w.sum(1)
        hp = h[:, 2 + N:]
        logits = self.policy(torch.cat([hp, g.unsqueeze(1).expand(-1, P, -1)], -1)).squeeze(-1)
        logits = logits.masked_fill(~(b.cand & b.part_valid), -math.inf)
        value = self.value(torch.cat([g, h[:, 0], h[:, 1]], -1)).squeeze(-1)
        return logits, value


class RunningScale:
    """Running mean/std of the cost-to-go targets, used to normalize the value head."""

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, xs) -> None:
        for x in np.asarray(xs, dtype=np.float64).ravel():
            self.n += 1
            dlt = x - self.mean
            self.mean += dlt / self.n
            self.m2 += dlt * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0

    def normalize(self, x):
        return (x - self.mean) / max(self.std, 1e-6)

    def denormalize(self, z):
        return z * max(self.std, 1e-6) + self.mean


# ------------------------------------------------------------------ policy
class NetPolicy:
    """Greedy (or sampled) dispatcher for :meth:`fastsim.FastSim.run`."""

    def __init__(self, net: DispatchNet, sample: bool = False, generator=None):
        self.net = net.eval()
        self.sample = sample
        self.gen = generator

    @torch.inference_mode()
    def __call__(self, sim, candidates):
        if len(candidates) == 1:
            return candidates[0]
        from rl2.features import encode
        obs = encode(sim, candidates)
        logits, _ = self.net(collate([obs]))
        if self.sample:
            i = int(torch.multinomial(torch.softmax(logits[0], -1), 1, generator=self.gen))
        else:
            i = int(torch.argmax(logits[0]))
        return obs.part_ids[i]


def save(net: DispatchNet, path, extra: dict | None = None) -> None:
    """Weights are always stored on the CPU, so a model trained on a GPU loads anywhere."""
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"config": asdict(net.cfg), "state": state, "extra": extra or {}}, path)


def resolve_device(name: str = "auto") -> "torch.device":
    """``auto``: CUDA if available, else Apple MPS, else CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def load(path) -> DispatchNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = DispatchNet(ModelConfig(**ck["config"]))
    net.load_state_dict(ck["state"])
    return net.eval()


def load_extra(path) -> dict:
    """The ``extra`` dict saved with the model (iteration, value scale ...)."""
    return torch.load(path, map_location="cpu", weights_only=False).get("extra", {})


def load_scale(path) -> RunningScale | None:
    """The value-head scale saved with the model (``extra["scale"]``), or None."""
    sc = load_extra(path).get("scale")
    if sc is None:
        return None
    r = RunningScale()
    r.n, r.mean, r.m2 = int(sc[0]), float(sc[1]), float(sc[2])
    return r
