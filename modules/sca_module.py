"""
SCA -- Statistics Caching Test-Time Adaptation (NeurIPS 2025), on top of IPL.

Ported faithfully from Yuqin-G/SCA `utils_sca.EntropyBasedMemoryBank.get_memory2`
and `main_sca.run_test_sca`, with the ONE change required by our task:

    SCA is single-label (softmax over classes, argmax top-1, per-class memory
    keyed by a pseudo-label). PTB-XL is MULTI-LABEL. We therefore:
      * replace softmax -> sigmoid for the pseudo-probabilities;
      * replace class-distribution entropy -> mean per-class BINARY entropy;
      * keep the ridge cache G,P -> W = (G+ridgeI)^-1 P^T unchanged (it is a
        linear least-squares map feats->per-class score and is label-agnostic).

Hook stage: TEST TIME ONLY (training-free, no grad). SCA takes the *fixed* text
classifier IPL produced and the (frozen or DGA-reprogrammed) image features, and
blends a running statistics-cache logit with the base logit by confidence.

Composition note w/ IPL Phase C (meta-net): SCA needs a fixed (C,D) classifier.
When IPL instance-conditioning is on, feed SCA the *shared* prompt (meta-net
bypassed) -- see `ipl_module.IPLBase.shared_text_features()`. This is handled by
the trainer/eval, which passes `text_features` of shape (C, D) to SCA.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AdaptationModule, BatchContext, BuildContext


def _binary_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean per-class binary entropy of independent sigmoids. logits: (B, C)."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    ent = -(p * p.log() + (1 - p) * (1 - p).log())     # (B, C)
    return ent.mean(dim=-1)                             # (B,)


class RidgeStatsCache:
    """Accumulating ridge-regression cache: W = (G + ridge*I)^-1 P^T."""

    def __init__(self, feat_dim: int, n_classes: int, ridge: float, device: str):
        self.d, self.C, self.ridge, self.device = feat_dim, n_classes, ridge, device
        self.reset()

    def reset(self):
        self.G = torch.zeros(self.d, self.d, device=self.device)
        self.P = torch.zeros(self.C, self.d, device=self.device)

    @torch.no_grad()
    def update_and_logit(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """feats: (B, D) L2-normalised; probs: (B, C) sigmoid pseudo-probs."""
        feats = feats.float()
        self.G += (feats.T @ feats) / feats.shape[0]
        self.P += (feats[:, None, :] * probs[:, :, None]).mean(dim=0)   # (C, D)
        I = torch.eye(self.d, device=self.device)
        W = torch.linalg.solve(self.G + self.ridge * I, self.P.T)       # (D, C)
        return F.normalize(feats, p=2, dim=-1) @ F.normalize(W, p=2, dim=0)  # (B, C)


class SCAModule(AdaptationModule):
    name = "sca"
    is_test_time = True          # no training stage

    def build(self, ctx: BuildContext):
        c = ctx.cfg
        self.ridge = c.get("ridge", 1e4)
        self.beta = c.get("beta", 5.0)         # entropy-sharpness for the blend
        self.tau = c.get("tau", 0.5)           # (kept for parity; unused in binary variant)
        self.device = ctx.device
        self.cache = None

    def on_eval_start(self, n_classes, feat_dim, device):
        self.cache = RidgeStatsCache(feat_dim, n_classes, self.ridge, device)

    @torch.no_grad()
    def refine_logits(self, bctx: BatchContext) -> torch.Tensor:
        base_logit = bctx.logits.float()                    # (B, C)
        feats = bctx.image_features.float()
        probs = torch.sigmoid(base_logit)

        cache_logit = self.cache.update_and_logit(feats, probs)   # (B, C)

        # entropy-weighted blend: trust whichever branch is more confident.
        e_cache = _binary_entropy(cache_logit)              # (B,)
        e_base = _binary_entropy(base_logit)                # (B,)
        coeff = F.softmax(torch.stack([e_cache, e_base], -1) * self.beta, dim=-1)
        alpha = coeff[:, 0:1]                               # (B, 1)
        return alpha * cache_logit + (1 - alpha) * base_logit
