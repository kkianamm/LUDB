"""
Composition framework for stacking optional adaptation modules ON TOP OF IPL.

Design contract
---------------
IPL is the *base* and is always present. SCA / MERGETUNE / DGA are optional
modules that hook the shared BiomedCLIP pipeline at DIFFERENT, non-overlapping
stages, which is exactly why they compose:

    pixels --(DGA)--> [frozen vision] --> image_features
                                              |
                        text_features <-- [frozen text + IPL prompt]   (IPL, MERGETUNE)
                                              |
                    logits = scale * cos(img, text)
                                              |
                        loss (BCE + IPL.anchor + DGA.align + MERGETUNE.lmc)
                                              |
                    test-time logit refinement (SCA)

Each module declares WHERE it hooks via the capability flags below, and the
orchestrator (`ModuleStack`) routes the forward/loss/inference accordingly.
There are only three optional modules and they hook at genuinely different
stages, so the stack uses explicit, readable branches rather than a maximally
generic hook bus (that would be false elegance and harder to trust).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BuildContext:
    """Everything a module might need at construction time."""
    clip_model: nn.Module
    tokenizer: object
    classnames: list
    class_prompts: list
    ipl_model: nn.Module          # the base IPLModel (already built)
    device: str
    cfg: dict                     # the module's own config sub-dict


@dataclass
class BatchContext:
    """Per-step state passed to module hooks during training/eval."""
    pixel_values: Optional[torch.Tensor] = None   # raw (un-normalised) images, if available
    image_features: Optional[torch.Tensor] = None # (B, D) L2-normalised (cached or freshly computed)
    text_features: Optional[torch.Tensor] = None  # (C, D) or (B, C, D) from IPL prompt learner
    logits: Optional[torch.Tensor] = None         # (B, C) current base logits
    targets: Optional[torch.Tensor] = None        # (B, C) multi-hot
    scale: float = 1.0                            # logit_scale.exp()
    stage: int = 1
    extras: dict = field(default_factory=dict)    # scratch space for module-to-orchestrator handoff


class AdaptationModule(nn.Module):
    """Base class. Every optional method subclasses this. All hooks are no-ops
    by default, so a module only overrides the stages it actually touches."""

    name: str = "base"

    # --- capability flags (read by ModuleStack to route work) ---
    wants_raw_pixels: bool = False   # DGA=True -> disables IPL feature caching
    train_stage: int = 0             # 0 = not a training module; 1 = joint-with-IPL; 2 = post-hoc
    is_test_time: bool = False       # SCA=True

    def build(self, ctx: BuildContext) -> None:
        """Construct sub-modules given the frozen backbone + IPL base."""
        raise NotImplementedError

    # ---- image stage (DGA overrides to reprogram pixels & recompute features) ----
    def image_stage(self, pixel_values, frozen_encode):
        """Return (image_features, mg_logits_or_None). Default: just encode.
        DGA returns fused multi-granularity logits in the 2nd slot."""
        return frozen_encode(pixel_values), None

    # ---- training loss contribution (DGA alignment/HKP; IPL's own anchor lives in IPLModel) ----
    def extra_loss(self, bctx: BatchContext):
        return torch.zeros((), device=bctx.targets.device), {}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- inference-time refinement (SCA overrides) ----
    def refine_logits(self, bctx: BatchContext) -> torch.Tensor:
        return bctx.logits

    # ---- lifecycle hooks (SCA resets its cache between eval passes) ----
    def on_eval_start(self, n_classes: int, feat_dim: int, device: str):
        pass


class ModuleStack:
    """Holds the ordered set of active optional modules and applies them at the
    right stage. IPL itself is NOT in the stack (it is the immovable base handed
    in separately); the stack only carries SCA/MERGETUNE/DGA when enabled."""

    def __init__(self, modules: list[AdaptationModule]):
        self.modules = modules

    # ---- routing helpers ----
    @property
    def wants_raw_pixels(self) -> bool:
        return any(m.wants_raw_pixels for m in self.modules)

    def image_module(self) -> Optional[AdaptationModule]:
        mods = [m for m in self.modules if m.wants_raw_pixels]
        return mods[0] if mods else None      # currently only DGA

    def train_modules(self, stage: int):
        return [m for m in self.modules if m.train_stage == stage]

    def test_time_modules(self):
        return [m for m in self.modules if m.is_test_time]

    def all_trainable_parameters(self, stage: int):
        params = []
        for m in self.train_modules(stage):
            params += list(m.trainable_parameters())
        return params

    def extra_losses(self, bctx: BatchContext, stage: int):
        total = torch.zeros((), device=bctx.targets.device)
        logs = {}
        for m in self.train_modules(stage):
            l, d = m.extra_loss(bctx)
            total = total + l
            logs.update({f"{m.name}.{k}": v for k, v in d.items()})
        return total, logs

    def refine_test_logits(self, bctx: BatchContext) -> torch.Tensor:
        logits = bctx.logits
        for m in self.test_time_modules():        # SCA is the outermost inference wrapper
            bctx.logits = logits
            logits = m.refine_logits(bctx)
        return logits
