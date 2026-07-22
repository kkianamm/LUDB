"""
MERGETUNE -- Continued Fine-Tuning via Linear Mode Connectivity (ICLR 2026),
applied post-hoc ON TOP OF IPL.

Ported from Surrey-UP-Lab/MERGETUNE `kgcoop_coop_LMC.forward_backward`:
    line_samples = arange(0.1, 1.01, 1/NUM_SAMPLES)
    feature_start = text(reference prompt)      # one endpoint
    feature_end   = text(trainable prompt)      # the continued model
    for t in line_samples:
        mid = start + (end-start)*t
        loss += task_loss(logits_from(mid)) / len(samples)
    loss = task_loss(end) + W_LMC * line_loss

Our mapping (multi-label, BiomedCLIP text tower):
    endpoint 1 (ŵ1)  = zero-shot BiomedCLIP text anchors  (already in IPLModel.text_anchors)
    endpoint 2 (ŵ2)  = IPL-trained prompt  (the continued/trainable model)
    task_loss        = BCEWithLogits  (was CrossEntropy in the single-label repo)

This is a STAGE-2 module: it runs as a *second* training phase after the base
(IPL, or IPL+DGA) has converged, continuing to tune the SAME IPL parameters so
the solution stays linearly mode-connected to zero-shot -> recovers pretrained
knowledge / novel-class generalisation. It touches only the text-feature space,
so it works on the fast cached-feature path and composes with everything.

`MERGETUNE without IPL` (baseline): valid only on top of *some* fine-tuned model
(e.g. Phase-A CoOp). On pure zero-shot there are not two endpoints to merge, so
that combination is degenerate and intentionally not offered.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import AdaptationModule, BatchContext, BuildContext


class MergeTuneModule(AdaptationModule):
    name = "mergetune"
    train_stage = 2              # post-hoc continued fine-tuning

    def build(self, ctx: BuildContext):
        c = ctx.cfg
        self.ipl = ctx.ipl_model
        self.w_lmc = c.get("w_lmc", 1.0)
        self.num_samples = c.get("num_samples", 10)
        self.line_samples = np.arange(0.1, 1.01, 1.0 / float(self.num_samples))
        # endpoint 1: frozen zero-shot text anchors (C, D)
        self.register_buffer("w_zeroshot", ctx.ipl_model.text_anchors.detach().clone())

    def trainable_parameters(self):
        # continue tuning the IPL prompt learner (context [+ meta-net]).
        return list(self.ipl.prompt_learner.parameters())

    def extra_loss(self, bctx: BatchContext):
        scale = bctx.scale
        feats = bctx.image_features                     # (B, D)

        # endpoint 2 = current IPL text (shared prompt; meta-net bypassed for a
        # fixed classifier along the interpolation line).
        w_ft = self.ipl.prompt_learner(image_features=None)     # (C, D), grad flows
        w_zs = self.w_zeroshot                                   # (C, D)

        line_loss = torch.zeros((), device=feats.device)
        for t in self.line_samples:
            w_mid = w_zs + (w_ft - w_zs) * float(t)
            w_mid = F.normalize(w_mid, dim=-1)
            logits_mid = scale * feats @ w_mid.t()              # (B, C)
            line_loss = line_loss + F.binary_cross_entropy_with_logits(
                logits_mid, bctx.targets.float()) / len(self.line_samples)

        return self.w_lmc * line_loss, {"lmc": float(line_loss)}
