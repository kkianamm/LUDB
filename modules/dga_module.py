"""
DGA -- Dual Granularity Alignment for Visual Reprogramming of CLIP
(CVPR 2026 Highlight), added as a VISION-SIDE module ON TOP OF IPL.

CRITICAL framing (the "do not replace IPL with DGA" requirement)
----------------------------------------------------------------
DGA's own repo is a *complete* pipeline: it learns pixel-space visual
reprogramming (VR) + text_projections + a class hierarchy, and uses its OWN
text side. If we imported it wholesale it would REPLACE IPL's prompt learner.
Instead we extract DGA's genuinely vision-side contributions and keep IPL's
learned prompt as the classifier:

    * PaddingVR                -> learnable pixel program added before the frozen
                                  vision tower (ported from methods/vp.py).
    * Visual granularity       -> multi-scale crops, each reprogrammed, logits
                                  fused by entropy weighting (from dga.VisualGranularity).
    * Text side                -> IPL's learned (C,D) prompt features. DGA does
                                  NOT introduce a competing text classifier here.

Because VR edits pixels, image features must be RECOMPUTED each step through the
(still frozen) vision tower -> this module sets `wants_raw_pixels=True`, which
makes the trainer bypass IPL's feature cache. The vision tower stays frozen;
only the VR programs (and optional projections) are trained.

Ported vs. needs-tuning
-----------------------
Ported & multi-label-safe: PaddingVR, multi-scale VR, entropy-weighted logit
fusion, BCE alignment against IPL text.
NOT auto-ported (single-label in the original; left as opt-in with a caveat):
the Semantic-Granularity hierarchy branch + HKP consistency (`dga.HKP`) assume
softmax/argmax over a class tree. For 5 PTB-XL superclasses a shallow 2-level
tree {normal, abnormal}->{NORM,MI,STTC,CD,HYP} is provided as a default, but the
HKP rescaling must be re-derived for multi-label before enabling it. Keep
`use_semantic_hierarchy=False` for the first working runs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .base import AdaptationModule, BatchContext, BuildContext

# BiomedCLIP normalisation stats (NOT OpenAI CLIP's). The original PaddingVR
# hard-codes OpenAI CLIP mean/std -- for BiomedCLIP we must use its own. If your
# preprocess already normalises, feed this module UN-normalised [0,1] pixels and
# let VR normalise, so the program is added in pixel space (as DGA intends).
BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class PaddingVR(nn.Module):
    """Learnable pixel program padded around the image (ported from methods/vp.py,
    normalisation swapped to BiomedCLIP)."""

    def __init__(self, out_size=224, input_size=192, init="zero"):
        super().__init__()
        self.out_size, self.input_size = out_size, input_size
        if init == "randn":
            self.program = nn.Parameter(torch.randn(3, out_size, out_size))
        else:
            self.program = nn.Parameter(torch.zeros(3, out_size, out_size))
        self.normalize = Normalize(BIOMEDCLIP_MEAN, BIOMEDCLIP_STD)
        self.l_pad = int((out_size - input_size + 1) / 2)
        self.r_pad = int((out_size - input_size) / 2)
        mask = torch.zeros(3, input_size, input_size)
        self.register_buffer("mask", F.pad(mask, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=1))

    def forward(self, x):
        x = F.pad(x, (self.l_pad, self.r_pad, self.l_pad, self.r_pad), value=0)
        x = x + torch.sigmoid(self.program) * self.mask
        return self.normalize(x)


class DGAModule(AdaptationModule):
    name = "dga"
    wants_raw_pixels = True       # -> disables IPL feature cache
    train_stage = 1               # trained jointly with IPL

    def build(self, ctx: BuildContext):
        c = ctx.cfg
        self.clip = ctx.clip_model
        self.device = ctx.device
        self.input_size = c.get("input_size", 192)
        self.num_scales = c.get("num_vg_scales", 2)     # multi-granularity crop scales
        self.lambda_align = c.get("lambda_align", 1.0)
        self.use_semantic_hierarchy = c.get("use_semantic_hierarchy", False)  # see caveat

        # one VR per visual-granularity scale
        self.visual_reprograms = nn.ModuleList([
            PaddingVR(224, self.input_size) for _ in range(self.num_scales)
        ])

        # cache handle to IPL text so alignment uses the learned prompt, not a
        # competing DGA text classifier.
        self._ipl = ctx.ipl_model

        if self.use_semantic_hierarchy:
            # Default shallow ECG tree; HKP for multi-label must be re-derived.
            self.superclass_of = c.get("superclass_map",
                                       {0: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # normal=0, abnormal=1
            raise NotImplementedError(
                "Semantic-hierarchy/HKP branch is single-label in the source; "
                "re-derive HKP for multi-label before enabling. Start with "
                "use_semantic_hierarchy=False.")

    # ---- crop helper (multi-scale visual granularity) ----
    def _crop_resize(self, x, crop_size):
        b, c, h, w = x.shape
        top = torch.randint(0, max(1, h - crop_size + 1), (1,)).item()
        left = torch.randint(0, max(1, w - crop_size + 1), (1,)).item()
        x = x[:, :, top:top + crop_size, left:left + crop_size]
        return F.interpolate(x, size=(self.input_size, self.input_size),
                             mode="bilinear", align_corners=False)

    @staticmethod
    def _entropy_weight(logits_list):
        """Entropy-weighted fusion across granularities (lower entropy -> more weight)."""
        ent = []
        for lg in logits_list:
            p = torch.sigmoid(lg).clamp(1e-6, 1 - 1e-6)
            ent.append((-(p * p.log() + (1 - p) * (1 - p).log())).mean(dim=-1))  # (B,)
        w = F.softmax(-torch.stack(ent, dim=-1), dim=-1)      # (B, S)
        return w

    def _encode_reprogrammed(self, x_scaled, vr):
        feats = self.clip.encode_image(vr(x_scaled))
        return F.normalize(feats, dim=-1)

    # ---- image stage: reprogram pixels, recompute features, fuse logits ----
    def image_stage(self, pixel_values, frozen_encode):
        """pixel_values: UN-normalised [0,1] images resized to `input_size`.
        Returns (fused_image_features, fused_multi_granularity_logits)."""
        x = pixel_values.to(self.device)
        text = F.normalize(self._ipl.prompt_learner(image_features=None), dim=-1)  # (C, D)
        scale = self._ipl.logit_scale.exp()

        feats_list, logits_list = [], []
        for s, vr in enumerate(self.visual_reprograms):
            crop = self.input_size if s == 0 else max(64, self.input_size - 32 * s)
            x_s = x if s == 0 else self._crop_resize(x, crop)
            if x_s.shape[-1] != self.input_size:
                x_s = F.interpolate(x_s, size=(self.input_size, self.input_size),
                                    mode="bilinear", align_corners=False)
            f = self._encode_reprogrammed(x_s, vr)           # (B, D)
            feats_list.append(f)
            logits_list.append(scale * f @ text.t())         # (B, C)

        w = self._entropy_weight(logits_list)                # (B, S)
        fused_logits = sum(w[:, i:i + 1] * logits_list[i] for i in range(len(logits_list)))
        fused_feats = sum(w[:, i:i + 1] * feats_list[i] for i in range(len(feats_list)))
        fused_feats = F.normalize(fused_feats, dim=-1)
        return fused_feats, fused_logits

    def extra_loss(self, bctx: BatchContext):
        # DGA's alignment is realised through the fused-logit BCE (computed by the
        # trainer on the fused logits). Additional cross-granularity consistency
        # can be added here; kept minimal & multi-label-safe for the first runs.
        return torch.zeros((), device=bctx.targets.device), {}

    def trainable_parameters(self):
        return list(self.visual_reprograms.parameters())
