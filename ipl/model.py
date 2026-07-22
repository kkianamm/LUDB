"""
IPL model = frozen BiomedCLIP  +  IPLPromptLearner  +  multi-label head logic.

Everything in the BiomedCLIP backbone is frozen (this is prompt learning).
Only the learnable context (and, in Phase C, the meta-net) are trained.

Losses mirror the IGPL phase structure:
  Phase A : L_cls  (multi-label BCE)
  Phase B : L_cls + lambda_anchor * L_anchor   (keep prompts near zero-shot text)
  Phase C : Phase B, but with the instance-conditioned meta-net active
"""

from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from .prompt_learner import IPLPromptLearner

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_biomedclip(device="cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


@torch.no_grad()
def zero_shot_text_anchors(clip_model, tokenizer, class_prompts, device):
    """
    Frozen zero-shot text embeddings, prompt-ensembled per class.
    `class_prompts` : list[list[str]] — several prompt strings per class.
    Returns (C, 512), L2-normalised. Used as the Phase-B anchor.
    """
    anchors = []
    for prompts in class_prompts:
        toks = tokenizer(prompts).to(device)
        feats = clip_model.encode_text(toks)
        feats = F.normalize(feats, dim=-1).mean(0)
        anchors.append(F.normalize(feats, dim=-1))
    return torch.stack(anchors, dim=0)


class IPLModel(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames,
        class_prompts,
        n_ctx: int = 8,
        ctx_init: str = "this is a twelve lead ecg showing",
        use_metanet: bool = True,
        lambda_anchor: float = 0.0,
        device: str = "cuda",
    ):
        super().__init__()
        self.clip = clip_model                      # frozen
        self.logit_scale = clip_model.logit_scale   # frozen scalar (log space)
        self.lambda_anchor = lambda_anchor
        self.device = device

        self.prompt_learner = IPLPromptLearner(
            clip_model, tokenizer, classnames,
            ctx_init=ctx_init, n_ctx=n_ctx, use_metanet=use_metanet,
        ).to(device)

        anchors = zero_shot_text_anchors(clip_model, tokenizer, class_prompts, device)
        self.register_buffer("text_anchors", anchors)   # (C, 512)

    # ---- image side (frozen; cache these for speed) ----
    @torch.no_grad()
    def encode_image(self, pixel_values):
        feats = self.clip.encode_image(pixel_values.to(self.device))
        return F.normalize(feats, dim=-1)

    def logits_from_features(self, image_features):
        """
        image_features : (B, 512) L2-normalised (frozen, possibly cached).
        Returns multi-label logits (B, C) = scale * cos(img, text).
        """
        scale = self.logit_scale.exp()
        text = self.prompt_learner(image_features)      # (C,512) or (B,C,512)

        if text.dim() == 2:                             # shared prompt
            logits = scale * image_features @ text.t()  # (B, C)
        else:                                           # per-image prompt
            logits = scale * torch.einsum("bd,bcd->bc", image_features, text)
        return logits

    def anchor_loss(self):
        """Keep the *shared* learned prompts close to the zero-shot anchors."""
        if self.lambda_anchor <= 0:
            return torch.zeros((), device=self.device)
        learned = self.prompt_learner(image_features=None)   # (C,512), metanet bypassed
        return (1.0 - (learned * self.text_anchors).sum(-1)).mean()

    def forward(self, image_features, targets):
        logits = self.logits_from_features(image_features)
        cls = F.binary_cross_entropy_with_logits(logits, targets.float())
        anchor = self.anchor_loss()
        loss = cls + self.lambda_anchor * anchor
        return loss, {"cls": cls.item(), "anchor": float(anchor)}

    def trainable_parameters(self):
        return [p for p in self.prompt_learner.parameters() if p.requires_grad]
