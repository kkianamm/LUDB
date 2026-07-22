"""
IPL prompt learner for BiomedCLIP (PubMedBERT text tower).

This is the port of the IPL / IGPL *mechanism* to BiomedCLIP:

  * learnable context tokens (CoOp)                      -> Phase A
  * text-anchor preservation to the frozen zero-shot text -> Phase B
  * instance-conditioned prompts via a meta-net (CoCoOp / IGPL
    "representation tokens" idea): each image shifts the context     -> Phase C

Why this file exists instead of reusing IGPL's trainer directly:
BiomedCLIP's text encoder is a HuggingFace BERT (PubMedBERT), not CLIP's
own text transformer. The usual CoOp trick of splicing learnable vectors
into `token_embedding` / `positional_embedding` does not apply. Instead we
feed `inputs_embeds` straight into the BERT and reuse BiomedCLIP's own
pooler + projection so the output lives in the exact same 512-d space as
`model.encode_text`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_bert(clip_model):
    """The HFTextEncoder wraps the actual BERT at `.transformer`."""
    text_encoder = clip_model.text
    bert = text_encoder.transformer
    return text_encoder, bert


class IPLPromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer,
        classnames,
        ctx_init: str = "this is a twelve lead ecg showing",
        n_ctx: int = 8,
        use_metanet: bool = True,
        metanet_hidden: int = 128,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.use_metanet = use_metanet
        self.n_classes = len(classnames)

        text_encoder, bert = _get_bert(clip_model)
        self.text_encoder = text_encoder            # for pooler + proj at forward
        word_emb: nn.Embedding = bert.get_input_embeddings()
        hidden = word_emb.embedding_dim             # 768 for PubMedBERT
        img_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, "output_dim") else 512

        hf_tok = tokenizer.tokenizer                # underlying HF WordPiece tokenizer
        cls_id = hf_tok.cls_token_id
        sep_id = hf_tok.sep_token_id
        self.pad_id = hf_tok.pad_token_id

        with torch.no_grad():
            emb_table = word_emb.weight.detach().cpu()
            cls_emb = emb_table[cls_id].clone()     # (hidden,)
            sep_emb = emb_table[sep_id].clone()

            # ---- initialise context ----
            if ctx_init:
                init_ids = hf_tok.encode(ctx_init, add_special_tokens=False)[:n_ctx]
                ctx_vectors = emb_table[torch.tensor(init_ids)].clone()
                if ctx_vectors.shape[0] < n_ctx:    # pad with small noise if prompt too short
                    extra = torch.empty(n_ctx - ctx_vectors.shape[0], hidden).normal_(std=0.02)
                    ctx_vectors = torch.cat([ctx_vectors, extra], dim=0)
            else:
                ctx_vectors = torch.empty(n_ctx, hidden).normal_(std=0.02)

            # ---- per-class name embeddings ----
            class_embs, class_lens = [], []
            for name in classnames:
                ids = hf_tok.encode(name, add_special_tokens=False)
                class_embs.append(emb_table[torch.tensor(ids)].clone())
                class_lens.append(len(ids))

        self.n_ctx = n_ctx
        self.hidden = hidden
        self.ctx = nn.Parameter(ctx_vectors.to(dtype))     # the ONLY text params trained in Phase A

        # fixed pieces (not trained) -> buffers so they move with .to(device)
        self.register_buffer("cls_emb", cls_emb.to(dtype))
        self.register_buffer("sep_emb", sep_emb.to(dtype))

        # Build padded suffix (class tokens + SEP) + attention masks, one row per class.
        max_class_len = max(class_lens)
        self.seq_len = 1 + n_ctx + max_class_len + 1        # CLS + ctx + class + SEP
        suffix = torch.zeros(self.n_classes, max_class_len + 1, hidden, dtype=dtype)
        attn = torch.zeros(self.n_classes, self.seq_len, dtype=torch.long)
        for i, (ce, cl) in enumerate(zip(class_embs, class_lens)):
            suffix[i, :cl] = ce.to(dtype)
            suffix[i, cl] = sep_emb.to(dtype)              # SEP right after class tokens
            attn[i, : 1 + n_ctx + cl + 1] = 1              # CLS + ctx + class + SEP are real
        self.register_buffer("suffix", suffix)             # (C, max_class_len+1, hidden)
        self.register_buffer("attn_mask", attn)            # (C, seq_len)

        # ---- instance conditioning (Phase C) ----
        if use_metanet:
            self.meta_net = nn.Sequential(
                nn.Linear(img_dim, metanet_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(metanet_hidden, hidden),
            ).to(dtype)
            # start as near-identity: zero the last layer so Phase-C init == Phase-B behaviour
            nn.init.zeros_(self.meta_net[-1].weight)
            nn.init.zeros_(self.meta_net[-1].bias)

    # ------------------------------------------------------------------ #
    def _assemble(self, ctx):
        """ctx: (C, n_ctx, hidden) -> inputs_embeds (C, seq_len, hidden)."""
        C = self.n_classes
        cls = self.cls_emb.expand(C, 1, self.hidden)
        return torch.cat([cls, ctx, self.suffix], dim=1)

    def _encode(self, inputs_embeds, attn_mask):
        """Run BERT with inputs_embeds, then BiomedCLIP's own pooler + proj."""
        out = self.text_encoder.transformer(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask
        )
        pooled = self.text_encoder.pooler(out, attn_mask)   # honours whatever pooler BiomedCLIP uses
        return self.text_encoder.proj(pooled)               # (N, 512)

    def forward(self, image_features: torch.Tensor | None = None):
        """
        Returns L2-normalised text features.
          * metanet OFF -> (C, 512)              shared prompt (CoOp)
          * metanet ON  -> (B, C, 512)           per-image prompt (IPL/CoCoOp)
        """
        if (not self.use_metanet) or image_features is None:
            ctx = self.ctx.unsqueeze(0).expand(self.n_classes, -1, -1)     # (C, n_ctx, hidden)
            feats = self._encode(self._assemble(ctx), self.attn_mask)
            return nn.functional.normalize(feats, dim=-1)

        # instance-conditioned: shift the shared ctx by a per-image bias
        B = image_features.shape[0]
        shift = self.meta_net(image_features.to(self.dtype))              # (B, hidden)
        ctx = self.ctx.unsqueeze(0) + shift.unsqueeze(1)                  # (B, n_ctx, hidden)
        ctx = ctx.unsqueeze(1).expand(B, self.n_classes, self.n_ctx, self.hidden)

        cls = self.cls_emb.view(1, 1, 1, self.hidden).expand(B, self.n_classes, 1, self.hidden)
        suffix = self.suffix.unsqueeze(0).expand(B, -1, -1, -1)
        seq = torch.cat([cls, ctx, suffix], dim=2)                        # (B, C, L, hidden)
        seq = seq.reshape(B * self.n_classes, self.seq_len, self.hidden)
        mask = self.attn_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * self.n_classes, self.seq_len)

        feats = self._encode(seq, mask).view(B, self.n_classes, -1)
        return nn.functional.normalize(feats, dim=-1)
