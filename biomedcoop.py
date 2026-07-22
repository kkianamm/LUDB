"""
BiomedCoOp for BiomedCLIP, as a standalone module that plugs into this repo.

This reproduces the method from Koleilat et al., "BiomedCoOp: Learning to Prompt
for Biomedical Vision-Language Models" (CVPR 2025), specifically the
`BiomedCoOp_BiomedCLIP` trainer, WITHOUT the Dassl framework or their forked
open_clip. It talks directly to the stock `open-clip-torch` BiomedCLIP model
(the one loaded in model_utils.load_biomedclip), so nothing else in your repo
has to change.

What BiomedCoOp learns
----------------------
Only a small set of `n_ctx` continuous "context" vectors (like CoOp). The whole
BiomedCLIP vision + text backbone stays frozen. On top of the classification
cross-entropy it adds two losses:

  * SCCM (Semantic Consistency by Contextual Mapping): pulls the learned prompt
    text embeddings toward the *mean* of an ensemble of LLM-written prompts for
    each class (MSE).  -> keeps the learned prompt semantically on-topic.

  * KDSP (Knowledge Distillation with Selective Prompting): distills the frozen
    model's predictions (built from an outlier-pruned subset of the LLM prompt
    ensemble) into the learned-prompt predictions (KL divergence).
    -> improves generalization, prunes weird LLM prompts.

Total loss = CE + SCCM_LAMBDA * SCCM + KDSP_LAMBDA * KDSP.

Multi-label note (important for PTB-XL)
---------------------------------------
The paper is single-label (softmax + cross-entropy). PTB-XL is MULTI-LABEL (a
record can be e.g. MI *and* STTC). This module supports both:
  * task="single": faithful reproduction (softmax CE, softmax-KL).
  * task="multi" : BCE-with-logits classification, per-class (sigmoid) KL for
    KDSP, SCCM unchanged. This is the mode that matches your existing
    zero_shot_eval / linear_probe pipeline and macro-AUROC metric.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Replicates the BiomedCoOp fork's `encode_text(prompts, inputs_embeds=True, y=...)`
# using only the submodules that stock open-clip-torch already exposes.
# ---------------------------------------------------------------------------
def encode_text_from_embeds(clip_model, inputs_embeds, tokenized_prompts):
    """Encode text from precomputed word embeddings (needed for prompt tuning).

    clip_model        : the BiomedCLIP model from create_model_from_pretrained
    inputs_embeds     : (n, L, d) assembled [SOS ; ctx ; classname ; ...] embeds
    tokenized_prompts : (n, L) token ids matching inputs_embeds, used ONLY to
                        build the attention mask (pad positions are masked out).
    returns           : (n, output_dim) text features (un-normalized)
    """
    text = clip_model.text  # HFTextEncoder (PubMedBERT + pooler + proj)
    attn_mask = (tokenized_prompts != text.config.pad_token_id).long().to(inputs_embeds.device)
    out = text.transformer(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
    pooled = text.pooler(out, attn_mask)
    return text.proj(pooled)


class PromptLearner(nn.Module):
    """Learns `n_ctx` context vectors shared across classes (CSC optional)."""

    def __init__(self, clip_model, tokenizer, classnames, class_templates,
                 n_ctx=4, ctx_init="a photo of a", csc=False,
                 class_token_position="end", context_length=256, device="cpu"):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.class_token_position = class_token_position
        self.context_length = context_length
        self.tokenizer = tokenizer

        wte = clip_model.text.transformer.embeddings.word_embeddings
        dtype = wte.weight.dtype
        ctx_dim = wte.weight.shape[1]  # 768 for PubMedBERT

        # ---- initialize context vectors -----------------------------------
        use_init = bool(ctx_init) and (n_ctx == len(ctx_init.split()))
        if use_init:
            init_ids = tokenizer([ctx_init], context_length=context_length).to(device)
            with torch.no_grad():
                init_emb = wte(init_ids).type(dtype)  # (1, L, d)
            # skip the [CLS] at position 0, take the next n_ctx word embeddings
            ctx_vectors = init_emb[0, 1:1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            if csc:
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # the ONLY trainable tensor

        # ---- build the fixed prefix/suffix around the class names ---------
        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = torch.cat(
            [tokenizer([p], context_length=context_length) for p in prompts]
        ).to(device)  # (n_cls, L)
        with torch.no_grad():
            embedding = wte(tokenized_prompts).type(dtype)  # (n_cls, L, d)
        self.register_buffer("token_prefix", embedding[:, :1, :])            # [CLS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])    # name + [SEP] + pad
        self.tokenized_prompts = tokenized_prompts

        # ---- precompute the frozen LLM prompt-ensemble text features ------
        # class_templates: list (len n_cls) of lists (len n_prompts) of sentences
        n_prompts = len(class_templates[0])
        teacher = []
        with torch.no_grad():
            for i in range(n_prompts):
                batch = [class_templates[c][i] for c in range(self.n_cls)]
                toks = tokenizer(batch, context_length=context_length).to(device)
                feats = clip_model.encode_text(toks)          # (n_cls, d)
                teacher.append(feats.unsqueeze(1))            # (n_cls, 1, d)
        # (n_cls, n_prompts, d)
        self.register_buffer("fixed_embeddings", torch.cat(teacher, dim=1))

    def _assemble(self, ctx):
        prefix, suffix = self.token_prefix, self.token_suffix
        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)
        raise NotImplementedError("Only class_token_position='end' is implemented.")

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:  # shared context -> broadcast to all classes
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = self._assemble(ctx)  # (n_cls, L, d)
        return encode_text_from_embeds(self._clip, prompts, self.tokenized_prompts)


class BiomedCoOpCLIP(nn.Module):
    """Frozen BiomedCLIP + learnable prompt, with BiomedCoOp's SCCM/KDSP losses."""

    def __init__(self, clip_model, tokenizer, classnames, class_templates,
                 task="multi", n_ctx=4, ctx_init="a photo of a", csc=False,
                 sccm_lambda=0.5, kdsp_lambda=0.25, tau=1.5,
                 context_length=256, ml_temperature=0.5, device="cpu"):
        super().__init__()
        assert task in ("single", "multi")
        self.task = task
        self.sccm_lambda = sccm_lambda
        self.kdsp_lambda = kdsp_lambda
        self.tau = tau
        self.ml_temperature = ml_temperature

        self.prompt_learner = PromptLearner(
            clip_model, tokenizer, classnames, class_templates,
            n_ctx=n_ctx, ctx_init=ctx_init, csc=csc,
            context_length=context_length, device=device,
        )
        # give the prompt learner a handle to the frozen model (not a submodule,
        # so it is not registered / re-saved with the learner's state_dict)
        object.__setattr__(self.prompt_learner, "_clip", clip_model)

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.embeddings.word_embeddings.weight.dtype

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def image_features(self, image):
        f = self.image_encoder(image.type(self.dtype))
        return f / f.norm(dim=-1, keepdim=True)

    def logits_from(self, image_features, text_features):
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()

        text_features = self.prompt_learner()                      # (n_cls, d)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        img_f = self.image_encoder(image.type(self.dtype))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        logits = logit_scale * img_f @ text_features.t()           # (B, n_cls)

        if not self.prompt_learner.training or label is None:
            return logits

        fixed = self.prompt_learner.fixed_embeddings               # (n_cls, P, d)
        fixed = fixed / fixed.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            # vision tower is frozen -> zero-shot img features == img_f
            zs_f = img_f.detach()

            # ---- KDSP statistics-based prompt (outlier) pruning -----------
            P = fixed.shape[1]
            scores = []
            for i in range(P):
                temp_logits = logit_scale * img_f @ fixed[:, i, :].t()   # (B, n_cls)
                scores.append(torch.max(temp_logits, dim=1).values.mean().item())
            scores_t = torch.tensor(scores)
            s_bar = torch.median(scores_t)
            d_bar = torch.median(torch.abs(scores_t - s_bar)) + 1e-8
            z = (scores_t - s_bar) / d_bar
            zz = (z - z.mean()) / (z.std() + 1e-8)
            mask = torch.abs(zz) <= self.tau
            if mask.sum() == 0:                      # safety: keep all if pruned everything
                mask = torch.ones_like(mask)
            selected = fixed[:, mask, :].mean(dim=1)                # (n_cls, d)
            selected = selected / selected.norm(dim=-1, keepdim=True)

        fixed_mean = fixed.mean(dim=1)                             # (n_cls, d)
        fixed_mean = fixed_mean / fixed_mean.norm(dim=-1, keepdim=True)
        zero_shot_logits = logit_scale * zs_f @ selected.t()      # (B, n_cls)

        # ---- SCCM: prompt <-> LLM-ensemble-mean consistency ---------------
        loss_sccm = F.mse_loss(text_features, fixed_mean) * self.sccm_lambda

        if self.task == "single":
            loss_ce = F.cross_entropy(logits, label)
            loss_kdsp = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction="sum", log_target=True,
            ) / logits.numel()
        else:  # multi-label
            # scale cosine sims into a BCE-friendly range (large logit_scale
            # would saturate the sigmoid), then BCE against the multi-hot label.
            s = (img_f @ text_features.t()) / self.ml_temperature
            loss_ce = F.binary_cross_entropy_with_logits(
                s, label.type(s.dtype), pos_weight=self._pos_weight_to(s.device)
            )
            # per-class (Bernoulli) KL distillation
            t_stud = (img_f @ text_features.t()) / self.ml_temperature
            t_teach = (zs_f @ selected.t()) / self.ml_temperature
            p_teach = torch.sigmoid(t_teach)
            logp_stud = F.logsigmoid(t_stud)
            logp_stud_neg = F.logsigmoid(-t_stud)
            logp_teach = F.logsigmoid(t_teach)
            logp_teach_neg = F.logsigmoid(-t_teach)
            loss_kdsp = (
                p_teach * (logp_teach - logp_stud)
                + (1 - p_teach) * (logp_teach_neg - logp_stud_neg)
            ).mean()

        loss_kdsp = loss_kdsp * self.kdsp_lambda
        return logits, loss_ce, loss_sccm, loss_kdsp

    # pos_weight for multi-label BCE (set from data by the trainer)
    _pos_weight = None

    def set_pos_weight(self, pos_weight):
        self._pos_weight = pos_weight

    def _pos_weight_to(self, device):
        return None if self._pos_weight is None else self._pos_weight.to(device)


def build_biomedcoop(clip_model, tokenizer, classnames, class_templates,
                     device, **kwargs):
    """Freeze everything except prompt_learner.ctx and return the model."""
    model = BiomedCoOpCLIP(clip_model, tokenizer, classnames, class_templates,
                           device=device, **kwargs)
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith("prompt_learner.ctx"))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {trainable}")
    return model.to(device)
