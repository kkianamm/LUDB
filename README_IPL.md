# IPL prompt learning for ECG-CLIP (BiomedCLIP × PTB-XL)

This adds the **IPL / IGPL prompt-learning strategy** to your `ecgclip` repo to
push accuracy above zero-shot and linear-probe, in both **few-shot** and
**full-data** settings — while keeping BiomedCLIP frozen.

## What IPL actually is (and what had to change)

The repo you linked (`kkianamm/IPL`) is **IGPL** — an *interpretable-guided
prompt-learning* trainer built on the **Dassl** framework, for **standard
OpenAI CLIP ViT-B/16**, **single-label** datasets, using concept caches from
SAE tools (`patchsae`, `splice`, …). Three things don't transfer to your setup,
so this is a faithful *port of the mechanism*, not a copy of their trainer:

| IGPL assumes | Your setup | What this port does |
|---|---|---|
| CLIP's own text transformer (CoOp token surgery) | BiomedCLIP = **PubMedBERT** text tower | Injects learnable context via `inputs_embeds` into PubMedBERT, then reuses BiomedCLIP's own pooler + projection so text lands in the same 512-d space as `encode_text` |
| Single-label softmax CE | **Multi-label** (5 superclasses) | Cosine-sim logits → `BCEWithLogitsLoss` per class |
| Concept cache from an SAE trained on CLIP | No SAE exists for BiomedCLIP's vision tower | Concept guidance replaced by **text-anchor preservation** (same regulariser role; see caveat below) |

The transferable core of IPL is preserved:

- **Learnable prompt context** (CoOp) — replaces hand-written prompts.
- **Instance-conditioned prompts** (the IGPL "representation tokens" / CoCoOp
  idea) — a meta-net turns each image's feature into a shift on the context, so
  every ECG gets a tailored prompt.
- **Anchor preservation** — keep the learned prompt near the frozen zero-shot
  text so it doesn't drift and lose BiomedCLIP's pretrained knowledge.

These map to three staged **phases**, exactly like IGPL's `PHASE=A/B/C`:

| Phase | Trains | Objective |
|---|---|---|
| **A** | learnable context only | `BCE` (CoOp) |
| **B** | learnable context | `BCE + λ·anchor` |
| **C** | context + meta-net | `BCE + λ·anchor`, prompts conditioned on each image (**full IPL**) |

## Files

```
ipl/prompt_learner.py   learnable prompts on PubMedBERT + instance meta-net
ipl/model.py            frozen BiomedCLIP + prompt learner + multi-label loss
ipl/data.py             labels.csv/images loader, few-shot sampler, feature cache
ipl/metrics.py          macro AUROC / per-class AUROC / F1  (PTB-XL standard)
config_ipl.py           class names, anchor prompts, phase presets, hyper-params
train_ipl.py            few-shot AND full training
eval_ipl.py             evaluate a checkpoint
```

Drop the `ipl/` folder and the three top-level files into your repo root (next
to `config.py`). Nothing in your existing repo is modified.

## Install

Your `requirements.txt` already covers most of it. IPL also needs:

```
pip install open-clip-torch>=2.23.0 timm>=0.9.8 transformers scikit-learn pandas pillow
```

## Run

First render images as you already do (`python prepare_data.py`). Then:

```bash
# Few-shot (e.g. 16 examples per class), full IPL:
python train_ipl.py --shots 16 --phase C

# Full training, full IPL:
python train_ipl.py --shots 0 --phase C

# Ablations / staging:
python train_ipl.py --shots 0 --phase A     # CoOp only
python train_ipl.py --shots 0 --phase B     # + anchor preservation

# Evaluate a saved checkpoint on the test fold:
python eval_ipl.py --ckpt work/checkpoints/ipl_C_train_full.pt
```

Backbone is frozen, so image features are cached once to `work/feat_cache/` and
reused across phases and seeds. Val-macro-AUROC selects the best epoch; the test
fold (fold 10) is scored only with that checkpoint.

## Recommended progression

1. `--phase A` full-data — sanity baseline; should beat zero-shot.
2. `--phase B` — anchor loss usually stabilises and helps generalisation.
3. `--phase C` — instance conditioning; the main IPL result. Compare few-shot
   (`--shots 1/2/4/8/16`) curves against `linear_probe.py`.

## Honest caveats

- **Concept pool (patchsae/SAE) is intentionally not ported.** It needs a sparse
  autoencoder trained on the vision tower's activations; none exists off-the-shelf
  for BiomedCLIP. The Phase-B text anchor fills the same regularisation role. If
  you later train an SAE on BiomedCLIP image features, the hook is
  `IPLModel.anchor_loss` — add a concept-alignment term there.
- **Ceiling.** As your own README notes, rendering a 1-D signal to a 224px image
  loses information; a purpose-built 1-D ECG model (~0.93 macro AUROC) will still
  beat any CLIP-on-plots approach. IPL is the right tool for *adapting BiomedCLIP*,
  not for beating 1-D models.
- **Phase C cost.** Instance conditioning encodes text per image (B×5 BERT passes
  per step). It's fine on GPU with cached image features; on CPU it's slow — start
  with `--phase B` there.
- **Config assumptions.** `config_ipl.py` reads `labels.csv`/`images` paths from
  your `config.py` if importable, and `data.py` handles either per-class 0/1
  columns or a list column plus `split`/`strat_fold`. If your `labels.csv` uses
  different names, adjust `CLASSNAMES` / the resolvers at the top of `data.py`.
```
