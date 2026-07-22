# BiomedCoOp for ecgclip (PTB-XL)

Drop-in prompt-learning on top of your existing BiomedCLIP × PTB-XL repo. No
Dassl, no forked `open_clip` — it uses the stock `open-clip-torch>=2.23.0` you
already have, and reuses your `config.py`, `model_utils.py`, `labels.csv`, and
`prepare_data.image_path_for`.

## Files
| File | Purpose |
|------|---------|
| `ecg_prompts.py` | Per-class LLM-style prompt ensembles for the 5 superclasses (BiomedCoOp's SCCM/KDSP teacher prompts). Extend/regenerate freely. |
| `biomedcoop.py` | Core method: `PromptLearner` (learns `n_ctx` context vectors only), CE/BCE + **SCCM** (semantic consistency) + **KDSP** (knowledge distillation with statistics-based prompt pruning). |
| `train_biomedcoop.py` | Few-shot **and** full training + evaluation, single- or multi-label. Uses your `zero_shot_eval.evaluate()` so numbers are directly comparable to your baselines. |

Put all three in your repo root, next to `config.py`.

## Prereq
```bash
python prepare_data.py          # writes work/labels.csv + work/images/*.png
```

## Few-shot (matches your macro-AUROC pipeline; multi-label)
```bash
# K shots per class, K in {1,2,4,8,16}; run 3 seeds and average
python train_biomedcoop.py --shots 16 --seed 1
python train_biomedcoop.py --shots 16 --seed 2
python train_biomedcoop.py --shots 16 --seed 3
```

## Full training (all of folds 1–8)
```bash
python train_biomedcoop.py --shots 0 --epochs 20 --batch-size 32
```

## Faithful single-label reproduction (paper's exact objective)
Uses only records with a single superclass, softmax CE + softmax-KL:
```bash
python train_biomedcoop.py --shots 16 --task single
```

## Evaluate a saved prompt
```bash
python train_biomedcoop.py --eval-only \
  --ckpt work/checkpoints/biomedcoop_multi_16shot_seed1.pt
```

## Key options (defaults follow the paper's few-shot config)
`--task {multi,single}` · `--n-ctx 4` · `--ctx-init "a photo of a"` ·
`--n-prompts 30` · `--tau 1.5` · `--sccm-lambda 0.5` · `--kdsp-lambda 0.25` ·
`--lr 0.0025` (SGD+cosine, 1-epoch warmup) · `--ml-temperature 0.5` (multi-label only).

## Two things to know
1. **Multi-label.** BiomedCoOp is single-label (softmax CE) in the paper. PTB-XL
   is multi-label, so `--task multi` (default) swaps CE→BCE and softmax-KL→
   per-class sigmoid-KL, keeping SCCM as-is. `--task single` is the faithful
   reproduction on single-superclass records only.
2. **Only the prompt is trained.** The BiomedCLIP vision + text towers stay
   frozen; the sole trainable tensor is `prompt_learner.ctx`. That's the whole
   point — big generalization gains, tiny parameter count, no backbone fine-tune.
   If you want to *also* move the backbone, run this first, then your existing
   `finetune_clip.py` / `linear_probe.py`.
