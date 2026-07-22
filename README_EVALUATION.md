# ECGCLIP evaluation patch

This patch adds one shared metric implementation for zero-shot BiomedCLIP,
BiomedCoOp prompt tuning, and the frozen-feature linear probe.

## Files

- `evaluation.py`: single-label and multi-label metrics, validation threshold
  selection, JSON output.
- `zero_shot_eval.py`: replacement that supports `--task single|multi`.
- `train_biomedcoop.py`: replacement with common evaluation, safer checkpoint
  loading, result JSON files, and deterministic few-shot sampling.
- `linear_probe.py`: replacement using the same multi-label metrics.
- `aggregate_results.py`: aggregates result JSON files across seeds.
- `run_fewshot_evaluation.sh`: runs K={1,2,4,8,16}, seeds={1,2,3} for both
  protocols.

## Install

Copy the files into the repository root, replacing the three scripts with the
same names:

```bash
cp evaluation.py /path/to/ecgclip/
cp aggregate_results.py /path/to/ecgclip/
cp run_fewshot_evaluation.sh /path/to/ecgclip/
cp zero_shot_eval.py /path/to/ecgclip/
cp train_biomedcoop.py /path/to/ecgclip/
cp linear_probe.py /path/to/ecgclip/
```

Or apply the supplied unified patch from the repository root:

```bash
git apply ecgclip_evaluation.patch
```

## Smoke tests

```bash
python -m py_compile evaluation.py zero_shot_eval.py \
  train_biomedcoop.py linear_probe.py aggregate_results.py

python zero_shot_eval.py --task multi --limit 200
python zero_shot_eval.py --task single --limit 200
python train_biomedcoop.py --task multi --shots 1 --seed 1 \
  --epochs 1 --limit 500
python train_biomedcoop.py --task single --shots 1 --seed 1 \
  --epochs 1 --limit 500
```

Do not use `--limit` for final reported results because taking the first rows of
fold 9 or fold 10 can remove positives from a class.

## Final experiments

```bash
./run_fewshot_evaluation.sh
```

Results are written to `work/results/*.json`; the aggregate table is written to
`work/results/biomedcoop_summary.csv`.

### Direct BiomedCoOp-style comparison

Use the single-label rows and report mean ± standard deviation across seeds:

- Accuracy
- Macro F1
- Balanced accuracy

The primary headline metric is accuracy.

### Native PTB-XL comparison

Use the multi-label rows and report mean ± standard deviation across seeds:

- Macro and micro AUROC
- Macro and micro AUPRC
- Macro and micro F1
- Per-class AUROC, AUPRC, sensitivity, and specificity

Fold 9 selects F1 thresholds. Fold 10 is only used for final evaluation.

## Important seed interpretation

Changing only an evaluation seed should not change deterministic test results.
Seeds should change few-shot sampling, prompt initialization, minibatch order,
or model training. Zero-shot evaluation has no training seed and is expected to
produce identical results for the same checkpoint and data.

## Base-to-novel

This patch does not claim an official base-to-novel result. With only five
PTB-XL superclasses, base/novel results are highly split-dependent. Add that as
a separate experiment only after defining fixed class splits and rebuilding the
prompt learner for novel class names while transferring a shared context
(`CSC=False`).
