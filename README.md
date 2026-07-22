# BiomedCLIP × PTB-XL — evaluate and fine-tune

Evaluate **BiomedCLIP** on the **PTB-XL** 12-lead ECG dataset (zero-shot), and
train/adapt it three ways: a linear probe on frozen features, and CLIP-style
contrastive fine-tuning.

---

## The one thing to understand first

**BiomedCLIP is an image–text model. PTB-XL is time-series signal data.**

BiomedCLIP was pretrained on 15M *figure–caption* pairs from PubMed Central. It
consumes 224×224 RGB **images** and text. PTB-XL ships raw 12-lead **waveforms**
in WFDB format. So the whole pipeline hinges on one bridge: **render each ECG
into an image** (`ecg_to_image.py`) that looks like the ECG figures BiomedCLIP
may have seen in papers, then feed those images to the model.

**Set expectations accordingly:**
- Zero-shot BiomedCLIP on ECG plots is a genuine out-of-distribution test. It
  will likely land only modestly above chance (macro AUROC ~0.55–0.70 on the 5
  superclasses, in our experience of similar setups) — it was never trained on
  ECG waveform plots.
- Linear probe and contrastive fine-tuning improve substantially.
- None of these will match a purpose-built **1D** ECG model (e.g. a ResNet1d/
  xresnet1d101 from the Strodthoff et al. benchmark reaches ~0.93 macro AUROC on
  superclasses). Rendering a signal to a 224px image discards information. If
  your goal is *best ECG accuracy*, a 1D model wins; if your goal is
  *evaluating/adapting BiomedCLIP specifically*, this repo is the right tool.

---

## Dataset (PTB-XL v1.0.3)

- 21,799 clinical 12-lead ECGs, 10 s each, 18,869 patients.
- WFDB signals at 500 Hz (`records500/`) and 100 Hz (`records100/`).
- Labels: 71 SCP-ECG statements, aggregated here into **5 diagnostic
  superclasses** — NORM, MI, STTC, CD, HYP. **Multi-label** (a record can have
  several).
- Official split via `strat_fold`: **folds 1–8 = train, 9 = val, 10 = test**
  (patient-disjoint; folds 9–10 are human-validated, highest quality).
- License CC-BY 4.0, open access (no credentialing needed).

### Download

```bash
# ~1.7 GB zip / ~3 GB uncompressed
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
# or:
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ./ptbxl
```

Point `DATA_DIR` in `config.py` (or `export PTBXL_DIR=...`) at the folder that
contains `ptbxl_database.csv`.

---

## Install

```bash
pip install -r requirements.txt
```
`open-clip-torch>=2.23.0` and `timm>=0.9.8` are required for BiomedCLIP to load.
A GPU is strongly recommended for feature extraction and fine-tuning (CPU works
but is slow). The first model load downloads weights from the Hugging Face Hub.

---

## Step by step

Everything is driven by `config.py`. Use `--limit N` on any script for a fast
smoke test before committing to the full ~22k records.

### 1. Prepare data + render images
```bash
python prepare_data.py --limit 500     # quick check first
python prepare_data.py                 # full: writes work/labels.csv + PNGs
```
Parses SCP codes → 5 superclasses, applies the fold split, writes
`work/labels.csv`, and renders every ECG to `work/images/<ecg_id>.png`.
Rendering all records takes a while (it's the slow step); PNGs are cached and
skipped on re-runs.

### 2. Zero-shot evaluation (no training)
```bash
python zero_shot_eval.py
```
Builds prompt-ensembled text embeddings for each class (see `PROMPT_TEMPLATES`
and `CLASS_DESCRIPTIONS` in `config.py`), encodes the test images, computes
image↔text cosine similarity, and reports **macro AUROC** (the standard PTB-XL
metric), per-class AUROC, macro/micro F1, and a loose top-1 accuracy.

*This is the answer to "evaluate BiomedCLIP on this dataset."*

### 3a. Train — linear probe (fast, the recommended first "training")
```bash
python extract_features.py     # cache frozen BiomedCLIP image embeddings
python linear_probe.py         # train a 5-way multi-label linear head
```
Freezes BiomedCLIP, trains only a linear classifier on its image features with
`BCEWithLogitsLoss` (+ positive weighting for class imbalance), selects on
validation macro AUROC, reports on test. Runs in seconds–minutes.

### 3b. Train — contrastive fine-tuning (this is "training BiomedCLIP")
```bash
python finetune_clip.py                     # captions from labels (English)
python finetune_clip.py --caption report    # use raw PTB-XL report strings
python zero_shot_eval.py --ckpt work/checkpoints/biomedclip_ft.pt
```
Fine-tunes with the **same symmetric InfoNCE loss CLIP was trained with**,
pairing each ECG image with a caption. By default it freezes the text tower
(PubMedBERT) and tunes the vision tower — see `FT_FREEZE_TEXT`, `FT_LR`,
`FT_EPOCHS` in `config.py`. Then re-run zero-shot with the checkpoint to measure
the lift.

> **Caption note:** many PTB-XL `report` strings are in **German**, but
> BiomedCLIP's text encoder is English. The default `--caption label` builds an
> English sentence from the superclasses (e.g. *"a 12-lead ECG showing
> myocardial infarction and ST/T wave change"*), which is usually the better
> signal. Use `--caption report` only if you translate the reports first.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All paths, class list, prompts, hyper-parameters |
| `ecg_to_image.py` | WFDB signal → ECG-paper-style PNG |
| `prepare_data.py` | Metadata, superclass labels, split, batch render |
| `model_utils.py` | Load BiomedCLIP; build class text embeddings |
| `zero_shot_eval.py` | Zero-shot classification + metrics |
| `extract_features.py` | Cache frozen image embeddings for the probe |
| `linear_probe.py` | Train/eval a linear head on frozen features |
| `finetune_clip.py` | Contrastive fine-tuning of BiomedCLIP |

---

## Design choices & knobs

- **Image style.** `render_ecg(style="grid")` mimics ECG paper (pink grid), the
  closest match to figures in papers; `style="plain"` is a clean alternative.
  Leads are stacked (12 rows, full 10 s each) so no signal is dropped. At 100 Hz
  this is plenty; 500 Hz (`SAMPLING_RATE=500`) gives finer detail at higher cost.
- **Multi-label metric.** Macro AUROC over the 5 superclasses is the field
  standard (Strodthoff et al. 2021) because the task is multi-label; top-1
  accuracy is only a loose sanity check.
- **Better supervised results.** For maximum accuracy, unfreeze the vision tower
  and train it end-to-end with a classification head + `BCEWithLogitsLoss`
  (combine the encoder from `model_utils` with the head from `linear_probe`).
  Contrastive fine-tuning first, then a probe, also works well.

## Citation

Wagner P, Strodthoff N, Bousseljot R-D, Samek W, Schaeffter T. *PTB-XL, a large
publicly available electrocardiography dataset* (v1.0.3). PhysioNet (2022).
doi:10.13026/kfzx-aw45. And the BiomedCLIP paper (Zhang et al., 2023).
