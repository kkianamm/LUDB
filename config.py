"""
Central configuration for the BiomedCLIP x PTB-XL project.

Edit DATA_DIR to point at the folder that contains `ptbxl_database.csv`
(i.e. the root of the downloaded PTB-XL dataset).
"""
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Root of the extracted PTB-XL dataset (contains ptbxl_database.csv,
# scp_statements.csv, records100/, records500/)
DATA_DIR = os.environ.get("DATA_DIR", "/lambda/nfs/Kiana2/ecgclip/data/ptbxl")

# Where rendered ECG images and cached features/checkpoints go
WORK_DIR = os.environ.get("WORK_DIR", "./work")
IMG_DIR = os.path.join(WORK_DIR, "images")          # rendered ECG PNGs
FEAT_DIR = os.path.join(WORK_DIR, "features")        # cached embeddings
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")     # fine-tuned weights

for _d in (WORK_DIR, IMG_DIR, FEAT_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# Data settings
# ----------------------------------------------------------------------------
SAMPLING_RATE = 100          # 100 -> records100 (fast), 500 -> records500 (hi-res)
FILENAME_COL = "filename_lr" if SAMPLING_RATE == 100 else "filename_hr"

# PTB-XL recommended split: folds 1-8 train, 9 val, 10 test
TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10

# The 5 diagnostic SUPERCLASSES (multi-label; a record can carry several)
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Human-readable descriptions used to build text prompts / captions
CLASS_DESCRIPTIONS = {
    "NORM": "normal ECG",
    "MI":   "myocardial infarction",
    "STTC": "ST/T wave change",
    "CD":   "conduction disturbance",
    "HYP":  "cardiac hypertrophy",
}

# ----------------------------------------------------------------------------
# Zero-shot prompt engineering
# ----------------------------------------------------------------------------
# BiomedCLIP's own template is "this is a photo of ".  We ensemble several
# templates and average the resulting text embeddings for robustness.
PROMPT_TEMPLATES = [
    "this is a photo of {}",
    "an electrocardiogram showing {}",
    "a 12-lead ECG with {}",
    "ECG tracing consistent with {}",
]

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256         # BiomedCLIP tokenizer default

# ----------------------------------------------------------------------------
# Training hyper-parameters (used by linear_probe.py and finetune_clip.py)
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4

# Linear probe
LP_EPOCHS = 50
LP_LR = 1e-3
LP_WEIGHT_DECAY = 1e-4

# Contrastive fine-tune
FT_EPOCHS = 5
FT_LR = 1e-5
FT_WEIGHT_DECAY = 0.1
FT_FREEZE_TEXT = True        # freeze PubMedBERT text tower, tune vision tower only
