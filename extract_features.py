"""
Extract frozen BiomedCLIP image embeddings for train/val/test and cache them.
These are the inputs to the (fast, CPU-friendly) linear probe.

    python extract_features.py
    python extract_features.py --limit 2000   # quick test

Outputs (in work/features/):
    X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import config as C
from model_utils import load_biomedclip, get_device
from zero_shot_eval import encode_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = get_device()
    model, preprocess, _ = load_biomedclip(device)

    labels_df = pd.read_csv(os.path.join(C.WORK_DIR, "labels.csv"),
                            index_col="ecg_id")
    if args.limit:
        labels_df = labels_df.iloc[:args.limit]

    splits = {
        "train": labels_df[labels_df.strat_fold.isin(C.TRAIN_FOLDS)],
        "val":   labels_df[labels_df.strat_fold == C.VAL_FOLD],
        "test":  labels_df[labels_df.strat_fold == C.TEST_FOLD],
    }

    for name, df in splits.items():
        if len(df) == 0:
            continue
        print(f"[{name}] {len(df)} records")
        feats = encode_images(model, preprocess, device, df.index.tolist())
        X = feats.numpy().astype(np.float32)
        y = df[C.CLASSES].values.astype(np.float32)
        np.save(os.path.join(C.FEAT_DIR, f"X_{name}.npy"), X)
        np.save(os.path.join(C.FEAT_DIR, f"y_{name}.npy"), y)
        print(f"   saved X_{name}.npy {X.shape}  y_{name}.npy {y.shape}")


if __name__ == "__main__":
    main()
