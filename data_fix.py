"""
data_fix.py -- corrected label resolution for PTB-XL superclasses.

Why this exists
---------------
`ipl/data.py:_resolve_targets` matches the label tokens in labels.csv against
`config_ipl.CLASSNAMES`. But CLASSNAMES are the *human-readable prompt phrases*
("normal ecg", ...), while PTB-XL's labels.csv stores the *codes* NORM/MI/STTC/
CD/HYP. They never match, so every target row is all-zeros -> the model learns to
predict 0 (tiny BCE), and every AUROC is NaN (constant columns). This module
decouples the two: prompts stay human-readable; targets are resolved via an alias
map, and an assertion makes the silent-all-zero failure impossible to miss.

Usage: in train_compose.py, import build_splits from here instead of from ipl.
Aliases are aligned to config_ipl.CLASSNAMES order; override by setting
`LABEL_ALIASES` in config_ipl.py if your vocabulary differs.
"""
from __future__ import annotations

import ast
import numpy as np
import pandas as pd

from ipl.data import ECGImageDataset, _resolve_split   # reuse the base plumbing

# code/phrase aliases, aligned to config_ipl.CLASSNAMES order:
# [NORM, MI, STTC, CD, HYP]
DEFAULT_ALIASES = [
    ["NORM", "normal", "normal ecg", "normal_ecg", "norm"],
    ["MI", "myocardial infarction", "myocardial_infarction", "mi"],
    ["STTC", "st t wave change", "st-t change", "st/t change", "stt", "sttc"],
    ["CD", "conduction disturbance", "conduction_disturbance", "cd"],
    ["HYP", "hypertrophy", "ventricular hypertrophy", "hyp"],
]


def _get_aliases(classnames):
    try:
        import config_ipl as C
        aliases = getattr(C, "LABEL_ALIASES", None)
        if aliases is not None:
            assert len(aliases) == len(classnames), \
                "LABEL_ALIASES length must match CLASSNAMES"
            return [[a.lower() for a in row] for row in aliases]
    except Exception:
        pass
    assert len(DEFAULT_ALIASES) == len(classnames), \
        "DEFAULT_ALIASES length must match CLASSNAMES; set LABEL_ALIASES in config_ipl.py"
    return [[a.lower() for a in row] for row in DEFAULT_ALIASES]


def _parse_list(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    if not isinstance(val, str):
        return []
    try:
        out = ast.literal_eval(val)
        return list(out) if isinstance(out, (list, tuple)) else [str(out)]
    except (ValueError, SyntaxError):
        return [t.strip(" '\"[]") for t in val.split(",") if t.strip(" '\"[]")]


def resolve_targets(df: pd.DataFrame, classnames):
    aliases = _get_aliases(classnames)
    token2idx = {tok: i for i, row in enumerate(aliases) for tok in row}
    cols_lower = {c.lower(): c for c in df.columns}
    Y = np.zeros((len(df), len(classnames)), dtype=np.float32)

    # --- Path A: one 0/1 column per class (matched via aliases) ---
    per_class_cols = []
    for row in aliases:
        hit = next((cols_lower[a] for a in row if a in cols_lower), None)
        per_class_cols.append(hit)
    if all(c is not None for c in per_class_cols):
        for i, col in enumerate(per_class_cols):
            Y[:, i] = df[col].astype(float).values
        return _guard(Y, classnames, source=f"per-class columns {per_class_cols}")

    # --- Path B: a single list-like column of codes/phrases ---
    list_col = next((cols_lower[c] for c in
                     ["labels", "superclass", "superclasses",
                      "diagnostic_superclass", "scp_superclass"]
                     if c in cols_lower), None)
    if list_col is not None:
        for r, val in enumerate(df[list_col].tolist()):
            for it in _parse_list(val):
                j = token2idx.get(str(it).strip().lower())
                if j is not None:
                    Y[r, j] = 1.0
        return _guard(Y, classnames, source=f"list column '{list_col}'")

    raise ValueError(
        f"labels.csv has no per-class columns nor a list column matching the "
        f"aliases. Columns present: {list(df.columns)}. "
        f"Set LABEL_ALIASES in config_ipl.py to match your vocabulary.")


def _guard(Y, classnames, source):
    counts = Y.sum(0)
    if Y.sum() == 0 or (counts == 0).any():
        raise ValueError(
            f"Degenerate targets from {source}: per-class positive counts = "
            f"{dict(zip(classnames, counts.tolist()))}. A zero count means the "
            f"label vocabulary in labels.csv does not match LABEL_ALIASES. Fix "
            f"the aliases before training (this is exactly what caused the NaN "
            f"AUROC / 0.0 F1).")
    print(f"[data_fix] targets OK from {source}: "
          f"{dict(zip(classnames, counts.astype(int).tolist()))}")
    return Y


def build_splits(labels_csv, image_dir, classnames, preprocess, id_col="ecg_id"):
    """Drop-in replacement for ipl.build_splits with corrected target resolution."""
    df = pd.read_csv(labels_csv)
    df["_split"] = _resolve_split(df)
    out = {}
    for name in ["train", "val", "test"]:
        sub = df[df["_split"] == name].reset_index(drop=True)
        ds = ECGImageDataset.__new__(ECGImageDataset)          # build without base resolver
        import torch
        ds.df = sub
        ds.image_dir = image_dir
        ds.preprocess = preprocess
        ds.ids = sub[id_col].tolist()
        ds.targets = torch.tensor(resolve_targets(sub, classnames), dtype=torch.float32)
        out[name] = ds
    return out
