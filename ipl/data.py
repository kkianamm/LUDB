"""
Data plumbing for IPL on your rendered PTB-XL images.

Reads `work/labels.csv` + `work/images/<ecg_id>.png` produced by your
`prepare_data.py`. Robust to two common layouts:

  (a) one 0/1 column per superclass (NORM, MI, STTC, CD, HYP)
  (b) a single list-like column ("labels"/"superclass"/"diagnostic_superclass")

Split resolution order: explicit `split` column -> `strat_fold`
(1-8 train, 9 val, 10 test, the official PTB-XL split).
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _label_code(name):
    """Map readable prompt names to PTB-XL superclass columns."""
    aliases = {
        "normal ecg": "NORM",
        "myocardial infarction": "MI",
        "st t wave change": "STTC",
        "st-t wave change": "STTC",
        "st/t wave change": "STTC",
        "conduction disturbance": "CD",
        "hypertrophy": "HYP",
    }

    value = str(name).strip()
    return aliases.get(value.lower(), value.upper())


def _resolve_targets(df: pd.DataFrame, classnames):
    target_codes = [_label_code(name) for name in classnames]

    # Preferred PTB-XL format:
    # NORM, MI, STTC, CD, HYP as separate binary columns.
    if all(code in df.columns for code in target_codes):
        values = (
            df[target_codes]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        return (values > 0).astype(np.float32)

    # Fallback: one list or pipe-separated label column.
    list_col = next(
        (
            column
            for column in [
                "labels",
                "superclass",
                "superclasses",
                "diagnostic_superclass",
                "scp_superclass",
            ]
            if column in df.columns
        ),
        None,
    )

    if list_col is None:
        raise ValueError(
            f"Could not find target columns {target_codes}. "
            f"Available columns: {list(df.columns)}"
        )

    index = {code: i for i, code in enumerate(target_codes)}
    targets = np.zeros(
        (len(df), len(target_codes)),
        dtype=np.float32,
    )

    for row, value in enumerate(df[list_col].tolist()):
        for item in _parse_list(value):
            code = _label_code(item)
            if code in index:
                targets[row, index[code]] = 1.0

    if targets.sum() == 0:
        examples = df[list_col].dropna().astype(str).head(10).tolist()
        raise ValueError(
            f"All resolved targets are zero. "
            f"Column: {list_col}; examples: {examples}"
        )

    return targets


def _parse_list(value):
    """Parse Python lists, comma-separated labels, or NORM|MI format."""
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value]

    if value is None or pd.isna(value):
        return []

    value = str(value).strip()
    if not value:
        return []

    if "|" in value:
        return [
            item.strip()
            for item in value.split("|")
            if item.strip()
        ]

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        return [
            item.strip(" '\"[]")
            for item in value.split(",")
            if item.strip(" '\"[]")
        ]


def _resolve_split(df: pd.DataFrame):
    if "split" in df.columns:
        s = df["split"].astype(str).str.lower()
        return s.map(lambda x: {"training": "train", "validation": "val",
                                "testing": "test"}.get(x, x))
    fold_col = next((c for c in ["strat_fold", "fold"] if c in df.columns), None)
    if fold_col is None:
        raise ValueError("labels.csv needs a `split` or `strat_fold` column.")
    f = df[fold_col].astype(int)
    return np.where(f <= 8, "train", np.where(f == 9, "val", "test"))


class ECGImageDataset(Dataset):
    def __init__(self, df, image_dir, classnames, preprocess, id_col="ecg_id"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.ids = self.df[id_col].tolist()
        self.targets = torch.tensor(_resolve_targets(self.df, classnames), dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def _path(self, ecg_id):
        for name in (f"{ecg_id}.png", f"{int(ecg_id):05d}.png", f"{ecg_id}.jpg"):
            p = os.path.join(self.image_dir, name)
            if os.path.exists(p):
                return p
        return os.path.join(self.image_dir, f"{ecg_id}.png")

    def __getitem__(self, i):
        img = Image.open(self._path(self.ids[i])).convert("RGB")
        return self.preprocess(img), self.targets[i], i


def build_splits(labels_csv, image_dir, classnames, preprocess, id_col="ecg_id"):
    df = pd.read_csv(labels_csv)
    df["_split"] = _resolve_split(df)
    out = {}
    for name in ["train", "val", "test"]:
        sub = df[df["_split"] == name]
        out[name] = ECGImageDataset(sub, image_dir, classnames, preprocess, id_col)
    return out


def few_shot_indices(dataset: ECGImageDataset, shots: int, seed: int = 1):
    """
    K-shot multi-label sampling: for each class collect `shots` positive
    examples. Because labels are multi-label, one record can satisfy several
    classes; we de-duplicate the final index set.
    """
    rng = np.random.default_rng(seed)
    Y = dataset.targets.numpy()
    chosen = set()
    for c in range(Y.shape[1]):
        pos = np.where(Y[:, c] == 1)[0]
        rng.shuffle(pos)
        chosen.update(pos[:shots].tolist())
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Frozen-feature cache. BiomedCLIP is frozen, so encode every image ONCE and
# train the prompt learner on cached features. This makes full-data training
# fast and mirrors your extract_features.py.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_features(model, dataset, batch_size=128, num_workers=8, indices=None):
    from torch.utils.data import DataLoader, Subset

    ds = dataset if indices is None else Subset(dataset, indices)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats, tgts = [], []
    for pixel_values, targets, _ in loader:
        feats.append(model.encode_image(pixel_values).cpu())
        tgts.append(targets)
    return torch.cat(feats), torch.cat(tgts)
