"""Multi-label metrics — macro AUROC is the PTB-XL field standard."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def multilabel_metrics(y_true, y_score, classnames, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    per_class = {}
    aurocs = []
    for i, c in enumerate(classnames):
        if len(np.unique(y_true[:, i])) < 2:      # AUROC undefined for a constant column
            per_class[c] = float("nan")
            continue
        a = roc_auc_score(y_true[:, i], y_score[:, i])
        per_class[c] = a
        aurocs.append(a)

    y_pred = (y_score >= threshold).astype(int)
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "per_class_auroc": per_class,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }
