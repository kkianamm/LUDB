"""Shared evaluation utilities for ECGCLIP/PTB-XL experiments.

The module intentionally separates:
1. ranking scores used by AUROC/AUPRC; and
2. probabilities used by threshold-dependent metrics such as F1.

Thresholds must be selected on the validation fold and then frozen before
calling ``evaluate_multilabel`` on the test fold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _as_2d_float(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {result.shape}")
    return result


def _check_same_shape(*arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Arrays must have identical shapes, got {sorted(shapes)}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def tune_multilabel_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    mode: str = "per-class",
    fixed_threshold: float = 0.5,
) -> np.ndarray:
    """Select F1-maximising thresholds using validation data only.

    Args:
        y_true: Binary matrix of shape ``[n_samples, n_classes]``.
        probabilities: Probability matrix with the same shape.
        mode: ``per-class``, ``global`` or ``fixed``.
        fixed_threshold: Used when ``mode='fixed'`` and as a fallback for
            degenerate validation classes.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = _as_2d_float(probabilities, "probabilities")
    _check_same_shape(y_true, probabilities)

    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("fixed_threshold must be between 0 and 1")

    n_classes = y_true.shape[1]
    if mode == "fixed":
        return np.full(n_classes, fixed_threshold, dtype=np.float64)

    if mode == "global":
        candidates = np.linspace(0.01, 0.99, 99)
        values = [
            f1_score(
                y_true,
                probabilities >= threshold,
                average="macro",
                zero_division=0,
            )
            for threshold in candidates
        ]
        return np.full(n_classes, candidates[int(np.argmax(values))])

    if mode != "per-class":
        raise ValueError("mode must be one of: per-class, global, fixed")

    thresholds = np.full(n_classes, fixed_threshold, dtype=np.float64)
    for class_index in range(n_classes):
        target = y_true[:, class_index]
        score = probabilities[:, class_index]

        # F1 threshold tuning is undefined if validation contains only one
        # target value. Keep the declared fallback in that case.
        if np.unique(target).size < 2:
            continue

        precision, recall, candidates = precision_recall_curve(target, score)
        if candidates.size == 0:
            continue

        f1_values = (
            2.0 * precision[:-1] * recall[:-1]
            / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        )
        thresholds[class_index] = candidates[int(np.nanargmax(f1_values))]

    return thresholds


def evaluate_multilabel(
    ranking_scores: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | float = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-label classifier.

    ``ranking_scores`` should be raw cosine similarities or logits. AUROC and
    AUPRC are computed from these values. ``probabilities`` are used only for
    threshold-dependent metrics.
    """
    ranking_scores = _as_2d_float(ranking_scores, "ranking_scores")
    probabilities = _as_2d_float(probabilities, "probabilities")
    y_true = np.asarray(y_true, dtype=np.int64)
    _check_same_shape(ranking_scores, probabilities, y_true)

    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(index) for index in range(n_classes)]
    if len(class_names) != n_classes:
        raise ValueError("class_names length must match the number of classes")

    if np.isscalar(thresholds):
        threshold_array = np.full(n_classes, float(thresholds))
    else:
        threshold_array = np.asarray(thresholds, dtype=np.float64)
        if threshold_array.shape != (n_classes,):
            raise ValueError(
                f"thresholds must have shape ({n_classes},), "
                f"got {threshold_array.shape}"
            )

    predictions = (probabilities >= threshold_array[None, :]).astype(np.int64)
    valid_roc = np.array(
        [np.unique(y_true[:, index]).size == 2 for index in range(n_classes)]
    )
    valid_pr = np.array([y_true[:, index].sum() > 0 for index in range(n_classes)])

    def macro_auroc() -> float:
        if not valid_roc.any():
            return float("nan")
        return float(
            roc_auc_score(
                y_true[:, valid_roc],
                ranking_scores[:, valid_roc],
                average="macro",
            )
        )

    def micro_auroc() -> float:
        if not valid_roc.any():
            return float("nan")
        return float(
            roc_auc_score(
                y_true[:, valid_roc],
                ranking_scores[:, valid_roc],
                average="micro",
            )
        )

    def macro_auprc() -> float:
        if not valid_pr.any():
            return float("nan")
        return float(
            average_precision_score(
                y_true[:, valid_pr],
                ranking_scores[:, valid_pr],
                average="macro",
            )
        )

    def micro_auprc() -> float:
        if not valid_pr.any():
            return float("nan")
        return float(
            average_precision_score(
                y_true[:, valid_pr],
                ranking_scores[:, valid_pr],
                average="micro",
            )
        )

    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, class_name in enumerate(class_names):
        target = y_true[:, index]
        prediction = predictions[:, index]
        score = ranking_scores[:, index]

        tn, fp, fn, tp = confusion_matrix(
            target,
            prediction,
            labels=[0, 1],
        ).ravel()

        per_class[str(class_name)] = {
            "auroc": (
                float(roc_auc_score(target, score))
                if np.unique(target).size == 2
                else None
            ),
            "auprc": (
                float(average_precision_score(target, score))
                if target.sum() > 0
                else None
            ),
            "f1": float(f1_score(target, prediction, zero_division=0)),
            "sensitivity": _safe_divide(tp, tp + fn),
            "specificity": _safe_divide(tn, tn + fp),
            "threshold": float(threshold_array[index]),
            "support": int(target.sum()),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    top1 = ranking_scores.argmax(axis=1)
    top1_hit = float(np.mean(y_true[np.arange(len(y_true)), top1] == 1))

    return {
        "macro_auroc": macro_auroc(),
        "micro_auroc": micro_auroc(),
        "macro_auprc": macro_auprc(),
        "micro_auprc": micro_auprc(),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "top1_hit_accuracy": top1_hit,
        "label_cardinality": float(y_true.sum(axis=1).mean()),
        "thresholds": threshold_array.tolist(),
        "per_class": per_class,
    }


def evaluate_single_label(
    scores: np.ndarray,
    y_true: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the BiomedCoOp-compatible single-label protocol."""
    scores = _as_2d_float(scores, "scores")
    target = np.asarray(y_true)

    if target.ndim == 2:
        if target.shape != scores.shape:
            raise ValueError(
                "One-hot y_true must have the same shape as scores, got "
                f"{target.shape} and {scores.shape}"
            )
        if not np.all(target.sum(axis=1) == 1):
            raise ValueError("Single-label evaluation requires one positive class per row")
        target_index = target.argmax(axis=1)
    elif target.ndim == 1:
        target_index = target.astype(np.int64)
    else:
        raise ValueError("y_true must be a 1-D index vector or 2-D one-hot matrix")

    n_classes = scores.shape[1]
    if class_names is None:
        class_names = [str(index) for index in range(n_classes)]
    if len(class_names) != n_classes:
        raise ValueError("class_names length must match the number of classes")

    prediction = scores.argmax(axis=1)
    labels = np.arange(n_classes)
    matrix = confusion_matrix(target_index, prediction, labels=labels)
    normalised = confusion_matrix(
        target_index,
        prediction,
        labels=labels,
        normalize="true",
    )

    per_class_accuracy = {}
    for index, class_name in enumerate(class_names):
        class_total = matrix[index].sum()
        per_class_accuracy[str(class_name)] = (
            float(matrix[index, index] / class_total) if class_total else None
        )

    accuracy = float(accuracy_score(target_index, prediction))
    return {
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "macro_f1": float(
            f1_score(
                target_index,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_index, prediction)
        ),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": normalised.tolist(),
    }


def print_metrics(metrics: dict[str, Any], task: str) -> None:
    """Print the common headline metrics in a stable format."""
    if task == "single":
        print(f"accuracy          : {metrics['accuracy']:.4f}")
        print(f"error rate        : {metrics['error_rate']:.4f}")
        print(f"macro F1          : {metrics['macro_f1']:.4f}")
        print(f"balanced accuracy : {metrics['balanced_accuracy']:.4f}")
        for class_name, value in metrics["per_class_accuracy"].items():
            text = "nan" if value is None else f"{value:.4f}"
            print(f"  {class_name:5s} accuracy: {text}")
        return

    for key, label in (
        ("macro_auroc", "macro AUROC"),
        ("micro_auroc", "micro AUROC"),
        ("macro_auprc", "macro AUPRC"),
        ("micro_auprc", "micro AUPRC"),
        ("macro_f1", "macro F1"),
        ("micro_f1", "micro F1"),
        ("top1_hit_accuracy", "top-1 hit acc"),
    ):
        print(f"{label:16s}: {metrics[key]:.4f}")

    print("per-class metrics:")
    for class_name, values in metrics["per_class"].items():
        auroc = values["auroc"]
        auprc = values["auprc"]
        auroc_text = "nan" if auroc is None else f"{auroc:.4f}"
        auprc_text = "nan" if auprc is None else f"{auprc:.4f}"
        print(
            f"  {class_name:5s} AUROC={auroc_text} AUPRC={auprc_text} "
            f"F1={values['f1']:.4f} sens={values['sensitivity']:.4f} "
            f"spec={values['specificity']:.4f} "
            f"thr={values['threshold']:.4f}"
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_result(
    path: str | Path,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Save metrics and experiment metadata as standards-compliant JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": _json_safe(metadata),
        "metrics": _json_safe(metrics),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path
