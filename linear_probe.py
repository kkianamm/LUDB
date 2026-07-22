"""Train and evaluate a multi-label linear probe on frozen ECG embeddings."""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C
from evaluation import (
    evaluate_multilabel,
    print_metrics,
    save_result,
    tune_multilabel_thresholds,
)


def load_split(name):
    features = np.load(os.path.join(C.FEAT_DIR, f"X_{name}.npy"))
    labels = np.load(os.path.join(C.FEAT_DIR, f"y_{name}.npy"))
    return torch.from_numpy(features), torch.from_numpy(labels)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument(
        "--threshold-mode",
        choices=["per-class", "global", "fixed"],
        default="per-class",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", default=os.path.join(C.WORK_DIR, "results"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_features, train_labels = load_split("train")
    val_features, val_labels = load_split("val")
    test_features, test_labels = load_split("test")
    feature_dim = train_features.shape[1]
    n_classes = train_labels.shape[1]
    print(
        f"feature dim {feature_dim} | classes {n_classes} | "
        f"train {len(train_features)} val {len(val_features)} "
        f"test {len(test_features)} | seed {args.seed}"
    )

    mean = train_features.mean(0, keepdim=True)
    std = train_features.std(0, keepdim=True) + 1e-6
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std

    classifier = nn.Linear(feature_dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=C.LP_LR,
        weight_decay=C.LP_WEIGHT_DECAY,
    )

    positive = train_labels.sum(0)
    negative = len(train_labels) - positive
    positive_weight = (negative / positive.clamp(min=1)).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    train_features = train_features.to(device)
    train_labels_device = train_labels.to(device)
    val_features_device = val_features.to(device)
    test_features_device = test_features.to(device)

    best_auroc = -float("inf")
    best_state = None
    generator = torch.Generator(device=device if device == "cuda" else "cpu")
    generator.manual_seed(args.seed)

    for epoch in range(C.LP_EPOCHS):
        classifier.train()
        permutation = torch.randperm(
            len(train_features),
            generator=generator,
            device=device,
        )
        for start in range(0, len(train_features), C.BATCH_SIZE):
            indices = permutation[start : start + C.BATCH_SIZE]
            optimizer.zero_grad()
            logits = classifier(train_features[indices])
            loss = loss_function(logits, train_labels_device[indices])
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_features_device).cpu().numpy()
            val_probabilities = torch.sigmoid(
                torch.from_numpy(val_logits)
            ).numpy()

        validation_metrics = evaluate_multilabel(
            val_logits,
            val_labels.numpy(),
            val_probabilities,
            thresholds=args.threshold,
            class_names=C.CLASSES,
        )
        validation_auroc = validation_metrics["macro_auroc"]
        if np.isfinite(validation_auroc) and validation_auroc > best_auroc:
            best_auroc = validation_auroc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

        if epoch % 5 == 0 or epoch == C.LP_EPOCHS - 1:
            print(
                f"epoch {epoch:3d} val macro AUROC {validation_auroc:.4f} "
                f"(best {best_auroc:.4f})"
            )

    if best_state is None:
        raise RuntimeError("No valid validation AUROC was produced")

    classifier.load_state_dict(best_state)
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_features_device).cpu().numpy()
        test_logits = classifier(test_features_device).cpu().numpy()

    val_probabilities = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
    test_probabilities = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
    thresholds = tune_multilabel_thresholds(
        val_labels.numpy(),
        val_probabilities,
        mode=args.threshold_mode,
        fixed_threshold=args.threshold,
    )
    metrics = evaluate_multilabel(
        test_logits,
        test_labels.numpy(),
        test_probabilities,
        thresholds=thresholds,
        class_names=C.CLASSES,
    )

    print("\n=== Linear probe on frozen BiomedCLIP features: test fold ===")
    print_metrics(metrics, "multi")

    checkpoint_path = Path(C.CKPT_DIR) / f"linear_probe_seed{args.seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": mean,
            "feature_std": std,
            "classes": list(C.CLASSES),
            "seed": args.seed,
            "best_validation_macro_auroc": best_auroc,
        },
        checkpoint_path,
    )
    print(f"Saved linear head -> {checkpoint_path}")

    result_path = Path(args.results_dir) / f"linear_probe_multi_seed{args.seed}.json"
    save_result(
        result_path,
        metrics,
        metadata={
            "method": "linear_probe",
            "task": "multi",
            "shots": 0,
            "seed": args.seed,
            "checkpoint": str(checkpoint_path),
            "classes": list(C.CLASSES),
            "threshold_mode": args.threshold_mode,
            "fixed_threshold": args.threshold,
            "thresholds": thresholds.tolist(),
            "validation_fold": C.VAL_FOLD,
            "test_fold": C.TEST_FOLD,
        },
    )
    print(f"Saved metrics -> {result_path}")


if __name__ == "__main__":
    main()
