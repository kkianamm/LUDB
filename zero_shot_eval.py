"""Zero-shot evaluation of BiomedCLIP on PTB-XL.

Examples:
    python zero_shot_eval.py --task multi
    python zero_shot_eval.py --task single
    python zero_shot_eval.py --task multi --ckpt work/checkpoints/biomedclip_ft.pt

For the multi-label protocol, fold 9 is used only to select F1 thresholds and
fold 10 is used once for the final report. AUROC/AUPRC use raw cosine scores;
threshold-dependent metrics use sigmoid probabilities.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import config as C
from evaluation import (
    evaluate_multilabel,
    evaluate_single_label,
    print_metrics,
    save_result,
    tune_multilabel_thresholds,
)
from model_utils import build_class_text_features, get_device, load_biomedclip
from prepare_data import image_path_for


@torch.no_grad()
def encode_images(model, preprocess, device, ecg_ids, batch_size=C.BATCH_SIZE):
    """Return L2-normalised image embeddings for the supplied ECG IDs."""
    features = []
    batch = []
    for ecg_id in tqdm(ecg_ids, desc="Encoding images"):
        image = Image.open(image_path_for(ecg_id)).convert("RGB")
        batch.append(preprocess(image))
        if len(batch) == batch_size:
            tensor = torch.stack(batch).to(device)
            embedding = model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            features.append(embedding.cpu())
            batch = []

    if batch:
        tensor = torch.stack(batch).to(device)
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        features.append(embedding.cpu())

    if not features:
        raise ValueError("No images were available for evaluation")
    return torch.cat(features, dim=0)


def filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[C.CLASSES].values.sum(axis=1) == 1
    return df.loc[mask]


def compute_scores(model, image_features, text_features, task, temperature):
    """Return ranking scores and probabilities.

    For multi-label evaluation, raw cosine similarity is used for ranking
    metrics and sigmoid(cosine / temperature) for threshold metrics.
    """
    cosine = image_features.to(text_features.device) @ text_features.t()

    if task == "single":
        logits = model.logit_scale.exp() * cosine
        probabilities = logits.softmax(dim=-1)
        return probabilities.cpu().numpy(), probabilities.cpu().numpy()

    probabilities = torch.sigmoid(cosine / temperature)
    return cosine.cpu().numpy(), probabilities.cpu().numpy()


def default_run_name(task: str, checkpoint: str | None) -> str:
    checkpoint_name = "pretrained" if checkpoint is None else Path(checkpoint).stem
    return f"zero_shot_{task}_{checkpoint_name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["multi", "single"], default="multi")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="optional fine-tuned BiomedCLIP checkpoint",
    )
    parser.add_argument(
        "--ml-temperature",
        type=float,
        default=0.5,
        help="sigmoid temperature for multi-label probabilities",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["per-class", "global", "fixed"],
        default="per-class",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", default=os.path.join(C.WORK_DIR, "results"))
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if args.ml_temperature <= 0:
        raise ValueError("--ml-temperature must be positive")

    device = get_device()
    print(f"Device: {device} | task={args.task}")

    labels_df = pd.read_csv(
        os.path.join(C.WORK_DIR, "labels.csv"),
        index_col="ecg_id",
    )
    val_df = labels_df[labels_df.strat_fold == C.VAL_FOLD]
    test_df = labels_df[labels_df.strat_fold == C.TEST_FOLD]

    if args.task == "single":
        val_df = filter_single_label(val_df)
        test_df = filter_single_label(test_df)

    if args.limit:
        val_df = val_df.iloc[: args.limit]
        test_df = test_df.iloc[: args.limit]

    print(f"Validation records: {len(val_df)} | test records: {len(test_df)}")

    model, preprocess, tokenizer = load_biomedclip(device, ckpt_path=args.ckpt)
    model.eval()
    text_features = build_class_text_features(model, tokenizer, device)

    test_image_features = encode_images(
        model,
        preprocess,
        device,
        test_df.index.tolist(),
    )
    test_ranking, test_probabilities = compute_scores(
        model,
        test_image_features,
        text_features,
        args.task,
        args.ml_temperature,
    )
    test_labels = test_df[C.CLASSES].values.astype(int)

    if args.task == "single":
        metrics = evaluate_single_label(
            test_probabilities,
            test_labels,
            class_names=C.CLASSES,
        )
        thresholds = None
    else:
        val_image_features = encode_images(
            model,
            preprocess,
            device,
            val_df.index.tolist(),
        )
        _, val_probabilities = compute_scores(
            model,
            val_image_features,
            text_features,
            args.task,
            args.ml_temperature,
        )
        val_labels = val_df[C.CLASSES].values.astype(int)
        thresholds = tune_multilabel_thresholds(
            val_labels,
            val_probabilities,
            mode=args.threshold_mode,
            fixed_threshold=args.threshold,
        )
        metrics = evaluate_multilabel(
            test_ranking,
            test_labels,
            test_probabilities,
            thresholds=thresholds,
            class_names=C.CLASSES,
        )

    print("\n=== Zero-shot BiomedCLIP on PTB-XL test fold ===")
    print_metrics(metrics, args.task)

    run_name = args.run_name or default_run_name(args.task, args.ckpt)
    output_path = Path(args.results_dir) / f"{run_name}.json"
    save_result(
        output_path,
        metrics,
        metadata={
            "method": "zero_shot",
            "task": args.task,
            "checkpoint": args.ckpt,
            "classes": list(C.CLASSES),
            "validation_fold": C.VAL_FOLD,
            "test_fold": C.TEST_FOLD,
            "validation_records": len(val_df),
            "test_records": len(test_df),
            "threshold_mode": args.threshold_mode if args.task == "multi" else None,
            "fixed_threshold": args.threshold if args.task == "multi" else None,
            "ml_temperature": args.ml_temperature if args.task == "multi" else None,
            "thresholds": thresholds.tolist() if thresholds is not None else None,
        },
    )
    print(f"Saved metrics -> {output_path}")


if __name__ == "__main__":
    main()
