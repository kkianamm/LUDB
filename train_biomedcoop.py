"""Train BiomedCoOp prompts on PTB-XL and evaluate consistently.

Examples:
    python train_biomedcoop.py --shots 16 --seed 1 --task multi
    python train_biomedcoop.py --shots 16 --seed 1 --task single
    python train_biomedcoop.py --eval-only \
        --ckpt work/checkpoints/biomedcoop_multi_16shot_seed1.pt

Single-label mode follows the official BiomedCoOp-style argmax evaluation.
Multi-label mode reports AUROC, AUPRC, and validation-tuned F1 metrics.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config as C
from biomedcoop import build_biomedcoop
from ecg_prompts import get_templates, readable_name
from evaluation import (
    evaluate_multilabel,
    evaluate_single_label,
    print_metrics,
    save_result,
    tune_multilabel_thresholds,
)
from model_utils import get_device, load_biomedclip
from prepare_data import image_path_for


class ECGImageDataset(Dataset):
    """Return a preprocessed ECG image and its multi-hot label vector."""

    def __init__(self, df, preprocess):
        self.preprocess = preprocess
        self.ids = df.index.tolist()
        self.labels = df[C.CLASSES].values.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = Image.open(image_path_for(self.ids[index])).convert("RGB")
        return self.preprocess(image), torch.from_numpy(self.labels[index])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def single_label_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[C.CLASSES].values.sum(axis=1) == 1]


def few_shot_subset(df, shots, seed, task):
    """Sample K positive examples per class from the training folds.

    For ``task='single'`` the records are first restricted to exactly one
    superclass. For ``task='multi'`` the union is deduplicated, so the total
    number of records can be smaller than ``K * n_classes``.
    """
    if task == "single":
        df = single_label_subset(df)
    if shots == 0:
        return df

    rng = np.random.RandomState(seed)
    selected = set()
    for class_name in C.CLASSES:
        positive_ids = df.index[df[class_name].values.astype(bool)].tolist()
        rng.shuffle(positive_ids)
        if len(positive_ids) < shots:
            raise ValueError(
                f"Class {class_name} has only {len(positive_ids)} eligible "
                f"records, fewer than --shots={shots}"
            )
        selected.update(positive_ids[:shots])

    return df.loc[sorted(selected)]


def multi_hot_to_single(labels):
    return labels.argmax(dim=1)


def make_scheduler(
    optimizer,
    max_epoch,
    warmup_epoch=1,
    warmup_lr=1e-5,
    base_lr=0.0025,
):
    def lr_lambda(epoch):
        if epoch < warmup_epoch:
            return warmup_lr / base_lr
        progress = (epoch - warmup_epoch) / max(1, max_epoch - warmup_epoch)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def score_split(model, df, preprocess, device, batch_size, task):
    """Return ranking scores, probabilities, and labels for a dataframe."""
    model.eval()
    dataset = ECGImageDataset(df, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=C.NUM_WORKERS,
    )

    text_features = model.prompt_learner()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    all_ranking_scores = []
    all_probabilities = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        image_features = model.image_encoder(images.type(model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine = image_features @ text_features.t()

        if task == "single":
            logits = model.logit_scale.exp() * cosine
            probabilities = logits.softmax(dim=-1)
            ranking_scores = probabilities
        else:
            probabilities = torch.sigmoid(cosine / model.ml_temperature)
            ranking_scores = cosine

        all_ranking_scores.append(ranking_scores.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_ranking_scores),
        np.concatenate(all_probabilities),
        np.concatenate(all_labels).astype(int),
    )


def evaluate_test(
    task,
    val_ranking,
    val_probabilities,
    val_labels,
    test_ranking,
    test_probabilities,
    test_labels,
    threshold_mode,
    fixed_threshold,
):
    if task == "single":
        metrics = evaluate_single_label(
            test_probabilities,
            test_labels,
            class_names=C.CLASSES,
        )
        return metrics, None

    thresholds = tune_multilabel_thresholds(
        val_labels,
        val_probabilities,
        mode=threshold_mode,
        fixed_threshold=fixed_threshold,
    )
    metrics = evaluate_multilabel(
        test_ranking,
        test_labels,
        test_probabilities,
        thresholds=thresholds,
        class_names=C.CLASSES,
    )
    return metrics, thresholds


def checkpoint_name(args) -> str:
    full = args.shots == 0
    suffix = "full" if full else f"{args.shots}shot"
    return args.ckpt or os.path.join(
        C.CKPT_DIR,
        f"biomedcoop_{args.task}_{suffix}_seed{args.seed}.pt",
    )


def apply_checkpoint_configuration(args, state):
    """Use saved architecture metadata during --eval-only.

    This avoids silently building a prompt learner with a different task,
    number of context tokens, or CSC shape before copying the learned tensor.
    Older checkpoints remain supported through defaults.
    """
    args.task = state.get("task", args.task)
    args.n_ctx = int(state.get("n_ctx", args.n_ctx))
    args.csc = bool(state.get("csc", args.csc))
    args.ctx_init = state.get("ctx_init", args.ctx_init)
    args.ml_temperature = float(
        state.get("ml_temperature", args.ml_temperature)
    )
    args.shots = int(state.get("shots", args.shots))
    args.seed = int(state.get("seed", args.seed))

    saved_classes = state.get("classes")
    if saved_classes is not None and list(saved_classes) != list(C.CLASSES):
        raise ValueError(
            f"Checkpoint classes {saved_classes} do not match config classes "
            f"{C.CLASSES}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shots",
        type=int,
        default=16,
        help="examples per class; 0 uses all records in folds 1-8",
    )
    parser.add_argument("--task", choices=["multi", "single"], default="multi")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument("--ctx-init", type=str, default="a photo of a")
    parser.add_argument("--csc", action="store_true", help="class-specific context")
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--sccm-lambda", type=float, default=0.5)
    parser.add_argument("--kdsp-lambda", type=float, default=0.25)
    parser.add_argument("--ml-temperature", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--threshold-mode",
        choices=["per-class", "global", "fixed"],
        default="per-class",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--results-dir", default=os.path.join(C.WORK_DIR, "results"))
    args = parser.parse_args()

    if args.shots < 0:
        raise ValueError("--shots must be non-negative")
    if args.ml_temperature <= 0:
        raise ValueError("--ml-temperature must be positive")

    ckpt_path = checkpoint_name(args)
    checkpoint_state = None
    if args.eval_only:
        checkpoint_state = torch.load(ckpt_path, map_location="cpu")
        apply_checkpoint_configuration(args, checkpoint_state)

    full = args.shots == 0
    epochs = args.epochs if args.epochs is not None else (20 if full else 100)
    batch_size = (
        args.batch_size if args.batch_size is not None else (32 if full else 4)
    )

    set_seed(args.seed)
    device = get_device()
    print(
        f"Device: {device} | task={args.task} | shots={args.shots} | "
        f"epochs={epochs} | batch_size={batch_size} | seed={args.seed}"
    )

    clip_model, preprocess, tokenizer = load_biomedclip(device)
    clip_model = clip_model.float()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    classnames = [readable_name(class_name) for class_name in C.CLASSES]
    class_templates = get_templates(C.CLASSES, args.n_prompts)
    model = build_biomedcoop(
        clip_model,
        tokenizer,
        classnames,
        class_templates,
        device,
        task=args.task,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        csc=args.csc,
        sccm_lambda=args.sccm_lambda,
        kdsp_lambda=args.kdsp_lambda,
        tau=args.tau,
        context_length=C.CONTEXT_LENGTH,
        ml_temperature=args.ml_temperature,
    )

    labels_df = pd.read_csv(
        os.path.join(C.WORK_DIR, "labels.csv"),
        index_col="ecg_id",
    )
    if args.limit:
        labels_df = labels_df.iloc[: args.limit]

    train_df = labels_df[labels_df.strat_fold.isin(C.TRAIN_FOLDS)]
    val_df = labels_df[labels_df.strat_fold == C.VAL_FOLD]
    test_df = labels_df[labels_df.strat_fold == C.TEST_FOLD]

    if args.task == "single":
        val_df = single_label_subset(val_df)
        test_df = single_label_subset(test_df)

    train_subset = (
        train_df
        if args.eval_only
        else few_shot_subset(train_df, args.shots, args.seed, args.task)
    )
    print(
        f"Train records used: {len(train_subset)} | "
        f"validation: {len(val_df)} | test: {len(test_df)}"
    )
    for class_name in C.CLASSES:
        print(f"  {class_name:5s} train positives: {int(train_subset[class_name].sum())}")

    if args.eval_only:
        assert checkpoint_state is not None
        saved_context = checkpoint_state["ctx"].to(device)
        current_context = model.prompt_learner.ctx
        if saved_context.shape != current_context.shape:
            raise ValueError(
                "Checkpoint context shape does not match the constructed model: "
                f"saved={tuple(saved_context.shape)}, "
                f"current={tuple(current_context.shape)}"
            )
        current_context.data.copy_(saved_context)
    else:
        if args.task == "multi":
            train_targets = train_subset[C.CLASSES].values
            positive = train_targets.sum(axis=0)
            negative = len(train_targets) - positive
            positive_weight = torch.tensor(
                negative / np.clip(positive, 1, None),
                dtype=torch.float32,
            )
            model.set_pos_weight(positive_weight)

        context = model.prompt_learner.ctx
        optimizer = torch.optim.SGD(
            [context],
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = make_scheduler(optimizer, epochs, base_lr=args.lr)

        train_dataset = ECGImageDataset(train_subset, preprocess)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=full,
            num_workers=C.NUM_WORKERS,
            generator=generator,
        )

        best_metric = -float("inf")
        best_context = None
        selection_name = "accuracy" if args.task == "single" else "macro AUROC"

        for epoch in range(epochs):
            model.prompt_learner.train()
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")

            for images, labels in progress:
                images = images.to(device)
                labels = labels.to(device)
                target = (
                    multi_hot_to_single(labels) if args.task == "single" else labels
                )

                _, loss_ce, loss_sccm, loss_kdsp = model(images, target)
                loss = loss_ce + loss_sccm + loss_kdsp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress.set_postfix(
                    loss=f"{loss.item():.3f}",
                    ce=f"{loss_ce.item():.3f}",
                    sccm=f"{loss_sccm.item():.3f}",
                    kdsp=f"{loss_kdsp.item():.3f}",
                )

            scheduler.step()
            val_ranking, val_probabilities, val_labels = score_split(
                model,
                val_df,
                preprocess,
                device,
                batch_size,
                args.task,
            )

            if args.task == "single":
                validation_metrics = evaluate_single_label(
                    val_probabilities,
                    val_labels,
                    class_names=C.CLASSES,
                )
                validation_metric = validation_metrics["accuracy"]
            else:
                validation_metrics = evaluate_multilabel(
                    val_ranking,
                    val_labels,
                    val_probabilities,
                    thresholds=args.threshold,
                    class_names=C.CLASSES,
                )
                validation_metric = validation_metrics["macro_auroc"]

            average_loss = running_loss / max(1, len(train_loader))
            print(
                f"epoch {epoch + 1}: train loss {average_loss:.4f} | "
                f"val {selection_name} {validation_metric:.4f}"
            )

            if np.isfinite(validation_metric) and validation_metric > best_metric:
                best_metric = validation_metric
                best_context = context.detach().clone()

        if best_context is None:
            raise RuntimeError("No valid validation metric was produced during training")

        model.prompt_learner.ctx.data.copy_(best_context)
        checkpoint_payload = {
            "ctx": best_context.cpu(),
            "task": args.task,
            "n_ctx": args.n_ctx,
            "csc": args.csc,
            "ctx_init": args.ctx_init,
            "ml_temperature": args.ml_temperature,
            "classes": list(C.CLASSES),
            "shots": args.shots,
            "seed": args.seed,
            "best_validation_metric": best_metric,
            "best_validation_metric_name": selection_name,
        }
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, ckpt_path)
        print(
            f"\nSaved prompt checkpoint -> {ckpt_path} "
            f"(best val {selection_name} {best_metric:.4f})"
        )

    val_ranking, val_probabilities, val_labels = score_split(
        model,
        val_df,
        preprocess,
        device,
        batch_size,
        args.task,
    )
    test_ranking, test_probabilities, test_labels = score_split(
        model,
        test_df,
        preprocess,
        device,
        batch_size,
        args.task,
    )

    metrics, thresholds = evaluate_test(
        args.task,
        val_ranking,
        val_probabilities,
        val_labels,
        test_ranking,
        test_probabilities,
        test_labels,
        args.threshold_mode,
        args.threshold,
    )

    print(f"\n=== BiomedCoOp on PTB-XL test fold: task={args.task} ===")
    print_metrics(metrics, args.task)

    suffix = "full" if full else f"{args.shots}shot"
    result_path = Path(args.results_dir) / (
        f"biomedcoop_{args.task}_{suffix}_seed{args.seed}.json"
    )
    save_result(
        result_path,
        metrics,
        metadata={
            "method": "biomedcoop",
            "task": args.task,
            "shots": args.shots,
            "seed": args.seed,
            "checkpoint": ckpt_path,
            "classes": list(C.CLASSES),
            "validation_fold": C.VAL_FOLD,
            "test_fold": C.TEST_FOLD,
            "train_records": len(train_subset),
            "validation_records": len(val_df),
            "test_records": len(test_df),
            "threshold_mode": args.threshold_mode if args.task == "multi" else None,
            "fixed_threshold": args.threshold if args.task == "multi" else None,
            "thresholds": thresholds.tolist() if thresholds is not None else None,
            "n_ctx": args.n_ctx,
            "csc": args.csc,
            "ml_temperature": args.ml_temperature,
        },
    )
    print(f"Saved metrics -> {result_path}")


if __name__ == "__main__":
    main()
