"""
Unified trainer: IPL base + optional SCA / MERGETUNE / DGA, selected by
experiment name. One command runs any of the 8 combinations or the baselines.

    python train_compose.py --exp ipl_sca_dga --shots 0
    python train_compose.py --exp ipl_mergetune --shots 16
    python train_compose.py --exp zeroshot            # baseline, eval only

Orchestration
-------------
* Feature caching is GATED: if any active module wants raw pixels (DGA), we train
  on raw images and recompute features each step through the frozen tower;
  otherwise we keep IPL's fast cached-feature path unchanged.
* Stage 1  : IPL (+ DGA if present) trained jointly (losses summed).
* Stage 2  : MERGETUNE continued fine-tuning (if present), post-hoc on stage-1 params.
* Eval     : DGA multi-granularity fusion (if present) then SCA test-time
             refinement (if present, outermost).
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config_ipl as C
from config_experiments import get_experiment
from ipl import (IPLModel, cache_features, few_shot_indices,
                 load_biomedclip)
from data_fix import build_splits          # corrected, alias-aware label resolution
from modules import BatchContext, build_stack
from evaluation import (
    evaluate_multilabel,
    print_metrics,
    save_result,
    tune_multilabel_thresholds,
)


# ------------------------------------------------------------------ #
# FIXED REPORT LABELS (output only)
#
# The model/dataset still use C.CLASSNAMES. These names are used only by
# evaluate_multilabel/print_metrics so the report is always displayed as:
# NORM, MI, STTC, CD, HYP.
# ------------------------------------------------------------------ #
REPORT_LABEL_FIX_VERSION = "NORM-MI-STTC-CD-HYP-v1"
_EXPECTED_REPORT_LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
_PTBXL_DISPLAY_LABELS = {
    "norm": "NORM",
    "normal": "NORM",
    "normal ecg": "NORM",
    "normal electrocardiogram": "NORM",
    "mi": "MI",
    "myocardial infarction": "MI",
    "sttc": "STTC",
    "st t wave change": "STTC",
    "st t changes": "STTC",
    "st t change": "STTC",
    "cd": "CD",
    "conduction disturbance": "CD",
    "conduction disorder": "CD",
    "hyp": "HYP",
    "hypertrophy": "HYP",
}


def metric_display_names(class_names):
    """Return canonical PTB-XL report labels without reordering columns.

    An unknown or duplicated class raises an error instead of silently printing
    the long internal label, making the output-label fix easy to verify.
    """
    display_names = []
    for class_name in class_names:
        normalized = str(class_name).strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = " ".join(normalized.split())
        if normalized not in _PTBXL_DISPLAY_LABELS:
            raise ValueError(
                f"No PTB-XL display-label mapping for class {class_name!r} "
                f"(normalized={normalized!r})"
            )
        display_names.append(_PTBXL_DISPLAY_LABELS[normalized])

    if tuple(display_names) != _EXPECTED_REPORT_LABELS:
        raise ValueError(
            "Unexpected PTB-XL class order. Expected internal classes to map to "
            f"{list(_EXPECTED_REPORT_LABELS)}, but got {display_names}. "
            f"C.CLASSNAMES={list(class_names)}"
        )
    return display_names

# ------------------------------------------------------------------ #
# raw-pixel preprocess for the DGA path: resize to input_size, [0,1], NO normalise
# (VR adds the program in pixel space then applies BiomedCLIP normalisation).
# ------------------------------------------------------------------ #
def raw_preprocess(input_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),                      # [0,1], un-normalised
    ])


def get_cached_features(model, dataset, tag, indices=None):
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    path = os.path.join(C.CACHE_DIR, f"{tag}.pt")
    if os.path.exists(path):
        blob = torch.load(path)
        # Guard against a stale cache written with the old all-zero targets.
        if blob["targets"].sum() > 0 and (blob["targets"].sum(0) > 0).all():
            return blob["feats"], blob["targets"]
        print(f"[compose] stale/degenerate cache {path} -> rebuilding")
    feats, targets = cache_features(model, dataset, C.BATCH_SIZE, C.NUM_WORKERS, indices)
    assert targets.sum() > 0, f"targets for '{tag}' are all zero after resolve; check LABEL_ALIASES"
    torch.save({"feats": feats, "targets": targets}, path)
    return feats, targets


@torch.no_grad()
def score_split(ipl, stack, feats_or_loader, targets, device, raw=False):
    """Return ranking scores, probabilities, and labels for one split.

    ``evaluation.evaluate_multilabel`` uses ranking scores for AUROC/AUPRC and
    probabilities for threshold-dependent metrics. Keeping both mirrors the
    BiomedCoOp evaluation path and produces the same complete metric report.
    """
    ipl.eval()
    dga = stack.image_module()

    # SCA keeps evaluation-time state, so reset it independently for every split.
    feat_dim = ipl.text_anchors.shape[1]
    for module in stack.test_time_modules():
        module.on_eval_start(len(C.CLASSNAMES), feat_dim, device)

    all_ranking_scores, all_probabilities, all_targets = [], [], []
    scale = ipl.logit_scale.exp().item()

    if raw:  # DGA present: iterate raw images.
        for batch in feats_or_loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            y_device = y.to(device)
            feats, logits = dga.image_stage(x, ipl.encode_image)
            bctx = BatchContext(
                image_features=feats,
                logits=logits,
                targets=y_device,
                scale=scale,
            )
            logits = stack.refine_test_logits(bctx)
            all_ranking_scores.append(logits.cpu())
            all_probabilities.append(torch.sigmoid(logits).cpu())
            all_targets.append(y.cpu())
    else:  # Cached-feature path.
        feats = feats_or_loader
        for i in range(0, len(feats), C.BATCH_SIZE):
            fb = feats[i:i + C.BATCH_SIZE].to(device)
            yb = targets[i:i + C.BATCH_SIZE]
            logits = ipl.logits_from_features(fb)
            bctx = BatchContext(
                image_features=fb,
                logits=logits,
                targets=yb.to(device),
                scale=scale,
            )
            logits = stack.refine_test_logits(bctx)
            all_ranking_scores.append(logits.cpu())
            all_probabilities.append(torch.sigmoid(logits).cpu())
            all_targets.append(yb.cpu())

    return (
        torch.cat(all_ranking_scores).numpy(),
        torch.cat(all_probabilities).numpy(),
        torch.cat(all_targets).numpy().astype(int),
    )


def train_stage(ipl, stack, stage, loader, epochs, lr, device, raw=False):
    """One training stage. stage=1 IPL(+DGA); stage=2 MERGETUNE continued."""
    params = list(ipl.trainable_parameters()) if stage == 1 else []
    params += stack.all_trainable_parameters(stage)
    if not params:
        return
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dga = stack.image_module()
    scale = ipl.logit_scale.exp().item()

    for epoch in range(epochs):
        ipl.train()
        for batch in loader:
            if raw:
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                feats, mg_logits = dga.image_stage(x, ipl.encode_image)
                logits = mg_logits
            else:
                fb, y = batch
                feats, y = fb.to(device), y.to(device)
                logits = ipl.logits_from_features(feats)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.float())
            loss = loss + ipl.lambda_anchor * ipl.anchor_loss()

            bctx = BatchContext(image_features=feats, logits=logits, targets=y, scale=scale)
            extra, _ = stack.extra_losses(bctx, stage)
            loss = loss + extra

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        print(f"  [stage {stage}] epoch {epoch+1}/{epochs} loss {loss.item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name from config_experiments.EXPERIMENTS")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--threshold-mode",
        choices=["per-class", "global", "fixed"],
        default="per-class",
        help="Tune F1 thresholds on validation data or use a fixed threshold.",
    )
    try:
        default_threshold = float(C.THRESHOLD)
    except (TypeError, ValueError):
        default_threshold = 0.5
    ap.add_argument("--threshold", type=float, default=default_threshold)
    default_results_dir = getattr(
        C,
        "RESULTS_DIR",
        os.path.join(os.path.dirname(C.CKPT_DIR) or "work", "results"),
    )
    ap.add_argument("--results-dir", default=default_results_dir)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device

    exp = get_experiment(args.exp)
    print(f"[compose] experiment={args.exp}  cfg={exp}")

    clip_model, preprocess, tokenizer = load_biomedclip(device)

    # IPL base (built even for base_only baselines; its prompt then stays frozen/at-init)
    phase = C.PHASES.get(exp["ipl_phase"], C.PHASES["C"]) if exp.get("ipl_phase") else \
            dict(use_metanet=False, lambda_anchor=0.0)
    ipl = IPLModel(clip_model, tokenizer, C.CLASSNAMES, C.CLASS_PROMPTS,
                   n_ctx=C.N_CTX, ctx_init=C.CTX_INIT,
                   use_metanet=phase["use_metanet"], lambda_anchor=phase["lambda_anchor"],
                   device=device).to(device)

    stack = build_stack(exp, clip_model, tokenizer, C.CLASSNAMES,
                        C.CLASS_PROMPTS, ipl, device)
    raw = stack.wants_raw_pixels
    print(f"[compose] modules={[m.name for m in stack.modules]}  raw_pixels={raw}")

    # ---- data ----
    splits = build_splits(C.LABELS_CSV, C.IMAGE_DIR, C.CLASSNAMES, preprocess)
    train_idx = few_shot_indices(splits["train"], args.shots, args.seed) if args.shots else None

    if raw:
        dga_cfg = exp["modules"]["dga"]
        rp = raw_preprocess(dga_cfg.get("input_size", 192))
        raw_splits = build_splits(C.LABELS_CSV, C.IMAGE_DIR, C.CLASSNAMES, rp)
        from torch.utils.data import Subset
        tr = raw_splits["train"] if train_idx is None else Subset(raw_splits["train"], train_idx)
        train_loader = DataLoader(tr, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=C.NUM_WORKERS)
        val_eval, test_eval = raw_splits["val"], raw_splits["test"]
        val_loader = DataLoader(val_eval, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)
        test_loader = DataLoader(test_eval, batch_size=C.BATCH_SIZE, num_workers=C.NUM_WORKERS)
        train_count, val_count, test_count = len(tr), len(val_eval), len(test_eval)
    else:
        tag = f"train_{args.shots}shot_seed{args.seed}" if args.shots else "train_full"
        tr_f, tr_y = get_cached_features(ipl, splits["train"], tag, train_idx)
        va_f, va_y = get_cached_features(ipl, splits["val"], "val")
        te_f, te_y = get_cached_features(ipl, splits["test"], "test")
        train_loader = DataLoader(TensorDataset(tr_f, tr_y), batch_size=C.BATCH_SIZE, shuffle=True)
        train_count, val_count, test_count = len(tr_y), len(va_y), len(te_y)

    # ---- baselines that don't train the IPL prompt ----
    if exp.get("base_only"):
        for p in ipl.prompt_learner.parameters():
            p.requires_grad_(False)      # keep prompt at ctx_init ~ hand prompt

    # ---- Stage 1: IPL (+ DGA) ----
    if not exp.get("base_only") or raw:   # raw+base_only still trains DGA's VR
        train_stage(ipl, stack, stage=1, loader=train_loader,
                    epochs=args.epochs, lr=args.lr, device=device, raw=raw)

    # ---- Stage 2: MERGETUNE continued fine-tuning ----
    if stack.train_modules(stage=2):
        mt_cfg = exp["modules"]["mergetune"]
        train_stage(ipl, stack, stage=2, loader=train_loader,
                    epochs=mt_cfg.get("epochs", 10), lr=mt_cfg.get("lr", args.lr),
                    device=device, raw=raw)

    # ---- Save ----
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(C.CKPT_DIR, f"{args.exp}_{'raw' if raw else 'cache'}_seed{args.seed}.pt")
    torch.save({"exp": args.exp, "prompt_learner": ipl.prompt_learner.state_dict(),
                "modules": {m.name: m.state_dict() for m in stack.modules}}, ckpt)

    # ---- Validation + test evaluation ----
    # Validation probabilities are used only to choose thresholds. Final metrics
    # are computed once on the untouched test fold.
    if raw:
        val_ranking, val_probabilities, val_labels = score_split(
            ipl, stack, val_loader, None, device, raw=True
        )
        test_ranking, test_probabilities, test_labels = score_split(
            ipl, stack, test_loader, None, device, raw=True
        )
    else:
        val_ranking, val_probabilities, val_labels = score_split(
            ipl, stack, va_f, va_y, device, raw=False
        )
        test_ranking, test_probabilities, test_labels = score_split(
            ipl, stack, te_f, te_y, device, raw=False
        )

    thresholds = tune_multilabel_thresholds(
        val_labels,
        val_probabilities,
        mode=args.threshold_mode,
        fixed_threshold=args.threshold,
    )
    report_class_names = metric_display_names(C.CLASSNAMES)
    print(f"[compose] report-label fix={REPORT_LABEL_FIX_VERSION}")
    print(f"[compose] metric labels={report_class_names}")
    metrics = evaluate_multilabel(
        test_ranking,
        test_labels,
        test_probabilities,
        thresholds=thresholds,
        class_names=report_class_names,
    )

    display_name = (
        "Zero-shot BiomedCLIP"
        if args.exp.lower() in {"zeroshot", "zero_shot", "zero-shot"}
        else args.exp
    )
    print(f"\n=== {display_name} on PTB-XL test fold ===")
    print_metrics(metrics, "multi")

    shot_suffix = "full" if args.shots == 0 else f"{args.shots}shot"
    result_path = Path(args.results_dir) / (
        f"compose_{args.exp}_{shot_suffix}_seed{args.seed}.json"
    )
    save_result(
        result_path,
        metrics,
        metadata={
            "method": args.exp,
            "task": "multi",
            "shots": args.shots,
            "seed": args.seed,
            "checkpoint": ckpt,
            "classes": report_class_names,
            "internal_classnames": list(C.CLASSNAMES),
            "modules": [module.name for module in stack.modules],
            "raw_pixels": raw,
            "train_records": train_count,
            "validation_records": val_count,
            "test_records": test_count,
            "threshold_mode": args.threshold_mode,
            "fixed_threshold": args.threshold,
            "thresholds": thresholds.tolist(),
        },
    )
    print(f"Saved metrics -> {result_path}")
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
