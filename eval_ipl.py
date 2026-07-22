"""
Evaluate a trained IPL checkpoint on the PTB-XL test split.

    python eval_ipl.py --ckpt work/checkpoints/ipl_C_train_full.pt
"""

import argparse
import os

import torch

import config_ipl as C
from ipl import IPLModel, build_splits, cache_features, load_biomedclip, multilabel_metrics


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu")
    clip_model, preprocess, tokenizer = load_biomedclip(args.device)

    ipl = IPLModel(
        clip_model, tokenizer, blob["classnames"], blob["class_prompts"],
        n_ctx=blob["n_ctx"], ctx_init=blob["ctx_init"],
        use_metanet=blob["use_metanet"], lambda_anchor=blob["lambda_anchor"],
        device=args.device,
    ).to(args.device)
    ipl.prompt_learner.load_state_dict(blob["prompt_learner"])
    ipl.eval()

    splits = build_splits(C.LABELS_CSV, C.IMAGE_DIR, blob["classnames"], preprocess)
    feats, targets = cache_features(ipl, splits[args.split], C.BATCH_SIZE, C.NUM_WORKERS)

    logits = []
    for i in range(0, len(feats), C.BATCH_SIZE):
        logits.append(ipl.logits_from_features(feats[i:i + C.BATCH_SIZE].to(args.device)).cpu())
    scores = torch.sigmoid(torch.cat(logits)).numpy()

    m = multilabel_metrics(targets.numpy(), scores, blob["classnames"], C.THRESHOLD)
    print(f"=== {args.split} ===")
    print(f"macro AUROC : {m['macro_auroc']:.4f}")
    for c, a in m["per_class_auroc"].items():
        print(f"  {c:24s} {a:.4f}")
    print(f"macro F1 {m['macro_f1']:.4f} | micro F1 {m['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
