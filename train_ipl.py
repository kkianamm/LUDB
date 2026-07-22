"""
Train BiomedCLIP with the IPL prompt-learning strategy on rendered PTB-XL ECGs.

Few-shot (e.g. 16-shot):   python train_ipl.py --shots 16 --phase C
Full training:             python train_ipl.py --shots 0  --phase C

Phases (staged like IGPL):
  A  learnable prompt only (CoOp)
  B  A + text-anchor preservation to zero-shot embeddings
  C  B + instance-conditioned prompt via meta-net  (full IPL)

The BiomedCLIP backbone is frozen throughout; only the prompt learner trains.
Image features are cached once (backbone is frozen) for speed.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import config_ipl as C
from ipl import (IPLModel, build_splits, cache_features, few_shot_indices,
                 load_biomedclip, multilabel_metrics)


def get_features(model, dataset, tag, cache_dir, indices=None, force=False):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}.pt")
    if os.path.exists(path) and not force:
        blob = torch.load(path)
        return blob["feats"], blob["targets"]
    feats, targets = cache_features(model, dataset, C.BATCH_SIZE, C.NUM_WORKERS, indices)
    torch.save({"feats": feats, "targets": targets}, path)
    return feats, targets


@torch.no_grad()
def evaluate(ipl, feats, targets, device):
    ipl.eval()
    logits = []
    for i in range(0, len(feats), C.BATCH_SIZE):
        fb = feats[i:i + C.BATCH_SIZE].to(device)
        logits.append(ipl.logits_from_features(fb).cpu())
    scores = torch.sigmoid(torch.cat(logits)).numpy()
    return multilabel_metrics(targets.numpy(), scores, C.CLASSNAMES, C.THRESHOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=0, help="K-shot per class; 0 = full training")
    ap.add_argument("--phase", choices=["A", "B", "C"], default="C")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--n_ctx", type=int, default=C.N_CTX)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = args.device
    phase = C.PHASES[args.phase]
    print(f"[IPL] phase={args.phase} shots={args.shots or 'full'} "
          f"metanet={phase['use_metanet']} lambda_anchor={phase['lambda_anchor']}")

    # ---- backbone + data ----
    clip_model, preprocess, tokenizer = load_biomedclip(device)
    splits = build_splits(C.LABELS_CSV, C.IMAGE_DIR, C.CLASSNAMES, preprocess)

    tag = f"train_{args.shots}shot_seed{args.seed}" if args.shots else "train_full"
    train_idx = few_shot_indices(splits["train"], args.shots, args.seed) if args.shots else None

    ipl = IPLModel(
        clip_model, tokenizer, C.CLASSNAMES, C.CLASS_PROMPTS,
        n_ctx=args.n_ctx, ctx_init=C.CTX_INIT,
        use_metanet=phase["use_metanet"], lambda_anchor=phase["lambda_anchor"],
        device=device,
    ).to(device)

    tr_f, tr_y = get_features(ipl, splits["train"], tag, C.CACHE_DIR, train_idx)
    va_f, va_y = get_features(ipl, splits["val"], "val", C.CACHE_DIR)
    te_f, te_y = get_features(ipl, splits["test"], "test", C.CACHE_DIR)
    print(f"[IPL] train={len(tr_f)} val={len(va_f)} test={len(te_f)}")

    loader = DataLoader(TensorDataset(tr_f, tr_y), batch_size=C.BATCH_SIZE,
                        shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(ipl.trainable_parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(C.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(C.CKPT_DIR, f"ipl_{args.phase}_{tag}.pt")
    best_auroc = -1.0

    for epoch in range(args.epochs):
        ipl.train()
        running = {"cls": 0.0, "anchor": 0.0}
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            loss, logs = ipl(fb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in running:
                running[k] += logs[k]
        sched.step()

        val = evaluate(ipl, va_f, va_y, device)
        n = len(loader)
        print(f"epoch {epoch+1:03d} | cls {running['cls']/n:.4f} "
              f"anchor {running['anchor']/n:.4f} | val macroAUROC {val['macro_auroc']:.4f}")

        if val["macro_auroc"] > best_auroc:
            best_auroc = val["macro_auroc"]
            torch.save({
                "prompt_learner": ipl.prompt_learner.state_dict(),
                "classnames": C.CLASSNAMES, "class_prompts": C.CLASS_PROMPTS,
                "n_ctx": args.n_ctx, "ctx_init": C.CTX_INIT,
                "phase": args.phase, "use_metanet": phase["use_metanet"],
                "lambda_anchor": phase["lambda_anchor"], "val_macro_auroc": best_auroc,
            }, ckpt_path)

    # ---- final test with the best checkpoint ----
    ipl.prompt_learner.load_state_dict(torch.load(ckpt_path)["prompt_learner"])
    test = evaluate(ipl, te_f, te_y, device)
    print("\n=== TEST (best val ckpt) ===")
    print(f"macro AUROC : {test['macro_auroc']:.4f}")
    for c, a in test["per_class_auroc"].items():
        print(f"  {c:24s} {a:.4f}")
    print(f"macro F1 {test['macro_f1']:.4f} | micro F1 {test['micro_f1']:.4f}")
    print(f"checkpoint  : {ckpt_path}")


if __name__ == "__main__":
    main()
