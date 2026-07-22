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

import torch
from torch.utils.data import DataLoader, TensorDataset

import config_ipl as C
from config_experiments import get_experiment
from ipl import (IPLModel, cache_features, few_shot_indices,
                 load_biomedclip, multilabel_metrics)
from data_fix import build_splits          # corrected, alias-aware label resolution
from modules import BatchContext, build_stack

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
def evaluate(ipl, stack, feats_or_loader, targets, device, raw=False):
    ipl.eval()
    dga = stack.image_module()
    # SCA needs cache init sized to feature dim / n_classes
    feat_dim = ipl.text_anchors.shape[1]
    for m in stack.test_time_modules():
        m.on_eval_start(len(C.CLASSNAMES), feat_dim, device)

    all_logits, all_tgts = [], []
    scale = ipl.logit_scale.exp().item()

    if raw:  # DGA present: iterate raw images
        loader = feats_or_loader
        for x, y, _ in loader:
            x = x.to(device)
            feats, mg_logits = dga.image_stage(x, ipl.encode_image)
            logits = mg_logits
            bctx = BatchContext(image_features=feats, logits=logits,
                                targets=y.to(device), scale=scale)
            logits = stack.refine_test_logits(bctx)
            all_logits.append(torch.sigmoid(logits).cpu()); all_tgts.append(y)
    else:    # cached-feature path
        feats = feats_or_loader
        for i in range(0, len(feats), C.BATCH_SIZE):
            fb = feats[i:i + C.BATCH_SIZE].to(device)
            yb = targets[i:i + C.BATCH_SIZE]
            logits = ipl.logits_from_features(fb)
            bctx = BatchContext(image_features=fb, logits=logits,
                                targets=yb.to(device), scale=scale)
            logits = stack.refine_test_logits(bctx)
            all_logits.append(torch.sigmoid(logits).cpu()); all_tgts.append(yb)

    scores = torch.cat(all_logits).numpy()
    tgts = torch.cat(all_tgts).numpy() if raw else targets.numpy()
    return multilabel_metrics(tgts, scores, C.CLASSNAMES, C.THRESHOLD)


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
    args = ap.parse_args()
    torch.manual_seed(args.seed)
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
    else:
        tag = f"train_{args.shots}shot_seed{args.seed}" if args.shots else "train_full"
        tr_f, tr_y = get_cached_features(ipl, splits["train"], tag, train_idx)
        va_f, va_y = get_cached_features(ipl, splits["val"], "val")
        te_f, te_y = get_cached_features(ipl, splits["test"], "test")
        train_loader = DataLoader(TensorDataset(tr_f, tr_y), batch_size=C.BATCH_SIZE, shuffle=True)

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

    # ---- Test ----
    if raw:
        test = evaluate(ipl, stack, test_loader, None, device, raw=True)
    else:
        test = evaluate(ipl, stack, te_f, te_y, device, raw=False)
    print(f"\n=== TEST [{args.exp}] ===  macro AUROC {test['macro_auroc']:.4f}  "
          f"macro F1 {test['macro_f1']:.4f}")
    for c, a in test["per_class_auroc"].items():
        print(f"  {c:24s} {a:.4f}")
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
