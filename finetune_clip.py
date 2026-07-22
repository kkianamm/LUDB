"""
Contrastive (CLIP-style) fine-tuning of BiomedCLIP on PTB-XL.

We pair each ECG image with a text caption and train with the standard symmetric
InfoNCE loss, exactly how CLIP/BiomedCLIP was originally trained — this is what
"training BiomedCLIP on this dataset" means in the CLIP sense.

Captions (choose with --caption):
    label  (default) : an English sentence built from the record's superclasses,
                       e.g. "a 12-lead ECG showing myocardial infarction and
                       ST/T wave change". Robust because BiomedCLIP's text tower
                       (PubMedBERT) is English; many raw PTB-XL reports are German.
    report           : the raw `report` string from ptbxl_database.csv.

    python finetune_clip.py                       # 5 epochs, vision tower only
    python finetune_clip.py --caption report
    python finetune_clip.py --limit 3000 --epochs 2

After training, evaluate the adapted model:
    python zero_shot_eval.py --ckpt work/checkpoints/biomedclip_ft.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as C
from model_utils import load_biomedclip, get_device
from prepare_data import image_path_for


def build_label_caption(superclass_str):
    """'MI|STTC' -> 'a 12-lead ECG showing myocardial infarction and ST/T wave change'"""
    codes = [c for c in str(superclass_str).split("|") if c]
    if not codes:
        return "a 12-lead ECG"
    phrases = [C.CLASS_DESCRIPTIONS.get(c, c) for c in codes]
    if len(phrases) == 1:
        body = phrases[0]
    else:
        body = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"a 12-lead ECG showing {body}"


class ECGTextDataset(Dataset):
    def __init__(self, df, preprocess, tokenizer, caption_mode="label"):
        self.df = df
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.caption_mode = caption_mode
        self.ecg_ids = df.index.tolist()

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, i):
        ecg_id = self.ecg_ids[i]
        img = Image.open(image_path_for(ecg_id)).convert("RGB")
        img = self.preprocess(img)

        if self.caption_mode == "report":
            text = str(self.df.loc[ecg_id, "report"]) or "a 12-lead ECG"
        else:
            text = build_label_caption(self.df.loc[ecg_id, "superclasses"])

        tokens = self.tokenizer([text], context_length=C.CONTEXT_LENGTH)[0]
        return img, tokens


def clip_contrastive_loss(image_features, text_features, logit_scale):
    """Standard symmetric InfoNCE loss (image<->text) over the batch."""
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(image_features), device=image_features.device)
    return 0.5 * (F.cross_entropy(logits_per_image, labels) +
                  F.cross_entropy(logits_per_text, labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", choices=["label", "report"], default="label")
    ap.add_argument("--epochs", type=int, default=C.FT_EPOCHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    device = get_device()
    print(f"Device: {device}")

    model, preprocess, tokenizer = load_biomedclip(device)
    model.train()

    # Optionally freeze the text tower (PubMedBERT) and tune the vision tower.
    if C.FT_FREEZE_TEXT:
        for name, p in model.named_parameters():
            if name.startswith("text"):
                p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    labels_df = pd.read_csv(os.path.join(C.WORK_DIR, "labels.csv"),
                            index_col="ecg_id")
    if args.limit:
        labels_df = labels_df.iloc[:args.limit]
    train_df = labels_df[labels_df.strat_fold.isin(C.TRAIN_FOLDS)]
    print(f"Training pairs: {len(train_df)}  (captions: {args.caption})")

    ds = ECGTextDataset(train_df, preprocess, tokenizer, args.caption)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=True,
                    num_workers=C.NUM_WORKERS, drop_last=True)

    opt = torch.optim.AdamW(trainable, lr=C.FT_LR, weight_decay=C.FT_WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(args.epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for images, texts in pbar:
            images, texts = images.to(device), texts.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                img_f = model.encode_image(images)
                txt_f = model.encode_text(texts)
                loss = clip_contrastive_loss(
                    img_f, txt_f, model.logit_scale.exp())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # keep logit_scale in the sane CLIP range
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"epoch {epoch+1} mean loss {running/len(dl):.4f}")

    out = os.path.join(C.CKPT_DIR, "biomedclip_ft.pt")
    torch.save(model.state_dict(), out)
    print(f"Saved fine-tuned model -> {out}")
    print("Evaluate with:  python zero_shot_eval.py --ckpt " + out)


if __name__ == "__main__":
    main()
