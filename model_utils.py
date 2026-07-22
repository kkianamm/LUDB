"""Shared helpers for loading BiomedCLIP and building text embeddings."""
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

import config as C


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_biomedclip(device=None, ckpt_path=None):
    """Load BiomedCLIP (model, preprocess, tokenizer).

    If `ckpt_path` is given, load fine-tuned weights on top of the base model.
    """
    device = device or get_device()
    model, preprocess = create_model_from_pretrained(C.BIOMEDCLIP_HF)
    tokenizer = get_tokenizer(C.BIOMEDCLIP_HF)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded fine-tuned weights from {ckpt_path}")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def build_class_text_features(model, tokenizer, device,
                              classes=C.CLASSES,
                              descriptions=C.CLASS_DESCRIPTIONS,
                              templates=C.PROMPT_TEMPLATES):
    """Return an (n_classes, dim) tensor of L2-normalised, prompt-ensembled
    text embeddings — one row per diagnostic superclass.
    """
    feats = []
    for c in classes:
        prompts = [t.format(descriptions[c]) for t in templates]
        tokens = tokenizer(prompts, context_length=C.CONTEXT_LENGTH).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.mean(dim=0)                      # ensemble across templates
        emb = emb / emb.norm()                     # renormalise
        feats.append(emb)
    return torch.stack(feats, dim=0)               # (n_classes, dim)
