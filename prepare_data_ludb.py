"""
LUDB adapter for the BiomedCLIP x ECG (ecgclip-IGPL-extended) repo.

The repo was written for PTB-XL. This script makes the *Lobachevsky University
Database* (LUDB, 200 records) look exactly like what the rest of the pipeline
expects, so `zero_shot_eval.py`, `extract_features.py`, `linear_probe.py`,
`finetune_clip.py`, and the whole `ipl/` (IPL/IGPL phases A/B/C) stack run
unchanged.

It does three things:
  1. Reads LUDB's `ludb.csv` (the aggregated per-patient diagnosis table).
  2. Maps the free-text diagnosis columns into a multi-label 0/1 target matrix
     -- either the 5 PTB-XL-style superclasses (default) or an 8-class
     LUDB-native taxonomy that keeps rhythm / ectopy / pacing.
  3. Assigns a stratified 10-fold split (folds 1-8 train, 9 val, 10 test, to
     match the repo's defaults) and renders every record to an ECG-paper PNG
     using the repo's own `ecg_to_image.render_to_file`.

Output: `work/labels.csv` (indexed by `ecg_id`, with `strat_fold`, a `split`
column, a synthesized English `report`, a pipe-joined `superclasses` string,
and one 0/1 column per class) + `work/images/<ecg_id:05d>.png`.

Usage
-----
    # point DATA_DIR at the LUDB root (the folder containing ludb.csv + data/)
    export DATA_DIR=/path/to/ludb/1.0.1
    export WORK_DIR=./work_ludb          # keep separate from any PTB-XL work dir

    python prepare_data_ludb.py                    # super5, render all
    python prepare_data_ludb.py --labelset ludb8   # richer 8-class taxonomy
    python prepare_data_ludb.py --no-render        # labels only (fast)
    python prepare_data_ludb.py --limit 20         # quick smoke test

Then run the existing pipeline exactly as the READMEs describe, e.g.:
    python zero_shot_eval.py --task multi
    python extract_features.py && python linear_probe.py
    python train_ipl.py --shots 0 --phase C
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config as C  # the repo's config.py; DATA_DIR/WORK_DIR/IMG_DIR come from here

# --------------------------------------------------------------------------- #
# LUDB diagnosis columns (exact headers in ludb.csv v1.0.1)
# --------------------------------------------------------------------------- #
COL_RHYTHM = "Rhythms"
COL_AXIS = "Electric axis of the heart"
COL_CD = "Conduction abnormalities"
COL_ECTO = "Extrasystolies"
COL_HYP = "Hypertrophies"
COL_PACE = "Cardiac pacing"
COL_ISCH = "Ischemia"
COL_STTC = "Non-specific repolarization abnormalities"
COL_OTHER = "Other states"

# Rhythms that are considered non-pathological for the purpose of "normal ECG".
SINUS_RHYTHMS = {
    "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
    "sinus arrhythmia", "irregular sinus rhythm",
}


def _items(value) -> list[str]:
    """A LUDB cell may hold several findings separated by newlines. Return them
    lowercased and stripped; empty cell -> []."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [x.strip().lower() for x in str(value).replace("\r", "\n").split("\n")
            if x.strip()]


def _scalar(value) -> str:
    """Collapse a (possibly multiline) cell to a single clean lowercase string."""
    return " ".join(_items(value))


# --------------------------------------------------------------------------- #
# Label-set definitions
# --------------------------------------------------------------------------- #
# super5: mirrors PTB-XL's NORM/MI/STTC/CD/HYP so config.py / config_ipl.py and
# every prompt in the repo apply verbatim. Note the semantic remap:
#   MI  <- LUDB "Ischemia" column (STEMI / ischemia / scar / undefined-NSTEMI)
#   STTC<- LUDB "Non-specific repolarization abnormalities"
#   CD  <- LUDB "Conduction abnormalities"
#   HYP <- LUDB "Hypertrophies" (incl. atrial/ventricular overload)
# Rhythm abnormalities, ectopy and pacing are NOT representable in super5; such
# records become all-zero rows (honest information loss -- reported at the end).
SUPER5 = ["NORM", "MI", "STTC", "CD", "HYP"]

# ludb8: keeps LUDB's clinical richness. Requires matching edits to config.py /
# config_ipl.py (the script prints the exact snippet to paste).
LUDB8 = ["NORM", "AFIB_FL", "CD", "ECTOPY", "HYP", "MI", "STTC", "PACE"]

LUDB8_DESCRIPTIONS = {
    "NORM": "normal ECG",
    "AFIB_FL": "atrial fibrillation or flutter",
    "CD": "conduction disturbance",
    "ECTOPY": "atrial or ventricular extrasystoles",
    "HYP": "cardiac hypertrophy",
    "MI": "myocardial ischemia or infarction",
    "STTC": "ST/T wave change",
    "PACE": "cardiac pacing",
}


def _base_flags(row) -> dict:
    """Compute the raw presence flags shared by both label sets."""
    rhythms = _items(row.get(COL_RHYTHM))
    sinus_only = len(rhythms) > 0 and all(r in SINUS_RHYTHMS for r in rhythms)
    return {
        "hyp": bool(_items(row.get(COL_HYP))),
        "cd": bool(_items(row.get(COL_CD))),
        "sttc": bool(_items(row.get(COL_STTC))),
        "mi": bool(_items(row.get(COL_ISCH))),
        "ectopy": bool(_items(row.get(COL_ECTO))),
        "pace": bool(_items(row.get(COL_PACE))),
        "rhythm_abn": (len(rhythms) > 0 and not sinus_only),
        "sinus_only": sinus_only,
    }


def map_super5(row) -> dict:
    f = _base_flags(row)
    any_abn = f["hyp"] or f["cd"] or f["sttc"] or f["mi"] or f["ectopy"] \
        or f["pace"] or f["rhythm_abn"]
    return {
        "NORM": int(f["sinus_only"] and not any_abn),
        "MI": int(f["mi"]),
        "STTC": int(f["sttc"]),
        "CD": int(f["cd"]),
        "HYP": int(f["hyp"]),
    }


def map_ludb8(row) -> dict:
    f = _base_flags(row)
    any_abn = f["hyp"] or f["cd"] or f["sttc"] or f["mi"] or f["ectopy"] \
        or f["pace"] or f["rhythm_abn"]
    return {
        "NORM": int(f["sinus_only"] and not any_abn),
        "AFIB_FL": int(f["rhythm_abn"]),
        "CD": int(f["cd"]),
        "ECTOPY": int(f["ectopy"]),
        "HYP": int(f["hyp"]),
        "MI": int(f["mi"]),
        "STTC": int(f["sttc"]),
        "PACE": int(f["pace"]),
    }


MAPPERS = {"super5": map_super5, "ludb8": map_ludb8}
CLASS_SETS = {"super5": SUPER5, "ludb8": LUDB8}


# --------------------------------------------------------------------------- #
# Human-readable report (used for captions in finetune_clip.py --caption report,
# and generally a better text signal than raw labels).
# --------------------------------------------------------------------------- #
def build_report(row) -> str:
    parts = []
    rhythm = _scalar(row.get(COL_RHYTHM))
    if rhythm:
        parts.append(rhythm)
    for col in (COL_CD, COL_HYP, COL_ISCH, COL_STTC, COL_ECTO, COL_PACE, COL_OTHER):
        for it in _items(row.get(col)):
            parts.append(it)
    if not parts:
        return "twelve lead electrocardiogram"
    return "a twelve lead ecg showing " + "; ".join(dict.fromkeys(parts))


# --------------------------------------------------------------------------- #
# Coverage-aware split (deterministic). LUDB has no official split and is tiny
# (200 records), so a naive fold assignment easily leaves a rare class with
# ZERO positives in the val or test fold. Both `data_fix._guard` and
# `train_compose.get_cached_features` abort in that case, so the split MUST put
# at least `min_per_class` positives of every class into val AND test.
#
# Returns fold ids 1..10 to match the repo convention: folds 1-8 = train,
# 9 = val, 10 = test. val/test can be made larger than 1/10 to cut the noise
# that comes with ~20-record folds.
# --------------------------------------------------------------------------- #
def _pick_covered(candidates, Y, target_n, min_per_class, rng) -> set:
    """Choose ~target_n indices from `candidates`, front-loading rare classes so
    each class reaches `min_per_class` positives if at all possible."""
    chosen: set = set()
    counts = np.zeros(Y.shape[1], dtype=int)
    # rarest class first
    for c in np.argsort(Y[candidates].sum(axis=0)):
        pos = [i for i in candidates if Y[i, c] == 1 and i not in chosen]
        rng.shuffle(pos)
        for i in pos:
            if counts[c] >= min_per_class or len(chosen) >= target_n:
                break
            chosen.add(i)
            counts += Y[i].astype(int)
    # fill the rest at random
    rest = [i for i in candidates if i not in chosen]
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= target_n:
            break
        chosen.add(i)
    return chosen


def assign_splits(Y: np.ndarray, seed=42, val_frac=0.15, test_frac=0.15,
                  min_per_class=1) -> np.ndarray:
    n, n_cls = Y.shape
    totals = Y.sum(axis=0).astype(int)
    need = 3 * min_per_class  # >=min_per_class in each of train/val/test
    thin = {int(c): int(totals[c]) for c in range(n_cls) if totals[c] < need}
    if thin:
        raise ValueError(
            f"These class indices have too few positives to guarantee "
            f">={min_per_class}/split: {thin}. Options: use --labelset super5, "
            f"lower --min-per-class, or merge/drop the rare class."
        )
    rng = np.random.default_rng(seed)
    all_idx = list(range(n))
    test = _pick_covered(all_idx, Y, round(test_frac * n), min_per_class, rng)
    remaining = [i for i in all_idx if i not in test]
    val = _pick_covered(remaining, Y, round(val_frac * n), min_per_class, rng)

    folds = np.zeros(n, dtype=int)
    for i in test:
        folds[i] = 10
    for i in val:
        folds[i] = 9
    train = [i for i in all_idx if folds[i] == 0]
    rng.shuffle(train)
    for k, i in enumerate(train):          # spread train across folds 1-8
        folds[i] = (k % 8) + 1
    return folds


def ludb_filename(ecg_id: int) -> str:
    """WFDB record path relative to DATA_DIR, e.g. 'data/1' (no extension)."""
    return os.path.join("data", str(int(ecg_id)))


def image_path_for(ecg_id: int) -> str:
    # identical convention to the repo's prepare_data.image_path_for
    return os.path.join(C.IMG_DIR, f"{int(ecg_id):05d}.png")


def render_all(df: pd.DataFrame, data_dir: str, fs: int, limit=None):
    from ecg_to_image import load_signal, render_to_file  # imported lazily
    from tqdm import tqdm
    ids = df.index.tolist()[: limit] if limit else df.index.tolist()
    for ecg_id in tqdm(ids, desc="Rendering ECG images"):
        out = image_path_for(ecg_id)
        if os.path.exists(out):
            continue
        signal, _ = load_signal(data_dir, ludb_filename(ecg_id))
        render_to_file(signal, fs, out)


def print_config_snippet(labelset: str):
    classes = CLASS_SETS[labelset]
    if labelset == "super5":
        print("\n[config] super5 matches the repo defaults -- no config edits needed.")
        return
    desc = LUDB8_DESCRIPTIONS
    print("\n" + "=" * 70)
    print("Paste into config.py (replace CLASSES / CLASS_DESCRIPTIONS):")
    print("=" * 70)
    print("CLASSES = " + repr(classes))
    print("CLASS_DESCRIPTIONS = {")
    for c in classes:
        print(f"    {c!r}: {desc[c]!r},")
    print("}")
    print("\nAnd in config_ipl.py set CLASSNAMES to the readable names:")
    print("CLASSNAMES = " + repr([desc[c] for c in classes]))
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelset", choices=list(MAPPERS), default="super5")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-per-class", type=int, default=1,
                    help="min positives per class guaranteed in EACH of val/test")
    ap.add_argument("--sampling-rate", type=int, default=None,
                    help="override fs for rendering; LUDB is natively 500 Hz")
    args = ap.parse_args()

    data_dir = C.DATA_DIR
    fs = args.sampling_rate or getattr(C, "SAMPLING_RATE", 500)
    csv_path = os.path.join(data_dir, "ludb.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Point DATA_DIR at the LUDB root "
            f"(the folder that contains ludb.csv and the data/ directory)."
        )

    df = pd.read_csv(csv_path)  # pandas handles the quoted multiline cells
    df["ID"] = df["ID"].astype(int)
    df = df.set_index("ID")
    if args.limit:
        df = df.iloc[: args.limit]

    mapper = MAPPERS[args.labelset]
    classes = CLASS_SETS[args.labelset]

    label_rows = [mapper(row) for _, row in df.iterrows()]
    Y = np.array([[r[c] for c in classes] for r in label_rows], dtype=np.float32)
    folds = assign_splits(Y, seed=args.seed, val_frac=args.val_frac,
                          test_frac=args.test_frac, min_per_class=args.min_per_class)

    meta = pd.DataFrame(index=df.index)
    meta.index.name = "ecg_id"
    meta["filename"] = [ludb_filename(i) for i in df.index]
    meta["strat_fold"] = folds
    meta["split"] = np.where(folds <= 8, "train",
                             np.where(folds == 9, "val", "test"))
    meta["report"] = [build_report(row) for _, row in df.iterrows()]
    for j, c in enumerate(classes):
        meta[c] = Y[:, j].astype(int)
    meta["superclasses"] = [
        "|".join(c for c in classes if r[c]) for r in label_rows
    ]

    os.makedirs(C.WORK_DIR, exist_ok=True)
    out_csv = os.path.join(C.WORK_DIR, "labels.csv")
    meta.to_csv(out_csv)

    # ---- report ----------------------------------------------------------- #
    print(f"LUDB root         : {data_dir}")
    print(f"Label set         : {args.labelset}  ({len(classes)} classes)")
    print(f"Records           : {len(meta)}  "
          f"(train {int((folds<=8).sum())} / val {int((folds==9).sum())} "
          f"/ test {int((folds==10).sum())})")
    print("Per-class positives (total | train | val | test):")
    tr_m, va_m, te_m = folds <= 8, folds == 9, folds == 10
    for j, c in enumerate(classes):
        print(f"  {c:8s}: {int(Y[:, j].sum()):3d} | "
              f"{int(Y[tr_m, j].sum()):3d} | {int(Y[va_m, j].sum()):3d} | "
              f"{int(Y[te_m, j].sum()):3d}")
    if min(Y[va_m].sum(0).min(), Y[te_m].sum(0).min()) == 0:
        print("  WARNING: a class is empty in val/test -> data_fix._guard will "
              "abort. Raise --min-per-class or --val/--test-frac.")
    all_zero = int((Y.sum(axis=1) == 0).sum())
    multi = int((Y.sum(axis=1) > 1).sum())
    print(f"Multi-label rows  : {multi}   |   all-zero rows: {all_zero}"
          + ("  <- records whose only findings are unrepresentable in super5"
             if args.labelset == "super5" and all_zero else ""))
    print(f"Saved labels      : {out_csv}")

    if not args.no_render:
        render_all(meta, data_dir, fs, limit=args.limit)
        print(f"Images            : {C.IMG_DIR}  (fs={fs} Hz)")

    print_config_snippet(args.labelset)


if __name__ == "__main__":
    main()
