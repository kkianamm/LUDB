"""
Load PTB-XL metadata, aggregate SCP codes into the 5 diagnostic superclasses,
apply the official fold split, and (optionally) pre-render every ECG to a PNG.

Run once:
    python prepare_data.py                 # build metadata + render ALL images
    python prepare_data.py --limit 500     # quick test on 500 records
    python prepare_data.py --no-render     # metadata only (no images)
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from ecg_to_image import load_signal, render_to_file


def load_metadata(data_dir=C.DATA_DIR):
    """Return a DataFrame indexed by ecg_id with a `superclasses` column
    (list of NORM/MI/STTC/CD/HYP) and the official `strat_fold`.
    """
    df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"),
                     index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # scp_statements maps each SCP code to a diagnostic_class superclass.
    scp = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]

    def to_superclasses(scp_dict):
        out = set()
        for code in scp_dict:
            if code in scp.index and isinstance(scp.loc[code, "diagnostic_class"], str):
                out.add(scp.loc[code, "diagnostic_class"])
        return sorted(out)

    df["superclasses"] = df.scp_codes.apply(to_superclasses)
    return df


def multilabel_matrix(df, classes=C.CLASSES):
    """Return an (n, 5) 0/1 numpy array of superclass labels."""
    y = np.zeros((len(df), len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for row, sc in enumerate(df["superclasses"]):
        for c in sc:
            if c in idx:
                y[row, idx[c]] = 1.0
    return y


def split_indices(df):
    """Boolean masks for train / val / test using strat_fold."""
    train = df.strat_fold.isin(C.TRAIN_FOLDS)
    val = df.strat_fold == C.VAL_FOLD
    test = df.strat_fold == C.TEST_FOLD
    return train, val, test


def image_path_for(ecg_id):
    return os.path.join(C.IMG_DIR, f"{int(ecg_id):05d}.png")


def render_all(df, data_dir=C.DATA_DIR, limit=None):
    """Render every record to WORK_DIR/images/<ecg_id>.png (skips existing)."""
    ids = df.index.tolist()
    if limit:
        ids = ids[:limit]
    for ecg_id in tqdm(ids, desc="Rendering ECG images"):
        out = image_path_for(ecg_id)
        if os.path.exists(out):
            continue
        filename = df.loc[ecg_id, C.FILENAME_COL]
        signal, _ = load_signal(data_dir, filename)
        render_to_file(signal, C.SAMPLING_RATE, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N records (quick test)")
    ap.add_argument("--no-render", action="store_true",
                    help="build metadata only, do not render images")
    args = ap.parse_args()

    print(f"Loading metadata from {C.DATA_DIR} ...")
    df = load_metadata()
    if args.limit:
        df = df.iloc[:args.limit]

    train, val, test = split_indices(df)
    print(f"Records: {len(df)} | train {train.sum()} "
          f"val {val.sum()} test {test.sum()}")

    y = multilabel_matrix(df)
    for i, c in enumerate(C.CLASSES):
        print(f"  {c:5s}: {int(y[:, i].sum())} positive")

    # Persist the label/split table so downstream scripts don't re-parse.
    meta = df[[C.FILENAME_COL, "strat_fold", "report"]].copy()
    meta["superclasses"] = df["superclasses"].apply(lambda x: "|".join(x))
    for i, c in enumerate(C.CLASSES):
        meta[c] = y[:, i]
    meta.to_csv(os.path.join(C.WORK_DIR, "labels.csv"))
    print(f"Saved labels -> {os.path.join(C.WORK_DIR, 'labels.csv')}")

    if not args.no_render:
        render_all(df, limit=args.limit)
        print(f"Images in {C.IMG_DIR}")


if __name__ == "__main__":
    main()
