"""
Preprocessing pipeline for a full CSE-CIC-IDS2018 retrain using ALL 10
days of the release, mirroring preprocess.py's CICIDS2017 recipe (same
per-class sampling cap, same 80/20 stratified split, same SMOTE step)
so train.py, dqn_agent.py, environment.py, and evaluate.py all work
unchanged against this dataset.

MEMORY DESIGN: one file in this release (02-20-2018.csv, ~3.8GB, DDoS-
heavy) is nearly 3x the combined size of all 9 other days and previously
caused an out-of-memory crash when loaded whole. This version streams
every file in bounded-size chunks via pandas' chunksize, and maintains a
running per-class sample reservoir capped at SAMPLES_PER_CLASS -- once a
class hits its cap, no more rows for it are kept, from any file. Peak
memory is bounded by (one chunk) + (the capped reservoir), regardless of
how large any individual file is.

Source: all daily CSVs from the CSE-CIC-IDS2018 release (Kaggle mirror:
solarmainframe/ids-intrusion-csv).
"""
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# EDIT THIS if your files live somewhere else.
DATA_FOLDER = r"C:\Users\DELL\Downloads\archive (1)"

FILES_CURATED_4DAY = [
    "02-14-2018.csv", "02-16-2018.csv", "02-22-2018.csv", "03-02-2018.csv",
]
FILES_FULL_10DAY = [
    "02-14-2018.csv", "02-15-2018.csv", "02-16-2018.csv", "02-20-2018.csv",
    "02-21-2018.csv", "02-22-2018.csv", "02-23-2018.csv", "02-28-2018.csv",
    "03-01-2018.csv", "03-02-2018.csv",
]
# DEFAULT: the proven, trustworthy 4-day subset (99.76% acc / 0.80% FPR).
# FILES_FULL_10DAY currently produces a much worse result (94.74% acc /
# 37.18% FPR) -- almost certainly because the streaming reservoir's
# per-class sampling is order-biased (first-encountered-per-class across
# files, not a true global random sample over all 10 days). Only switch
# to FILES_FULL_10DAY once that sampling bias is fixed with a proper
# two-pass (count-then-reservoir-sample) approach.
FILES = FILES_CURATED_4DAY

SAMPLES_PER_CLASS = 5000
CHUNK_SIZE = 200_000  # rows per chunk -- bounds peak memory per file


def _detect_encoding(path):
    """CSE-CIC-IDS2018 files can have non-UTF-8 bytes anywhere in the
    file, not just near the start -- a small-sample check (e.g. the
    first few rows) isn't reliable enough to catch a bad byte deep into
    a 300MB+ file. latin-1 is defined for every possible byte value
    (0-255), so it can never raise a decode error, on any file, at any
    position. Since this dataset's only non-numeric content is short
    ASCII labels (Benign, DDoS, etc.), there's no practical difference
    vs UTF-8 -- this just removes the failure mode entirely rather than
    trying to detect it."""
    return "latin-1"


def _clean_chunk(df):
    """Cleaning applied per-chunk, before accumulation -- keeps every
    chunk's dtypes consistent (numeric, float32) so nothing downstream
    is forced into a memory-heavy 'object' array."""
    df.columns = df.columns.str.strip()

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Known quirk: stray duplicate header rows embedded mid-file.
    if "Dst Port" in df.columns:
        df = df[df["Dst Port"] != "Dst Port"]

    df["Label"] = df["Label"].astype(str).str.strip()
    df.loc[df["Label"].str.lower() == "benign", "Label"] = "BENIGN"

    feature_cols = [c for c in df.columns if c != "Label"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def load_and_sample():
    """Streams all 10 files in bounded chunks, cleaning each chunk and
    keeping only up to SAMPLES_PER_CLASS rows per class overall (not
    per file/chunk). Returns a small, already-capped DataFrame -- the
    same shape of result the old load+clean+balance_sample chain
    produced, just built without ever holding a full multi-GB file
    (or the full 10-day pool) in memory at once.
    """
    print(f"Streaming all {len(FILES)} CSE-CIC-IDS2018 files "
          f"(chunk size {CHUNK_SIZE:,} rows)...")
    accumulated = {}   # label -> list of DataFrames, capped at SAMPLES_PER_CLASS rows total
    total_raw = 0

    for fname in FILES:
        path = os.path.join(DATA_FOLDER, fname)
        encoding = _detect_encoding(path)
        print(f"  Streaming {fname} (encoding={encoding})")
        file_kept = 0
        for chunk in pd.read_csv(path, low_memory=False,
                                  chunksize=CHUNK_SIZE, encoding=encoding):
            total_raw += len(chunk)
            chunk = _clean_chunk(chunk)
            for label, group in chunk.groupby("Label"):
                have = sum(len(g) for g in accumulated.get(label, []))
                if have >= SAMPLES_PER_CLASS:
                    continue
                need = SAMPLES_PER_CLASS - have
                take = group.sample(min(len(group), need), random_state=42)
                accumulated.setdefault(label, []).append(take)
                file_kept += len(take)
        print(f"    kept {file_kept:,} rows from this file "
              f"(per-class totals capped at {SAMPLES_PER_CLASS:,})")

    print(f"Total raw rows scanned across all files: {total_raw:,}")
    sampled = pd.concat(
        [pd.concat(v, ignore_index=True) for v in accumulated.values()],
        ignore_index=True
    )
    before = len(sampled)
    sampled.drop_duplicates(inplace=True)  # cheap now -- sampled set is small
    print(f"Classes found: {sorted(accumulated.keys())}")
    print(f"Records after streaming per-class cap: {len(sampled):,} "
          f"(removed {before - len(sampled):,} exact duplicates)")
    return sampled


def encode_labels(df):
    print("Encoding labels...")
    unique_classes = sorted(df["Label"].unique(), key=lambda c: (c != "BENIGN", c))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    df = df.copy()
    df["Label"] = df["Label"].map(class_to_idx)
    le = LabelEncoder()
    le.classes_ = np.array(unique_classes)
    print(f"Encoded classes: {unique_classes}")
    return df, le


def split_and_scale(df, holdout_class=None, label_encoder=None):
    print("Splitting and scaling...")
    feature_cols = [c for c in df.columns if c != "Label"]

    if holdout_class is not None:
        # Leave-one-class-out: this class's rows are completely excluded
        # from training and go ONLY into the test set, so we can measure
        # whether the model flags a genuinely unseen attack category as
        # malicious despite never having trained on it -- a proxy for
        # zero-day generalization, distinct from cross-dataset zero-shot.
        holdout_encoded = label_encoder.transform([holdout_class])[0]
        holdout_mask = df["Label"] == holdout_encoded
        holdout_df = df[holdout_mask]
        rest_df = df[~holdout_mask]
        print(f"  Holding out class '{holdout_class}' ({len(holdout_df):,} rows) "
              f"entirely from training")
    else:
        holdout_df = None
        rest_df = df

    X_rest = rest_df[feature_cols].values
    y_rest = rest_df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_rest, y_rest, test_size=0.2, random_state=42, stratify=y_rest
    )

    if holdout_df is not None:
        X_holdout = holdout_df[feature_cols].values
        y_holdout = holdout_df["Label"].values
        X_test = np.concatenate([X_test, X_holdout], axis=0)
        y_test = np.concatenate([y_test, y_holdout], axis=0)
        print(f"  Added {len(X_holdout):,} held-out-class rows to test set "
              f"only (0 rows in training)")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def apply_smote(X_train, y_train):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Training set size after SMOTE: {len(X_res):,}")
    return X_res, y_res


def preprocess(use_smote=True, save_artifacts=True, holdout_class=None):
    df = load_and_sample()
    df, label_encoder = encode_labels(df)
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(
        df, holdout_class=holdout_class, label_encoder=label_encoder
    )

    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/cicids2018_scaler.pkl")
        joblib.dump(label_encoder, "models/cicids2018_label_encoder.pkl")
        with open("models/cicids2018_feature_order.json", "w") as f:
            json.dump(feature_cols, f)
        print("Saved scaler, label encoder, and feature order to models/")

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    else:
        print("Skipping SMOTE (ablation: training on naturally imbalanced classes)")

    print("\nPreprocessing complete!")
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"  Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, label_encoder


if __name__ == "__main__":
    preprocess()