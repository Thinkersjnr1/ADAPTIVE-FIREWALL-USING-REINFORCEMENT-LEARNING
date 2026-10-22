"""
Zero-shot evaluation: score CIC-DDoS2019 traffic using a model trained
ONLY on CICIDS2017, with no retraining. Tests whether the learned policy
transfers to attack traffic from an entirely different capture -- distinct
from the full-retrain experiments, which test whether the training recipe
itself reproduces on a new dataset.

Usage:
    python -m src.zeroshot_eval --model models/dqn_firewall_seed0.keras \
        --data "C:\\Users\\DELL\\Downloads\\archive\\Syn-testing.parquet"

Multiple --data files are combined into one evaluation set.
"""
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# DDoS2019 (cleaned Kaggle release) -> CICIDS2017 column name mapping.
# Only columns whose NAME changed between CICFlowMeter versions are
# listed here; identically-named columns need no entry.
RENAME_MAP = {
    "Fwd Packets Length Total": "Total Length of Fwd Packets",
    "Bwd Packets Length Total": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",
    "Avg Packet Size": "Average Packet Size",
    "Init Fwd Win Bytes": "Init_Win_bytes_forward",
    "Init Bwd Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Packets": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}


def align_columns(df, feature_order, dest_port_mean):
    df = df.rename(columns=RENAME_MAP)

    # 'Protocol' exists in DDoS2019 but was never a CICIDS2017 training
    # feature -- drop it.
    if "Protocol" in df.columns:
        df = df.drop(columns=["Protocol"])

    # 'Fwd Header Length.1' was a duplicate of 'Fwd Header Length' in the
    # original CICIDS2017 CSVs (a known CICFlowMeter quirk that produced
    # two identically-valued columns). DDoS2019 only has it once, so this
    # recovers the duplicate losslessly by copying it -- not an
    # approximation, it's literally what that column always was.
    if "Fwd Header Length" in df.columns:
        df["Fwd Header Length.1"] = df["Fwd Header Length"]

    # 'Destination Port' doesn't exist in this DDoS2019 release at all.
    # Imputed with CICIDS2017's own training-set mean (saved by
    # preprocess.py) rather than an arbitrary constant like zero.
    df["Destination Port"] = dest_port_mean

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Still missing required columns after alignment: {missing}")

    return df[feature_order]  # exact order the model's input layer expects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the trained .keras model")
    parser.add_argument("--data", required=True, nargs="+",
                         help="One or more DDoS2019 parquet/csv files to evaluate on")
    args = parser.parse_args()

    print("=" * 55)
    print("  ZERO-SHOT EVALUATION: CIC-DDoS2019 on CICIDS2017-trained model")
    print(f"  Model: {args.model}")
    print("=" * 55)

    scaler = joblib.load("models/cicids2017_scaler.pkl")
    with open("models/cicids2017_feature_order.json") as f:
        feature_order = json.load(f)
    with open("models/cicids2017_feature_stats.json") as f:
        stats = json.load(f)
    dest_port_mean = stats["Destination Port_train_mean"]

    frames = []
    for path in args.data:
        print(f"  Loading {path}")
        if path.endswith(".parquet"):
            frames.append(pd.read_parquet(path))
        else:
            frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"  Total rows loaded: {len(df):,}")

    # Ground truth: 0 = ALLOW (Benign), 1 = BLOCK (any attack label).
    # Defined directly from DDoS2019's own string labels -- NOT reusing
    # CICIDS2017's LabelEncoder, since DDoS2019 contains attack classes
    # (e.g. 'Syn') that encoder never saw during training.
    y_true = (df["Label"].astype(str).str.strip().str.lower() != "benign").astype(int).values

    X = df.drop(columns=["Label"])
    X = X.replace([np.inf, -np.inf], np.nan)
    before = len(X)
    valid_mask = ~X.isna().any(axis=1)
    X, y_true = X[valid_mask], y_true[valid_mask]
    print(f"  Dropped {before - len(X):,} rows with inf/NaN values")

    X = align_columns(X, feature_order, dest_port_mean)
    X_scaled = scaler.transform(X.values)  # transform ONLY -- never re-fit

    model = tf.keras.models.load_model(args.model)
    q_values = model(X_scaled, training=False).numpy()
    y_pred = np.argmax(q_values, axis=1)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 55)
    print("  ZERO-SHOT PERFORMANCE METRICS")
    print("=" * 55)
    print(f"  Accuracy           : {acc*100:.2f}%")
    print(f"  Precision          : {prec*100:.2f}%")
    print(f"  Recall             : {rec*100:.2f}%")
    print(f"  F1-Score           : {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print("=" * 55)
    print("  CONFUSION MATRIX")
    print("=" * 55)
    print(f"  True Negatives  (Benign correctly allowed) : {tn:,}")
    print(f"  False Positives (Benign wrongly blocked)   : {fp:,}")
    print(f"  False Negatives (Attack missed)            : {fn:,}")
    print(f"  True Positives  (Attack correctly blocked) : {tp:,}")
    print("=" * 55)
    print("\n  Reminder: this dataset's curator notes a trivial single-rule")
    print("  classifier already reaches ~0.991 AUROC on this data (a known")
    print("  shortcut-separability issue). Report this result alongside")
    print("  that caveat rather than as unqualified generalization evidence.")


if __name__ == "__main__":
    main()