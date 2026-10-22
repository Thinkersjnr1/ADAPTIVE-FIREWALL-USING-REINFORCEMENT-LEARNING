"""
Preprocessing pipeline for a full CIC-DDoS2019 retrain, mirroring
preprocess.py's CICIDS2017 recipe exactly (same per-class sampling cap,
same 80/20 stratified split, same SMOTE step) so train.py, dqn_agent.py,
environment.py, and evaluate.py all work unchanged against this dataset.

Source: the per-attack-type *.parquet files from the cleaned CIC-DDoS2019
Kaggle release (dhoogla/cicddos2019). ALL *.parquet files found in
DATA_FOLDER are pooled together and re-split by this script -- this does
NOT preserve the original dataset's training-day/testing-day attack-type
split, since this pipeline creates its own held-out test set the same
way it does for CICIDS2017 (consistency across datasets matters more
here than reusing someone else's split).

Note: this dataset has 77 features (no Destination Port, but has
Protocol), one fewer than CICIDS2017's 78. This is fine -- DQNAgent
builds its input layer dynamically from state_size at construction time,
so a model trained here will simply have a differently-shaped input
layer. No architecture changes needed.
"""
import pandas as pd
import numpy as np
import os
import glob
import json
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# EDIT THIS to wherever you downloaded/extracted the CIC-DDoS2019 parquet
# files, e.g. r"C:\Users\DELL\Downloads\archive"
DATA_FOLDER = r"C:\Users\DELL\Downloads\archive"
SAMPLES_PER_CLASS = 5000


def load_data():
    print("Loading CIC-DDoS2019 parquet files...")
    files = glob.glob(os.path.join(DATA_FOLDER, "*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No .parquet files found in {DATA_FOLDER} -- edit DATA_FOLDER "
            f"at the top of this file to point at your downloaded dataset."
        )
    frames = []
    for f in files:
        print(f"  Reading {os.path.basename(f)}")
        frames.append(pd.read_parquet(f))
    data = pd.concat(frames, ignore_index=True)
    print(f"Total records loaded: {len(data):,}")
    return data


def clean_data(df):
    print("Cleaning data...")
    df.columns = df.columns.str.strip()
    # Standardize the benign label to CICIDS2017's convention ('BENIGN',
    # uppercase) in case any downstream code has a hardcoded comparison.
    df["Label"] = df["Label"].astype(str).str.strip()
    df.loc[df["Label"].str.lower() == "benign", "Label"] = "BENIGN"
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Records after cleaning: {len(df):,}")
    return df


def encode_labels(df):
    print("Encoding labels...")
    unique_classes = sorted(df["Label"].unique(), key=lambda c: (c != "BENIGN", c))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    df = df.copy()
    df["Label"] = df["Label"].map(class_to_idx)
    le = LabelEncoder()
    le.classes_ = np.array(unique_classes)
    print(f"Classes found: {unique_classes}")
    return df, le


def balance_sample(df):
    print(f"Sampling up to {SAMPLES_PER_CLASS} rows per class...")
    groups = []
    for label, group in df.groupby("Label"):
        n = min(len(group), SAMPLES_PER_CLASS)
        groups.append(group.sample(n, random_state=42))
    sampled = pd.concat(groups, ignore_index=True)
    print(f"Records after sampling: {len(sampled):,}")
    return sampled


def split_and_scale(df):
    print("Splitting and scaling...")
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    if holdout_class is not None:
        raise NotImplementedError(
            "Leave-one-class-out (--holdout-class) is currently only "
            "implemented for --dataset cicids2018."
        )
    df = load_data()
    df = clean_data(df)
    df, label_encoder = encode_labels(df)
    df = balance_sample(df)
    X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(df)

    if save_artifacts:
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/ddos2019_scaler.pkl")
        joblib.dump(label_encoder, "models/ddos2019_label_encoder.pkl")
        with open("models/ddos2019_feature_order.json", "w") as f:
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