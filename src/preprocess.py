import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_FOLDER = "data"
SAMPLES_PER_CLASS = 5000

def load_data():
    print("Loading CSV files...")
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    frames = []
    for f in files:
        print(f"  Reading {f}")
        df = pd.read_csv(os.path.join(DATA_FOLDER, f), low_memory=False)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    print(f"Total records loaded: {len(data):,}")
    return data

def clean_data(df):
    print("Cleaning data...")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Records after cleaning: {len(df):,}")
    return df

def encode_labels(df):
    print("Encoding labels...")
    # FIX: force BENIGN to index 0 explicitly rather than relying on
    # alphabetical sort (which broke on UNSW-NB15's "Analysis" class).
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

    # Captured for cross-dataset zero-shot use: some external datasets
    # (e.g. the cleaned CIC-DDoS2019 release) don't include Destination
    # Port at all, so we impute it with this dataset's own training-set
    # mean rather than an arbitrary constant like zero.
    dest_port_idx = feature_cols.index("Destination Port")
    dest_port_train_mean = float(X_train[:, dest_port_idx].mean())

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols, dest_port_train_mean

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
    X_train, X_test, y_train, y_test, scaler, feature_cols, dest_port_mean = split_and_scale(df)

    if save_artifacts:
        # Persist everything needed to correctly evaluate a DIFFERENT
        # dataset against this trained model later: the exact fitted
        # scaler (must be reused, never refit, for a valid comparison),
        # the exact feature column order the model expects, and a
        # fallback value for any feature a new dataset might be missing.
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/cicids2017_scaler.pkl")
        joblib.dump(label_encoder, "models/cicids2017_label_encoder.pkl")
        with open("models/cicids2017_feature_order.json", "w") as f:
            json.dump(feature_cols, f)
        with open("models/cicids2017_feature_stats.json", "w") as f:
            json.dump({"Destination Port_train_mean": dest_port_mean}, f)
        print("Saved scaler, label encoder, feature order, and feature stats to models/")

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