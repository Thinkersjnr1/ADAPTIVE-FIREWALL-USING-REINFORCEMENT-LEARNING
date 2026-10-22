"""
Preprocessing pipeline for a full UNSW-NB15 retrain, mirroring the same
recipe as the other pipelines (per-class sampling cap, 80/20 stratified
split, SMOTE) so train.py, dqn_agent.py, environment.py, and evaluate.py
all work unchanged against this dataset.

Two genuine differences from every CICIDS-family dataset so far:

1. proto/service/state are real text categories (tcp/udp, http/dns/-,
   FIN/CON/etc), not numeric -- these need label-encoding before they can
   feed into MinMaxScaler. This is a different, coarser encoding than
   one-hot: it keeps feature count fixed and small, consistent with the
   rest of the pipeline, at the cost of implying an arbitrary ordinal
   relationship between categories that isn't really there. Acceptable
   for a dense MLP; worth noting as a modeling simplification in the
   paper if this dataset is reported in detail.

2. The benign label is 'Normal' here, and is frequently blank/NaN in the
   raw attack_cat column for benign rows (a known UNSW-NB15 quirk -- the
   dataset relies on a separate binary `label` column, 0=benign/1=attack,
   for that distinction instead). Alphabetically 'Normal' would sort
   AFTER most attack category names (Backdoor, DoS, Exploits, Fuzzers,
   Generic), which would break the "label==0 means benign" assumption
   environment.py and dashboard.py both rely on. Normalized to 'BENIGN'
   here, matching the same fix already applied in the DDoS2019 and
   CSE-CIC-IDS2018 pipelines.

Note: this dataset has 42 features, not ~78 -- DQNAgent builds its input
layer dynamically from state_size, so this just produces a differently-
shaped model. No architecture changes needed.

Source: UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv (the
pre-extracted, pre-split release). Both are pooled and re-split by this
script -- consistent with how CIC-DDoS2019 and CSE-CIC-IDS2018 are
handled, rather than reusing UNSW's own train/test split.
"""
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# EDIT THIS to wherever you downloaded/extracted the UNSW-NB15 CSVs.
DATA_FOLDER = r"C:\Users\DELL\Downloads\OneDrive_2_9-13-2026"
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE  = "UNSW_NB15_testing-set.csv"

CATEGORICAL_COLS = ["proto", "service", "state"]
SAMPLES_PER_CLASS = 5000


def load_data():
    print("Loading UNSW-NB15 CSV files...")
    train_path = os.path.join(DATA_FOLDER, TRAIN_FILE)
    test_path  = os.path.join(DATA_FOLDER, TEST_FILE)
    df_train = pd.read_csv(train_path, low_memory=False)
    df_test  = pd.read_csv(test_path, low_memory=False)
    print(f"  {TRAIN_FILE}: {len(df_train):,} rows")
    print(f"  {TEST_FILE}: {len(df_test):,} rows")
    # Pooled and re-split by this script's own 80/20 logic below --
    # consistent with how CIC-DDoS2019 and CSE-CIC-IDS2018 are handled,
    # rather than reusing UNSW's own train/test split.
    data = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total records loaded: {len(data):,}")
    return data


def clean_data(df):
    print("Cleaning data...")
    df.columns = df.columns.str.strip()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Normalize the benign label to 'BENIGN' (see module docstring for why:
    # 'Normal' doesn't reliably sort to encoded index 0, which the reward/
    # environment logic elsewhere assumes). attack_cat is often blank for
    # benign rows, so fall back to it explicitly using the reliable binary
    # `label` column (0 = benign, 1 = attack) rather than trusting
    # attack_cat's presence/absence alone.
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df.loc[(df["label"] == 0) | (df["attack_cat"].str.lower().isin(["normal", "nan", ""])), "attack_cat"] = "BENIGN"
    df = df.rename(columns={"attack_cat": "Label"})
    df = df.drop(columns=["label"])  # redundant now that Label carries this info

    # Encode the three real categorical columns. Label-encoded (not
    # one-hot) to keep feature count fixed and small -- see module
    # docstring for the tradeoff this implies.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = LabelEncoder().fit_transform(df[col])

    feature_cols = [c for c in df.columns if c != "Label"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Records after cleaning: {len(df):,} (dropped {before - len(df):,} invalid/duplicate rows)")
    return df


def encode_labels(df):
    print("Encoding labels...")
    # FIX: do not rely on LabelEncoder's alphabetical sort to put BENIGN at
    # index 0 -- that broke here specifically because UNSW-NB15 has an
    # "Analysis" attack category, and 'A' sorts before 'B'. Every reward/
    # decision check elsewhere in this codebase assumes label==0 means
    # benign; relying on alphabetical luck for that is fragile and already
    # failed once. This forces BENIGN to index 0 explicitly, with every
    # other class alphabetically ordered after it, regardless of naming.
    unique_classes = sorted(df["Label"].unique(), key=lambda c: (c != "BENIGN", c))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    df = df.copy()
    df["Label"] = df["Label"].map(class_to_idx)
    le = LabelEncoder()
    le.classes_ = np.array(unique_classes)  # keeps le.transform()/inverse_transform() consistent with this mapping
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
        joblib.dump(scaler, "models/unsw_scaler.pkl")
        joblib.dump(label_encoder, "models/unsw_label_encoder.pkl")
        with open("models/unsw_feature_order.json", "w") as f:
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