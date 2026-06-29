import pandas as pd
import numpy as np
import os
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
    le = LabelEncoder()
    df = df.copy()
    df["Label"] = le.fit_transform(df["Label"])
    print(f"Classes found: {list(le.classes_)}")
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
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def apply_smote(X_train, y_train):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"Training set size after SMOTE: {len(X_res):,}")
    return X_res, y_res

def preprocess():
    df = load_data()
    df = clean_data(df)
    df, label_encoder = encode_labels(df)
    df = balance_sample(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    X_train, y_train = apply_smote(X_train, y_train)

    print("\nPreprocessing complete!")
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"  Features         : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, label_encoder

if __name__ == "__main__":
    preprocess()