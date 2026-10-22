import pandas as pd

DATA_FOLDER = r"C:\Users\DELL\Downloads\OneDrive_2_9-13-2026"
df_train = pd.read_csv(f"{DATA_FOLDER}\\UNSW_NB15_training-set.csv", low_memory=False)
df_test = pd.read_csv(f"{DATA_FOLDER}\\UNSW_NB15_testing-set.csv", low_memory=False)
df = pd.concat([df_train, df_test], ignore_index=True)
if "id" in df.columns:
    df = df.drop(columns=["id"])

benign = df[df["label"] == 0]
attack = df[df["label"] == 1]

print(f"Raw benign rows: {len(benign):,}")
print(f"Unique benign rows (all columns, excluding id): {benign.drop_duplicates().shape[0]:,}")
print(f"Raw attack rows: {len(attack):,}")
print(f"Unique attack rows (all columns, excluding id): {attack.drop_duplicates().shape[0]:,}")

# Also check duplicates on feature columns only (excluding attack_cat/label,
# which is closer to what actually happens in our pipeline after attack_cat
# gets overwritten to 'BENIGN' for all benign rows -- making benign rows
# even MORE likely to collide with each other on the remaining columns)
feature_cols = [c for c in df.columns if c not in ("attack_cat", "label")]
print(f"\nUnique benign rows (feature columns only): {benign[feature_cols].drop_duplicates().shape[0]:,}")
print(f"Unique attack rows (feature columns only): {attack[feature_cols].drop_duplicates().shape[0]:,}")