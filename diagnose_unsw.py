import pandas as pd

DATA_FOLDER = r"C:\Users\DELL\Downloads\OneDrive_2_9-13-2026"
df_train = pd.read_csv(f"{DATA_FOLDER}\\UNSW_NB15_training-set.csv", low_memory=False)
df_test = pd.read_csv(f"{DATA_FOLDER}\\UNSW_NB15_testing-set.csv", low_memory=False)
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"Total raw rows: {len(df):,}")
print(f"\nRaw label==0 (benign) count: {(df['label']==0).sum():,}")
print(f"Raw label==1 (attack) count: {(df['label']==1).sum():,}")

check_cols = ['is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_state_ttl', 'response_body_len']
print("\n--- Checking for blank/non-numeric values in suspect columns ---")
for col in check_cols:
    if col not in df.columns:
        continue
    numeric = pd.to_numeric(df[col], errors='coerce')
    n_bad = numeric.isna().sum()
    if n_bad > 0:
        bad_rows = df[numeric.isna()]
        benign_bad = (bad_rows['label']==0).sum()
        attack_bad = (bad_rows['label']==1).sum()
        print(f"{col}: {n_bad:,} rows fail numeric conversion (benign={benign_bad:,}, attack={attack_bad:,})")
    else:
        print(f"{col}: all values numeric, OK")