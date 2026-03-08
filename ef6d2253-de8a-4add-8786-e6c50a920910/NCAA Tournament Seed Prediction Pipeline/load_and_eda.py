
import pandas as pd
import numpy as np

# ── Load all files ──────────────────────────────────────────────────────────
train_df       = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
test_df        = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")
submission_df  = pd.read_csv("submission_template2.0.csv")
data_dict_df   = pd.read_excel("FFAC Data Dictionary.xlsx")

# ── Basic shapes ────────────────────────────────────────────────────────────
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)
print(f"  Training set   : {train_df.shape[0]:>5} rows × {train_df.shape[1]:>3} cols")
print(f"  Test set       : {test_df.shape[0]:>5} rows × {test_df.shape[1]:>3} cols")
print(f"  Submission tmpl: {submission_df.shape[0]:>5} rows × {submission_df.shape[1]:>3} cols")
print(f"  Data dictionary: {data_dict_df.shape[0]:>5} rows × {data_dict_df.shape[1]:>3} cols")

# ── Data Dictionary preview ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATA DICTIONARY — columns & first rows")
print("=" * 70)
print(data_dict_df.columns.tolist())
print(data_dict_df.head(20).to_string())

# ── Training set columns ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING COLUMNS & DTYPES")
print("=" * 70)
print(train_df.dtypes.to_string())

# ── Missing values ───────────────────────────────────────────────────────────
missing_train = train_df.isnull().sum()
missing_pct   = (missing_train / len(train_df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing_train, "missing_%": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0].sort_values("missing_count", ascending=False)

print("\n" + "=" * 70)
print(f"MISSING VALUES IN TRAINING SET  ({len(missing_report)} cols with gaps)")
print("=" * 70)
if len(missing_report) > 0:
    print(missing_report.to_string())
else:
    print("  ✅ No missing values in training set")

# ── Target variable: Overall Seed ───────────────────────────────────────────
target_col = None
for c in train_df.columns:
    if "overall" in c.lower() and "seed" in c.lower():
        target_col = c
        break
    if c.lower() == "seed":
        target_col = c

if target_col is None:
    # fallback search
    seed_cols = [c for c in train_df.columns if "seed" in c.lower()]
    print(f"\nSeed-related columns: {seed_cols}")
    target_col = seed_cols[0] if seed_cols else None

print("\n" + "=" * 70)
print(f"TARGET COLUMN: '{target_col}'")
print("=" * 70)
if target_col:
    print(train_df[target_col].describe().to_string())
    print(f"\nUnique values ({train_df[target_col].nunique()}):")
    vc = train_df[target_col].value_counts().sort_index()
    print(vc.to_string())

# ── Sample rows ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING SET — SAMPLE ROWS (first 5)")
print("=" * 70)
print(train_df.head(5).to_string())

# ── Key summary stats ────────────────────────────────────────────────────────
numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
print("\n" + "=" * 70)
print(f"NUMERIC FEATURE SUMMARY ({len(numeric_cols)} numeric cols)")
print("=" * 70)
print(train_df[numeric_cols].describe().T.round(3).to_string())

# ── Feature inventory: map training cols to data dictionary ─────────────────
print("\n" + "=" * 70)
print("FEATURE INVENTORY — Data Dictionary Mapping")
print("=" * 70)
# Guess the column name column in the data dict
name_col_candidates = [c for c in data_dict_df.columns if any(k in c.lower() for k in ["variable", "feature", "field", "column", "name"])]
if not name_col_candidates:
    name_col_candidates = [data_dict_df.columns[0]]
dict_name_col = name_col_candidates[0]
print(f"Using data dict column '{dict_name_col}' for feature name matching.\n")

dict_features = set(data_dict_df[dict_name_col].dropna().astype(str).str.strip())
train_features = set(train_df.columns)
matched   = sorted(train_features & dict_features)
unmatched = sorted(train_features - dict_features)
dict_only = sorted(dict_features - train_features)

print(f"  Training cols matched in dict : {len(matched)}")
print(f"  Training cols NOT in dict     : {len(unmatched)}")
print(f"  Dict entries NOT in training  : {len(dict_only)}")
print(f"\n  MATCHED features:\n  {matched}")
print(f"\n  TRAINING-ONLY (not in dict):\n  {unmatched}")
print(f"\n  DICT-ONLY (not in training):\n  {dict_only}")

# Store for downstream blocks
target_column   = target_col
numeric_features = [c for c in numeric_cols if c != target_col]

print("\n" + "=" * 70)
print("EDA COMPLETE ✅")
print("=" * 70)
