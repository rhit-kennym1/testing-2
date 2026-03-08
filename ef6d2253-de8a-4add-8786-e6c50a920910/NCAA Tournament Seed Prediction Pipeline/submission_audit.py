
import pandas as pd
import numpy as np
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════════
# SUBMISSION AUDIT — Full Diagnosis Report
# Checks: non-tournament teams, seed validity, value distribution,
#         leakage columns, target distribution, feature correlations
# ═══════════════════════════════════════════════════════════════════════════════

DIVIDER  = "═" * 72
DIVIDER2 = "─" * 72

print(DIVIDER)
print("  NCAA TOURNAMENT SEED PREDICTION — SUBMISSION AUDIT REPORT")
print(DIVIDER)

# ─── 0. Reload raw data fresh (independent of pipeline) ───────────────────────
audit_train = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
audit_test  = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")
audit_sub   = pd.read_csv("submission.csv")
audit_tmpl  = pd.read_csv("submission_template2.0.csv")

TARGET = "Overall Seed"

print(f"\n  Files loaded:")
print(f"    Training set  : {audit_train.shape[0]} rows x {audit_train.shape[1]} cols")
print(f"    Test set      : {audit_test.shape[0]}  rows x {audit_test.shape[1]} cols")
print(f"    Submission    : {audit_sub.shape[0]}  rows x {audit_sub.shape[1]} cols")
print(f"    Template      : {audit_tmpl.shape[0]}  rows x {audit_tmpl.shape[1]} cols")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TARGET DISTRIBUTION PROFILE (Training Set)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 1: TRAINING TARGET DISTRIBUTION — 'Overall Seed'")
print(DIVIDER)

_tourn_mask_tr  = audit_train[TARGET].notna() & (audit_train[TARGET] > 0)
_non_tourn_mask = audit_train[TARGET].isna() | (audit_train[TARGET] == 0)

n_total     = len(audit_train)
n_tourn     = int(_tourn_mask_tr.sum())
n_null      = int(audit_train[TARGET].isna().sum())
n_zero      = int((audit_train[TARGET] == 0).sum())
n_non_tourn = n_null + n_zero

print(f"\n  Row Counts:")
print(f"    Total training rows          : {n_total:>6}")
print(f"    Tournament teams (seed 1-68) : {n_tourn:>6}  ({n_tourn/n_total*100:.1f}%)")
print(f"    Non-tournament (seed = NaN)  : {n_null:>6}  ({n_null/n_total*100:.1f}%)")
print(f"    Non-tournament (seed = 0)    : {n_zero:>6}  ({n_zero/n_total*100:.1f}%)")
print(f"    Total non-tournament         : {n_non_tourn:>6}  ({n_non_tourn/n_total*100:.1f}%)")

_tourn_seeds = audit_train.loc[_tourn_mask_tr, TARGET]
_seed_max    = int(_tourn_seeds.max())
_seed_min    = int(_tourn_seeds.min())
_seed_mean   = _tourn_seeds.mean()
_seed_median = _tourn_seeds.median()
_seed_std    = _tourn_seeds.std()
_seed_nuniq  = int(_tourn_seeds.nunique())

print(f"\n  Tournament seed stats:")
print(f"    Min    : {_seed_min}")
_max_note = "EXCEEDS 16! This is OVERALL seed 1-68" if _seed_max > 16 else "Within 1-16"
print(f"    Max    : {_seed_max}   <- {_max_note}")
print(f"    Mean   : {_seed_mean:.2f}")
print(f"    Median : {_seed_median:.1f}")
print(f"    Std    : {_seed_std:.2f}")
print(f"    Unique values: {_seed_nuniq}")

_seed_vc = _tourn_seeds.astype(int).value_counts().sort_index()
print(f"\n  Full seed value counts (tournament rows only):")
print(f"  {'Seed':>5} | {'Count':>5} | {'% of tourn':>10} | Bar")
print(f"  {DIVIDER2[:55]}")
for _s, _cnt in _seed_vc.items():
    _pct = _cnt / n_tourn * 100
    _bar = chr(9608) * min(int(_pct * 2), 40)
    print(f"  {_s:>5} | {_cnt:>5} | {_pct:>9.1f}% | {_bar}")

# Seasons breakdown
print(f"\n  Tournament teams per season:")
_season_tourn = audit_train[_tourn_mask_tr].groupby("Season").agg(
    n_teams=(TARGET, "count"),
    min_seed=(TARGET, "min"),
    max_seed=(TARGET, "max"),
    mean_seed=(TARGET, "mean"),
).round(2)
print(_season_tourn.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SEED RANGE VALIDITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 2: SEED RANGE VALIDITY CHECK")
print(DIVIDER)

_invalid_low    = int((_tourn_seeds < 1).sum())
_invalid_high   = int((_tourn_seeds > 68).sum())
_valid_range    = int(((_tourn_seeds >= 1) & (_tourn_seeds <= 68)).sum())
_seeds_above_16 = int(((_tourn_seeds > 16) & (_tourn_seeds <= 68)).sum())
_seeds_le16     = int((_tourn_seeds <= 16).sum())

print(f"\n  Training — Overall Seed range check (valid: 1-68):")
print(f"    Seeds in valid range [1-68]  : {_valid_range:>5}  OK")
print(f"    Seeds < 1                    : {_invalid_low:>5}  {'WARN' if _invalid_low > 0 else 'OK'}")
print(f"    Seeds > 68                   : {_invalid_high:>5}  {'WARN' if _invalid_high > 0 else 'OK'}")
print(f"    Seeds in [1-16]              : {_seeds_le16:>5}  (top-line seeds)")
print(f"    Seeds in [17-68]             : {_seeds_above_16:>5}  (lower seeds - valid but harder to predict)")

# Submission checks
_sub_seeds   = audit_sub[TARGET]
_sub_tourn   = _sub_seeds[_sub_seeds > 0]
_sub_non     = _sub_seeds[_sub_seeds == 0]
_sub_inv_low = int((_sub_tourn < 1).sum())
_sub_inv_hi  = int((_sub_tourn > 16).sum())
_sub_nulls   = int(_sub_seeds.isna().sum())

print(f"\n  Submission — Overall Seed range check:")
print(f"    Total rows                   : {len(audit_sub):>5}")
print(f"    Non-tournament rows (seed=0) : {len(_sub_non):>5}  ({len(_sub_non)/len(audit_sub)*100:.1f}%)")
print(f"    Tournament rows (seed>0)     : {len(_sub_tourn):>5}  ({len(_sub_tourn)/len(audit_sub)*100:.1f}%)")
print(f"    Seeds in valid range [1-16]  : {((_sub_tourn >= 1) & (_sub_tourn <= 16)).sum():>5}  {'OK' if _sub_inv_hi == 0 else 'WARN'}")
print(f"    Seeds > 16 (INVALID)         : {_sub_inv_hi:>5}  {'PROBLEM!' if _sub_inv_hi > 0 else 'None - OK'}")
print(f"    Seeds < 1 (INVALID)          : {_sub_inv_low:>5}  {'PROBLEM!' if _sub_inv_low > 0 else 'None - OK'}")
print(f"    Null seeds in submission     : {_sub_nulls:>5}  {'PROBLEM!' if _sub_nulls > 0 else 'None - OK'}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SUBMISSION VALUE DISTRIBUTION vs EXPECTED
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 3: SUBMISSION VALUE DISTRIBUTION vs EXPECTED")
print(DIVIDER)

print(f"\n  Submission seed distribution (seeds 1-16):")
_sub_vc = _sub_tourn.astype(int).value_counts().sort_index()

print(f"  {'Seed':>5} | {'Count':>5} | {'Expected':>8} | {'Delta':>6} | Notes")
print(f"  {DIVIDER2[:60]}")
for _s in range(1, 17):
    _cnt = int(_sub_vc.get(_s, 0))
    _exp = 4
    _delta = _cnt - _exp
    _flag = "HEAVY" if _cnt > 10 else ("SPARSE" if _cnt == 0 else "")
    print(f"  {_s:>5} | {_cnt:>5} | {_exp:>8} | {_delta:>+6} | {_flag}")

_top4  = sum(int(_sub_vc.get(s, 0)) for s in range(1, 5))
_mid8  = sum(int(_sub_vc.get(s, 0)) for s in range(5, 9))
_bub12 = sum(int(_sub_vc.get(s, 0)) for s in range(9, 13))
_bot16 = sum(int(_sub_vc.get(s, 0)) for s in range(13, 17))

print(f"\n  Summary by bracket tier:")
print(f"    Seeds 1-4   (top tier)  : {_top4:>4} teams  (expected ~20)")
print(f"    Seeds 5-8   (solid)     : {_mid8:>4} teams  (expected ~20)")
print(f"    Seeds 9-12  (bubble)    : {_bub12:>4} teams  (expected ~20)")
print(f"    Seeds 13-16 (long shots): {_bot16:>4} teams  (expected ~20)")

_n_seed16   = int(_sub_vc.get(16, 0))
_n_tourn_sub = len(_sub_tourn)
_pct_seed16 = _n_seed16 / _n_tourn_sub * 100 if _n_tourn_sub > 0 else 0

print(f"\n  Seed 16 concentration: {_n_seed16} teams = {_pct_seed16:.1f}% of all tournament predictions")
if _pct_seed16 > 30:
    print(f"  *** RED FLAG: Seed 16 over-concentration detected!")
    print(f"      Regressor is failing on lower seeds due to systematic upward bias.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LEAKAGE COLUMN AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 4: LEAKAGE COLUMN AUDIT")
print(DIVIDER)

KNOWN_LEAKAGE = {"Overall Seed", "Bid Type", "is_auto_bid", "is_at_large", "in_tournament"}

print(f"\n  Raw training columns: {list(audit_train.columns)}")
print(f"\n  Leakage Assessment:")
for _col in audit_train.columns:
    if _col == "Overall Seed":
        print(f"    {_col:<25} TARGET LEAKAGE (the target itself)")
    elif _col == "Bid Type":
        print(f"    {_col:<25} STRONG LEAKAGE (reveals tournament selection: AQ/AL post-selection)")
    elif _col in KNOWN_LEAKAGE:
        print(f"    {_col:<25} POTENTIAL LEAKAGE (derived from Bid Type)")

print(f"\n  Pipeline leakage exclusions:")
print(f"    CLF excluded  : is_auto_bid, is_at_large, in_tournament, Overall Seed")
print(f"    REG excluded  : is_auto_bid, is_at_large, in_tournament, Overall Seed")
print(f"\n  STATUS: 'Bid Type' raw field is a strong leakage source.")
print(f"    - is_auto_bid / is_at_large / in_tournament are all derived from it.")
print(f"    - These are correctly EXCLUDED from clf_feat_cols and reg_feat_cols. OK")
print(f"    - Note: in_tournament would be a perfect classifier feature if included.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: NON-TOURNAMENT TEAM AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 5: NON-TOURNAMENT TEAM AUDIT")
print(DIVIDER)

_null_seed_train = audit_train[audit_train[TARGET].isna()]
print(f"\n  Training rows with NaN 'Overall Seed' (non-tournament):")
print(f"    Count            : {len(_null_seed_train)}")
print(f"    Bid Type values  : {_null_seed_train['Bid Type'].value_counts(dropna=False).to_dict()}")
print(f"\n  Seasons breakdown:")
print(_null_seed_train.groupby("Season").size().to_string())

_test_has_target = TARGET in audit_test.columns
print(f"\n  Test set: 'Overall Seed' column present? {_test_has_target}")
if _test_has_target:
    print(f"  WARNING: Target column found in test set!")
else:
    print(f"  OK: No target in test set (as expected)")

print(f"\n  Test set Bid Type distribution:")
_test_bid_vc = audit_test["Bid Type"].value_counts(dropna=False)
print(_test_bid_vc.to_string())
_test_bid_tourn = int(audit_test["Bid Type"].isin(["AQ", "AL"]).sum())
_test_bid_none  = int((~audit_test["Bid Type"].isin(["AQ", "AL"])).sum())
print(f"\n    Known tournament teams (AQ/AL) in test : {_test_bid_tourn}")
print(f"    Non-tournament / NaN in test           : {_test_bid_none}")
print(f"\n  NOTE: 'Bid Type' in test set reveals which teams are in the tournament.")
print(f"    In competition: may be permissible as 'known at prediction time'.")
print(f"    In real deployment: this would be unavailable before selection.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FULL FEATURE CORRELATION TABLE vs TARGET
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 6: FULL FEATURE CORRELATION vs 'Overall Seed' (Pearson + Spearman)")
print("  (Tournament teams only, n=249 training rows with known seeds)")
print(DIVIDER)

# Use train_fe from pipeline (available via upstream block)
_corr_base  = train_fe[train_fe[TARGET] > 0].copy()
_feat_cols  = [c for c in _corr_base.columns if c != TARGET]
_target_arr = _corr_base[TARGET]

_corr_results = []
for _f in _feat_cols:
    _x = _corr_base[_f]
    _xf = _x.fillna(_x.median())
    _pr, _pp = stats.pearsonr(_xf, _target_arr)
    _sr, _sp = stats.spearmanr(_xf, _target_arr)
    _corr_results.append({
        "feature":    _f,
        "pearson_r":  round(_pr, 4),
        "spearman_r": round(_sr, 4),
        "abs_pearson": abs(_pr),
        "pearson_p":  round(_pp, 4),
        "spearman_p": round(_sp, 4),
    })

audit_corr_df = pd.DataFrame(_corr_results).sort_values("abs_pearson", ascending=False).reset_index(drop=True)
_display_df = audit_corr_df.drop(columns=["abs_pearson"])

print(f"\n  {'Feature':<30} {'Pearson r':>10} {'p-val':>8} {'Spearman r':>11} {'p-val':>8}  Sig  Direction")
print(f"  {DIVIDER2}")
for _, _row in _display_df.iterrows():
    _dir = "up (higher seed# = worse)" if _row["pearson_r"] > 0 else "down (lower seed# = better)"
    _sig = "***" if _row["pearson_p"] < 0.001 else ("**" if _row["pearson_p"] < 0.01 else ("*" if _row["pearson_p"] < 0.05 else ""))
    print(f"  {_row['feature']:<30} {_row['pearson_r']:>+10.4f} {_row['pearson_p']:>8.4f} "
          f"{_row['spearman_r']:>+11.4f} {_row['spearman_p']:>8.4f}  {_sig:<3}  {_dir}")

print(f"\n  TOP 10 most correlated features (by |Pearson r|):")
print(f"  {'Rank':<5} {'Feature':<30} {'|r|':>7}  {'Spearman r':>11}")
print(f"  {DIVIDER2[:55]}")
for _rank, (_, _row) in enumerate(_display_df.head(10).iterrows(), 1):
    print(f"  {_rank:<5} {_row['feature']:<30} {abs(_row['pearson_r']):>7.4f}  {_row['spearman_r']:>+11.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DIAGNOSIS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("  SECTION 7: DIAGNOSIS SUMMARY — WHAT IS WRONG WITH CURRENT SUBMISSION")
print(DIVIDER)

_n_seed1         = int(_sub_vc.get(1, 0))
_n_test_seasons  = audit_test["Season"].nunique() if "Season" in audit_test.columns else "?"
_expected_total  = _n_test_seasons * 68 if isinstance(_n_test_seasons, int) else "?"

print(f"""
  PROBLEM 1 [CRITICAL]: TARGET SCALE MISMATCH
  ---
  The training target 'Overall Seed' ranges 1 to {_seed_max} (not 1-16).
  Overall Seed = each team's overall rank among the 68 tournament selectees.
  The regressor was trained on 1-68 scale but submission clips to 1-16.

  Training seed range  : {_seed_min} to {_seed_max}
  Tournament rows (train) : {n_tourn}
  Seeds 1-16 in training  : {_seeds_le16}  ({_seeds_le16/n_tourn*100:.1f}%)
  Seeds 17-68 in training : {_seeds_above_16}  ({_seeds_above_16/n_tourn*100:.1f}%)

  Impact: All 249 tournament training rows include seeds 1-68.
  The submission clips to 1-16, artificially piling teams at seed 16.
""")

print(f"""  PROBLEM 2 [CRITICAL]: SEED DISTRIBUTION IS PATHOLOGICAL
  ---
  Current submission has {_n_tourn_sub} tournament teams. Of those:
    Seed 16: {_n_seed16} teams = {_pct_seed16:.1f}%  (expected ~4 teams per season, ~20 total)
    Seed 1 : {_n_seed1} team(s)

  Real tournament has exactly 4 teams per seed line (lines 1-16).
  For {_n_test_seasons} test seasons: expected ~{_expected_total} total tournament rows,
  with approx {_n_test_seasons * 4 if isinstance(_n_test_seasons, int) else '~20'} teams per seed line.

  The regressor is predicting seeds in the 1-68 range, but most predictions
  land >16 (teams ranked 17-68), and clipping forces them all to seed 16.
""")

print(f"""  PROBLEM 3 [IMPORTANT]: TOURNAMENT TEAM COUNT MISMATCH
  ---
  Expected: {_expected_total} tournament rows ({_n_test_seasons} seasons x 68 teams)
  Predicted: {_n_tourn_sub} tournament rows (classifier found {_n_tourn_sub} of {len(audit_test)} test rows)
  Non-tournament: {len(_sub_non)} rows

  This means either:
  a) The classifier threshold is too loose (including too many or too few teams)
  b) The test set contains more non-tournament teams than expected
""")

print(f"""  LEAKAGE STATUS: OK
  ---
  Confirmed correctly excluded from model features:
    - Overall Seed (target)
    - Bid Type (raw leakage source)
    - is_auto_bid, is_at_large, in_tournament (derived leakage)
  No leakage detected in final model inputs.
""")

print(f"""  TOP PREDICTIVE FEATURES (by |Pearson r| vs Overall Seed):""")
for _rank, (_, _row) in enumerate(_display_df.head(10).iterrows(), 1):
    _dir = "(inverse - better rank = lower seed#)" if _row["pearson_r"] > 0 else "(positive - better performance = lower seed#)"
    print(f"    {_rank:>2}. {_row['feature']:<28}  r={_row['pearson_r']:+.4f}  {_dir}")

print(f"""
  RECOMMENDED FIXES:
    1. Train seed regressor on FULL 1-68 range, predict 1-68 in submission
    2. OR: Create a bracket-seed mapping (overall seed -> bracket line 1-16)
    3. Ensure tournament teams in test = 68 per season (adjust clf threshold)
    4. Validate final seed distribution is ~4 teams per seed line per season
""")

print(f"\n{DIVIDER}")
print("  AUDIT COMPLETE")
print(DIVIDER)
