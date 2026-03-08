
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — NCAA Tournament Seed Prediction
# Applies identical transformations to train and test sets.
# ══════════════════════════════════════════════════════════════════════════════

TARGET = "Overall Seed"

# ─── Helper: parse "W-L" strings ─────────────────────────────────────────────
def _parse_wl(s):
    """Return (wins, losses) as ints; (NaN, NaN) on failure."""
    try:
        parts = str(s).strip().split("-")
        if len(parts) == 2:
            w, l = int(parts[0]), int(parts[1])
            return w, l
    except (ValueError, AttributeError):
        pass
    return np.nan, np.nan


def engineer_features(df_raw):
    """
    Apply all feature engineering steps and return a numeric feature DataFrame.
    Works on both train and test DataFrames.
    """
    df = df_raw.copy()

    # ── 1. Parse all W-L string columns ─────────────────────────────────────
    wl_cols = {
        "WL":                   "ovr",
        "Conf.Record":          "conf",
        "Non-ConferenceRecord": "nc",
        "RoadWL":               "road",
        "Quadrant1":            "q1",
        "Quadrant2":            "q2",
        "Quadrant3":            "q3",
        "Quadrant4":            "q4",
    }
    for col, pfx in wl_cols.items():
        parsed = df[col].apply(_parse_wl)
        df[f"{pfx}_w"] = [x[0] for x in parsed]
        df[f"{pfx}_l"] = [x[1] for x in parsed]
        tot = df[f"{pfx}_w"] + df[f"{pfx}_l"]
        df[f"{pfx}_wp"] = np.where(tot > 0, df[f"{pfx}_w"] / tot, np.nan)

    # ── 2. Quad W/L ratios & combined quad scores ────────────────────────────
    # Ratio: wins / (losses + 1) avoids div-by-zero
    for q in ["q1", "q2", "q3", "q4"]:
        df[f"{q}_ratio"] = df[f"{q}_w"] / (df[f"{q}_l"] + 1)

    # Combined quad score: weight Q1 most, Q4 least (quality wins matter)
    df["quad_combined_score"] = (
        df["q1_w"] * 4 + df["q2_w"] * 3 + df["q3_w"] * 2 + df["q4_w"] * 1
        - df["q1_l"] * 4 - df["q2_l"] * 3 - df["q3_l"] * 1 - df["q4_l"] * 0.5
    )
    # Bad-loss penalty: Q3 + Q4 losses are especially damaging for seeding
    df["bad_loss_score"] = df["q3_l"] * 2 + df["q4_l"] * 4

    # ── 3. NET Rank tier encoding ─────────────────────────────────────────────
    def _net_tier(rank):
        if pd.isna(rank):
            return np.nan
        if rank <= 25:   return 1
        if rank <= 50:   return 2
        if rank <= 100:  return 3
        return 4

    df["net_rank_tier"]      = df["NET Rank"].apply(_net_tier)
    df["prev_net_rank_tier"] = df["PrevNET"].apply(_net_tier)

    # NET rank improvement (positive = improved rank = smaller number)
    df["net_rank_delta"] = df["PrevNET"] - df["NET Rank"]  # positive → improving

    # ── 4. Win% features: overall, conference, non-conference ────────────────
    # (already computed as ovr_wp, conf_wp, nc_wp in step 1)
    # Add road win% and home win% proxy (ovr - road)
    df["road_wp"]        = df["road_wp"]  # already done
    df["home_win_proxy"] = df["ovr_w"] - df["road_w"]  # ~home + neutral wins

    # Win% differential: non-conf vs conf (tests scheduling strength)
    df["nc_vs_conf_wp_diff"] = df["nc_wp"] - df["conf_wp"]

    # ── 5. SOS composite ────────────────────────────────────────────────────
    # Lower NETSOS = tougher schedule
    # Composite: average of NETSOS and NETNonConfSOS (both normalised)
    # Also: opponent quality differential
    df["sos_composite"]     = (df["NETSOS"] + df["NETNonConfSOS"]) / 2
    df["opp_quality_delta"] = df["AvgOppNETRank"] - df["AvgOppNET"]

    # ── 6. Offensive / Defensive efficiency deltas ──────────────────────────
    # We don't have raw O/D efficiency, but we can proxy:
    #   - AvgOppNET (quality of opposition) vs NET Rank delta
    # "Outperformance": beating better opponents → strong offense + defense
    df["net_vs_opp_delta"]   = df["AvgOppNET"] - df["NET Rank"]   # positive = beating better teams
    df["net_vs_avgopp_rank"] = df["AvgOppNETRank"] - df["NET Rank"]  # positive = winning vs better

    # ── 7. Recent form proxies from available stats ──────────────────────────
    # PrevNET → NET Rank change captures form trajectory
    # Q1 win% relative to overall win% = clutch performance index
    df["q1_vs_ovr_wp"]   = df["q1_wp"] - df["ovr_wp"]   # positive = better vs top teams
    df["q1_dominance"]   = df["q1_w"] - df["q1_l"]       # net Q1 record
    df["road_vs_ovr_wp"] = df["road_wp"] - df["ovr_wp"]  # road performance relative to overall

    # ── 8. is_auto_bid flag ─────────────────────────────────────────────────
    # "Bid Type" column: AQ = auto bid, AL = at-large; NaN = not in tournament
    df["is_auto_bid"] = (df["Bid Type"].astype(str).str.upper().str.strip() == "AQ").astype(int)
    df["is_at_large"] = (df["Bid Type"].astype(str).str.upper().str.strip() == "AL").astype(int)
    df["in_tournament"] = ((df["is_auto_bid"] + df["is_at_large"]) > 0).astype(int)

    # ── 9. Conference strength tier encoding ────────────────────────────────
    # Based on historical NCAA tournament representation & program strength
    tier1_conf = {
        "big ten", "big east", "sec", "acc", "big 12",
        "atlantic coast", "southeastern", "pac-12", "pac 12"
    }
    tier2_conf = {
        "american", "atlantic 10", "west coast", "mountain west",
        "conference usa", "c-usa", "sun belt", "mac",
        "mid-american", "ohio valley", "colonial"
    }

    def _conf_tier(conf):
        if pd.isna(conf):
            return 3
        c = str(conf).strip().lower()
        if c in tier1_conf:   return 1
        if c in tier2_conf:   return 2
        return 3

    df["conf_tier"] = df["Conference"].apply(_conf_tier)
    # Binary: power conference flag
    df["is_power_conf"] = (df["conf_tier"] == 1).astype(int)

    # ── 10. Compound interaction features ───────────────────────────────────
    # NET rank × SOS: good rank against tough schedule = elite team
    df["net_sos_interaction"] = df["NET Rank"] * df["NETSOS"]
    # Q1 win% × total games proxy: quality and volume
    df["q1_vol_score"] = df["q1_wp"] * (df["q1_w"] + df["q1_l"])
    # Quad1+2 combined win rate (Tier A wins)
    tier_ab_w = df["q1_w"] + df["q2_w"]
    tier_ab_g = df["q1_w"] + df["q1_l"] + df["q2_w"] + df["q2_l"]
    df["tier_ab_wp"] = np.where(tier_ab_g > 0, tier_ab_w / tier_ab_g, np.nan)

    return df


# ── Apply to train and test ───────────────────────────────────────────────────
train_raw = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
test_raw  = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")

train_eng = engineer_features(train_raw)
test_eng  = engineer_features(test_raw)

# ── Define final feature columns (exclude ID, target, raw strings) ───────────
EXCLUDE_COLS = {
    "RecordID", "Season", "Team", "Conference",
    "Overall Seed", "Bid Type",
    "WL", "Conf.Record", "Non-ConferenceRecord", "RoadWL",
    "Quadrant1", "Quadrant2", "Quadrant3", "Quadrant4",
}
fe_cols = [c for c in train_eng.columns if c not in EXCLUDE_COLS]

# ── Build feature matrices ────────────────────────────────────────────────────
train_X = train_eng[fe_cols].copy()
test_X  = test_eng[fe_cols].copy()

# ── Fit median imputation on TRAIN only, apply to both ───────────────────────
fe_medians = train_X.median()
train_X = train_X.fillna(fe_medians)
test_X  = test_X.fillna(fe_medians)

# Attach target to train
train_X["Overall Seed"] = train_eng["Overall Seed"].values

# Final outputs
train_fe = train_X.copy()
test_fe  = test_X.copy()

# ── Validation: null check ────────────────────────────────────────────────────
train_nulls = train_fe.drop(columns=["Overall Seed"]).isnull().sum().sum()
test_nulls  = test_fe.isnull().sum().sum()

print("=" * 65)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 65)
print(f"  train_fe shape : {train_fe.shape}")
print(f"  test_fe  shape : {test_fe.shape}")
print(f"  Feature columns: {len(fe_cols)}")
print(f"  Null values in train features : {train_nulls}")
print(f"  Null values in test features  : {test_nulls}")
print(f"\n  Feature list ({len(fe_cols)} features):")
for i, f in enumerate(fe_cols, 1):
    print(f"    {i:>2}. {f}")

# ── Mutual Information Scores vs Overall Seed ─────────────────────────────────
# Only use rows where the target is known (tournament teams)
mi_df = train_fe.dropna(subset=["Overall Seed"])
mi_X  = mi_df[fe_cols]
mi_y  = mi_df["Overall Seed"]

mi_scores = mutual_info_regression(mi_X, mi_y, random_state=42, n_neighbors=5)
mi_series = pd.Series(mi_scores, index=fe_cols).sort_values(ascending=False)

print("\n" + "=" * 65)
print("MUTUAL INFORMATION SCORES vs Overall Seed (top 25)")
print("=" * 65)
print(f"{'Feature':<30}  {'MI Score':>9}")
print("-" * 42)
for feat, score in mi_series.head(25).items():
    bar = "█" * int(score * 30)
    print(f"  {feat:<28}  {score:>7.4f}  {bar}")

print("\n" + "=" * 65)
print("BOTTOM 10 MI SCORES (potentially weak features)")
print("=" * 65)
for feat, score in mi_series.tail(10).items():
    print(f"  {feat:<28}  {score:>7.4f}")

print("\n✅ train_fe and test_fe ready for modelling.")
