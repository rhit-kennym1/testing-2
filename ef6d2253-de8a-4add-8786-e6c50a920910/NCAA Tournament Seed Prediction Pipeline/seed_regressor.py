
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: SEED REGRESSOR
# Target: Overall Seed (1–16), tournament teams only (Overall Seed > 0)
# Method: Leave-One-Year-Out (LOYO) CV
# Models: GradientBoostingRegressor + Ridge → weighted average ensemble
# (GBR is sklearn's native equivalent of XGBoost — same tree boosting logic)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Re-read Season for LOYO splits ─────────────────────────────────────────
_train_raw_s2 = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")

# ── 2. Filter to tournament teams only (Overall Seed > 0) ─────────────────────
SEED_TARGET = "Overall Seed"
_seed_mask = train_fe[SEED_TARGET] > 0
reg_train = train_fe[_seed_mask].copy()
reg_train["season_year"] = _train_raw_s2.loc[_seed_mask.values, "Season"].values

print(f"Tournament teams (seed > 0): {len(reg_train)} / {len(train_fe)} total rows")
print(f"Seed range in training: {reg_train[SEED_TARGET].min():.0f} – {reg_train[SEED_TARGET].max():.0f}")
print(f"Seasons: {sorted(reg_train['season_year'].unique())}\n")

# ── 3. Feature columns for regression ─────────────────────────────────────────
REG_LEAKAGE = {"Overall Seed", "is_auto_bid", "is_at_large", "in_tournament"}
reg_feat_cols = [c for c in fe_cols if c not in REG_LEAKAGE]
print(f"Regression features: {len(reg_feat_cols)}")

# ── 4. Prepare arrays ──────────────────────────────────────────────────────────
reg_X = reg_train[reg_feat_cols].values
reg_y = reg_train[SEED_TARGET].values
reg_seasons = reg_train["season_year"].values
reg_season_list = sorted(reg_train["season_year"].unique())

# ── 5. LOYO Cross-Validation ───────────────────────────────────────────────────
loyo_reg_results = []
oof_preds_gbr   = np.zeros(len(reg_train))
oof_preds_ridge = np.zeros(len(reg_train))
oof_true        = reg_y.copy()

ENSEMBLE_W_GBR   = 0.65   # GBR weight
ENSEMBLE_W_RIDGE = 0.35   # Ridge weight

print("\n" + "═" * 72)
print("LEAVE-ONE-YEAR-OUT CV — SEED REGRESSOR (GBR + Ridge Ensemble)")
print("═" * 72)
print(f"{'Season':<12} {'N_test':>7} {'GBR RMSE':>10} {'Ridge RMSE':>12} {'Ens RMSE':>10}")
print("─" * 72)

for _year in reg_season_list:
    _mask_val = reg_seasons == _year
    _mask_tr  = ~_mask_val

    _X_tr  = reg_X[_mask_tr]
    _y_tr  = reg_y[_mask_tr]
    _X_val = reg_X[_mask_val]
    _y_val = reg_y[_mask_val]

    # ── Gradient Boosting Regressor ───────────────────────────────────────────
    _gbr = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-4,
    )
    _gbr.fit(_X_tr, _y_tr)
    _pred_gbr = _gbr.predict(_X_val)

    # ── Ridge Regression ──────────────────────────────────────────────────────
    _scaler = StandardScaler()
    _X_tr_sc  = _scaler.fit_transform(_X_tr)
    _X_val_sc = _scaler.transform(_X_val)
    _ridge = Ridge(alpha=5.0)
    _ridge.fit(_X_tr_sc, _y_tr)
    _pred_ridge = _ridge.predict(_X_val_sc)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    _pred_ens = ENSEMBLE_W_GBR * _pred_gbr + ENSEMBLE_W_RIDGE * _pred_ridge

    # Store OOF
    oof_preds_gbr[_mask_val]   = _pred_gbr
    oof_preds_ridge[_mask_val] = _pred_ridge

    _rmse_gbr   = np.sqrt(mean_squared_error(_y_val, _pred_gbr))
    _rmse_ridge = np.sqrt(mean_squared_error(_y_val, _pred_ridge))
    _rmse_ens   = np.sqrt(mean_squared_error(_y_val, _pred_ens))

    loyo_reg_results.append({
        "season": _year,
        "n_test": int(_mask_val.sum()),
        "rmse_gbr": _rmse_gbr,
        "rmse_ridge": _rmse_ridge,
        "rmse_ensemble": _rmse_ens,
    })
    print(f"  {_year:<10} {int(_mask_val.sum()):>7} {_rmse_gbr:>10.4f} {_rmse_ridge:>12.4f} {_rmse_ens:>10.4f}")

loyo_reg_df = pd.DataFrame(loyo_reg_results)

# ── 6. Overall OOF RMSE ───────────────────────────────────────────────────────
oof_preds_ensemble = ENSEMBLE_W_GBR * oof_preds_gbr + ENSEMBLE_W_RIDGE * oof_preds_ridge

overall_rmse_gbr   = np.sqrt(mean_squared_error(oof_true, oof_preds_gbr))
overall_rmse_ridge = np.sqrt(mean_squared_error(oof_true, oof_preds_ridge))
overall_rmse_ens   = np.sqrt(mean_squared_error(oof_true, oof_preds_ensemble))

print("─" * 72)
print(f"  {'OVERALL OOF':<10} {len(reg_train):>7} {overall_rmse_gbr:>10.4f} {overall_rmse_ridge:>12.4f} {overall_rmse_ens:>10.4f}")
print("═" * 72)

# ── 7. RMSE breakdown by seed bin ─────────────────────────────────────────────
_bins = [(1, 4), (5, 8), (9, 12), (13, 16)]
print("\n" + "═" * 55)
print("RMSE BREAKDOWN BY SEED BIN (Ensemble OOF)")
print("═" * 55)
print(f"{'Seed Bin':<15} {'N':>5} {'RMSE':>10} {'Mean Actual':>12} {'Mean Pred':>10}")
print("─" * 55)
bin_rmse_data = []
for _lo, _hi in _bins:
    _m = (oof_true >= _lo) & (oof_true <= _hi)
    if _m.sum() == 0:
        continue
    _r = np.sqrt(mean_squared_error(oof_true[_m], oof_preds_ensemble[_m]))
    _ma = oof_true[_m].mean()
    _mp = oof_preds_ensemble[_m].mean()
    print(f"  Seeds {_lo:>2}–{_hi:<2}   {_m.sum():>5} {_r:>10.4f} {_ma:>12.2f} {_mp:>10.2f}")
    bin_rmse_data.append({"bin": f"{_lo}–{_hi}", "n": int(_m.sum()),
                          "rmse": _r, "mean_actual": _ma, "mean_pred": _mp})
print("─" * 55)
print(f"  {'ALL SEEDS':<14} {len(reg_train):>5} {overall_rmse_ens:>10.4f}")
print("═" * 55)
print(f"\n✅ Overall CV RMSE (Ensemble): {overall_rmse_ens:.4f}")
print(f"   GBR RMSE:   {overall_rmse_gbr:.4f}")
print(f"   Ridge RMSE: {overall_rmse_ridge:.4f}")
print(f"   Ensemble weights: GBR={ENSEMBLE_W_GBR}, Ridge={ENSEMBLE_W_RIDGE}\n")

# ── 8. Train FINAL models on ALL tournament training data ──────────────────────
print("Training final GBR + Ridge on all tournament training data...")
reg_final_gbr = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    min_samples_leaf=5,
    max_features=0.8,
    random_state=42,
)
reg_final_gbr.fit(reg_X, reg_y)

reg_final_scaler = StandardScaler()
_X_all_sc = reg_final_scaler.fit_transform(reg_X)
reg_final_ridge = Ridge(alpha=5.0)
reg_final_ridge.fit(_X_all_sc, reg_y)
print("  Final models trained.\n")

# ── 9. Generate seed predictions for test tournament teams ─────────────────────
PROBA_THRESHOLD = 0.50   # tunable Stage 1 threshold
_test_tourn_mask = tournament_proba >= PROBA_THRESHOLD
_n_test_tourn    = _test_tourn_mask.sum()

print(f"  Threshold={PROBA_THRESHOLD} → {_n_test_tourn} test teams selected for seed prediction")

_test_tourn_X    = test_fe[reg_feat_cols].values[_test_tourn_mask]
_pred_test_gbr   = reg_final_gbr.predict(_test_tourn_X)
_pred_test_ridge = reg_final_ridge.predict(reg_final_scaler.transform(_test_tourn_X))
_pred_test_seed  = ENSEMBLE_W_GBR * _pred_test_gbr + ENSEMBLE_W_RIDGE * _pred_test_ridge

# Clip to valid seed range [1, 16] and round to nearest integer
seed_predictions_raw     = _pred_test_seed.copy()
seed_predictions_clipped = np.clip(np.round(_pred_test_seed), 1, 16).astype(int)

# Build prediction DataFrame
_test_raw_s2 = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")
seed_pred_df = pd.DataFrame({
    "RecordID":       _test_raw_s2["RecordID"].values[_test_tourn_mask],
    "predicted_seed": seed_predictions_clipped,
    "seed_raw_pred":  np.round(seed_predictions_raw, 2),
    "clf_proba":      tournament_proba[_test_tourn_mask],
})
if "Team" in _test_raw_s2.columns:
    seed_pred_df.insert(1, "Team", _test_raw_s2["Team"].values[_test_tourn_mask])
if "Season" in _test_raw_s2.columns:
    seed_pred_df.insert(1, "Season", _test_raw_s2["Season"].values[_test_tourn_mask])

print(f"\n  Seed prediction sample (top 10 by confidence):")
print(seed_pred_df.sort_values("clf_proba", ascending=False).head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Residual Analysis — OOF Predicted vs Actual + Mean Residual by Seed
# ══════════════════════════════════════════════════════════════════════════════
_residuals = oof_preds_ensemble - oof_true

fig_residuals, (ax_res1, ax_res2) = plt.subplots(1, 2, figsize=(16, 6))
fig_residuals.patch.set_facecolor("#1D1D20")
for _ax in (ax_res1, ax_res2):
    _ax.set_facecolor("#1D1D20")
    _ax.spines["bottom"].set_color("#444")
    _ax.spines["left"].set_color("#444")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.tick_params(colors="#909094")

# Panel A: Predicted vs Actual with jitter to show density at integer seeds
_rng_jitter = np.random.default_rng(42)
_jitter = _rng_jitter.uniform(-0.25, 0.25, len(oof_true))
_sc = ax_res1.scatter(oof_true + _jitter, oof_preds_ensemble,
                      c=np.abs(_residuals), cmap="RdYlGn_r",
                      alpha=0.65, s=35, edgecolors="none", vmin=0, vmax=5)
ax_res1.plot([1, 16], [1, 16], color="#ffd400", linewidth=1.8,
             linestyle="--", label="Perfect prediction")
ax_res1.set_xlabel("Actual Seed", color="#909094", fontsize=11)
ax_res1.set_ylabel("Predicted Seed (Ensemble)", color="#909094", fontsize=11)
ax_res1.set_title("Predicted vs Actual Seed\n(OOF LOYO — Tournament Teams)",
                  color="#fbfbff", fontsize=12, fontweight="bold", pad=10)
ax_res1.set_xticks(range(1, 17))
ax_res1.set_xticklabels(range(1, 17), color="#909094", fontsize=8)
ax_res1.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff", fontsize=9)
_cbar = fig_residuals.colorbar(_sc, ax=ax_res1, pad=0.02)
_cbar.ax.tick_params(colors="#909094")
_cbar.set_label("|Residual|", color="#909094", fontsize=9)

# Panel B: Mean residual per actual seed (bias analysis)
_bin_means_res = []
_bin_labels_res = []
for _seed in range(1, 17):
    _m = oof_true == _seed
    if _m.sum() > 0:
        _bin_means_res.append(_residuals[_m].mean())
        _bin_labels_res.append(_seed)
_bc = ["#17b26a" if v >= 0 else "#f04438" for v in _bin_means_res]
ax_res2.bar(_bin_labels_res, _bin_means_res, color=_bc, edgecolor="#1D1D20", width=0.7)
ax_res2.axhline(0, color="#ffd400", linewidth=1.5, linestyle="--")
ax_res2.set_xlabel("Actual Seed", color="#909094", fontsize=11)
ax_res2.set_ylabel("Mean Residual (Pred − Actual)", color="#909094", fontsize=11)
ax_res2.set_title("Mean Residual by Seed\n(Green = over-predicted, Red = under-predicted)",
                  color="#fbfbff", fontsize=12, fontweight="bold", pad=10)
ax_res2.set_xticks(range(1, 17))
ax_res2.set_xticklabels(range(1, 17), color="#909094", fontsize=8)
_leg_res = [
    mpatches.Patch(color="#17b26a", label="Over-predicted (too high seed #)"),
    mpatches.Patch(color="#f04438", label="Under-predicted (too low seed #)"),
]
ax_res2.legend(handles=_leg_res, facecolor="#2a2a2d", edgecolor="#444",
               labelcolor="#fbfbff", fontsize=9)

fig_residuals.suptitle(
    f"Stage 2 Seed Regressor — Residual Analysis\n"
    f"OOF RMSE: {overall_rmse_ens:.4f}  |  GBR: {overall_rmse_gbr:.4f}  |  Ridge: {overall_rmse_ridge:.4f}",
    color="#fbfbff", fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("seed_regressor_residuals.png", dpi=150, bbox_inches="tight", facecolor="#1D1D20")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Test Seed Distribution + LOYO RMSE per Season
# ══════════════════════════════════════════════════════════════════════════════
fig_seed_dist, (ax_sd1, ax_sd2) = plt.subplots(1, 2, figsize=(16, 6))
fig_seed_dist.patch.set_facecolor("#1D1D20")
for _ax in (ax_sd1, ax_sd2):
    _ax.set_facecolor("#1D1D20")
    _ax.spines["bottom"].set_color("#444")
    _ax.spines["left"].set_color("#444")
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.tick_params(colors="#909094")

# Panel A: Distribution of predicted seeds for test tournament teams
_pred_series = pd.Series(seed_predictions_clipped).value_counts().sort_index()
_all_seeds_counts = pd.Series(0, index=range(1, 17))
_all_seeds_counts.update(_pred_series)

_bar_colors_dist = []
for _s in range(1, 17):
    if   _s <= 4:  _bar_colors_dist.append("#17b26a")
    elif _s <= 8:  _bar_colors_dist.append("#A1C9F4")
    elif _s <= 12: _bar_colors_dist.append("#FFB482")
    else:          _bar_colors_dist.append("#f04438")

ax_sd1.bar(range(1, 17), _all_seeds_counts.values,
           color=_bar_colors_dist, edgecolor="#1D1D20", width=0.72)
ax_sd1.set_xlabel("Predicted Seed", color="#909094", fontsize=11)
ax_sd1.set_ylabel("Number of Teams", color="#909094", fontsize=11)
ax_sd1.set_title(
    f"Predicted Seed Distribution — Test Teams\n(Threshold={PROBA_THRESHOLD}, N={_n_test_tourn} teams)",
    color="#fbfbff", fontsize=12, fontweight="bold", pad=10
)
ax_sd1.set_xticks(range(1, 17))
ax_sd1.set_xticklabels(range(1, 17), color="#909094", fontsize=9)
for _xi, _yi in zip(range(1, 17), _all_seeds_counts.values):
    if _yi > 0:
        ax_sd1.text(_xi, _yi + 0.15, str(_yi), ha="center", va="bottom",
                    color="#fbfbff", fontsize=8.5, fontweight="bold")
_leg_bins = [
    mpatches.Patch(color="#17b26a", label="Seeds 1–4  (top line)"),
    mpatches.Patch(color="#A1C9F4", label="Seeds 5–8"),
    mpatches.Patch(color="#FFB482", label="Seeds 9–12"),
    mpatches.Patch(color="#f04438", label="Seeds 13–16"),
]
ax_sd1.legend(handles=_leg_bins, facecolor="#2a2a2d", edgecolor="#444",
              labelcolor="#fbfbff", fontsize=9)

# Panel B: LOYO RMSE per season — GBR vs Ridge vs Ensemble
_x_pos = np.arange(len(reg_season_list))
_w = 0.25
ax_sd2.bar(_x_pos - _w, loyo_reg_df["rmse_gbr"],     width=_w, color="#A1C9F4",
           label="GBR",      edgecolor="#1D1D20")
ax_sd2.bar(_x_pos,       loyo_reg_df["rmse_ridge"],   width=_w, color="#FFB482",
           label="Ridge",    edgecolor="#1D1D20")
ax_sd2.bar(_x_pos + _w,  loyo_reg_df["rmse_ensemble"],width=_w, color="#8DE5A1",
           label="Ensemble", edgecolor="#1D1D20")
ax_sd2.axhline(overall_rmse_ens, color="#ffd400", linewidth=1.8, linestyle="--",
               label=f"Overall RMSE={overall_rmse_ens:.3f}")
ax_sd2.set_xticks(_x_pos)
ax_sd2.set_xticklabels(loyo_reg_df["season"], rotation=30, ha="right",
                       color="#909094", fontsize=9)
ax_sd2.set_ylabel("RMSE (seed units)", color="#909094", fontsize=11)
ax_sd2.set_title("LOYO RMSE per Season\nGBR vs Ridge vs Ensemble",
                 color="#fbfbff", fontsize=12, fontweight="bold", pad=10)
ax_sd2.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff", fontsize=9)

fig_seed_dist.suptitle(
    "Stage 2: Seed Regression — Test Predictions & CV Performance",
    color="#fbfbff", fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("seed_regressor_distribution.png", dpi=150, bbox_inches="tight", facecolor="#1D1D20")
plt.show()

print("\n✅ Stage 2 Seed Regressor complete.")
print(f"   Overall CV RMSE (Ensemble): {overall_rmse_ens:.4f}")
print(f"   Test teams with seed predictions: {len(seed_pred_df)}")
print(f"   'seed_pred_df' available for downstream use.")
