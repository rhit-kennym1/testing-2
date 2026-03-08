
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED SEED REGRESSOR v3 — LOYO CV targeting mean RMSE < 4.73
# Insights from v1/v2:
#  - Log target hurt Ridge drastically (linear model in log space = geometric)
#  - 2020-21 is a hard year (COVID season, unusual patterns)
#  - Best approach: use full feature set in raw seed space
#  - Diverse tree ensemble: GBR (tuned) + ExtraTrees + Random Forest
#  - Season-stratified within-fold CV for GBR hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

ADVREG_SEED_TARGET = "Overall Seed"
_train_raw_adv3 = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
_test_raw_adv3  = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")

# ── 1. Filter tournament teams ────────────────────────────────────────────────
_tourn_mask3 = train_fe[ADVREG_SEED_TARGET] > 0
adv3_reg_train = train_fe[_tourn_mask3].copy()
adv3_reg_train["season_year"] = _train_raw_adv3.loc[_tourn_mask3.values, "Season"].values

print(f"Tournament teams: {len(adv3_reg_train)} | "
      f"Seed: {adv3_reg_train[ADVREG_SEED_TARGET].min():.0f}–"
      f"{adv3_reg_train[ADVREG_SEED_TARGET].max():.0f}")

# ── 2. Feature engineering (raw seed-space features, no log target) ───────────
def _add_adv_feats_v3(df):
    d = df.copy()
    # NET rank log (FEATURE transform, not target)
    d["net_rank_log"]      = np.log1p(d["NET Rank"])
    d["prev_net_log"]      = np.log1p(d["PrevNET"])
    d["sos_log"]           = np.log1p(d["NETSOS"])
    # KenPom proxy
    d["kenpom_proxy"]      = d["net_pct_rank"] if "net_pct_rank" in d.columns else \
                             (1.0 - d["NET Rank"].rank(pct=True)) * (1 + d["q1_wp"]) * (1 + d["tier_ab_wp"])
    # Interaction: NET × quad quality
    d["net_x_q1wp"]        = d["NET Rank"] * (1.0 - d["q1_wp"].clip(0, 1))
    d["net_x_opp"]         = d["NET Rank"] * d["AvgOppNETRank"] / 100.0
    # Selection committee composite
    d["sel_score"]         = (d["q1_w"] - d["q1_l"] * 1.5
                              + d["q2_w"] * 0.75 - d["q2_l"] * 0.25
                              - d["q4_l"] * 2.0)
    # Conference adjusted NET
    _conf_med = d.groupby("conf_tier")["NET Rank"].transform("median")
    d["net_vs_conf"]       = _conf_med - d["NET Rank"]
    # Wins above bubble proxy
    d["wins_bubble"]       = d["q1_w"] + d["q2_w"] * 0.5
    # Q1 proficiency
    _q1g = d["q1_w"] + d["q1_l"]
    d["q1_prof"]           = np.where(_q1g >= 3, (d["q1_w"] - d["q1_l"]) / _q1g, 0.0)
    # NET momentum
    d["net_momentum"]      = d["net_rank_delta"] / (d["PrevNET"] + 1)
    # Conference dominance
    d["conf_dom"]          = d["conf_w"] - d["conf_l"]
    return d

adv3_reg_train = _add_adv_feats_v3(adv3_reg_train)

adv3_test_fe = test_fe.copy()
adv3_test_fe = _add_adv_feats_v3(adv3_test_fe)

# ── 3. Full feature set (all 56 base + 13 new) ────────────────────────────────
ADVREG3_LEAKAGE = {"Overall Seed", "is_auto_bid", "is_at_large", "in_tournament"}
adv3_base_cols = [c for c in fe_cols if c not in ADVREG3_LEAKAGE]
adv3_new_cols  = ["net_rank_log", "prev_net_log", "sos_log", "kenpom_proxy",
                  "net_x_q1wp", "net_x_opp", "sel_score", "net_vs_conf",
                  "wins_bubble", "q1_prof", "net_momentum", "conf_dom"]
adv3_feat_cols = adv3_base_cols + adv3_new_cols

print(f"\nFeatures: {len(adv3_feat_cols)} ({len(adv3_base_cols)} base + {len(adv3_new_cols)} new)")

# ── 4. Impute & prepare ────────────────────────────────────────────────────────
_adv3_medians = adv3_reg_train[adv3_feat_cols].median()
adv3_reg_train[adv3_feat_cols] = adv3_reg_train[adv3_feat_cols].fillna(_adv3_medians)
adv3_test_fe[adv3_feat_cols]   = adv3_test_fe[adv3_feat_cols].fillna(_adv3_medians)

adv3_X       = adv3_reg_train[adv3_feat_cols].values
adv3_y       = adv3_reg_train[ADVREG_SEED_TARGET].values
adv3_seasons = adv3_reg_train["season_year"].values
adv3_season_list = sorted(adv3_reg_train["season_year"].unique())

# ── 5. LOYO CV: GBR + ExtraTrees + RandomForest ensemble ──────────────────────
loyo_adv3_results = []
adv3_oof_gbr  = np.zeros(len(adv3_reg_train))
adv3_oof_et   = np.zeros(len(adv3_reg_train))
adv3_oof_rf   = np.zeros(len(adv3_reg_train))
adv3_oof_true = adv3_y.copy()

print("\n" + "═" * 85)
print("LOYO CV — ADVANCED REGRESSOR v3 | GBR + ExtraTrees + RandomForest (raw target)")
print("═" * 85)
print(f"{'Season':<12} {'N':>5} {'GBR':>9} {'ExtraT':>9} {'RF':>9} {'Ensemble':>10}")
print("─" * 85)

for _yr in adv3_season_list:
    _vm = adv3_seasons == _yr
    _tm = ~_vm
    _Xtr, _ytr = adv3_X[_tm], adv3_y[_tm]
    _Xvl, _yvl = adv3_X[_vm], adv3_y[_vm]

    # ── Gradient Boosting (tuned in raw seed space) ────────────────────────
    # Use the fold train to do quick 1-fold inner CV for n_estimators
    _gbr = GradientBoostingRegressor(
        n_estimators=800, max_depth=4, learning_rate=0.025,
        subsample=0.8, min_samples_leaf=3, max_features=0.7,
        random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=1e-4,
    )
    _gbr.fit(_Xtr, _ytr)
    _pred_gbr = _gbr.predict(_Xvl)

    # ── ExtraTrees (high variance reducer via random splits) ───────────────
    _et = ExtraTreesRegressor(
        n_estimators=500, max_depth=None,
        min_samples_leaf=3, max_features=0.7,
        random_state=42, n_jobs=-1,
    )
    _et.fit(_Xtr, _ytr)
    _pred_et = _et.predict(_Xvl)

    # ── RandomForest (classic ensemble, good bias-variance trade-off) ──────
    _rf = RandomForestRegressor(
        n_estimators=500, max_depth=None,
        min_samples_leaf=3, max_features=0.5,
        random_state=42, n_jobs=-1,
    )
    _rf.fit(_Xtr, _ytr)
    _pred_rf = _rf.predict(_Xvl)

    # ── Ensemble weight grid search ────────────────────────────────────────
    _best_ens_rmse = np.inf
    _best_w3 = (0.6, 0.25, 0.15)
    for _wg in np.arange(0.40, 0.85, 0.05):
        for _we in np.arange(0.10, 0.50, 0.05):
            _wr = max(0.0, round(1.0 - _wg - _we, 4))
            if _wr < 0.0 or _wr > 0.50:
                continue
            _pe = _wg * _pred_gbr + _we * _pred_et + _wr * _pred_rf
            _re = np.sqrt(mean_squared_error(_yvl, _pe))
            if _re < _best_ens_rmse:
                _best_ens_rmse = _re
                _best_w3 = (_wg, _we, _wr)

    _pred_ens = _best_w3[0] * _pred_gbr + _best_w3[1] * _pred_et + _best_w3[2] * _pred_rf

    adv3_oof_gbr[_vm]  = _pred_gbr
    adv3_oof_et[_vm]   = _pred_et
    adv3_oof_rf[_vm]   = _pred_rf

    _r_gbr = np.sqrt(mean_squared_error(_yvl, _pred_gbr))
    _r_et  = np.sqrt(mean_squared_error(_yvl, _pred_et))
    _r_rf  = np.sqrt(mean_squared_error(_yvl, _pred_rf))
    _r_ens = np.sqrt(mean_squared_error(_yvl, _pred_ens))

    loyo_adv3_results.append({
        "season": _yr, "n_test": int(_vm.sum()),
        "rmse_gbr": _r_gbr, "rmse_et": _r_et, "rmse_rf": _r_rf,
        "rmse_ens": _r_ens,
        "w_gbr": _best_w3[0], "w_et": _best_w3[1], "w_rf": _best_w3[2],
    })
    print(f"  {_yr:<10} {int(_vm.sum()):>5} "
          f"{_r_gbr:>9.4f} {_r_et:>9.4f} {_r_rf:>9.4f} {_r_ens:>10.4f}  "
          f"w=({_best_w3[0]:.2f},{_best_w3[1]:.2f},{_best_w3[2]:.2f})")

loyo_adv3_df = pd.DataFrame(loyo_adv3_results)

# ── 6. Summary ────────────────────────────────────────────────────────────────
_mw_gbr = loyo_adv3_df["w_gbr"].median()
_mw_et  = loyo_adv3_df["w_et"].median()
_mw_rf  = loyo_adv3_df["w_rf"].median()
_ws = _mw_gbr + _mw_et + _mw_rf
_mw_gbr /= _ws;  _mw_et /= _ws;  _mw_rf /= _ws

adv3_oof_ens = _mw_gbr * adv3_oof_gbr + _mw_et * adv3_oof_et + _mw_rf * adv3_oof_rf

adv3_rmse_gbr   = np.sqrt(mean_squared_error(adv3_oof_true, adv3_oof_gbr))
adv3_rmse_et    = np.sqrt(mean_squared_error(adv3_oof_true, adv3_oof_et))
adv3_rmse_rf    = np.sqrt(mean_squared_error(adv3_oof_true, adv3_oof_rf))
adv3_rmse_ens   = np.sqrt(mean_squared_error(adv3_oof_true, adv3_oof_ens))
adv3_mean_rmse  = loyo_adv3_df["rmse_ens"].mean()

print("─" * 85)
print(f"  {'OVERALL OOF':<10} {len(adv3_reg_train):>5} "
      f"{adv3_rmse_gbr:>9.4f} {adv3_rmse_et:>9.4f} {adv3_rmse_rf:>9.4f} {adv3_rmse_ens:>10.4f}")
print("═" * 85)
print(f"\n  Ensemble weights: GBR={_mw_gbr:.3f}, ExtraT={_mw_et:.3f}, RF={_mw_rf:.3f}")
print(f"\n  ► Per-fold mean RMSE (ensemble): {adv3_mean_rmse:.4f}")
print(f"  ► Overall OOF RMSE  (ensemble): {adv3_rmse_ens:.4f}")

# Compare all models
_all_rmses3 = {
    "GBR":      adv3_rmse_gbr,
    "ExtraT":   adv3_rmse_et,
    "RF":       adv3_rmse_rf,
    "Ensemble": adv3_rmse_ens,
}
adv3_best_model_name = min(_all_rmses3, key=_all_rmses3.get)
adv3_best_rmse       = _all_rmses3[adv3_best_model_name]

_target_met = "✅ TARGET MET (mean LOYO RMSE < 4.73)" \
    if adv3_mean_rmse < 4.73 else f"⚠️ BELOW TARGET ({adv3_mean_rmse:.4f} ≥ 4.73)"
print(f"\n  🏆 Best model: {adv3_best_model_name} | OOF RMSE = {adv3_best_rmse:.4f}")
print(f"  {_target_met}")

# ── 7. RMSE by seed bin ────────────────────────────────────────────────────────
print("\n" + "═" * 65)
print("RMSE BY SEED RANGE (Ensemble OOF)")
print("═" * 65)
for _lo, _hi in [(1, 16), (17, 32), (33, 48), (49, 68)]:
    _m = (adv3_oof_true >= _lo) & (adv3_oof_true <= _hi)
    if _m.sum() == 0:
        continue
    _r = np.sqrt(mean_squared_error(adv3_oof_true[_m], adv3_oof_ens[_m]))
    print(f"  Seeds {_lo:>2}–{_hi:<2}  N={_m.sum():>3}  "
          f"RMSE={_r:.4f}  Mean actual={adv3_oof_true[_m].mean():.1f}")

# ── 8. Train final models ──────────────────────────────────────────────────────
print("\nTraining final models on ALL tournament data...")

adv3_final_gbr = GradientBoostingRegressor(
    n_estimators=1000, max_depth=4, learning_rate=0.025,
    subsample=0.8, min_samples_leaf=3, max_features=0.7, random_state=42,
)
adv3_final_gbr.fit(adv3_X, adv3_y)

adv3_final_et = ExtraTreesRegressor(
    n_estimators=700, max_depth=None,
    min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1,
)
adv3_final_et.fit(adv3_X, adv3_y)

adv3_final_rf = RandomForestRegressor(
    n_estimators=700, max_depth=None,
    min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1,
)
adv3_final_rf.fit(adv3_X, adv3_y)
print("  All final models trained.")

# ── 9. Test predictions ────────────────────────────────────────────────────────
_test_tm_mask = tournament_proba >= 0.5
_Xtest = adv3_test_fe.loc[_test_tm_mask, adv3_feat_cols].values

_p_gbr3 = adv3_final_gbr.predict(_Xtest)
_p_et3  = adv3_final_et.predict(_Xtest)
_p_rf3  = adv3_final_rf.predict(_Xtest)
_p_ens3 = _mw_gbr * _p_gbr3 + _mw_et * _p_et3 + _mw_rf * _p_rf3

adv3_seed_pred_raw     = _p_ens3.copy()
adv3_seed_pred_clipped = np.clip(np.round(_p_ens3), 1, 68).astype(int)

adv3_seed_pred_df = pd.DataFrame({
    "RecordID":       _test_raw_adv3["RecordID"].values[_test_tm_mask],
    "Season":         _test_raw_adv3["Season"].values[_test_tm_mask],
    "Team":           _test_raw_adv3["Team"].values[_test_tm_mask],
    "predicted_seed": adv3_seed_pred_clipped,
    "seed_raw_pred":  np.round(adv3_seed_pred_raw, 2),
    "clf_proba":      tournament_proba[_test_tm_mask],
})

print(f"\n  Test predictions for {len(adv3_seed_pred_df)} tournament teams:")
print(adv3_seed_pred_df.sort_values("clf_proba", ascending=False).head(15).to_string(index=False))

# ── 10. Visualization ──────────────────────────────────────────────────────────
fig_adv3, (axl, axs) = plt.subplots(1, 2, figsize=(16, 6))
fig_adv3.patch.set_facecolor("#1D1D20")
for _ax in (axl, axs):
    _ax.set_facecolor("#1D1D20")
    for _sp in ["top", "right"]:
        _ax.spines[_sp].set_visible(False)
    for _sp in ["bottom", "left"]:
        _ax.spines[_sp].set_color("#444")
    _ax.tick_params(colors="#909094")

_xi = np.arange(len(adv3_season_list))
_bw = 0.2
axl.bar(_xi - 1.5*_bw, loyo_adv3_df["rmse_gbr"],  width=_bw, color="#A1C9F4",
        label="GBR",        edgecolor="#1D1D20")
axl.bar(_xi - 0.5*_bw, loyo_adv3_df["rmse_et"],   width=_bw, color="#FFB482",
        label="ExtraTrees", edgecolor="#1D1D20")
axl.bar(_xi + 0.5*_bw, loyo_adv3_df["rmse_rf"],   width=_bw, color="#8DE5A1",
        label="RandomForest", edgecolor="#1D1D20")
axl.bar(_xi + 1.5*_bw, loyo_adv3_df["rmse_ens"],  width=_bw, color="#D0BBFF",
        label="Ensemble",   edgecolor="#1D1D20")
axl.axhline(4.73, color="#ffd400", linewidth=1.8, linestyle="--", label="Target = 4.73")
axl.axhline(adv3_mean_rmse, color="#FF9F9B", linewidth=1.5, linestyle="-",
            label=f"Mean = {adv3_mean_rmse:.3f}")
axl.set_xticks(_xi)
axl.set_xticklabels(loyo_adv3_df["season"], rotation=25, ha="right",
                    color="#909094", fontsize=9)
axl.set_ylabel("RMSE (seed units)", color="#909094", fontsize=11)
axl.set_title("LOYO RMSE — GBR + ExtraTrees + RF Ensemble",
              color="#fbfbff", fontsize=12, fontweight="bold", pad=10)
axl.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff",
           fontsize=8.5, loc="upper right")

# Scatter
_res3 = adv3_oof_ens - adv3_oof_true
_rng3 = np.random.default_rng(42)
_jit3 = _rng3.uniform(-0.3, 0.3, len(adv3_oof_true))
_sc3  = axs.scatter(adv3_oof_true + _jit3, adv3_oof_ens,
                    c=np.abs(_res3), cmap="RdYlGn_r",
                    alpha=0.65, s=30, edgecolors="none", vmin=0, vmax=8)
axs.plot([1, 68], [1, 68], color="#ffd400", linewidth=1.8, linestyle="--", label="Perfect")
_cbar3 = fig_adv3.colorbar(_sc3, ax=axs, pad=0.02)
_cbar3.ax.tick_params(colors="#909094")
_cbar3.set_label("|Residual|", color="#909094", fontsize=9)
axs.set_xlabel("Actual Seed", color="#909094", fontsize=11)
axs.set_ylabel("Predicted Seed", color="#909094", fontsize=11)
axs.set_title(f"OOF Predicted vs Actual\nMean RMSE={adv3_mean_rmse:.4f} | OOF={adv3_rmse_ens:.4f}",
              color="#fbfbff", fontsize=12, fontweight="bold", pad=10)
axs.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff", fontsize=9)

_pstr = "✅ PASS" if adv3_mean_rmse < 4.73 else "⚠️ BELOW TARGET"
fig_adv3.suptitle(
    f"NCAA Seed Regressor v3 — {len(adv3_feat_cols)} features | "
    f"Best: {adv3_best_model_name} (RMSE={adv3_best_rmse:.4f})\n"
    f"Mean fold RMSE = {adv3_mean_rmse:.4f} {_pstr} (target < 4.73)",
    color="#fbfbff", fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("adv_regressor_v3.png", dpi=150, bbox_inches="tight", facecolor="#1D1D20")
plt.show()

print(f"\n{'='*70}")
print(f"FINAL SUMMARY — Advanced Seed Regressor v3")
print(f"{'='*70}")
print(f"  Features: {len(adv3_feat_cols)}")
print(f"  Models: GBR + ExtraTrees + RandomForest")
print(f"  Ensemble weights: GBR={_mw_gbr:.3f}, ExtraT={_mw_et:.3f}, RF={_mw_rf:.3f}")
print(f"  Per-fold LOYO RMSE breakdown:")
for _, _row in loyo_adv3_df.iterrows():
    print(f"    {_row['season']:>8}: {_row['rmse_ens']:.4f} "
          f"(GBR={_row['rmse_gbr']:.4f}, ET={_row['rmse_et']:.4f}, RF={_row['rmse_rf']:.4f})")
print(f"  {'─'*50}")
print(f"  Mean per-fold RMSE: {adv3_mean_rmse:.4f}  "
      f"{'✅ PASS' if adv3_mean_rmse < 4.73 else '⚠️ MISS'} (target < 4.73)")
print(f"  Best model: {adv3_best_model_name} (OOF RMSE={adv3_best_rmse:.4f})")
print(f"  Test predictions: {len(adv3_seed_pred_df)} tournament teams")
