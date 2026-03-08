
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: TOURNAMENT SELECTION CLASSIFIER
# Binary target: in_tournament (1 = tournament team, 0 = not)
# Method: Leave-One-Year-Out (LOYO) cross-validation
# Model: GradientBoostingClassifier (sklearn) with balanced class weights
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Reload CSVs to get the Season column for LOYO splits ──────────────────
_train_raw = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
_test_raw  = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")

# ── 2. Define binary classification target ───────────────────────────────────
BINARY_TARGET = "in_tournament"

# ── 3. Define model features — drop leakage cols ─────────────────────────────
# is_auto_bid + is_at_large directly encode tournament selection (same info as target)
# Overall Seed is the downstream regression target, not a classifier input
LEAKAGE_COLS = {"is_auto_bid", "is_at_large", "in_tournament", "Overall Seed"}
clf_feat_cols = [c for c in fe_cols if c not in LEAKAGE_COLS]

print(f"Classification features: {len(clf_feat_cols)}")
print(f"Leakage cols dropped: {sorted(LEAKAGE_COLS)}\n")

# ── 4. Assemble full training frame with Season ───────────────────────────────
clf_train = train_fe[clf_feat_cols].copy()
clf_train["season_year"] = _train_raw["Season"].values
clf_train[BINARY_TARGET] = train_fe[BINARY_TARGET].values

season_years = sorted(clf_train["season_year"].unique())
print(f"Seasons available: {season_years}")
print(f"Total rows: {len(clf_train)}")
class_counts = clf_train[BINARY_TARGET].value_counts()
print(f"Class distribution — 0 (non-tournament): {class_counts[0]}, 1 (tournament): {class_counts[1]}")

# ── 5. Class balance ratio ────────────────────────────────────────────────────
_n_neg = class_counts[0]
_n_pos = class_counts[1]
scale_pos_weight_global = round(_n_neg / _n_pos, 2)
print(f"\nClass weight ratio (neg/pos): {scale_pos_weight_global}")

# ── 6. Leave-One-Year-Out Cross-Validation ────────────────────────────────────
loyo_results   = []
all_oof_proba   = np.zeros(len(clf_train))
all_oof_true    = clf_train[BINARY_TARGET].values
all_oof_predict = np.zeros(len(clf_train), dtype=int)

print("\n" + "═" * 70)
print("LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
print("═" * 70)
print(f"{'Season':<12} {'N_test':>7} {'Pos%':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("─" * 70)

for _year in season_years:
    _mask_val = clf_train["season_year"] == _year
    _mask_tr  = ~_mask_val

    _X_tr  = clf_train.loc[_mask_tr,  clf_feat_cols]
    _y_tr  = clf_train.loc[_mask_tr,  BINARY_TARGET]
    _X_val = clf_train.loc[_mask_val, clf_feat_cols]
    _y_val = clf_train.loc[_mask_val, BINARY_TARGET]

    _model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    _model.fit(_X_tr, _y_tr)

    _proba = _model.predict_proba(_X_val)[:, 1]
    _pred  = (_proba >= 0.5).astype(int)

    # Store OOF predictions
    all_oof_proba[_mask_val]   = _proba
    all_oof_predict[_mask_val] = _pred

    # Per-year metrics
    _prec = precision_score(_y_val, _pred, zero_division=0)
    _rec  = recall_score(_y_val, _pred, zero_division=0)
    _f1   = f1_score(_y_val, _pred, zero_division=0)
    _auc  = roc_auc_score(_y_val, _proba) if len(_y_val.unique()) == 2 else float("nan")
    _pos_pct = 100 * _y_val.mean()

    loyo_results.append({
        "season": _year, "n_test": len(_y_val),
        "pos_pct": _pos_pct,
        "precision": _prec, "recall": _rec, "f1": _f1, "auc": _auc,
    })
    print(f"  {_year:<10} {len(_y_val):>7} {_pos_pct:>5.1f}% {_prec:>7.3f} {_rec:>7.3f} {_f1:>7.3f} {_auc:>7.3f}")

loyo_df = pd.DataFrame(loyo_results)

# ── 7. Overall OOF metrics ────────────────────────────────────────────────────
oof_prec = precision_score(all_oof_true, all_oof_predict, zero_division=0)
oof_rec  = recall_score(all_oof_true, all_oof_predict, zero_division=0)
oof_f1   = f1_score(all_oof_true, all_oof_predict, zero_division=0)
oof_auc  = roc_auc_score(all_oof_true, all_oof_proba)
mean_auc = loyo_df["auc"].mean()

print("─" * 70)
print(f"  {'OVERALL OOF':<10} {len(clf_train):>7} {'':>6} {oof_prec:>7.3f} {oof_rec:>7.3f} {oof_f1:>7.3f} {oof_auc:>7.3f}")
print("═" * 70)
print(f"\n  Mean LOYO AUC : {mean_auc:.4f} {'✅ PASS' if mean_auc > 0.85 else '⚠️  BELOW TARGET'}")
print(f"  OOF AUC       : {oof_auc:.4f}")
print(f"  OOF Precision : {oof_prec:.4f}")
print(f"  OOF Recall    : {oof_rec:.4f}")
print(f"  OOF F1        : {oof_f1:.4f}")

# ── 8. Confusion Matrix (OOF) ─────────────────────────────────────────────────
_cm = confusion_matrix(all_oof_true, all_oof_predict)
print("\n" + "═" * 40)
print("CONFUSION MATRIX (OOF - all folds)")
print("═" * 40)
print(f"              Predicted")
print(f"              Non-Tourn  Tourn")
print(f"  Non-Tourn     {_cm[0,0]:>5}      {_cm[0,1]:>5}")
print(f"  Tourn         {_cm[1,0]:>5}      {_cm[1,1]:>5}")
_tn, _fp, _fn, _tp = _cm.ravel()
print(f"\n  True Neg  (TN): {_tn}  |  False Pos (FP): {_fp}")
print(f"  False Neg (FN): {_fn}  |  True Pos  (TP): {_tp}")
print(f"  Specificity: {_tn/(_tn+_fp):.3f} | Sensitivity: {_tp/(_tp+_fn):.3f}")

# ── 9. Train FINAL model on ALL training data ─────────────────────────────────
print("\n" + "═" * 60)
print("TRAINING FINAL MODEL ON ALL DATA...")
X_all = clf_train[clf_feat_cols]
y_all = clf_train[BINARY_TARGET]

clf_final = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)
clf_final.fit(X_all, y_all)
print(f"  Final model trained | n_features={len(clf_feat_cols)}")

# ── 10. Predict on test set ───────────────────────────────────────────────────
_X_test          = test_fe[clf_feat_cols].copy()
tournament_proba = clf_final.predict_proba(_X_test)[:, 1]
tournament_pred  = (tournament_proba >= 0.5).astype(int)

print(f"\n  Test set: {len(tournament_proba)} teams")
print(f"  Predicted in tournament: {tournament_pred.sum()} ({100*tournament_pred.mean():.1f}%)")
print(f"  Proba range: [{tournament_proba.min():.3f}, {tournament_proba.max():.3f}]")

# ── 11. Feature Importance Chart ─────────────────────────────────────────────
_imp     = pd.Series(clf_final.feature_importances_, index=clf_feat_cols).sort_values(ascending=False)
_top_imp = _imp.head(20)

# Color-code by category
_NET_FEATS  = {"NET Rank", "PrevNET", "net_rank_tier", "prev_net_rank_tier",
               "net_rank_delta", "net_sos_interaction"}
_QUAD_PRFX  = ("q1_", "q2_", "q3_", "q4_", "quad_", "bad_", "tier_ab")
_SOS_FEATS  = {"NETSOS", "NETNonConfSOS", "AvgOppNETRank", "AvgOppNET",
               "sos_composite", "opp_quality_delta", "net_vs_opp_delta",
               "net_vs_avgopp_rank"}
_CONF_FEATS = {"conf_tier", "is_power_conf"}

def _feat_color(fn):
    if fn in _NET_FEATS:  return "#A1C9F4"
    if any(fn.startswith(p) for p in _QUAD_PRFX): return "#FFB482"
    if fn in _SOS_FEATS:  return "#8DE5A1"
    if fn in _CONF_FEATS: return "#D0BBFF"
    return "#FF9F9B"

_rev_names       = list(_top_imp.index)[::-1]
_rev_vals        = list(_top_imp.values)[::-1]
_feat_colors_top = [_feat_color(fn) for fn in _rev_names]

fig_clf_imp, ax_clf = plt.subplots(figsize=(12, 8))
fig_clf_imp.patch.set_facecolor("#1D1D20")
ax_clf.set_facecolor("#1D1D20")

_bars_imp = ax_clf.barh(range(len(_rev_names)), _rev_vals, color=_feat_colors_top,
                        edgecolor="#1D1D20", height=0.72)
ax_clf.set_yticks(range(len(_rev_names)))
ax_clf.set_yticklabels(_rev_names, fontsize=10, color="#fbfbff")
ax_clf.set_xlabel("Gradient Boosting Feature Importance (mean decrease impurity)", color="#909094", fontsize=11)
ax_clf.set_title(
    "Top 20 Features — GradientBoosting Tournament Selection Classifier\n"
    "(Leave-One-Year-Out | binary: in_tournament)",
    color="#fbfbff", fontsize=13, fontweight="bold", pad=14
)
ax_clf.tick_params(colors="#909094")
ax_clf.spines["bottom"].set_color("#444")
ax_clf.spines["left"].set_color("#444")
ax_clf.spines["top"].set_visible(False)
ax_clf.spines["right"].set_visible(False)

_max_v = max(_rev_vals) if _rev_vals else 1
for _v, _rect in zip(_rev_vals, _bars_imp):
    ax_clf.text(_v + _max_v * 0.01, _rect.get_y() + _rect.get_height() / 2,
                f"{_v:.4f}", va="center", ha="left", color="#fbfbff", fontsize=8.5)

_leg = [
    mpatches.Patch(color="#A1C9F4", label="NET / Rank"),
    mpatches.Patch(color="#FFB482", label="Quadrant"),
    mpatches.Patch(color="#8DE5A1", label="SOS / Opp Quality"),
    mpatches.Patch(color="#D0BBFF", label="Conference"),
    mpatches.Patch(color="#FF9F9B", label="Win % / Record"),
]
ax_clf.legend(handles=_leg, facecolor="#2a2a2d", edgecolor="#444",
              labelcolor="#fbfbff", fontsize=9, loc="lower right")
ax_clf.set_xlim(0, _max_v * 1.18)
plt.tight_layout()
plt.savefig("tournament_classifier_importance.png", dpi=150, bbox_inches="tight",
            facecolor="#1D1D20")
plt.show()

# ── 12. LOYO AUC per season chart ─────────────────────────────────────────────
fig_auc, ax_auc = plt.subplots(figsize=(12, 5))
fig_auc.patch.set_facecolor("#1D1D20")
ax_auc.set_facecolor("#1D1D20")

_seasons = loyo_df["season"]
_aucs    = loyo_df["auc"]
_bar_c   = ["#17b26a" if v > 0.85 else "#f04438" for v in _aucs]

ax_auc.bar(range(len(_seasons)), _aucs, color=_bar_c, edgecolor="#1D1D20", width=0.6)
ax_auc.axhline(0.85, color="#ffd400", linestyle="--", linewidth=1.5, label="Target AUC = 0.85")
ax_auc.axhline(mean_auc, color="#A1C9F4", linestyle="-", linewidth=1.5,
               label=f"Mean AUC = {mean_auc:.3f}")

ax_auc.set_xticks(range(len(_seasons)))
ax_auc.set_xticklabels(_seasons, rotation=45, ha="right", color="#fbfbff", fontsize=9)
ax_auc.set_ylabel("ROC-AUC", color="#909094", fontsize=11)
ax_auc.set_title("LOYO Cross-Validation AUC per Season\n(Tournament Selection Classifier)",
                 color="#fbfbff", fontsize=13, fontweight="bold", pad=12)
ax_auc.set_ylim(0.5, 1.05)
ax_auc.tick_params(colors="#909094")
ax_auc.spines["bottom"].set_color("#444")
ax_auc.spines["left"].set_color("#444")
ax_auc.spines["top"].set_visible(False)
ax_auc.spines["right"].set_visible(False)
ax_auc.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff", fontsize=9)

for _xi, _yi in zip(range(len(_seasons)), _aucs):
    ax_auc.text(_xi, _yi + 0.008, f"{_yi:.3f}", ha="center", va="bottom",
                color="#fbfbff", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig("tournament_classifier_loyo_auc.png", dpi=150, bbox_inches="tight",
            facecolor="#1D1D20")
plt.show()

print(f"\n✅ tournament_proba generated for {len(tournament_proba)} test samples.")
print(f"✅ Mean LOYO AUC: {mean_auc:.4f} | OOF AUC: {oof_auc:.4f}")
