
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUBMISSION v2: Best Model → adv3 ensemble (GBR + ExtraTrees + RF)
# - Predicts Overall Seed in 1–68 range (the actual competition target)
# - Only rows in submission_template get predictions (tournament teams)
# - Non-tournament rows receive seed = 0.0
# - Clips to valid range 1–68; decimal precision preserved (no rounding)
# ══════════════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  FINAL SUBMISSION BUILD — Overall Seed 1-68 (adv3 ensemble, float)")
print("═" * 72)

# ── 1. Load template to get all 451 RecordIDs in correct order ────────────────
_template_v2 = pd.read_csv("submission_template2.0.csv")
print(f"\nTemplate: {_template_v2.shape[0]} rows | columns: {list(_template_v2.columns)}")

# ── 2. Model selection: compare OOF RMSE across models ───────────────────────
print("\n── Model Comparison ────────────────────────────────────────────────────")
print(f"  basic GBR+Ridge ensemble    OOF RMSE = {float(overall_rmse_ens):.4f}")
print(f"  adv3  GBR+ExtraTrees+RF     OOF RMSE = {float(adv3_rmse_ens):.4f} (65 features)")
print(f"  adv3  mean per-fold RMSE    = {float(adv3_mean_rmse):.4f}")

# Select best model: lower overall OOF RMSE wins
_use_adv3 = float(adv3_rmse_ens) < float(overall_rmse_ens)
_best_model_label = "adv3 (GBR+ExtraTrees+RF)" if _use_adv3 else "basic (GBR+Ridge)"
print(f"\n  ✅ Selected: {_best_model_label} (lower OOF RMSE)")

# ── 3. Build base submission ──────────────────────────────────────────────────
sub_v2 = _template_v2[["RecordID"]].copy()
sub_v2["Overall Seed"] = 0.0   # default = non-tournament (float)

# ── 4. Apply seed predictions for tournament teams ───────────────────────────
# Use raw continuous predictions (1-68 scale) from the best model
_raw_preds = adv3_seed_pred_raw if _use_adv3 else seed_predictions_raw

# Build map: RecordID → clipped float seed (NO rounding)
_test_raw_build = pd.read_csv("NCAA_Seed_Test_Set2.0.csv")
_tournament_mask_build = tournament_proba >= 0.5  # same threshold as model
_tourn_record_ids_build = _test_raw_build["RecordID"].values[_tournament_mask_build]

# Clip to [1, 68] — decimal precision preserved
_seeds_float = np.clip(_raw_preds, 1.0, 68.0)
_seed_map_v2 = dict(zip(_tourn_record_ids_build, _seeds_float))

# Apply via vectorised map
_sub_map = sub_v2["RecordID"].map(_seed_map_v2)
_tourn_mask_sub = _sub_map.notna()
sub_v2.loc[_tourn_mask_sub, "Overall Seed"] = _sub_map[_tourn_mask_sub]
_n_applied_v2 = int(_tourn_mask_sub.sum())

print(f"\n  Applied predictions to {_n_applied_v2} tournament teams")

# ── 5. Reorder to template ordering exactly ────────────────────────────────────
sub_v2 = sub_v2.set_index("RecordID").reindex(_template_v2["RecordID"]).reset_index()
sub_v2["Overall Seed"] = sub_v2["Overall Seed"].fillna(0.0)

# ── 6. Validate ───────────────────────────────────────────────────────────────
assert len(sub_v2) == 451,                   f"Row count error: {len(sub_v2)}"
assert sub_v2["RecordID"].isna().sum() == 0, "Missing RecordIDs!"
assert sub_v2["Overall Seed"].isna().sum() == 0, "Null seeds!"

_tourn_seeds_sub = sub_v2.loc[sub_v2["Overall Seed"] > 0, "Overall Seed"]
assert float(_tourn_seeds_sub.min()) >= 1.0,  f"Seed below 1: {_tourn_seeds_sub.min()}"
assert float(_tourn_seeds_sub.max()) <= 68.0, f"Seed above 68: {_tourn_seeds_sub.max()}"

# Confirm no rounding was applied (at least some values should be non-integer)
_non_int_count = int(((_tourn_seeds_sub % 1) != 0).sum())
print(f"\n  ✅ Validation passed:")
print(f"     Rows: {len(sub_v2)} (expected 451)")
print(f"     Tournament teams (seed > 0): {int((sub_v2['Overall Seed'] > 0).sum())}")
print(f"     Non-tournament (seed = 0.0):  {int((sub_v2['Overall Seed'] == 0).sum())}")
print(f"     Seed range: {float(_tourn_seeds_sub.min()):.4f} – {float(_tourn_seeds_sub.max()):.4f}")
print(f"     Float seeds (non-integer): {_non_int_count} / {_n_applied_v2}  ✅ decimal precision preserved")

# ── 7. Save submission ─────────────────────────────────────────────────────────
sub_v2.to_csv("submission.csv", index=False)
print(f"\n  ✅ Saved → submission.csv  (Overall Seed column = float64)")

# ── 8. Print seed distribution (binned) + compare to training ─────────────────
print("\n" + "═" * 72)
print("  SEED DISTRIBUTION SANITY CHECK: Predictions vs Training")
print("═" * 72)

_train_tourn     = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
_train_seeds     = _train_tourn["Overall Seed"].dropna()
_train_seeds     = _train_seeds[_train_seeds > 0]
_n_train_seasons = int(_train_tourn["Season"].nunique())
_n_test_seasons  = int(_test_raw_build["Season"].nunique())

_tier_defs = [
    ("Seeds  1–16  (top tier)",   1, 16),
    ("Seeds 17–32  (solid)",     17, 32),
    ("Seeds 33–48  (bubble)",    33, 48),
    ("Seeds 49–68  (long shots)", 49, 68),
]

print(f"\n  Training: {len(_train_seeds)} tournament rows | {_n_train_seasons} seasons")
print(f"  Predicted: {_n_applied_v2} tournament rows | {_n_test_seasons} seasons")
print(f"  Seed dtype in submission: {sub_v2['Overall Seed'].dtype}")
print(f"  Sample float seeds (first 5): {[round(v, 3) for v in _tourn_seeds_sub.values[:5].tolist()]}")

print(f"\n  Tier Breakdown (using float seed values):")
print(f"  {'Tier':<30} {'Pred':>6} {'Pred%':>7} {'Train':>7} {'Train%':>7} {'Expected':>9}")
print(f"  {'─'*70}")

for _tier_lbl, _lo, _hi in _tier_defs:
    _pred_cnt  = int(((_tourn_seeds_sub >= _lo) & (_tourn_seeds_sub <= _hi)).sum())
    _train_cnt = int(((_train_seeds >= _lo) & (_train_seeds <= _hi)).sum())
    _pred_pct  = _pred_cnt  / _n_applied_v2 * 100
    _train_pct = _train_cnt / len(_train_seeds) * 100
    _expected  = round(_train_cnt / _n_train_seasons * _n_test_seasons)
    print(f"  {_tier_lbl:<30} {_pred_cnt:>6} {_pred_pct:>7.1f}% {_train_cnt:>7} {_train_pct:>7.1f}% {_expected:>9}")

# Per-seed-line distribution (floor for display bucketing only)
print(f"\n  Distribution by seed line (floor-bucketed for display):")
print(f"  {'Seed':>5} | {'Pred':>5} | {'Train/season':>12} | {'Delta':>7} | Pred mean seed in line")
print(f"  {'─'*65}")

_train_vc = _train_seeds.apply(np.floor).astype(int).value_counts().sort_index()
for _s in range(1, 69):
    # Count predictions that fall in [s, s+1) bucket (floor == s)
    _pred_bucket = _tourn_seeds_sub[(_tourn_seeds_sub >= _s) & (_tourn_seeds_sub < _s + 1)]
    _pred_cnt  = len(_pred_bucket)
    _train_cnt = int(_train_vc.get(_s, 0))
    _train_avg = _train_cnt / _n_train_seasons
    _delta     = _pred_cnt - round(_train_avg * _n_test_seasons)
    _mean_in_bucket = f"{_pred_bucket.mean():.3f}" if _pred_cnt > 0 else "—"
    if _pred_cnt > 0 or _train_cnt > 0:
        print(f"  {_s:>5} | {_pred_cnt:>5} | {_train_avg:>12.2f} | {_delta:>+7} | {_mean_in_bucket}")

print(f"\n  ── Summary Statistics ──────────────────────────────────────────────")
print(f"  Predicted: mean={_tourn_seeds_sub.mean():.3f}, median={_tourn_seeds_sub.median():.3f}, "
      f"std={_tourn_seeds_sub.std():.3f}, min={_tourn_seeds_sub.min():.3f}, max={_tourn_seeds_sub.max():.3f}")
print(f"  Training:  mean={_train_seeds.mean():.3f}, median={_train_seeds.median():.3f}, "
      f"std={_train_seeds.std():.3f}")

# ── 9. Distribution Visualization ─────────────────────────────────────────────
_BG = "#1D1D20"

def _tier_color(s):
    if   s <= 16: return "#17b26a"
    elif s <= 32: return "#A1C9F4"
    elif s <= 48: return "#FFB482"
    else:         return "#f04438"

# Build floor-bucketed counts for visualization (bucket by floor of seed)
_seeds_full      = np.arange(1, 69)
_pred_full       = np.array([
    len(_tourn_seeds_sub[(_tourn_seeds_sub >= s) & (_tourn_seeds_sub < s + 1)])
    for s in _seeds_full
])
_train_full_avg  = np.array([
    int(_train_vc.get(s, 0)) / _n_train_seasons * _n_test_seasons
    for s in _seeds_full
])
_bar_clrs = [_tier_color(s) for s in _seeds_full]

fig_submission_dist, (ax_sub, ax_compare) = plt.subplots(1, 2, figsize=(20, 7))
fig_submission_dist.patch.set_facecolor(_BG)

for _axi in (ax_sub, ax_compare):
    _axi.set_facecolor(_BG)
    _axi.spines["bottom"].set_color("#444")
    _axi.spines["left"].set_color("#444")
    _axi.spines["top"].set_visible(False)
    _axi.spines["right"].set_visible(False)
    _axi.tick_params(colors="#909094")

# Panel A: Predicted seed distribution (floor-bucketed)
ax_sub.bar(_seeds_full, _pred_full, color=_bar_clrs, edgecolor=_BG, width=0.75, linewidth=0.4)
for _xi, _yi in zip(_seeds_full, _pred_full):
    if _yi > 0:
        ax_sub.text(_xi, _yi + 0.05, str(_yi), ha="center", va="bottom",
                    color="#fbfbff", fontsize=7, fontweight="bold")

ax_sub.set_xlabel("Predicted Overall Seed (1–68, float bucketed by floor)", color="#909094", fontsize=11)
ax_sub.set_ylabel("Number of Teams", color="#909094", fontsize=11)
ax_sub.set_xlim(0.5, 68.5)
ax_sub.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64, 68])
ax_sub.set_xticklabels([1, 8, 16, 24, 32, 40, 48, 56, 64, 68], color="#909094", fontsize=9)
ax_sub.set_title(
    f"Predicted Seed Distribution — Test Set (Float Seeds)\n"
    f"{_n_applied_v2} Tournament Teams | Model: {_best_model_label}",
    color="#fbfbff", fontsize=12, fontweight="bold", pad=12
)
_leg_items = [
    mpatches.Patch(color="#17b26a", label="Seeds 1–16  (top tier)"),
    mpatches.Patch(color="#A1C9F4", label="Seeds 17–32 (solid)"),
    mpatches.Patch(color="#FFB482", label="Seeds 33–48 (bubble)"),
    mpatches.Patch(color="#f04438", label="Seeds 49–68 (long shots)"),
]
ax_sub.legend(handles=_leg_items, facecolor="#2a2a2d", edgecolor="#444",
              labelcolor="#fbfbff", fontsize=9, loc="upper right")

# Panel B: Predicted vs Training (per-season avg, scaled to test seasons)
_w2 = 0.38
ax_compare.bar(_seeds_full - _w2/2, _pred_full,      width=_w2, color="#A1C9F4",
               label="Predicted (test)", edgecolor=_BG, alpha=0.9)
ax_compare.bar(_seeds_full + _w2/2, _train_full_avg, width=_w2, color="#FFB482",
               label="Training avg (scaled)", edgecolor=_BG, alpha=0.9)
ax_compare.set_xlabel("Overall Seed (1–68)", color="#909094", fontsize=11)
ax_compare.set_ylabel("Team Count", color="#909094", fontsize=11)
ax_compare.set_xlim(0.5, 68.5)
ax_compare.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64, 68])
ax_compare.set_xticklabels([1, 8, 16, 24, 32, 40, 48, 56, 64, 68], color="#909094", fontsize=9)
ax_compare.set_title(
    f"Predictions vs Training Distribution\n"
    f"Training avg scaled to {_n_test_seasons} seasons",
    color="#fbfbff", fontsize=12, fontweight="bold", pad=12
)
ax_compare.legend(facecolor="#2a2a2d", edgecolor="#444", labelcolor="#fbfbff",
                  fontsize=9, loc="upper right")

fig_submission_dist.suptitle(
    f"Final Submission — Overall Seed 1-68 Distribution (Float, No Rounding)\n"
    f"Predicted mean={_tourn_seeds_sub.mean():.3f}  |  Training mean={_train_seeds.mean():.3f}  "
    f"|  OOF RMSE={float(adv3_rmse_ens):.4f}",
    color="#fbfbff", fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("submission_seed_distribution.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

print(f"\n{'═'*72}")
print(f"  SUBMISSION COMPLETE")
print(f"{'═'*72}")
print(f"  File: submission.csv")
print(f"  Total rows: {len(sub_v2)}")
print(f"  Tournament teams: {int((sub_v2['Overall Seed'] > 0).sum())}")
print(f"  Non-tournament:   {int((sub_v2['Overall Seed'] == 0).sum())}")
print(f"  Seed dtype: {sub_v2['Overall Seed'].dtype}  (float64 — decimal precision preserved)")
print(f"  Seed range: {float(_tourn_seeds_sub.min()):.4f} – {float(_tourn_seeds_sub.max()):.4f}")
print(f"  Predicted mean: {_tourn_seeds_sub.mean():.4f} (training mean: {_train_seeds.mean():.4f})")
print(f"  Float seeds (non-integer values): {_non_int_count} / {_n_applied_v2}")
print(f"  Model: {_best_model_label} | OOF RMSE: {float(adv3_rmse_ens):.4f}")
