
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Load & engineer numeric features ────────────────────────────────────────
corr_df = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
target_col = "Overall Seed"

# ── Parse W-L strings into numeric wins / losses / win_pct ─────────────────
def parse_wl(s):
    """Return (wins, losses) from a 'W-L' string."""
    try:
        w, l = str(s).split("-")
        return int(w), int(l)
    except Exception:
        return np.nan, np.nan

for col, prefix in [("WL", "ovr"), ("Conf.Record", "conf"),
                    ("Non-ConferenceRecord", "nc"), ("RoadWL", "road"),
                    ("Quadrant1", "q1"), ("Quadrant2", "q2"),
                    ("Quadrant3", "q3"), ("Quadrant4", "q4")]:
    wins, losses = zip(*corr_df[col].apply(parse_wl))
    corr_df[f"{prefix}_wins"]   = wins
    corr_df[f"{prefix}_losses"] = losses
    w_arr = np.array(wins, dtype=float)
    l_arr = np.array(losses, dtype=float)
    total = w_arr + l_arr
    corr_df[f"{prefix}_win_pct"] = np.where(total > 0, w_arr / total, np.nan)

# ── Build numeric feature matrix ────────────────────────────────────────────
base_numeric = ["NET Rank", "PrevNET", "AvgOppNETRank", "AvgOppNET",
                "NETSOS", "NETNonConfSOS"]
derived = [c for c in corr_df.columns if any(c.startswith(p) for p in
           ["ovr_", "conf_", "nc_", "road_", "q1_", "q2_", "q3_", "q4_"])]
all_features = base_numeric + derived

# Keep only rows with a valid target
analysis = corr_df[all_features + [target_col]].dropna(subset=[target_col])

# ── Pearson correlation with target ─────────────────────────────────────────
corr_with_target = (analysis
    .apply(pd.to_numeric, errors="coerce")
    .corr()[target_col]
    .drop(target_col)
    .sort_values())

print("Correlation with 'Overall Seed' (top positive & negative):")
print(corr_with_target.round(3).to_string())

# ── Select top 15 by absolute correlation ───────────────────────────────────
top_features = corr_with_target.abs().nlargest(15).index.tolist()
top_features_ordered = corr_with_target[top_features].sort_values().index.tolist()

heatmap_cols = top_features_ordered + [target_col]
corr_matrix = (analysis[heatmap_cols]
               .apply(pd.to_numeric, errors="coerce")
               .corr())

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor("#1D1D20")

# Left: full correlation matrix heatmap
ax_heat = axes[0]
ax_heat.set_facecolor("#1D1D20")

cmap = plt.cm.RdYlBu_r
im = ax_heat.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors="#909094")
cbar.set_label("Pearson r", color="#909094", fontsize=10)

ax_heat.set_xticks(range(len(heatmap_cols)))
ax_heat.set_yticks(range(len(heatmap_cols)))
ax_heat.set_xticklabels(heatmap_cols, rotation=45, ha="right",
                         fontsize=8, color="#909094")
ax_heat.set_yticklabels(heatmap_cols, fontsize=8, color="#909094")
ax_heat.set_title("Correlation Matrix — Top 15 Features vs Overall Seed",
                  color="#fbfbff", fontsize=12, fontweight="bold", pad=12)
ax_heat.spines[:].set_color("#444")

# Annotate cells with correlation value
for i in range(len(heatmap_cols)):
    for j in range(len(heatmap_cols)):
        val = corr_matrix.values[i, j]
        txt_color = "#1D1D20" if abs(val) > 0.6 else "#fbfbff"
        ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color=txt_color)

# Right: horizontal bar – correlation with target only
ax_bar = axes[1]
ax_bar.set_facecolor("#1D1D20")

corr_vals = corr_with_target[top_features_ordered]
bar_colors = ["#FF9F9B" if v > 0 else "#A1C9F4" for v in corr_vals.values]
bars = ax_bar.barh(range(len(corr_vals)), corr_vals.values,
                   color=bar_colors, edgecolor="#1D1D20", height=0.7)
ax_bar.set_yticks(range(len(corr_vals)))
ax_bar.set_yticklabels(corr_vals.index.tolist(), fontsize=9, color="#fbfbff")
ax_bar.axvline(0, color="#555", linewidth=1)
ax_bar.set_xlabel("Pearson Correlation with Overall Seed", color="#909094", fontsize=10)
ax_bar.set_title("Feature Correlation with Overall Seed\n(sorted by strength)",
                 color="#fbfbff", fontsize=12, fontweight="bold", pad=12)
ax_bar.tick_params(colors="#909094")
ax_bar.spines["bottom"].set_color("#444")
ax_bar.spines["left"].set_color("#444")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

import matplotlib.patches as mpatches
pos_patch = mpatches.Patch(color="#FF9F9B", label="Positive correlation")
neg_patch = mpatches.Patch(color="#A1C9F4", label="Negative correlation")
ax_bar.legend(handles=[pos_patch, neg_patch], facecolor="#2a2a2d",
              edgecolor="#444", labelcolor="#fbfbff", fontsize=9, loc="lower right")

plt.tight_layout(pad=2)
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor="#1D1D20")
plt.show()
print("\nCorrelation heatmap rendered ✅")
print(f"\nTop 5 strongest correlations with Overall Seed:")
print(corr_with_target.abs().nlargest(5).round(3).to_string())
