
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Top 20 features by MI score ──────────────────────────────────────────────
top_n = 20
mi_top = mi_series.head(top_n)

# Colour-code by feature group
NET_RANK_FEATS = {"NET Rank", "PrevNET", "net_rank_tier", "prev_net_rank_tier",
                  "net_rank_delta", "net_sos_interaction"}
QUAD_PREFIXES  = ("q1_", "q2_", "q3_", "q4_", "quad_", "bad_", "tier_ab")
SOS_FEATS      = {"NETSOS", "NETNonConfSOS", "AvgOppNETRank", "AvgOppNET",
                  "sos_composite", "opp_quality_delta", "net_vs_opp_delta",
                  "net_vs_avgopp_rank"}
CONF_FEATS     = {"conf_tier", "is_power_conf", "is_auto_bid", "is_at_large",
                  "in_tournament"}

def _group_color(feat_name):
    feat_str = str(feat_name)
    if feat_str in NET_RANK_FEATS:
        return "#A1C9F4"
    if any(feat_str.startswith(p) for p in QUAD_PREFIXES):
        return "#FFB482"
    if feat_str in SOS_FEATS:
        return "#8DE5A1"
    if feat_str in CONF_FEATS:
        return "#D0BBFF"
    return "#FF9F9B"

mi_top_features = list(mi_top.index)
mi_top_values   = list(mi_top.values)
bar_colors = [_group_color(f) for f in mi_top_features]

fig_mi, ax = plt.subplots(figsize=(12, 8))
fig_mi.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

# Reverse so highest value appears at top
rev_features = mi_top_features[::-1]
rev_values   = mi_top_values[::-1]
rev_colors   = bar_colors[::-1]

bars = ax.barh(range(top_n), rev_values, color=rev_colors,
               edgecolor="#1D1D20", height=0.72)

ax.set_yticks(range(top_n))
ax.set_yticklabels(rev_features, fontsize=10, color="#fbfbff")
ax.set_xlabel("Mutual Information Score", color="#909094", fontsize=11)
ax.set_title(
    f"Top {top_n} Features — Mutual Information vs Overall Seed\n"
    "(Higher = more predictive of tournament seeding)",
    color="#fbfbff", fontsize=13, fontweight="bold", pad=14
)
ax.tick_params(colors="#909094")
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate bars with score values
max_val = max(rev_values)
for val, rect in zip(rev_values, bars):
    ax.text(val + max_val * 0.01, rect.get_y() + rect.get_height() / 2,
            f"{val:.3f}", va="center", ha="left",
            color="#fbfbff", fontsize=8.5)

# Legend
legend_items = [
    mpatches.Patch(color="#A1C9F4", label="NET / Rank"),
    mpatches.Patch(color="#FFB482", label="Quadrant"),
    mpatches.Patch(color="#8DE5A1", label="SOS / Opp. Quality"),
    mpatches.Patch(color="#D0BBFF", label="Conference / Bid"),
    mpatches.Patch(color="#FF9F9B", label="Win % / Record"),
]
ax.legend(handles=legend_items, facecolor="#2a2a2d", edgecolor="#444",
          labelcolor="#fbfbff", fontsize=9, loc="lower right")

ax.set_xlim(0, max_val * 1.18)
plt.tight_layout()
plt.savefig("mi_importance_chart.png", dpi=150, bbox_inches="tight",
            facecolor="#1D1D20")
plt.show()

print(f"MI importance chart saved.")
print(f"Top feature: '{mi_series.index[0]}' (MI={mi_series.iloc[0]:.4f})")
print(f"Features with MI > 0.1: {(mi_series > 0.1).sum()}")
print(f"Features with MI = 0   : {(mi_series == 0).sum()}")
