
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Reload training data
train_df = pd.read_csv("NCAA_Seed_Training_Set2.0.csv")
target_col = "Overall Seed"

# ── Seed value counts ───────────────────────────────────────────────────────
seed_counts = train_df[target_col].value_counts().sort_index()

# Assign bracket region colors by seed range
def seed_color(seed):
    if seed <= 17:   return "#A1C9F4"   # light blue  – top seeds
    elif seed <= 34: return "#8DE5A1"   # green       – mid seeds
    elif seed <= 51: return "#FFB482"   # orange      – lower seeds
    else:            return "#FF9F9B"   # coral       – play-in / lowest

colors = [seed_color(s) for s in seed_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("#1D1D20")

# ── Left: bar chart of seed frequency ──────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#1D1D20")
bars = ax1.bar(seed_counts.index.astype(int), seed_counts.values, color=colors,
               edgecolor="#1D1D20", linewidth=0.5, width=0.8)
ax1.set_xlabel("Overall Seed", color="#909094", fontsize=11)
ax1.set_ylabel("Count (seasons)", color="#909094", fontsize=11)
ax1.set_title("NCAA Tournament — Overall Seed Distribution", color="#fbfbff", fontsize=13, fontweight="bold", pad=14)
ax1.tick_params(colors="#909094")
ax1.spines["bottom"].set_color("#444")
ax1.spines["left"].set_color("#444")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(0, seed_counts.index.max() + 1)

legend_patches = [
    mpatches.Patch(color="#A1C9F4", label="Seeds 1–17"),
    mpatches.Patch(color="#8DE5A1", label="Seeds 18–34"),
    mpatches.Patch(color="#FFB482", label="Seeds 35–51"),
    mpatches.Patch(color="#FF9F9B", label="Seeds 52–68"),
]
ax1.legend(handles=legend_patches, facecolor="#2a2a2d", edgecolor="#444",
           labelcolor="#fbfbff", fontsize=9, loc="upper right")

# ── Right: describe + boxplot ──────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#1D1D20")
bp = ax2.boxplot(train_df[target_col].dropna(), vert=True, patch_artist=True,
                 medianprops=dict(color="#ffd400", linewidth=2),
                 boxprops=dict(facecolor="#A1C9F4", color="#A1C9F4", alpha=0.7),
                 whiskerprops=dict(color="#909094"),
                 capprops=dict(color="#909094"),
                 flierprops=dict(marker="o", color="#FF9F9B", alpha=0.5, markersize=4))
ax2.set_title("Overall Seed — Box Plot", color="#fbfbff", fontsize=13, fontweight="bold", pad=14)
ax2.set_ylabel("Overall Seed value", color="#909094", fontsize=11)
ax2.set_xticks([])
ax2.tick_params(colors="#909094")
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Annotate stats
stats = train_df[target_col].describe()
stat_txt = (f"Count : {int(stats['count'])}\n"
            f"Mean  : {stats['mean']:.1f}\n"
            f"Median: {train_df[target_col].median():.1f}\n"
            f"Std   : {stats['std']:.1f}\n"
            f"Min   : {stats['min']:.0f}\n"
            f"Max   : {stats['max']:.0f}")
ax2.text(1.4, stats["75%"], stat_txt, fontsize=10, color="#fbfbff",
         va="center", family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2a2d", edgecolor="#444"))

plt.tight_layout(pad=2)
plt.savefig("target_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#1D1D20")
plt.show()
print("Target distribution chart rendered ✅")
print(f"\nTarget '{target_col}' describe:\n{train_df[target_col].describe().round(2).to_string()}")
print(f"\nTop-5 most frequent seeds:\n{seed_counts.head(5).to_string()}")
