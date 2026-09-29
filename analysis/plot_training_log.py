"""
Training curves from models/training_log.csv: losses, validation dice, layout accuracy.

Run:  python analysis/metrics/plot_training_log.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths -----------------------------
HERE = Path(__file__).parent                               # analysis/metrics
MODELS = HERE.parents[1] / "models"                        # U-net seg/models
log = pd.read_csv(MODELS / "training_log.csv")             # one row per epoch

fig, (ax_loss, ax_dice, ax_layout) = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. losses: log scale, since they fall by orders of magnitude -----------------------------
ax_loss.plot(log["epoch"], log["loss"], color="black", lw=2, label="total")
ax_loss.plot(log["epoch"], log["seg_ce"], ls="--", label="pixel CE")
ax_loss.plot(log["epoch"], log["seg_dice"], ls="--", label="dice loss")
ax_loss.plot(log["epoch"], log["cls_ce"], ls="--", label="layout CE")
ax_loss.set(yscale="log", xlabel="epoch", title="training loss (log scale)")

# 2. validation dice (lead classes, background excluded) -----------------------------
ax_dice.plot(log["epoch"], log["dice"], marker="o", ms=3)
ax_dice.set(ylim=(0, 1), xlabel="epoch", title="validation dice")

# 3. layout accuracy -----------------------------
ax_layout.plot(log["epoch"], log["layout_acc"], marker="s", ms=3)
ax_layout.set(ylim=(0, 1.05), xlabel="epoch", title="layout accuracy")

for ax in (ax_loss, ax_dice, ax_layout):
    ax.grid(alpha=0.3)
ax_loss.legend(fontsize=8)

fig.tight_layout()
fig.savefig(MODELS / "training_curves.png", dpi=150)
print("saved", MODELS / "training_curves.png")
