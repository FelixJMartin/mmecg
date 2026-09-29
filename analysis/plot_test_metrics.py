"""
Validation dice per lead over training, from models/training_log_per_lead.csv.
Limb leads solid, chest leads dashed, so no two lines look alike.

Run:  python analysis/metrics/plot_test_metrics.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths -----------------------------
HERE = Path(__file__).parent                               # analysis/metrics
MODELS = HERE.parents[1] / "models"                        # U-net seg/models
log = pd.read_csv(MODELS / "training_log_per_lead.csv")    # epoch, dice_I, dice_II, ..., dice_V6

LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
CHEST = ["V1", "V2", "V3", "V4", "V5", "V6"]

fig, ax = plt.subplots(figsize=(11, 5))

# one line per lead: same 6 colours for both groups, told apart by solid vs dashed -----------------------------
for i, lead in enumerate(LIMB):
    ax.plot(log["epoch"], log[f"dice_{lead}"], color=f"C{i}", ls="-", label=lead)
for i, lead in enumerate(CHEST):
    ax.plot(log["epoch"], log[f"dice_{lead}"], color=f"C{i}", ls="--", label=lead)

# final-epoch score per lead, weakest first, printed for reference -----------------------------
last = log.iloc[-1]
for lead in sorted(LIMB + CHEST, key=lambda name: last[f"dice_{name}"]):
    print(f"{lead:>3}: {last[f'dice_{lead}']:.3f}")

ax.set(ylim=(0, 1), xlabel="epoch", ylabel="validation dice", title="validation dice per lead")
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(MODELS / "per_lead_dice.png", dpi=150)
print("saved", MODELS / "per_lead_dice.png")
