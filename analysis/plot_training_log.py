"""
Training curves from the four logs train.py writes into models/, in one figure (6 panels).

    training_log.csv            one row per epoch: losses + validation metrics
    training_log_per_lead.csv   one row per epoch: validation dice per lead
    training_log_per_group.csv  one row per epoch x persona / layout: validation metrics per group
    training_log_per_image.csv  one row per epoch x validation image (only used for the printout)

Limb leads solid, chest leads dashed, so no two lead lines look alike.

Run:  python analysis/plot_training_log.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths -----------------------------
HERE = Path(__file__).parent                               # analysis/
MODELS = HERE.parent / "models"                            # U-net seg/models
log = pd.read_csv(MODELS / "training_log.csv")             # epoch, loss, ..., dice, lead_error, layout_acc
per_lead = pd.read_csv(MODELS / "training_log_per_lead.csv")      # epoch, dice_I, ..., dice_V6
per_group = pd.read_csv(MODELS / "training_log_per_group.csv")    # epoch, group_by, group, dice, ...
per_image = pd.read_csv(MODELS / "training_log_per_image.csv")    # epoch, name, persona, template, dice, ...

LIMB = ["I", "II", "III", "aVR", "aVL", "aVF"]
CHEST = ["V1", "V2", "V3", "V4", "V5", "V6"]

fig, ax = plt.subplots(3, 2, figsize=(14, 13))

# 1. training loss and its parts: log scale, they should all go down -----------------------------
ax[0, 0].plot(log["epoch"], log["loss"], color="black", lw=2, label="total")
ax[0, 0].plot(log["epoch"], log["seg_ce"], ls="--", label="pixel CE")
ax[0, 0].plot(log["epoch"], log["seg_dice"], ls="--", label="dice loss")
ax[0, 0].plot(log["epoch"], log["cls_ce"], ls="--", label="layout CE")
ax[0, 0].set(yscale="log", xlabel="epoch", title="training loss (log scale)")

# 2. validation, the big picture: is the ink found, and given the right lead? -----------------------------
for col, label in [("trace_recall", "trace recall (ink found)"),
                   ("trace_precision", "trace precision (ink is real)"),
                   ("dice", "lead dice (right ink, right lead)"),
                   ("column_coverage", "column coverage"),
                   ("layout_acc", "layout accuracy")]:
    ax[0, 1].plot(log["epoch"], log[col], label=label)
ax[0, 1].plot(log["epoch"], log["lead_error"], color="k", ls=":", label="lead error (lower is better)")
ax[0, 1].set(ylim=(0, 1), xlabel="epoch", title="validation: overall")

# 3. validation dice per lead: same 6 colours for both groups, solid vs dashed -----------------------------
for i, lead in enumerate(LIMB):
    ax[1, 0].plot(per_lead["epoch"], per_lead[f"dice_{lead}"], color=f"C{i}", ls="-", label=lead)
for i, lead in enumerate(CHEST):
    ax[1, 0].plot(per_lead["epoch"], per_lead[f"dice_{lead}"], color=f"C{i}", ls="--", label=lead)
ax[1, 0].set(ylim=(0, 1), xlabel="epoch", ylabel="validation dice", title="validation dice per lead")

# 4. voltage error after digitising the predicted trace, in mV: lower is better -----------------------------
ax[1, 1].plot(log["epoch"], log["centre_rmse_mv"], label="RMSE (mV)")
ax[1, 1].plot(log["epoch"], log["centre_mae_mv"], label="MAE (mV)")
ax[1, 1].set(xlabel="epoch", ylabel="mV", title="trace position error (lower is better)")

# 5 + 6. lead dice per persona and per layout: which kind of image / page is hardest? -----------------------------
for a, group_by in [(ax[2, 0], "persona"), (ax[2, 1], "template")]:
    rows = per_group[per_group["group_by"] == group_by]
    for group, g in rows.groupby("group"):
        a.plot(g["epoch"], g["dice"], label=group)
    a.set(ylim=(0, 1), xlabel="epoch", ylabel="validation dice", title=f"validation dice per {group_by}")

for a in ax.flat:
    a.grid(alpha=0.3)
    a.legend(fontsize=8, ncol=2 if a is ax[1, 0] else 1, loc="best")

fig.tight_layout()
fig.savefig(MODELS / "training_metrics.png", dpi=150)
print("saved", MODELS / "training_metrics.png")

# printout: final epoch, weakest first -----------------------------
last = per_lead.iloc[-1]
epoch = int(last["epoch"])
print(f"\nepoch {epoch}: lead dice {log['dice'].iloc[-1]:.3f}, layout accuracy {log['layout_acc'].iloc[-1]:.3f}")

print("\nper lead:")
for lead in sorted(LIMB + CHEST, key=lambda name: last[f"dice_{name}"]):
    print(f"  {lead:>3}: {last[f'dice_{lead}']:.3f}")

print("\nper persona / layout:")
final_groups = per_group[per_group["epoch"] == epoch].sort_values(["group_by", "dice"])
for _, row in final_groups.iterrows():
    print(f"  {row['group_by']:>8} {row['group']:<15} {row['dice']:.3f}  ({row['n_images']} images)")

print("\n5 hardest validation images:")
final_images = per_image[per_image["epoch"] == epoch].nsmallest(5, "dice")
for _, row in final_images.iterrows():
    print(f"  {row['name']:<28} {row['template']:<6} guessed {row['pred_layout']:<6} dice {row['dice']:.3f}")
