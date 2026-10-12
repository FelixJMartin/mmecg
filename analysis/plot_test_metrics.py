"""
Test-set results from Predictions/test/score_report.csv (predict.py --score / the resume run):
5 PTB-XL fold-10 pages x 6 personas = 30 images the model never saw in training.

    dice       trace found vs truth, whatever lead it was given
    lead_dice  trace found AND given the right lead (the harder, more useful number)

Run:  python analysis/plot_test_metrics.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths -----------------------------
HERE = Path(__file__).parent                               # analysis/
PRED = HERE.parent / "Predictions" / "test"                # where the test predictions + report are
report = pd.read_csv(PRED / "score_report.csv")            # record, persona, true_layout, pred_layout, dice, lead_dice, ...
report["page"] = report["record"].str.split("_").str[0]    # ecg01031_photo_clean -> ecg01031

PERSONAS = ["scan_clean", "scan_marked", "photo_clean", "photo_folded", "photo_wrinkled", "photo_redacted"]
pages = report.groupby("page")["lead_dice"].mean().sort_values().index        # weakest page first

fig, (ax_persona, ax_page, ax_grid) = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})

# 1. per persona: trace dice vs lead dice side by side -----------------------------
by_persona = report.groupby("persona")[["dice", "lead_dice"]].mean().loc[PERSONAS]
by_persona.plot.bar(ax=ax_persona, rot=30, color=["lightgrey", "C0"])
ax_persona.set(ylim=(0, 1), ylabel="mean over the 5 pages", title="per persona", xlabel="")
ax_persona.legend(["trace dice", "lead dice"], fontsize=8, loc="lower right")

# 2. per page (each page has one layout): the page matters much more than the persona -----------------------------
by_page = report.groupby("page").agg(dice=("dice", "mean"), lead_dice=("lead_dice", "mean"),
                                     layout=("true_layout", "first")).loc[pages]
by_page[["dice", "lead_dice"]].plot.bar(ax=ax_page, rot=0, color=["lightgrey", "C0"], legend=False)
ax_page.set_xticklabels([f"{p}\n{layout}" for p, layout in zip(by_page.index, by_page["layout"])], fontsize=8)
ax_page.set(ylim=(0, 1), title="per page (layout underneath)", xlabel="")

# 3. every image: lead dice, page x persona -----------------------------
grid = report.pivot(index="page", columns="persona", values="lead_dice").loc[pages, PERSONAS]
im = ax_grid.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn")
ax_grid.set_xticks(range(len(PERSONAS)), PERSONAS, rotation=30, ha="right", fontsize=8)
ax_grid.set_yticks(range(len(pages)), pages, fontsize=8)
for y in range(len(pages)):
    for x in range(len(PERSONAS)):
        ax_grid.text(x, y, f"{grid.iloc[y, x]:.2f}", ha="center", va="center", fontsize=8)
ax_grid.set_title("lead dice per image")
fig.colorbar(im, ax=ax_grid, fraction=0.035)

for ax in (ax_persona, ax_page):
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(PRED / "test_metrics.png", dpi=150)
print("saved", PRED / "test_metrics.png")

# printout -----------------------------
print(f"\n{len(report)} test images: trace dice {report['dice'].mean():.3f}, precision {report['precision'].mean():.3f}, "
      f"recall {report['recall'].mean():.3f}, lead dice {report['lead_dice'].mean():.3f}")
known = report[report["pred_layout"].notna()]                 # first run didn't save its layout guesses ("n/a" reads as empty)
print(f"layout guess right: {int((known['pred_layout'] == known['true_layout']).sum())} / {len(known)} (where known)")
