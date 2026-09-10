"""Plot training curves from one or more training_log.csv files.

Each run logs epoch, loss, dice and layout_acc. Loss is on a log scale because
it drops by two orders of magnitude in the first few epochs; dice and accuracy
share a linear 0-1 axis. Passing several runs overlays them for comparison.

Note: precision/recall are NOT logged per epoch -- they are computed per record
by `predict.py --score`. Only loss/dice/layout_acc are available here.

    python plot_training_log.py run1.csv run2.csv -o curves.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_runs(csv_paths, out_path, labels=None):
    # name each run after its directory (os.path, so a Windows backslash path works too)
    labels = labels or [os.path.basename(os.path.dirname(p)) for p in csv_paths]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # the three loss terms, when train.py logged them separately
    parts = [("seg_ce", "pixel CE"), ("seg_dice", "dice loss"), ("cls_ce", "layout CE")]

    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        axes[0].plot(df["epoch"], df["loss"], marker="o", lw=2, color="black",
                     label=f"{label}: total" if len(csv_paths) > 1 else "total (sum)")
        # layout CE is ~100x the segmentation terms, so a total alone hides whether
        # segmentation is converging -- show each term when the columns are present
        for col, name in parts:
            if col in df and df[col].abs().sum() > 0:
                axes[0].plot(df["epoch"], df[col], marker=".", ls="--", lw=1.2, label=name)
        axes[1].plot(df["epoch"], df["dice"], marker="o", label=label)
        # layout_acc is blank for segmentation-only runs -- drop those rows
        acc = df.dropna(subset=["layout_acc"]) if "layout_acc" in df else df.iloc[0:0]
        if not acc.empty:
            axes[2].plot(acc["epoch"], acc["layout_acc"], marker="s", label=label)

    axes[0].set(yscale="log", xlabel="Epoch", ylabel="Training loss",
                title="Loss components (log scale)")
    axes[1].set(xlabel="Epoch", ylabel="Dice", title="Validation Dice", ylim=(0, 1))
    axes[2].set(xlabel="Epoch", ylabel="Accuracy", title="Layout accuracy", ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", default=["training_log.csv"],
                        help="one or more training_log.csv files")
    parser.add_argument("-o", "--out", default="analysis/metrics/curves.png",
                        help="output image path (analysis/metrics/ holds these figures)")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="legend label per run")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_runs(args.csv_paths, args.out, args.labels)
    print(f"wrote {args.out}")
