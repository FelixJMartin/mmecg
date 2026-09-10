"""Summarise a test-set score_report.csv: per-record metric plots + confusion matrix.

Unlike the training curves (which are per-epoch), these are per-record on the
held-out test set. Records are grouped by true layout, because a mean over all
records hides whether one layout is systematically worse than the others.

    python analysis/metrics/plot_test_metrics.py
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Layout names in class-index order, from the shared contract at the repo root,
# so this script cannot disagree with the model or the row splitter.
KEYS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "layout_keys.csv")
LAYOUTS = pd.read_csv(KEYS_CSV).sort_values("class_index")["template"].tolist()


def load(csv_path):
    df = pd.read_csv(csv_path)
    return df[df["record"] != "MEAN"].reset_index(drop=True)   # drop the summary row


def plot_metrics(df, out_path):
    """Left: each metric per record, coloured by layout. Right: distribution per layout."""
    fig, (ax_rec, ax_box) = plt.subplots(1, 2, figsize=(15, 5))

    colors = dict(zip(LAYOUTS, ["tab:blue", "tab:orange", "tab:green"]))
    for metric, marker in (("dice", "o"), ("precision", "s"), ("recall", "^")):
        ax_rec.plot(range(len(df)), df[metric], marker=marker, ms=3, lw=1, label=metric)
    ax_rec.set(xlabel="test record (in order)", ylabel="score",
               title="Per-record segmentation metrics", ylim=(0, 1))
    ax_rec.grid(alpha=0.3)
    ax_rec.legend(fontsize=8)

    # one box per layout per metric, so a weak layout stands out
    data, labels, box_colors = [], [], []
    for metric in ("dice", "precision", "recall"):
        for layout in LAYOUTS:
            vals = df.loc[df["true_layout"] == layout, metric]
            if len(vals):
                data.append(vals)
                labels.append(f"{metric[:4]}\n{layout}")
                box_colors.append(colors[layout])
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax_box.set(ylabel="score", title="Distribution by layout", ylim=(0, 1))
    ax_box.grid(alpha=0.3, axis="y")
    ax_box.tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_text(df):
    """Layout confusion matrix as text: rows = true, cols = predicted."""
    counts = np.zeros((len(LAYOUTS), len(LAYOUTS)), dtype=int)
    for _, row in df.iterrows():
        if row["true_layout"] in LAYOUTS and row["pred_layout"] in LAYOUTS:
            counts[LAYOUTS.index(row["true_layout"])][LAYOUTS.index(row["pred_layout"])] += 1

    width = max(len(n) for n in LAYOUTS) + 1
    lines = [f"Layout confusion matrix  (n={len(df)})", "",
             " " * (width + 7) + "predicted",
             " " * (width + 6) + " ".join(f"{n:>7s}" for n in LAYOUTS)]
    for i, name in enumerate(LAYOUTS):
        tag = "true " if i == 0 else " " * 5
        lines.append(f"{tag}{name:<{width}s} " + " ".join(f"{v:7d}" for v in counts[i]))

    correct = int(np.trace(counts))
    lines += ["", f"accuracy: {correct}/{counts.sum()} = {correct / max(counts.sum(), 1):.4f}", ""]
    for metric in ("dice", "precision", "recall"):
        per = "  ".join(f"{lay}={df.loc[df['true_layout'] == lay, metric].mean():.4f}" for lay in LAYOUTS)
        lines.append(f"mean {metric:9s} overall={df[metric].mean():.4f}   {per}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", nargs="?", default="analysis/metrics/test_score_report.csv")
    ap.add_argument("-o", "--out-dir", default="analysis/metrics")
    args = ap.parse_args()

    df = load(args.csv_path)
    os.makedirs(args.out_dir, exist_ok=True)

    png = os.path.join(args.out_dir, "test_metrics.png")
    plot_metrics(df, png)
    print(f"wrote {png}")

    txt = os.path.join(args.out_dir, "test_confusion_matrix.txt")
    text = confusion_text(df)
    with open(txt, "w") as f:
        f.write(text)
    print(f"wrote {txt}")
    print()
    print(text)
