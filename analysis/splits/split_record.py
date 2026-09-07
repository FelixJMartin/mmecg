"""Split ONE record into per-row pieces, cutting midway between lead baselines.

Baselines are the density MAXIMA (each lead's isoelectric line makes a sharp
spike); the cut goes halfway between neighbouring baselines. The outer edges sit
half a pitch beyond the first and last baseline, so every piece is one row tall.

Nothing here assumes the paper's dpi, margins or size - the pitch is measured
from the image by autocorrelation. Only n_rows comes from outside (the layout).

    python analysis/splits/split_record.py example_501
"""
import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from row_structure import N_ROWS, autocorr_pitch, find_baselines, row_density


def cut_rows(peaks, pitch, height):
    """Cut lines: halfway between baselines, plus half a pitch past each end."""
    mids = ((peaks[:-1] + peaks[1:]) / 2).round().astype(int)
    top = max(0, int(round(peaks[0] - pitch / 2)))
    bottom = min(height, int(round(peaks[-1] + pitch / 2)))
    return np.concatenate([[top], mids, [bottom]])


def split(record, mask_dir, suffix, index_csv, out_dir, plot_path):
    with open(index_csv) as f:
        layout = {r["name"]: r["template"] for r in csv.DictReader(f)}[record]
    n_rows = N_ROWS[layout]

    mask_path = os.path.join(mask_dir, record + suffix + ".png")
    img = Image.open(mask_path).convert("RGB")
    density = row_density(mask_path)
    pitch = autocorr_pitch(density, n_rows)
    peaks, heights = find_baselines(density, n_rows, pitch=pitch)
    cuts = cut_rows(peaks, pitch, img.size[1])
    os.makedirs(out_dir, exist_ok=True)

    # peaks csv, so the numbers can be inspected outside this script
    csv_path = os.path.join(out_dir, f"{record}_peaks.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "baseline_row", "peak_height", "cut_above", "cut_below"])
        for i, (p, h) in enumerate(zip(peaks, heights)):
            w.writerow([i + 1, int(p), int(h), int(cuts[i]), int(cuts[i + 1])])
    print(f"layout={layout}  n_rows={n_rows}  pitch={pitch:.1f}px")
    print(f"baselines: {peaks.tolist()}")
    print(f"cuts:      {cuts.tolist()}")
    print(f"wrote {csv_path}")

    for i in range(len(cuts) - 1):
        piece = img.crop((0, cuts[i], img.size[0], cuts[i + 1]))
        path = os.path.join(out_dir, f"{record}_row{i + 1:02d}.png")
        piece.save(path)
        print(f"  row {i + 1:2d}: rows {cuts[i]:5d}-{cuts[i + 1]:5d}  ({piece.size[1]:4d}px tall)  {path}")

    # mask + density, baselines red, cut lines blue dashed
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(15, 8),
                                          gridspec_kw={"width_ratios": [7, 1]})
    ax_img.imshow(np.asarray(img))
    ax_img.axis("off")
    ax_img.set_title(f"{record} ({layout}) - red = baselines, blue dashed = cuts")
    ax_hist.barh(np.arange(len(density)), density, height=1, color="black")
    ax_hist.set_ylim(len(density), 0)
    ax_hist.set_xlabel("trace px / row")
    ax_hist.set_yticks([])
    for p in peaks:
        ax_img.axhline(p, color="red", lw=1)
        ax_hist.axhline(p, color="red", lw=1)
    for c in cuts:
        ax_img.axhline(c, color="tab:blue", lw=1, ls="--")
        ax_hist.axhline(c, color="tab:blue", lw=1, ls="--")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", nargs="?", default="example_501")
    ap.add_argument("--mask-dir", default="Predictions/raw")
    ap.add_argument("--suffix", default="_epoch9")
    ap.add_argument("--index-csv", default="unet-src/data/test_index.csv")
    ap.add_argument("--out-dir", default="Predictions/pieces")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()
    plot = args.plot or f"analysis/splits/{args.record}_split.png"
    split(args.record, args.mask_dir, args.suffix, args.index_csv, args.out_dir, plot)
