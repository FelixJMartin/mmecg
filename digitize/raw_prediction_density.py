"""
Row-density diagnostic for a raw predicted mask (Predictions/raw/*.png) --
same technique as the density panel in replot_prediction.py (which computes
it on its own redrawn pmecg chart), but applied straight to the raw
prediction instead. Counts predicted-trace pixels per row and shows them as
a horizontal bar chart alongside the image, row-aligned: bands of high
density mark where leads/labels/artifacts cluster vertically, which is a
first, cheap way to eyeball where lead rows might be separable without
needing a full tlbr_boxes layout.
"""
import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from pmecg.utils.data import _TEMPLATE_CONFIGURATIONS

INDEX_CSVS = [
    "dataset_index.csv", "test_index.csv",
    "unet-src/data/dataset_index.csv", "unet-src/data/test_index.csv",
]


def load_predicted_lookup(report_csv):
    '''record name -> layout PREDICTED by the model, from predict.py --score's
    score_report.csv. This is what closes the loop: n_rows no longer comes from
    the ground-truth index CSV ("stealing from truth"), it comes from the net.'''
    with open(report_csv) as f:
        return {row["record"]: row["pred_layout"] for row in csv.DictReader(f)
                if row["record"] != "MEAN" and row["pred_layout"] not in ("", "n/a")}


def load_template_lookup():
    '''record name (e.g. "example_1001") -> template, merged from every index CSV found.'''
    lookup = {}
    for path in INDEX_CSVS:
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    lookup[row["name"]] = row["template"]
    return lookup


def find_row_minima(row_density, n_minima, min_distance):
    '''Pick exactly n_minima row-boundary positions. find_peaks on the negated
    density surfaces every local dip as a candidate; keeping only the
    n_minima deepest of those (not just the first n_minima found) avoids
    picking several shallow candidates that sit within the same real valley.'''
    candidates, _ = find_peaks(-row_density.astype(float), distance=min_distance)
    if len(candidates) < n_minima:
        # Not enough distinct local dips (flat/noisy density) -- fall back to
        # the n_minima lowest-density rows overall, wherever they land.
        chosen = np.argsort(row_density)[:n_minima]
    else:
        depths = row_density[candidates]
        chosen = candidates[np.argsort(depths)[:n_minima]]
    return np.sort(chosen)


def density_plot(pred_path, out_dir, template=None):
    name = os.path.splitext(os.path.basename(pred_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    mask_rgb = np.asarray(Image.open(pred_path).convert("RGB"))
    is_trace = (mask_rgb != 0).any(axis=-1)  # predicted-foreground (neon) pixels
    row_density = is_trace.sum(axis=1)

    minima = None
    if template is not None:
        # "Steal from truth" for now: the template tells us exactly how many
        # rows to expect, so how many boundaries (n_rows - 1) to look for is
        # known rather than guessed -- a real layout-detector model would
        # supply n_rows here instead, once one exists.
        n_rows = len(_TEMPLATE_CONFIGURATIONS[template])
        n_minima = n_rows - 1
        if n_minima > 0:
            h = row_density.shape[0]
            min_distance = max(5, h // (n_rows * 3))
            minima = find_row_minima(row_density, n_minima, min_distance)

    # Scale the figure to this image's own true aspect ratio (h/w), instead of a
    # fixed box -- different templates have very different shapes (e.g. 1x12 is
    # much taller/narrower than 4x3+1), and forcing them all into the same box
    # distorted that; each output should now look proportioned like its own input.
    h, w = mask_rgb.shape[:2]
    img_width_in = 14 * (10 / 11)  # image panel gets 10/11 of the figure's width (width_ratios below)
    fig_height = img_width_in * (h / w)
    fig, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(14, fig_height), gridspec_kw={"width_ratios": [10, 1]}, sharey=True
    )
    ax_img.imshow(mask_rgb)  # default aspect="equal" -- no stretching, true pixel proportions
    ax_img.axis("off")
    ax_hist.barh(np.arange(len(row_density)), row_density, color="black", height=1)
    ax_hist.set_xlabel("predicted trace px / row")
    if minima is not None:
        for y in minima:
            ax_img.axhline(y, color="red", linewidth=1)
            ax_hist.axhline(y, color="red", linewidth=1)
    fig.tight_layout()
    # tight_layout can give ax_hist a different box height than ax_img (it has an
    # xlabel/ticks eating vertical space that ax_img doesn't) -> force them equal
    # so row 0 and row h line up exactly top-to-bottom between the two panels.
    pos_img = ax_img.get_position()
    pos_hist = ax_hist.get_position()
    ax_hist.set_position([pos_hist.x0, pos_img.y0, pos_hist.width, pos_img.height])

    out_path = f"{out_dir}/{name}_density.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved {out_path}" + (f"  row boundaries: {minima.tolist()}" if minima is not None else ""))

    if minima is not None:
        with open(f"{out_dir}/{name}_row_boundaries.csv", "w") as f:
            f.write("row_boundary\n")
            for y in minima:
                f.write(f"{y}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", default="Predictions/raw", help="directory of predicted masks")
    ap.add_argument("--out-dir", default="Predictions/replotted", help="where to write density plots + boundary CSVs")
    ap.add_argument("--report", default=None,
                    help="score_report.csv from predict.py --score; use the model's PREDICTED "
                         "layout for n_rows instead of the ground-truth index CSV")
    args = ap.parse_args()

    lookup = load_predicted_lookup(args.report) if args.report else load_template_lookup()
    source = "predicted" if args.report else "ground truth"
    print(f"layout source: {source}")
    for pred_path in sorted(glob.glob(f"{args.pred_dir}/*.png")):
        pred_name = os.path.splitext(os.path.basename(pred_path))[0]
        record_name = pred_name.split("_epoch")[0]
        template = lookup.get(record_name)
        if template is None:
            print(f"  skipping {pred_name}: no layout for {record_name}")
            continue
        density_plot(pred_path, args.out_dir, template=template)
