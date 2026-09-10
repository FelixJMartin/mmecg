"""Vectorize the per-row prediction pieces from the row splitter into signals.

Input is one record's `Predictions/pieces/{record}_rowNN.png` (predicted mask,
one paper row each) plus `analysis/splits/{record}_peaks.csv` (baseline + cut
rows, written by analysis/splits/rows.py). Each piece is passed straight to
vectorize_single_lead.

NOTE: only 1x12 has one lead per row. For 2x6 and 4x3+1 a row holds several
leads side by side, so each output signal is those leads concatenated -- the
horizontal split is a separate step.

    python digitize/digitize_pieces.py example_501
"""
import argparse
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# convert corser grid to mm
# mm_to_mv = 0.5 / 5
# mm_to_s= 0.2 / 5
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg-preprocessing"))
from ecgprep.img_helpers import vectorize_single_lead, mm_to_mv, mm_to_s


# Physical distance between two lead baselines, per layout (pmecg's
# row_distance * voltage, already snapped to a multiple of 5mm by the renderer).
ROW_PITCH_MM = {"1x12": 15.0, "2x6": 30.0, "4x3+1": 45.0}


def fill_edge_nans(signal):
    """Flat-extend the first/last real value over leading/trailing NaN runs.

    Internal gaps (a genuine momentary dropout in the predicted trace) are left
    as NaN -- only the edges, where there's no value on one side to interpolate
    from, get filled, so every signal starts and ends on a real number.
    """
    valid = np.where(~np.isnan(signal))[0]
    if len(valid) == 0:
        return signal
    signal = signal.copy()
    signal[:valid[0]] = signal[valid[0]]
    signal[valid[-1] + 1:] = signal[valid[-1]]
    return signal


def load_peaks(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [(int(r["cut_above"]), int(r["cut_below"])) for r in rows], \
           np.array([int(r["baseline_row"]) for r in rows])


def load_layout(index_csv, record):
    """Which template this record was rendered as."""
    with open(index_csv) as f:
        return {r["name"]: r["template"] for r in csv.DictReader(f)}[record]


def calibrate(peaks_csv, layout):
    """px/mm and sample rate, from the measured spacing of the baselines.

    Pitch in pixels is measured per record; pitch in mm comes from the layout.
    Nothing here assumes a dpi.
    """
    cuts, baselines = load_peaks(peaks_csv)
    pitch_px = float(np.mean(np.diff(baselines)))
    px_per_mm = pitch_px / ROW_PITCH_MM[layout]
    fs = px_per_mm / mm_to_s
    print(f"layout={layout}  pitch={pitch_px:.1f}px / {ROW_PITCH_MM[layout]}mm "
          f"= {px_per_mm:.2f} px/mm  ->  fs={fs:.1f} Hz")
    return cuts, px_per_mm, fs


def piece_paths(pieces_dir, record):
    """The row PNGs for one record, in row order (so row10 sorts after row9)."""
    paths = glob.glob(os.path.join(pieces_dir, f"{record}_row*.png"))
    return sorted(paths, key=lambda p: int(re.search(r"_row(\d+)\.png$", p).group(1)))


def vectorize_piece(mask_path, page, top, bottom, px_per_mm):
    """One row piece -> one signal in mV, centred on its own baseline."""
    mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
    page_crop = page[top:bottom, :]     # darkness tie-break uses the ORIGINAL page
    assert page_crop.shape == mask.shape, f"{mask_path}: {page_crop.shape} vs {mask.shape}"

    signal = fill_edge_nans(vectorize_single_lead(page_crop, mask))
    # Centre before scaling: the raw value is a row position inside this crop,
    # with no zero-reference shared between pieces.
    centred = signal - np.nanmedian(signal)
    return centred * mm_to_mv / px_per_mm


def save_preview(png_path, signals, fs, title):
    """One panel per paper row, seconds on the x axis."""
    t = np.arange(len(signals[0])) / fs
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 1.4 * len(signals)),
                             sharex=True, squeeze=False)
    for i, (ax, signal) in enumerate(zip(axes[:, 0], signals), start=1):
        ax.plot(t, signal, color="black", linewidth=0.8)
        ax.set_ylabel(f"row {i}", rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("seconds")
    axes[0, 0].set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"wrote {png_path}")


def digitize(record, pieces_dir, peaks_dir, img_dir, index_csv, out_dir):
    layout = load_layout(index_csv, record)
    cuts, px_per_mm, fs = calibrate(os.path.join(peaks_dir, f"{record}_peaks.csv"), layout)

    page = np.asarray(Image.open(os.path.join(img_dir, record + ".png")).convert("L"))
    paths = piece_paths(pieces_dir, record)
    assert len(paths) == len(cuts), f"{len(paths)} pieces but {len(cuts)} rows in the peaks csv"

    signals = []
    for i, (path, (top, bottom)) in enumerate(zip(paths, cuts), start=1):
        signal = vectorize_piece(path, page, top, bottom, px_per_mm)
        signals.append(signal)
        print(f"  row {i:2d}: rows {top:5d}-{bottom:5d}  {len(signal)} cols, "
              f"{np.isnan(signal).sum()} internal NaN")

    os.makedirs(out_dir, exist_ok=True)
    matrix = np.array(signals)
    csv_path = os.path.join(out_dir, f"{record}_digitized.csv")
    pd.DataFrame(matrix).to_csv(csv_path, index=False, header=False)
    print(f"wrote {csv_path}  shape={matrix.shape}  "
          f"range={np.nanmin(matrix):.2f}..{np.nanmax(matrix):.2f} mV")

    save_preview(os.path.join(out_dir, f"{record}_digitized.png"), signals, fs,
                 f"{record} ({layout}) - vectorized predicted mask, mV")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", nargs="?", default="example_501")
    ap.add_argument("--pieces-dir", default="Predictions/pieces")
    ap.add_argument("--peaks-dir", default="analysis/splits", help="where the peaks CSV lives")
    ap.add_argument("--img-dir", default="unet-src/data/test_imgs")
    ap.add_argument("--index-csv", default="unet-src/data/test_index.csv")
    ap.add_argument("--out-dir", default="Predictions/final")
    args = ap.parse_args()
    digitize(args.record, args.pieces_dir, args.peaks_dir, args.img_dir,
             args.index_csv, args.out_dir)
