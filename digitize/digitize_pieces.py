"""
Digitize example_1000's per-lead prediction pieces (Predictions/pieces/) via
vectorize_single_lead. Each piece already IS one lead's cropped mask region
(cut_prediction_pieces.py), so the full vectorize()/tlbr_boxes machinery isn't
needed here -- just the core per-column height-extraction, run once per piece.

Saves the resulting (n_leads, W) matrix as a CSV, plus a stacked preview plot
so every lead's reconstruction can be sanity-checked visually in one image.
"""
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg-preprocessing"))
from ecgprep.img_helpers import vectorize_single_lead, ScaleFromPixels

NAME = "example_1000_epoch10"
SOURCE_IMG = "unet-src/data/test_imgs/example_1000.png"
BOUNDARIES_CSV = f"Predictions/replotted/{NAME}_row_boundaries.csv"
PIECES_GLOB = f"Predictions/pieces/{NAME}_piece*.png"
OUT_DIR = "Predictions/pieces"


def load_boundaries(csv_path):
    with open(csv_path) as f:
        return [int(row["row_boundary"]) for row in csv.DictReader(f)]


def piece_index(path):
    return int(re.search(r"piece(\d+)\.png$", path).group(1))


def fill_edge_nans(signal):
    '''Flat-extend the first/last real value over any leading/trailing NaN run.
    Internal NaN gaps (a genuine momentary dropout in the predicted trace,
    which real ECGs can have too) are left untouched -- only the start/end
    edges, where there's no real value on one side to interpolate from, get
    filled, so every lead starts and ends on an actual number.'''
    valid = np.where(~np.isnan(signal))[0]
    if len(valid) == 0:
        return signal
    first, last = valid[0], valid[-1]
    signal = signal.copy()
    signal[:first] = signal[first]
    signal[last + 1:] = signal[last]
    return signal


def main():
    source = np.asarray(Image.open(SOURCE_IMG).convert("L"))
    boundaries = load_boundaries(BOUNDARIES_CSV)
    cuts = [0] + boundaries + [source.shape[0]]  # same cut points cut_prediction_pieces.py used

    piece_paths = sorted(glob.glob(PIECES_GLOB), key=piece_index)
    signals = []
    for i, piece_path in enumerate(piece_paths):
        top, bottom = cuts[i], cuts[i + 1]
        img_crop = source[top:bottom, :]  # same row range, but from the ORIGINAL image (for darkness tie-break)
        mask_rgb = np.asarray(Image.open(piece_path).convert("RGB"))
        mask_crop = (mask_rgb != 0).any(axis=-1).astype(np.uint8)

        signal = vectorize_single_lead(img_crop, mask_crop)
        signal = fill_edge_nans(signal)

        # OLD: left as raw pixel units here, mV-conversion deferred to plot_digitized.py
        # using the FULL PAGE height for every lead -- wrong, since n_grid_h assumes the
        # image height IS one lead-strip's physical height, not 12 leads stacked into it.
        # signals.append(signal)

        # OLD (first fix, still incomplete): scaled the raw pixel value directly, with no
        # baseline centering. Each piece's raw height is just "row position within THIS
        # piece's own crop" (0..piece_h) -- there's no shared zero-reference between
        # pieces, so different leads ended up at wildly different absolute mV offsets
        # (e.g. lead I ~2mV, lead V6 ~10mV) instead of each centered near its own 0mV
        # baseline like a real ECG -- that's what broke pmecg's row layout.
        # piece_h = bottom - top
        # signal_mv, _ = ScaleFromPixels(piece_h, source.shape[1])(signal)
        # signals.append(signal_mv)

        # Center this lead on its own baseline (median -- most columns sit at rest,
        # only QRS spikes deviate) BEFORE scaling to mV, same as a real ECG signal is
        # centered near 0 per lead, with no meaningful absolute row-position carried over.
        piece_h = bottom - top
        signal_centered = signal - np.nanmedian(signal)
        signal_mv, _ = ScaleFromPixels(piece_h, source.shape[1])(signal_centered)
        signals.append(signal_mv)
        print(f"piece {i + 1}: {len(signal)} columns, {np.isnan(signal).sum()} NaN remaining (internal only), piece_h={piece_h}")

    matrix = np.array(signals)  # (n_leads, W), now in real mV (per-piece calibrated)
    csv_path = f"{OUT_DIR}/{NAME}_digitized.csv"
    pd.DataFrame(matrix).to_csv(csv_path, index=False, header=False)
    print(f"saved {csv_path}  shape={matrix.shape}")

    # Stacked preview: every reconstructed lead, one row per piece, for a quick visual check.
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 1.2 * len(signals)), sharex=True)
    for i, (ax, signal) in enumerate(zip(axes, signals)):
        ax.plot(signal, color="black", linewidth=0.8)
        ax.set_ylabel(f"{i + 1}", rotation=0, labelpad=15)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    preview_path = f"{OUT_DIR}/{NAME}_digitized_preview.png"
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"saved {preview_path}")


if __name__ == "__main__":
    main()
