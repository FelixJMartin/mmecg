"""Re-render a digitized CSV back onto ECG paper with pmecg.

Closes the loop: image -> UNet mask -> row pieces -> signals -> paper again.
The output is directly comparable to the original test image, so anything the
pipeline lost (a dropped segment, a wrong baseline, a bad scale) shows up as a
visible difference rather than a number.

The digitized CSV has one signal PER PAPER ROW, but a 4x3+1 row holds four leads
side by side (["I","aVR","V1","V4"], ...), so each row is split into four equal
column blocks and each block is written into its lead's own slot. pmecg then
slices each lead back out at exactly the same offsets when it draws the row.

The horizontal span comes from the piece masks (first/last column with any
trace), not from the CSV, whose edges were flat-filled by digitize_pieces.py.

    python digitize/replot_digitized.py example_501
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

# Same knob as plott_ecg.py / make_dataset.py -- must match or the replot lands
# on a differently-sized page than the image it is being compared against.
ROW_DISTANCE = {"1x12": 1.5, "2x6": 3.0, "4x3+1": 4.5}
ROW_PITCH_MM = {"1x12": 15.0, "2x6": 30.0, "4x3+1": 45.0}
MM_PER_S = 25.0


def trace_span(pieces_dir, record):
    """First and last column carrying any predicted trace, across all pieces."""
    lo, hi = np.inf, -np.inf
    for path in glob.glob(os.path.join(pieces_dir, f"{record}_row*.png")):
        cols = np.where((np.asarray(Image.open(path).convert("L")) > 0).any(axis=0))[0]
        if len(cols):
            lo, hi = min(lo, cols[0]), max(hi, cols[-1])
    assert np.isfinite(lo), f"no trace pixels in any {record} piece"
    return int(lo), int(hi) + 1


def build_frame(rows, configuration):
    """Scatter each paper row's signal into the per-lead columns pmecg expects."""
    n = rows.shape[1]
    leads = sorted({l for entry in configuration for l in ([entry] if isinstance(entry, str) else entry)})
    df = pd.DataFrame(np.nan, index=range(n), columns=leads)
    for signal, entry in zip(rows, configuration):
        if isinstance(entry, str):          # full-width rhythm strip
            # A rhythm lead also appears as one block in a grid row (II is both the
            # 4x3+1 strip and row 2's first block). The strip wins: it is the same
            # lead over the same window, digitized from a full-width trace.
            df[entry] = signal
            continue
        edges = np.linspace(0, n, len(entry) + 1).round().astype(int)
        for lead, a, b in zip(entry, edges[:-1], edges[1:]):
            df.loc[a:b - 1, lead] = signal[a:b]
    return df


def replot(record, csv_path, pieces_dir, peaks_dir, index_csv, out_path):
    with open(index_csv) as f:
        layout = {r["name"]: r["template"] for r in csv.DictReader(f)}[record]

    matrix = pd.read_csv(csv_path, header=None).to_numpy()
    c0, c1 = trace_span(pieces_dir, record)
    rows = matrix[:, c0:c1]

    # px/mm is recovered the same way digitize_pieces.py did it, so the replot's
    # time axis matches the original paper speed instead of assuming a dpi.
    peaks = pd.read_csv(os.path.join(peaks_dir, f"{record}_peaks.csv"))
    px_per_mm = float(np.mean(np.diff(peaks["baseline_row"]))) / ROW_PITCH_MM[layout]
    fs = px_per_mm / (1.0 / MM_PER_S)
    print(f"layout={layout}  columns {c0}-{c1} ({rows.shape[1]} = {rows.shape[1] / fs:.1f}s)  fs={fs:.1f} Hz")

    # A canonical frame only to resolve the template's lead names, then the real one.
    stub = pd.DataFrame(np.zeros((1, 12)), columns=["I", "II", "III", "aVR", "aVL", "aVF",
                                                    "V1", "V2", "V3", "V4", "V5", "V6"])
    configuration = pmecg.template_factory(layout, stub, leads_map=None)
    assert len(configuration) == rows.shape[0], \
        f"{layout} has {len(configuration)} rows but the csv has {rows.shape[0]}"
    for i, entry in enumerate(configuration):
        print(f"  row {i + 1}: {entry}")

    df = build_frame(rows, configuration)
    plotter = pmecg.ECGPlotter(row_distance=ROW_DISTANCE[layout])
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", nargs="?", default="example_501")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pieces-dir", default="Predictions/pieces")
    ap.add_argument("--peaks-dir", default="analysis/splits", help="where the peaks CSV lives")
    ap.add_argument("--index-csv", default="unet-src/data/test_index.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    csv_path = args.csv or f"Predictions/final/{args.record}_digitized.csv"
    out = args.out or f"Predictions/final/{args.record}_replot.png"
    replot(args.record, csv_path, args.pieces_dir, args.peaks_dir, args.index_csv, out)
