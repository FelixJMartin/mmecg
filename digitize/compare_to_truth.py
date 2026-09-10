"""Score a digitized CSV against the h5 signal it was rendered from.

The paper image is only an intermediate: every test record came from a row of
ptb_preprocessed.h5, so that row is the real ground truth. Comparing against it
(rather than against the rendered PNG) measures the whole chain at once --
render -> UNet mask -> row split -> vectorize -> mV scaling -- in millivolts.

ALIGNMENT
    A 4x3+1 grid row holds four leads side by side, so the digitized row is cut
    into four equal column blocks, block j belonging to lead j over the j-th
    quarter of the recording. pmecg splits the record the same way when drawing,
    so truth block j is just the j-th quarter of that lead's samples.

    The column span of the drawn signal is first guessed from the piece masks,
    but that guess is a few pixels wide: the trace is stroked with a finite line
    width, so its mask spills past the real first and last sample, and taking the
    min/max across four rows compounds it. Left uncorrected that is a linear
    time-base error -- on example_501 it cost lead I a correlation of 0.50 while
    the waveform itself was near-perfect. refine_span() measures it (per-block
    cross-correlation lag, fitted linearly) and corrects the span.

    Both scores are reported. `raw` is what the pipeline produces unaided; the
    aligned score isolates waveform error from the span estimate. The fitted
    correction is printed in pixels -- it is a diagnostic of trace_span(), not a
    free parameter, and because it consults the truth it belongs in this analysis
    script and NOT in the digitize pipeline.

Writes an overlay+error figure and a per-lead metrics CSV to
Predictions/digitized_vs_truth/.

    python digitize/compare_to_truth.py example_501
"""
import argparse
import csv
import glob
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

H5_PATH = "ptb-xl/ptb_preprocessed.h5"
FS_TRUTH = 400   # Hz, the rate make_dataset.py rendered at
MAX_LAG = 60     # samples; the span guess is only ever wrong by a few pixels

# ecgprep names its leads DI/DII/...; pmecg (and the templates) expect I/II/...
ECGPREP_LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RENAME = {"DI": "I", "DII": "II", "DIII": "III", "AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}


def trace_span(pieces_dir, record):
    """First and last column carrying any predicted trace, across all pieces.

    Taken from the masks, not the CSV: digitize_pieces.py flat-fills the CSV's
    edges out to the page margin, so the CSV alone cannot say where the drawn
    signal starts. Overshoots by roughly half a stroke width -- see refine_span.
    """
    lo, hi = np.inf, -np.inf
    for path in glob.glob(os.path.join(pieces_dir, f"{record}_row*.png")):
        cols = np.where((np.asarray(Image.open(path).convert("L")) > 0).any(axis=0))[0]
        if len(cols):
            lo, hi = min(lo, cols[0]), max(hi, cols[-1])
    assert np.isfinite(lo), f"no trace pixels in any {record} piece"
    return float(lo), float(hi + 1)


def blocks(configuration, n_samples):
    """(lead, first_sample, last_sample) for every lead block, row by row.

    Mirrors how pmecg lays a row out: the leads named in a row split the
    recording into that many equal consecutive windows, and a bare string is a
    rhythm strip that gets the whole recording.
    """
    for entry in configuration:
        leads = [entry] if isinstance(entry, str) else entry
        edges = np.linspace(0, n_samples, len(leads) + 1).round().astype(int)
        yield from ((lead, a, b) for lead, a, b in zip(leads, edges[:-1], edges[1:]))


def row_of_each_block(rows, configuration):
    """The paper-row signal each block in blocks() was cut from."""
    for signal, entry in zip(rows, configuration):
        for _ in ([entry] if isinstance(entry, str) else entry):
            yield signal


def read_block(row, span, s0, s1, n_samples):
    """Digitized values at the columns where truth samples s0..s1 were drawn."""
    a, b = span
    cols = a + np.arange(s0, s1) * (b - a) / (n_samples - 1)
    finite = ~np.isnan(row)
    return np.interp(cols, np.flatnonzero(finite), row[finite])


def pair_leads(rows, configuration, truth, span):
    """(lead, truth_segment, digitized_segment) triples, sample-aligned."""
    n = len(truth)
    for (lead, s0, s1), signal in zip(blocks(configuration, n),
                                      row_of_each_block(rows, configuration)):
        yield lead, truth[lead].to_numpy()[s0:s1], read_block(signal, span, s0, s1, n)


def best_lag(y_true, y_hat):
    """Sample shift that best aligns y_hat onto y_true, by correlation."""
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    scores = []
    for lag in lags:
        a = y_hat[max(0, lag):len(y_hat) + min(0, lag)]
        b = y_true[max(0, -lag):len(y_true) + min(0, -lag)]
        scores.append(np.corrcoef(a, b)[0, 1])
    return int(lags[int(np.argmax(scores))])


def refine_span(rows, configuration, truth, span):
    """Correct the column span using the lag each block shows against truth.

    A block read at truth-sample s actually carries truth(s - lag(s)); fitting
    lag linearly over s and re-deriving the span removes both the constant
    offset (a wrong left edge) and the slope (a wrong width).
    """
    n = len(truth)
    centres, lags = [], []
    for (lead, s0, s1), (_, y_true, y_hat) in zip(
            blocks(configuration, n), pair_leads(rows, configuration, truth, span)):
        centres.append((s0 + s1) / 2)
        lags.append(best_lag(y_true, y_hat))
    slope, intercept = np.polyfit(centres, lags, 1)
    a, b = span
    width = b - a
    a2 = a + intercept * width / n
    b2 = a2 + (1 + slope) * width
    print(f"span refine: lags {lags} -> shift {intercept:+.2f} samples, scale {1 + slope:.5f}")
    print(f"             columns [{a:.1f}, {b:.1f}) -> [{a2:.1f}, {b2:.1f})  "
          f"(left {a2 - a:+.1f}px, right {b2 - b:+.1f}px)")
    return a2, b2


def score(pairs):
    out = []
    for lead, y_true, y_hat in pairs:
        err = y_hat - y_true
        out.append({"lead": lead, "n": len(y_true),
                    "mse_mv2": float(np.mean(err ** 2)),
                    "rmse_mv": float(np.sqrt(np.mean(err ** 2))),
                    "offset_mv": float(np.median(err)),
                    "max_abs_err_mv": float(np.max(np.abs(err))),
                    "corr": float(np.corrcoef(y_true, y_hat)[0, 1]),
                    "truth_ptp_mv": float(np.ptp(y_true))})
    return out


def overall(rs):
    """Sample-weighted MSE across every block."""
    return float(sum(r["mse_mv2"] * r["n"] for r in rs) / sum(r["n"] for r in rs))


def compare(record, csv_path, pieces_dir, index_csv, out_dir):
    with open(index_csv) as f:
        meta = {r["name"]: r for r in
                csv.DictReader(f, fieldnames=["name", "index", "template"])}[record]
    layout, h5_row = meta["template"], int(meta["index"])

    with h5py.File(H5_PATH, "r") as f:
        truth = pd.DataFrame(f["tracings"][h5_row],
                             columns=[RENAME.get(l, l) for l in ECGPREP_LEADS])

    rows = pd.read_csv(csv_path, header=None).to_numpy()
    configuration = pmecg.template_factory(layout, truth, leads_map=None)
    assert len(configuration) == rows.shape[0], \
        f"{layout} draws {len(configuration)} rows but the csv has {rows.shape[0]}"

    span = trace_span(pieces_dir, record)
    print(f"{record}: layout={layout}  h5 row {h5_row}  "
          f"{len(truth)} samples @ {FS_TRUTH}Hz  mask span [{span[0]:.0f}, {span[1]:.0f})")
    raw = score(pair_leads(rows, configuration, truth, span))
    span = refine_span(rows, configuration, truth, span)
    pairs = list(pair_leads(rows, configuration, truth, span))
    aligned = score(pairs)

    metrics = pd.DataFrame(aligned)
    metrics["rmse_mv_raw"] = [r["rmse_mv"] for r in raw]
    metrics["corr_raw"] = [r["corr"] for r in raw]
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{record}_metrics.csv")
    metrics.to_csv(metrics_path, index=False, float_format="%.5f")
    print()
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for label, rs in (("raw    ", raw), ("aligned", aligned)):
        mse = overall(rs)
        print(f"overall {label}: MSE {mse:.5f} mV^2   RMSE {np.sqrt(mse):.4f} mV   "
              f"mean r {np.mean([r['corr'] for r in rs]):.4f}")
    print(f"wrote {metrics_path}")

    # One panel per lead: truth vs digitized, with the residual on the same axes.
    fig, axes = plt.subplots(len(pairs), 1, figsize=(15, 1.9 * len(pairs)), squeeze=False)
    for ax, (lead, y_true, y_hat), row in zip(axes[:, 0], pairs, aligned):
        t = np.arange(len(y_true)) / FS_TRUTH
        ax.plot(t, y_true, color="black", lw=0.9, label="truth (h5)")
        ax.plot(t, y_hat, color="tab:red", lw=0.9, alpha=0.8, label="digitized")
        ax.plot(t, y_hat - y_true, color="tab:blue", lw=0.7, alpha=0.7, label="error")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_xticks([])
        ax.text(0.995, 0.05, f"RMSE {row['rmse_mv']:.3f} mV   r={row['corr']:.4f}",
                transform=ax.transAxes, ha="right", fontsize=8, color="dimgrey")
    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0, 0].set_title(f"{record} ({layout}) - digitized vs h5 truth, mV")
    axes[-1, 0].set_xlabel("seconds within each lead's own window")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{record}_vs_truth.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", nargs="?", default="example_501")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pieces-dir", default="Predictions/pieces")
    ap.add_argument("--index-csv", default="unet-src/data/test_index.csv")
    ap.add_argument("--out-dir", default="Predictions/digitized_vs_truth")
    args = ap.parse_args()
    compare(args.record, args.csv or f"Predictions/final/{args.record}_digitized.csv",
            args.pieces_dir, args.index_csv, args.out_dir)
