"""Study the row structure of GROUND-TRUTH masks before trusting it on predictions.

Question: are lead rows periodic, and are their baselines (density MAXIMA) easier
to localise than the gaps between them (minima)?

Each lead spends most of its time on its isoelectric baseline, a horizontal line
spanning the page - so it makes a sharp, narrow spike in the row-density profile.
A gap, by contrast, is a wide flat near-zero region with no well-defined minimum,
which is why picking minima was unstable.

Reports per record: the pitch between detected baselines, how regular it is, and
an independent pitch estimate from autocorrelation (which uses the whole profile
at once instead of n separate decisions).

    python analysis/row_structure.py                                  # ground-truth masks
    python analysis/row_structure.py --mask-dir Predictions/raw_v2 --suffix _epoch9
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

N_ROWS = {"1x12": 12, "2x6": 6, "4x3+1": 4}


def row_density(mask_path):
    """Foreground pixels per image row."""
    rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    return (rgb != 0).any(axis=-1).sum(axis=1)


def find_baselines(density, n_rows, pitch=None):
    """The n_rows strongest density peaks - one lead baseline per row.

    `pitch` is MEASURED from the image (autocorrelation), not assumed from the
    canvas height: margins, dpi and paper size all vary on a real scan, so
    canvas/n_rows is not a usable estimate. Falls back to canvas/n_rows only if
    no pitch is supplied.

    Returns (positions, heights). The heights are a free quality signal - a peak
    much weaker than its neighbours usually means a missed row or a flat lead.
    """
    if pitch is None or not np.isfinite(pitch) or pitch <= 0:
        pitch = len(density) / n_rows
    # keep peaks ~60% of a pitch apart so two can't land inside one row
    peaks, _ = find_peaks(density.astype(float), distance=max(3, int(0.6 * pitch)))
    if len(peaks) > n_rows:
        peaks = peaks[np.argsort(density[peaks])[-n_rows:]]
    peaks = np.sort(peaks)
    return peaks, density[peaks]


def autocorr_pitch(density, n_rows):
    """Pitch from the profile's periodicity, independent of peak picking."""
    d = density.astype(float) - density.mean()
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    expected = len(density) / n_rows
    lo, hi = int(0.5 * expected), int(1.5 * expected)
    return lo + int(np.argmax(ac[lo:hi])) if hi < len(ac) else float("nan")


def analyse(mask_dir, suffix, index_csv, out_png, limit=None):
    with open(index_csv) as f:
        truth = {r["name"]: r["template"] for r in csv.DictReader(f)}
    names = sorted(truth, key=lambda n: (len(n), n))[:limit]

    print(f"{'record':12s} {'layout':7s} {'rows':>4s} {'found':>5s} "
          f"{'pitch':>7s} {'std':>6s} {'CV':>6s} {'autocorr':>8s} {'weakest':>8s}")
    rows = []
    for name in names:
        path = os.path.join(mask_dir, name + suffix + ".png")
        if not os.path.exists(path):
            continue
        layout = truth[name]
        n = N_ROWS[layout]
        d = row_density(path)
        ac = autocorr_pitch(d, n)              # measure the pitch first...
        peaks, heights = find_baselines(d, n, pitch=ac)   # ...then use it to pick peaks
        gaps = np.diff(peaks)
        pitch, std = (gaps.mean(), gaps.std()) if len(gaps) else (np.nan, np.nan)
        cv = std / pitch if pitch else np.nan
        # weakest peak relative to the strongest: low = a row may have been missed
        weakest = heights.min() / heights.max() if len(heights) else np.nan
        print(f"{name:12s} {layout:7s} {n:4d} {len(peaks):5d} {pitch:7.1f} {std:6.1f} "
              f"{cv:6.3f} {ac:8.1f} {weakest:8.2f}")
        rows.append(dict(record=name, layout=layout, pitch=pitch, cv=cv,
                         autocorr=ac, weakest=weakest, found=len(peaks), n_rows=n))

    print()
    for layout in N_ROWS:
        sel = [r for r in rows if r["layout"] == layout]
        if sel:
            miss = sum(r["found"] != r["n_rows"] for r in sel)
            print(f"{layout:7s} n={len(sel):2d}  mean CV={np.mean([r['cv'] for r in sel]):.4f}  "
                  f"pitch={np.mean([r['pitch'] for r in sel]):6.1f}  "
                  f"autocorr={np.mean([r['autocorr'] for r in sel]):6.1f}  "
                  f"weakest={np.mean([r['weakest'] for r in sel]):.2f}  "
                  f"wrong count={miss}")

    # mask on the left, its row-density on the right, row-aligned, baselines in red
    fig, axes = plt.subplots(len(N_ROWS), 2, figsize=(14, 13),
                             gridspec_kw={"width_ratios": [7, 1]})
    for (ax_img, ax_hist), layout in zip(axes, N_ROWS):
        first = next((r for r in rows if r["layout"] == layout), None)
        if not first:
            continue
        path = os.path.join(mask_dir, first["record"] + suffix + ".png")
        img = np.asarray(Image.open(path).convert("RGB"))
        d = row_density(path)
        peaks, _ = find_baselines(d, N_ROWS[layout], autocorr_pitch(d, N_ROWS[layout]))

        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(f"{layout} - {first['record']}  ({len(peaks)} baselines)", fontsize=10)

        ax_hist.barh(np.arange(len(d)), d, height=1, color="black")
        ax_hist.set_ylim(len(d), 0)            # top-to-bottom, matching the image
        ax_hist.set_xlabel("trace px / row", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.set_yticks([])

        for p in peaks:
            ax_img.axhline(p, color="red", lw=0.9)
            ax_hist.axhline(p, color="red", lw=0.9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask-dir", default="unet-src/data/test_masks")
    ap.add_argument("--suffix", default="_mask", help="'_mask' for truth, '_epoch9' for predictions")
    ap.add_argument("--index-csv", default="unet-src/data/test_index.csv")
    ap.add_argument("--out", default="analysis/row_structure_truth.png")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    analyse(args.mask_dir, args.suffix, args.index_csv, args.out, args.limit)
