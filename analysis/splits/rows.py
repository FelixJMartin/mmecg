"""Row geometry for ECG masks: find the lead baselines, then cut the page on them.

Splits one record into per-row pieces, writing the peaks CSV and one PNG per row.

python analysis/splits/rows.py example_501 --help

"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks


KEYS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "layout_keys.csv")


def load_layout_keys(csv_path=KEYS_CSV):
    """template -> rows on the page, from the shared contract."""
    with open(csv_path) as f:
        return {r["template"]: int(r["n_rows"]) for r in csv.DictReader(f)}


N_ROWS = load_layout_keys()

PRED_MASKS = ("Predictions/raw", "_epoch9")
PRED_LAYOUTS = "Predictions/test_score_report.csv"    # column: pred_layout

TRUTH_MASKS = ("unet-src/data/test_masks", "_mask")
TRUTH_LAYOUTS = "unet-src/data/test_index.csv"             # column: template



# ---------------------------------------------------------------- geometry

def row_density(mask_path):
    """
    iterate over all the rows in the mask and count the nonzero lines
    THis creates the histogram to the right of the plots
    
    """
    mask = np.asarray(Image.open(mask_path).convert("L"))
    height = mask.shape[0]

    counts = np.zeros(height, dtype=int)
    for y in range(height):
        counts[y] = np.count_nonzero(mask[y])   # trace pixels in row y
    return counts


def autocorr_dist(density, n_rows):
    """
    Estimate the vertical distance between two lead baselines, in pixels.
    """
    # Subtract the mean so empty rows count AGAINST a bad shift instead of just
    # contributing zero:
    #   density:  [  0,   0, 440, 577, 460, ...]   mean ~ 28
    #   profile:  [-28, -28, 412, 549, 432, ...]
    profile = density.astype(float) - density.mean()

    guess = len(density) / n_rows      # 2274 / 4 ~ 568 px on a 4x3+1 page
    smallest = int(0.5 * guess)        # 284
    largest = int(1.5 * guess)         # 852
    if largest >= len(profile):        # only reachable for a 1-row layout, where
        return float("nan")            # there is no spacing to measure at all

    # autocorrelation; ac[k] = the old loop's score at shift k
    ac = np.correlate(profile, profile, "full")[len(profile) - 1:]
    return smallest + int(np.argmax(ac[smallest:largest]))


def find_baselines(density, n_rows, dist=None):
    """
    Find the one baseline row belonging to each lead row, top to bottom.
    A baseline is a long horizontal line of ink, so it shows up as a peak in the
    density profile. 

    Returns (positions, heights). 
    """
    # Fall back to canvas/n_rows only if no usable dist was measured.
    if dist is None or not np.isfinite(dist) or dist <= 0:
        dist = len(density) / n_rows


    # Two peaks must sit at least ~60% of a dist apart.
    min_gap = 0.6 * dist
    peaks, _ = find_peaks(density.astype(float), distance=min_gap)


    #should not happen
    if len(peaks) > n_rows:
        heights = density[peaks]
        strongest = np.argsort(heights)[-n_rows:]
        peaks = peaks[strongest]

    #Example
    # peaks           [ 523,  933, 1340, 1752]   row indices  (where)
    # density[peaks]  [ 577,  538,  641,  772]   pixel counts (how strong)

    peaks = np.sort(peaks)          # argsort left them by height; want top-to-bottom
    heights = density[peaks]

    return peaks, heights    # baseline row indices, and the ink count at each


def cut_rows(peaks, dist):
    """Split the page into row bands.

    Args:
        peaks:  y-coordinates of the row separators.
        dist:  spacing between rows, in px (from autocorr_dist).
    
    PIL / crop boxes are (x, y):
        (0, 0)      top left
        (0, 2273)   bottom left 
        (3317, 0)   top right

    """

    height = len(density)
    cuts = []

    # top edge: half a dist above the first baseline, clipped to the page
    cuts.append(max(0, int(round(peaks[0] - dist / 2))))

    # between each pair of neighbouring baselines, cut in the middle
    for lower, upper in zip(peaks[1:], peaks[:-1]):
        cuts.append(int(round((upper + lower) / 2)))

    # bottom edge: half a dist below the last baseline
    cuts.append(min(height, int(round(peaks[-1] + dist / 2))))

    return np.array(cuts)                                           #returns a list of cuts, top is half a dist above base, bottom is half a dist below base


def measure(mask_path, n_rows):
    """
    (density, dist, peaks, heights) for one mask - the whole geometry step.
    A function calling other functions
    
    """

    density = row_density(mask_path)                               # Extract density profile
    dist = autocorr_dist(density, n_rows)                        # Compute the dist of the mask
    peaks, heights = find_baselines(density, n_rows, dist=dist)  # Get the 2 arrays needed

    return density, dist, peaks, heights                          # Return all in a compact way


# ---------------------------------------------------------------- drawing

def draw_rows(ax_img, ax_hist, img, density, peaks, cuts=None, title=None):
    """
    Mask on the left, its row-density on the right, sharing the same y axis.
    And also red for peaks.
    Blue for cuts.
    Saves when running save img
    """
    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=10)

    ax_hist.barh(np.arange(len(density)), density, height=1, color="black")
    ax_hist.set_ylim(len(density), 0)                                # top-to-bottom, matching the image
    ax_hist.set_xlabel("trace px / row", fontsize=8)
    ax_hist.tick_params(labelsize=7)
    ax_hist.set_yticks([])

    for p in peaks:
        ax_img.axhline(p, color="red", lw=0.9)
        ax_hist.axhline(p, color="red", lw=0.9)
    for c in cuts if cuts is not None else []:
        ax_img.axhline(c, color="tab:blue", lw=1, ls="--")
        ax_hist.axhline(c, color="tab:blue", lw=1, ls="--")


# ---------------------------------------------------------------- modes

def load_layouts(csv_path):

    """Map record name -> layout, from either the truth index or a score report.
    The two files label their columns differently, so take whichever is there:
        test_index.csv         name    , template      (ground truth)
        test_score_report.csv  record  , pred_layout   (what the model said)
    """

    layouts = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row["name"] if "name" in row else row["record"]
            layout = row["template"] if "template" in row else row["pred_layout"]
            if layout in N_ROWS:     
                layouts[name] = layout

    # {'example_500': '1x12',
    # 'example_501': '4x3+1',
    # 'example_502': '2x6',
    #   ...}                    

    return layouts #  # 50 entries for the test set


def write_peaks_csv(csv_path, peaks, heights, cuts):
    """
    One row per baseline: 
    where it is, how strong, and the crop it belongs to.
    saved to: C:analysis splits example_501_peaks.csv
    """
    with open(csv_path, "w", newline="") as f:                                              
        w = csv.writer(f)
        w.writerow(["row_index", "baseline_row", "peak_height", "cut_above", "cut_below"])
        for i, (peak, height) in enumerate(zip(peaks, heights)):
            w.writerow([i + 1, int(peak), int(height), int(cuts[i]), int(cuts[i + 1])])
    print(f"wrote {csv_path}")


def save_pieces(img, cuts, out_dir, record):
    """Crop the mask at the cut lines and save one PNG per row."""
    width = img.size[0]
    for i, (top, bottom) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        piece = img.crop((0, top, width, bottom))
        path = os.path.join(out_dir, f"{record}_row{i:02d}.png")
        piece.save(path)
        print(f"  row {i:2d}: rows {top:5d}-{bottom:5d}  "
              f"({piece.size[1]:4d}px tall)  {path}")


def save_figure(plot_path, img, density, peaks, cuts, title):
    """
    Mask and density profile side by side, baselines and cuts drawn on.
    
    """
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(15, 8),
                                          gridspec_kw={"width_ratios": [7, 1]})
    draw_rows(ax_img, ax_hist, np.asarray(img), density, peaks, cuts=cuts, title=title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {plot_path}")


def split(record, mask_dir, suffix, layouts_csv, out_dir, peaks_dir, plot_path):
    """Cut one record into per-row pieces.

    Row PNGs go to out_dir (the pipeline reads those); the peaks CSV and the
    figure go to peaks_dir.
    """
    layout = load_layouts(layouts_csv)[record]
    n_rows = N_ROWS[layout]

    mask_path = os.path.join(mask_dir, record + suffix + ".png")
    img = Image.open(mask_path).convert("RGB")
    density, dist, peaks, heights = measure(mask_path, n_rows)
    cuts = cut_rows(peaks, dist)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(peaks_dir, exist_ok=True)
    write_peaks_csv(os.path.join(peaks_dir, f"{record}_peaks.csv"), peaks, heights, cuts)
    save_pieces(img, cuts, out_dir, record)
    save_figure(plot_path, img, density, peaks, cuts,
                f"{record} ({layout}) - red = baselines, blue dashed = cuts")


if __name__ == "__main__":

    # Checking dist and density functions ------------------------------------------
    BASE = r"C:\Users\felix\Downloads\Research\U-net seg"
    RAW  = os.path.join(BASE, "Predictions", "raw")
    CUT_img = os.path.join(BASE, "Analysis", "splits")

    record = "example_501"                                      #choose img for testing
    mask_path = os.path.join(RAW, f"{record}_epoch9.png")

    with open(os.path.join(BASE, PRED_LAYOUTS)) as f:
        layout = next(r["pred_layout"] for r in csv.DictReader(f) if r["record"] == record)

    n_rows = N_ROWS[layout]
    density = row_density(mask_path)

    dist = autocorr_dist(density, n_rows)
    print(f"dist for this one is {dist}")

    # Testing find baselines --------------------------------------------------------

    peaks, heights = find_baselines(density, n_rows, dist)
    print(f"the peaks are located on rows: {peaks}")
    print(f"the height of these peaks are: {heights}")    

    # testing the cuts function -----------------------------------------------------

    cuts = cut_rows(peaks, dist)
    print(f"cuts will therefor be at these values: {cuts}")

    #testing the mesure function that calls all of them ----------------------------
    # des, dist, pek, heigh = measure(mask_path, n_rows)

    #testing the plotting of the first part: ---------------------------------------

    img = Image.open(mask_path).convert("RGB")
    save_figure(os.path.join(CUT_img, f"{record}_rows.png"), img, density, peaks, cuts, f"{record} ({layout})")




    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("record", help="which record to split, e.g. example_501")

    ap.add_argument("--mask-dir", default=PRED_MASKS[0])

    ap.add_argument("--suffix", default=PRED_MASKS[1], help="'_mask' for truth masks")

    ap.add_argument("--layouts-csv", default=PRED_LAYOUTS,
                    help=f"pass {TRUTH_LAYOUTS} to use ground truth")

    ap.add_argument("--out-dir", default="Predictions/pieces", help="where row PNGs go")

    ap.add_argument("--peaks-dir", default="analysis/splits",
                    help="where the peaks CSV and figure go")

    args = ap.parse_args()

    plot = os.path.join(args.peaks_dir, f"{args.record}_split.png")
    split(args.record, args.mask_dir, args.suffix, args.layouts_csv,
          args.out_dir, args.peaks_dir, plot)