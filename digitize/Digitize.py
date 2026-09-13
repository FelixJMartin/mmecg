"""Vectorize the per-row prediction pieces from the row splitter into signals.

Input is one record's `Predictions/pieces/{record}_rowNN.png` (predicted mask,
one paper row each) plus `analysis/splits/{record}_peaks.csv` (baseline + cut
rows, written by analysis/splits/rows.py).

NOTE: only 1x12 has one lead per row. For 2x6 and 4x3+1 a row holds several
leads side by side, so each output signal is those leads concatenated -- the
horizontal split is a separate step.

python digitize/digitize_pieces.py example_501; --help 


Instructions on convertions:

1. mm per mV — the gain. voltage in pmecg defaults to 10 mm/mV, the clinical standard (a 1 mV calibration pulse is 10 mm tall).

2. distance is in mV, not mm. So converting to paper distance:

dist_mm = row_distance × voltage
    1.5 mV × 10 mm/mV = 15 mm
    3.0 mV × 10 mm/mV = 30 mm
    4.5 mV × 10 mm/mV = 45 mm

3. Where 1.5 / 3.0 / 4.5 came from — the constraint in make_dataset.py: every layout must render to the same canvas height, or image height alone leaks the layout label. Rows on the page are 12, 6, 4:

    n_rows × dist_mm = const
    12 × 15 = 180 mm
    6 × 30 = 180 mm
    4 × 45 = 180 mm

4. So the whole table is one formula:

dist_mm      = 180 / n_rows
row_distance  = dist_mm / 10  =  18 / n_rows

180 mm is just the page budget you picked; any constant works, but this one is convenient because 180/12, 180/6 and 180/4 all land on multiples of 5 mm — which matters because pmecg snaps row_distance × voltage to a 5 mm multiple. Pick something like 200 mm and 200/12 = 16.67 gets snapped to 15 or 20, and your nominal dist no longer matches what's on the paper.

"""
import argparse
import csv
import glob
import os
import re
import sys

import pmecg
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import glob
import os


# convert corser grid to mm
# mm_to_mv = 0.5 / 5
# mm_to_s= 0.2 / 5

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg-preprocessing"))          #Hack for now
from ecgprep.img_helpers import vectorize_single_lead, mm_to_mv, mm_to_s


# Physical distance between two lead baselines, per layout (pmecg's row_distance * voltage, already snapped to a multiple of 5mm by the renderer).
# Need to think about these
ROW_dist_MM = {"1x12": 15.0, "2x6": 30.0, "4x3+1": 45.0}

                                                              
def load_peaks(peaks_csv):                               
    """(cuts, baselines) for one record.
    
    --peaks-dir", default="analysis/splits",

    cuts:      [(top, bottom), ...] pixel band per row
    baselines: array of baseline row positions
    """
    cuts = []                                           #emtpy, will have tuples
    baselines = []                                      #emtpy, will have values

    with open(peaks_csv) as f:
        for row in csv.DictReader(f):
            top = int(row["cut_above"])                 #read cut abouve
            bottom = int(row["cut_below"])              #read cut below
            cuts.append((top, bottom))                  #cuts get touple
            baselines.append(int(row["baseline_row"]))  #add the baselines also

    return cuts, np.array(baselines)


# Same source rows.py cut the pieces from, so cuts and template always agree.
# unet-src/data/test_index.csv is ground truth and belongs to compare_to_truth.py only.
LAYOUTS_CSV = "Predictions/test_score_report.csv"   # record, pred_layout, ...
KEYS_CSV = "layout_keys.csv"                              # template, n_rows, ...


def load_layout(record, layouts_csv=LAYOUTS_CSV):
    """The layout the classifier predicted for this record."""
    return pd.read_csv(layouts_csv).set_index("record").loc[record, "pred_layout"]


def expected_rows(layout, keys_csv=KEYS_CSV):
    """Rows on the page for a layout, from the shared contract."""
    return int(pd.read_csv(keys_csv).set_index("template").loc[layout, "n_rows"])


def load_paths(pieces_dir, record):
    """
    The row PNGs for one record, in row order (so row10 sorts after row9).
    """

    def row_number(path):
        """Pull NN out of '..._rowNN.png' as an int."""
        match = re.search(r"_row(\d+)\.png$", path)
        return int(match.group(1))

    pattern = os.path.join(pieces_dir, f"{record}_row*.png")
    paths = glob.glob(pattern)
    
    return sorted(paths, key=row_number)


def calibrate(peaks_csv, layout):
    """px/mm and sample rate, from the measured spacing of the baselines.
    dist in pixels is measured per record; dist in mm comes from the layout.

    returns: 
        cuts: says where to cut the image 
        px_per_mm: how many pixels are there in on mm
        fs: quotient

    """
    cuts, baselines = load_peaks(peaks_csv)       
    # print(f"cuts: {cuts}, baselines: {baselines}")

    dist_px = float(np.mean(np.diff(baselines)))
    # print(f"not the dist but axtual avg dist between baselines: {dist_px}")

    px_per_mm = dist_px / ROW_dist_MM[layout] #those 15, 30, 45. this means that if we take those pitches X px_per_mm we get teh distance_px
    # print(f"px_per_mm: {px_per_mm}, row dist:{ROW_dist_MM[layout]}, dist_px: {dist_px} ")

    fs = px_per_mm / mm_to_s  
    # mm_to_s= 0.2 / 5
    # print(f"convertion is {fs}")
    
    return cuts, px_per_mm, fs


def vectorize_piece(mask_path, page, top, bottom, px_per_mm):
    """
    One row piece -> one signal in mV, centred on its own baseline.
    """

    mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)             # Open it as a matrix of values
    page_crop = page[top:bottom, :]                                                          # Crop it accordingly to top and bottom values in cuts 
    assert page_crop.shape == mask.shape, f"{mask_path}: {page_crop.shape} vs {mask.shape}"  # For safety, so mask and page cropped have same dims

    signal = vectorize_single_lead(page_crop, mask)                                          # Vectorise it and fill th eedges w/ values.                
    
    centred = signal - np.nanmedian(signal)                                                  # And then put it in the middle, using median, most common value, not mean.
    return centred * mm_to_mv / px_per_mm                                                    # Pixels becomes mV! id by height over baseline 


def digitize(record, pieces_dir, peaks_dir, img_dir, layouts_csv, out_dir):
    """One record: predicted row masks + baseline cuts -> per-row mV signals.

    Calibrates px/mm from the baseline spacing, vectorizes each row piece
    against the original page, and writes Predictions/final/{record}_digitized.csv
    (one row per paper row, one column per pixel column).
    """

    layout = load_layout(record, layouts_csv)                                                           # predicted, e.g. "4x3+1"
    cuts, px_per_mm, _ = calibrate(os.path.join(peaks_dir, f"{record}_peaks.csv"), layout)             # Defaults analysis/splits/example_501_peaks.csv
    assert len(cuts) == expected_rows(layout), f"{layout} expects {expected_rows(layout)} rows, peaks csv has {len(cuts)}"

    page = np.asarray(Image.open(os.path.join(img_dir, record + ".png")).convert("L"))                  # Img_dir = unet-src/data/test_imgs, takes test image and (height, width) with values of darkness.
    paths = load_paths(pieces_dir, record)                                                             # take paths to cut-up pieces
    assert len(paths) == len(cuts), f"{len(paths)} pieces but {len(cuts)} rows in the peaks csv"        # For safety mesures

    signals = []

    for i in range(len(paths)):          # For all paths (0-4 eg)
        path = paths[i]                  # Predictions/pieces/example_501_row01.png
        top, bottom = cuts[i]            # (318, 728) - the band this piece came from

        signal = vectorize_piece(path, page, top, bottom, px_per_mm)
        signals.append(signal)           #adds array of that signal, so array of arryas, matrix.

        print(f"  row {i:2d}: rows {top:5d}-{bottom:5d}  {len(signal)} cols, "f"{np.isnan(signal).sum()} internal NaN")

    os.makedirs(out_dir, exist_ok=True)
    matrix = np.array(signals)                                                          # Make matrix out of the signal
    csv_path = os.path.join(out_dir, f"{record}_digitized.csv")                         # Put matrix in the digitized csv inside outdir, standardized to Predictions/final

    pd.DataFrame(matrix).to_csv(csv_path, index=False, header=False)
    print(f"wrote {csv_path}  shape={matrix.shape}")


def trace_span(pieces_dir, record):
    """First and last column carrying any predicted trace, across all pieces."""

    lo = np.inf        # smallest column index seen so far
    hi = -np.inf       # largest column index seen so far
    pattern = os.path.join(pieces_dir, f"{record}_row*.png")       #defaults to "Predictions/pieces"

    for path in glob.glob(pattern):                                #one iter per row piece
        piece = np.asarray(Image.open(path).convert("L"))          # (height, width)
        is_trace = piece > 0                                       # True where ink, so boolen
        column_has_trace = is_trace.any(axis=0)                    # (width,) one bool per column, makes it one long row
        cols = np.where(column_has_trace)[0]                       # indices of those columns, where true

        if len(cols) > 0:                                          # skip an all-blank piece
            lo = min(lo, cols[0])                                  # cols is sorted, so [0] is leftmost
            hi = max(hi, cols[-1])                                 # and [-1] is rightmost

    assert np.isfinite(lo), f"no trace pixels in any {record} piece"

    return int(lo), int(hi) + 1                                    #returns first and last trace value from the mask


def build_frame(rows, configuration):
    """Scatter each paper row's signal into the per-lead columns pmecg expects."""

    n = rows.shape[1]                       # samples per paper row, so hoe many colomn

    print(f"n is row shape 1: {n}")

    # every lead name mentioned anywhere in the configuration
    leads = set()
    for entry in configuration:

        # entry = ['I', 'aVR', 'V1', 'V4']
        # ['II', 'aVL', 'V2', 'V5']
        # ['III', 'aVF', 'V3', 'V6']
        # II

        if isinstance(entry, str):          # a bare rhythm-strip lead, e.g. "II"
            leads.add(entry)                #ends up last
        else:                               # a list of leads sharing one row
            for lead in entry:
                leads.add(lead)

    print(f"leads then looks like {leads}")
    leads = sorted(leads)
    print(f"sorting it then looks like {leads}")  #['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']


    # one column per lead, all NaN until filled, Result is a (2275, 12) frame of NaN
    df = pd.DataFrame(np.nan, index=range(n), columns=leads)


        #     " I  II  III  V1  V2  V3  V4  V5  V6  aVF  aVL  aVR
        # 0    NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 1    NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 2    NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 3    NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 4    NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # ...   ..  ..  ...  ..  ..  ..  ..  ..  ..  ...  ...  ...
        # 2327 NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 2328 NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 2329 NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 2330 NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN
        # 2331 NaN NaN  NaN NaN NaN NaN NaN NaN NaN  NaN  NaN  NaN

        # [2332 rows x 12 columns]

    for row_index in range(len(rows)):     #0,1,2,3 4 rows for 4 vectoriced ones
        signal = rows[row_index]            # signal is the first row, then the second etc
        entry = configuration[row_index]    # "II"  or  ["I", "aVR", "V1", "V4"] or the other arrays, should match by default! 12 signal rows, 12 configs

        if isinstance(entry, str):          # if its just a string, thereby a sole entry, append that df entry that signal
            df[entry] = signal 
            continue

        # split the row into len(entry) equal column blocks
        n_blocks = len(entry)                       # eg 4 in this case, 4 block on that row
        edges = np.linspace(0, n, n_blocks + 1)     # e.g. [0, 829.5, 1659, 2488.5, 3318]  split in 4 completely equal parts
        edges = edges.round().astype(int)           #      [0, 830,   1659, 2489,   3318]  round the edges so they become perfect pixelvalues

        for block_index in range(n_blocks):         # 0,
            print(block_index)
            lead = entry[block_index]           
            start = edges[block_index]
            stop = edges[block_index + 1]
            df.loc[start:stop - 1, lead] = signal[start:stop]

                    
        #                 block 0   block 1   block 2   block 3     ← time →
        # row 1  entry = [ "I",     "aVR",    "V1",     "V4"  ]
        # row 2  entry = [ "II",    "aVL",    "V2",     "V5"  ]
        # row 3  entry = [ "III",   "aVF",    "V3",     "V6"  ]
        # row 4  entry =  "II"  ────────── full width ─────────

    return df


def replot(record, csv_path, pieces_dir, peaks_dir, layouts_csv, out_path):

    layout = load_layout(record, layouts_csv)

    #take matrix csv, lo, hi from the mask, then cut the matrix where lo, hi is.
    matrix = pd.read_csv(csv_path, header=None).to_numpy() 

    print(f" matrix looks like : {matrix}")
                                                         
    lo, hi = trace_span(pieces_dir, record)                                                                     
    rows = matrix[:, lo:hi]                                                                                    

    print(f" matrix slimmed to only values looks like : {rows}")


    # px/mm and fs come from digitize_pieces.calibrate, so the replot's time axis
    # matches the original paper speed instead of assuming a dpi.
    _, px_per_mm, fs = calibrate(os.path.join(peaks_dir, f"{record}_peaks.csv"), layout)
    # print(f"layout={layout}  columns {lo}-{hi} " f"({rows.shape[1]} = {rows.shape[1] / fs:.1f}s)  fs={fs:.1f} Hz")

    # A canonical frame only to resolve the template's lead names, then the real one.
    stub = pd.DataFrame(np.zeros((1, 12)), columns=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"])


    #takes the stub names and gets what that array should look like for a given template    
    configuration = pmecg.template_factory(layout, stub, leads_map=None)
    print(configuration)


    assert len(configuration) == rows.shape[0], \
        f"{layout} has {len(configuration)} rows but the csv has {rows.shape[0]}"


    df = build_frame(rows, configuration)

    # Same layout as ptb-xl/example_record0.txt (samples x leads, h5 lead order, mV)
    h5_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    txt_path = out_path.replace("_replot.png", "_leads.txt")
    np.savetxt(txt_path, df[h5_order].to_numpy(), fmt="%.6f", delimiter="	",  header=f"{record} digitized ({len(df)} samples x 12 leads at {fs:.1f} Hz), one row per sample " + "	".join(h5_order))
 
    plotter = pmecg.ECGPlotter(row_distance=ROW_dist_MM[layout] / 10 )                       # pmecg wants mV; 10 mm/mV
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", nargs="?", default="example_501")
    ap.add_argument("--pieces-dir", default="Predictions/pieces")
    ap.add_argument("--peaks-dir", default="analysis/splits", help="where the peaks CSV lives")
    ap.add_argument("--img-dir", default="unet-src/data/test_imgs")
    ap.add_argument("--layouts-csv", default=LAYOUTS_CSV, help="score report with the predicted layout per record")
    ap.add_argument("--out-dir", default="Predictions/final")
    args = ap.parse_args()

    # Run to make the matrix
    # digitize(args.record, args.pieces_dir, args.peaks_dir, args.img_dir, args.layouts_csv, args.out_dir)

    #Run to make the replotted image
    csv_path = os.path.join(args.out_dir, f"{args.record}_digitized.csv")
    out = os.path.join(args.out_dir, f"{args.record}_replot.png")
    replot(args.record, csv_path, args.pieces_dir, args.peaks_dir, args.layouts_csv, out)
