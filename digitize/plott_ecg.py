"""
Render one ECG per layout template, all onto (nearly) the same canvas size.

WHY THIS EXISTS
---------------
The mixed-layout training set had a leak: each template came out at its own
image height (4x3+1 -> 1535px, 2x6 -> 2244px, 1x12 -> 4370px), and no two
templates ever shared a height. So "how tall is this image" WAS the layout
label. A classifier could score 100% by reading the image dimensions without
ever looking at the ECG -- and that shortcut doesn't exist on real scans, where
the paper is a fixed size and the layout varies within it.

This script renders all three templates at a matched height instead, so the only
way to tell them apart is by looking at the content (how many rows, how many
columns, where the separators fall).

HOW THE HEIGHT IS CONTROLLED
----------------------------
pmecg builds the figure at true physical size. From pmecg/utils/plot.py:

    row_distance_mm  = row_distance * voltage
    total_height_mm  = top_margin + (n_rows - 1) * row_distance_mm + bottom_margin

So height is driven by (n_rows - 1) * pitch. n_rows is fixed by the template,
which leaves `row_distance` as the knob: give each template its own value so the
products come out equal. Fewer rows -> more space each; more rows -> tighter.
That is also how real ECG paper behaves -- 12 stacked leads on a fixed page are
printed closer together, the page does not grow.

    template   n_rows   pitch      (n_rows-1) * pitch
    1x12         12     15 mm      11 * 15 = 165 mm
    2x6           6     30 mm       5 * 30 = 150 mm
    4x3+1         5     45 mm       4 * 45 = 180 mm

(4x3+1 renders 5 rows, not 4: four rows of three leads plus the rhythm strip.)

TWO GOTCHAS THAT COST ME A WRONG ANSWER FIRST TIME
--------------------------------------------------
1. pmecg snaps row_distance * voltage to a multiple of 5mm
   (_adjust_row_distance in pmecg/plot.py). Height is therefore a STEP function
   of row_distance -- you cannot dial in an arbitrary height, and solving for one
   numerically will bounce around. The three values below are the closest match
   the 5mm lattice allows, not an exact one (30px spread, ~1.3%).

2. ECGPlotter.plot() defaults to show=True, and the show path RESIZES the
   figure. Measuring the result without show=False reports a size that has
   nothing to do with what savefig writes. make_dataset.py already passes
   show=False; so does this script.

Run from the repo root. Writes layout_preview/<template>.png.
"""
import os

import h5py
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed; we only ever savefig
import pandas as pd
import pmecg
from PIL import Image

fs = 400        # PTB-XL preprocessed sampling rate (Hz)
RECORD = 0      # which h5 row to draw; any record works, this is just a preview

# The knob. pitch_mm = row_distance * voltage, and voltage defaults to 10 mm/mV,
# so these are 15mm / 30mm / 45mm row pitches respectively.
ROW_DISTANCE = {"1x12": 1.5, "2x6": 3.0, "4x3+1": 4.5}

# ecgprep names its leads DI/DII/...; pmecg expects I/II/... -- rename on the way in.
ecgprep_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
rename = {'DI': 'I', 'DII': 'II', 'DIII': 'III', 'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
df = pd.DataFrame(f["tracings"][RECORD], columns=[rename.get(l, l) for l in ecgprep_leads])

os.makedirs("layout_preview", exist_ok=True)
print(f"{'template':8s} {'row_dist':>9s} {'pitch':>7s} {'size (px)':>14s}")
for template, row_distance in ROW_DISTANCE.items():
    # row_distance is the ONLY thing that differs per template here -- same data,
    # same gain, same paper speed, so any visual difference is layout alone.
    plotter = pmecg.ECGPlotter(row_distance=row_distance)
    configuration = pmecg.template_factory(template, df, leads_map=None)
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)

    out = f"layout_preview/{template}.png"
    fig.savefig(out, dpi=300)  # no bbox_inches="tight": it crops to drawn content,
                               # which would reintroduce a per-template size difference
    # Force greyscale, same as make_dataset.py -- the UNet takes n_channels=1, so an
    # RGBA preview would blow up at the first conv with a 4-vs-1 channel mismatch.
    Image.open(out).convert("L").save(out)
    w_px, h_px = (d * 300 for d in fig.get_size_inches())
    print(f"{template:8s} {row_distance:9.1f} {row_distance*10:6.0f}mm {w_px:7.0f} x {h_px:<5.0f}")
