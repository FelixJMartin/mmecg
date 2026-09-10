"""
Render one ECG per layout template, all onto (nearly) the same canvas size.

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
    fig.savefig(out, dpi=300) 
    Image.open(out).convert("L").save(out)
    w_px, h_px = (d * 300 for d in fig.get_size_inches())
    print(f"{template:8s} {row_distance:9.1f} {row_distance*10:6.0f}mm {w_px:7.0f} x {h_px:<5.0f}")
