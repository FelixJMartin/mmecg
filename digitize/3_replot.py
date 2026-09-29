"""Replot a digitized.csv as an ECG printout with pmecg, to compare by eye with the original page."""
from pathlib import Path

import pandas as pd
import pmecg

HERE = Path(__file__).parent
PAGE = HERE.parent / "Predictions" / "digitized" / "example_56"          # folder with digitized.csv
LAYOUT = "4x3+1"
ROW_DIST_MM = {"1x12": 15, "2x6": 30, "4x3+1": 45}                    # same row spacing as the training pages

df = pd.read_csv(PAGE / "csv" /"digitized.csv", index_col=0)                 # 4096 samples x 12 leads, mV, 400 Hz
config = pmecg.template_factory(LAYOUT, df, leads_map=None)           # which lead goes where on the page

plotter = pmecg.ECGPlotter(row_distance=ROW_DIST_MM[LAYOUT] / 10)     # pmecg wants mV: 45 mm / 10 mm per mV
fig = plotter.plot(df, configuration=config, sampling_frequency=400, show=False)
fig.savefig(PAGE / "replot.png", dpi=150)
print("saved", PAGE / "replot.png")
