"""
Replot a digitized prediction matrix (Predictions/pieces/*_digitized.csv) with
pmecg -- same minimal style as plott_ecg.py, just reading from our own
digitized CSV instead of the h5 file, and converting from raw pixel units to
real mV via ScaleFromPixels (the same physical calibration vectorize() uses).
"""
import os
import sys

import pandas as pd
import pmecg
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecg-preprocessing"))
from ecgprep.img_helpers import ScaleFromPixels

NAME = "example_1000_epoch10"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # "1x12" row order
SOURCE_IMG = "unet-src/data/test_imgs/example_1000.png"

matrix = pd.read_csv(f"Predictions/pieces/{NAME}_digitized.csv", header=None).to_numpy()

w_full, h_full = Image.open(SOURCE_IMG).size  # width still needed for sample rate below

# OLD: also used this full-page-height instance to scale the matrix into mV -- wrong,
# since digitize_pieces.py's matrix is now ALREADY in mV, calibrated per-piece there.
# ecg, fs = ScaleFromPixels(h_full, w_full)(matrix)
# df = pd.DataFrame(ecg.T, columns=LEADS)

# fs only depends on width (pixels_per_mm_w), so it's unaffected by the per-piece-height
# fix above -- the discarded first return value would be double-mV-scaled, so ignore it.
_, fs = ScaleFromPixels(h_full, w_full)(matrix)
df = pd.DataFrame(matrix.T, columns=LEADS)

plotter = pmecg.ECGPlotter()
configuration = pmecg.template_factory("1x12", df, leads_map=None)
fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs)

os.makedirs("Predictions/final", exist_ok=True)
fig.savefig(f"Predictions/final/{NAME}.png", dpi=300, bbox_inches="tight")
