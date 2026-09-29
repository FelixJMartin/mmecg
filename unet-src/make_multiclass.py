"""Build a 13-class ECG dataset: image + per-lead label mask, any layout.

    python unet-src/Dataset_build/make_multiclass.py 120     # build 120 records

Runs from any directory; output always lands in unet-src/data/.

Why per-lead classes instead of one binary trace class:


Output, nnU-Net-ish but plain PNGs so the existing U-Net can read them:

    unet-src/data/imgs/{name}.png     styled printout, greyscale
    unet-src/data/labels/{name}.png   uint8, 0 = background, 1..12 = lead (palette PNG)
    unet-src/data/index.csv           name, record, template
"""
import os
import random
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent            # .../unet-src/data, the output folder
ROOT = HERE.parents[1]                            # .../U-net seg
H5 = str(ROOT / "ptb-xl" / "ptb_preprocessed.h5")
OUT = str(HERE)

# ---------------------------------------------------------------- what to build
N_RECORDS = 120
FIRST_RECORD = 600                  # h5 row to start at
SEED = 0
DPI = 300                           # -> 3318 x 2274 px pages

# ---------------------------------------------------------------- leads and class ids
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LABEL = {
    "I":   1,  "II":  2,  "III": 3,      # limb leads
    "aVR": 4,  "aVL": 5,  "aVF": 6,      # augmented limb leads
    "V1":  7,  "V2":  8,  "V3":  9,      # chest leads
    "V4": 10,  "V5": 11,  "V6": 12,
}
assert LABEL == {lead: i + 1 for i, lead in enumerate(LEADS)}, "LABEL must follow LEADS order"
N_CLASSES = len(LABEL) + 1               # 13: background + 12 leads

# the h5 holds the same 12 leads in the same order, under ecgprep's names
ECGPREP_LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RENAME = dict(zip(ECGPREP_LEADS, LEADS))

# ---------------------------------------------------------------- page geometry
VOLTAGE = 10.0                      # mm per mV
CANVAS_H = 2274                     # every layout is padded to this height

ROW_dist_MM = {"1x12": 15.0, "2x6": 30.0, "4x3+1": 45.0}
TEMPLATES = list(ROW_dist_MM)

#colour palette for the leads
PALETTE = [(255, 255, 255),                                       # 0 background
           (228, 26, 28), (55, 126, 184), (77, 175, 74),          # I, II, III
           (152, 78, 163), (255, 127, 0), (166, 206, 27),         # aVR, aVL, aVF
           (166, 86, 40), (247, 129, 191), (0, 206, 209),         # V1, V2, V3
           (120, 120, 120), (0, 0, 255), (255, 0, 255)]           # V4, V5, V6


def make_pair(df, template, name):
    # 1. the printout the model will see: grid, labels, calibration, all of it
    page = render(df, template)                       # (2273, 3318) or (2244, 3318), 255 = white
    img = pad_bottom(page, CANVAS_H, 255)             # (2274, 3318), padded with white paper

    labels = np.zeros_like(img)                      # 0 everywhere = background

    for lead in LEADS:
        # Blank every other lead. pmecg skips NaN samples, so the layout is laid out
        # exactly as before but only this one lead actually gets a line drawn.
        other_leads = [name for name in LEADS if name != lead]
        only = df.copy()
        only[other_leads] = np.nan                  #turn all other leads off

        # Same geometry as `img`, so the pixels line up one to one.
        trace = render(only, template, trace_only=True)   # since trace only is true, it turns everything off
        trace = pad_bottom(trace, CANVAS_H, 255)
        drawn = trace < 255                               # True where this lead has ink

        # Stamp those pixels with this lead's class id. Where two leads cross, the later
        # one overwrites the earlier: one pixel can only carry one label.
        labels[drawn] = LABEL[lead]                         

    Image.fromarray(img).save(f"{OUT}/imgs/{name}.png")   # the input the model sees
    save_labels(labels, f"{OUT}/labels/{name}.png")       # palette PNG: values 0..12
    return labels


def render(df, template, trace_only=False):
    """One figure -> greyscale array (255 = white paper).

    trace_only=True switches off everything except the signal (grid, calibration pulse,
    lead labels, separators, time axis), which is what the label masks are built from.
    """

    #set config acc to template and df
    configuration = pmecg.template_factory(template, df, leads_map=None)

    # how far apart the rows sit, in mV: pmecg wants mV, ROW_dist_MM is in mm
    # (15 mm at 10 mm/mV -> 1.5 mV, 30 -> 3.0, 45 -> 4.5)
    dist = ROW_dist_MM[template] / VOLTAGE

    #depending on trace only true or not says what to save
    if trace_only:
        # labels are built from this: signal only, nothing else on the page
        plotter = pmecg.ECGPlotter(row_distance=dist,
                                   grid_mode=None,
                                   show_calibration=False,
                                   show_leads_labels=False,
                                   show_separators=False,
                                   show_time_axis=False)
    else:
        # the printout the model sees: grid, calibration pulse, lead labels, separators
        plotter = pmecg.ECGPlotter(row_distance=dist)

    #we plot the df on that config
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=400, show=False)
    fig.set_dpi(DPI)                                     # 300 dpi -> 3318 x 2274 px
    fig.canvas.draw()                                       # rasterise the figure

    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # (H, W, 4) RGBA -> drop alpha
    plt.close(fig)                                        # free it: 13 figures per record

    return rgb.mean(axis=2).astype(np.uint8)              # (H, W) grey, 255 = blank paper  


def pad_bottom(arr, height, fill):
    """Add blank rows at the bottom so every layout ends up the same height.

    currently needed to not ahve the model just learn image sizes. 
    """
    current_height, width = arr.shape
    missing_rows = height - current_height
    assert missing_rows >= 0, "image is taller than the canvas"

    if missing_rows == 0:
        return arr

    blank = np.full((missing_rows, width), fill, dtype=arr.dtype)
    return np.concatenate([arr, blank])


def save_labels(labels, path):
    """Write the class-index mask as a palette PNG: same numbers, but a viewer shows colours."""
    out = Image.fromarray(labels, mode="P")
    out.putpalette([channel for colour in PALETTE for channel in colour] + [0] * (256 - len(PALETTE)) * 3)
    out.save(path)



if __name__ == "__main__":

        #how many images
        n = 120

        #make the necissary imgs and labels directories
        os.makedirs(f"{OUT}/imgs", exist_ok=True)
        os.makedirs(f"{OUT}/labels", exist_ok=True)

        #make seed to reproduce
        rng = random.Random(SEED)

        #lets go through all the data in the h5 preprocessed file: 
        #lets go through all the data in the h5 preprocessed file. Both files stay open
        #for the whole loop: the index gets one line per record as that record is written.
        with h5py.File(H5, "r") as f, open(f"{OUT}/index.csv", "w") as index:
            index.write("name,record,template\n")         # header, once

            for i in range(FIRST_RECORD, FIRST_RECORD + n):
                #chooses one tempalte
                template = rng.choice(TEMPLATES)

                signal = f["tracings"][i]                                           #(4096, 12): since we have 21k, 4096, 12
                lead_names = [RENAME.get(lead, lead) for lead in ECGPREP_LEADS]
                df = pd.DataFrame(signal, columns=lead_names)                       #make a pd dataframe out of it so we can handle it better

                #already here we make the imgs and labels
                name = f"example_{i}"                                               # give it a name also
                labels = make_pair(df, template, name)                              # writes imgs/{name}.png and labels/{name}.png
                index.write(f"{name},{i},{template}\n")                             # fill the csv with actuall values

                # what this template says should be on the page: each entry is either a
                # list of leads sharing a row, or a bare lead name for a rhythm strip
                configuration = pmecg.template_factory(template, df, leads_map=None)
                expected = set()

                for entry in configuration:           # entry is the list or jsut names of leads
                    if isinstance(entry, str):        # a rhythm strip: one lead, full width
                        expected.add(entry)
                    else:                             # a row of leads side by side
                        for lead in entry:
                            expected.add(lead)

                present = set()

                for class_id in np.unique(labels):
                    if class_id > 0:                        # 0 is background, not a lead
                        present.add(LEADS[class_id - 1])    # since class 1 is class 0

                # Fail loudly rather than train on a record with a lead silently missing.
                assert present == expected, \
                    f"{name} {template}: labelled {sorted(present)}, expected {sorted(expected)}"

                print(f"{name} {template:6s} {len(present):2d} leads")

        print(f"wrote {n} pairs to {OUT}")
    
