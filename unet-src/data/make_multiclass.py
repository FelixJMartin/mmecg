"""Build a 13-class ECG dataset: image + per-lead label mask, any layout.

    python unet-src/data/make_multiclass.py     # 500 train (600-1099) + 50 test_ (1100-1149)

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


# Paths resolve from this file, not the working directory, so the builder runs from
# anywhere: the h5 lives at the project root, the output next to train.py.
ROOT = Path(__file__).resolve().parents[2]                # C:\Users\felix\Downloads\Research\U-net seg
H5 = str(ROOT / "ptb-xl" / "ptb_preprocessed.h5")
OUT = str(Path(__file__).resolve().parents[1] / "data")   # .../unet-src/data


#settings
DPI = 300
SEED = 0                            # to generate the dataset
FIRST_RECORD = 600                  # past the 0-549 the binary dataset already uses


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LABEL = {
    "I":   1,  "II":  2,  "III": 3,      # limb leads
    "aVR": 4,  "aVL": 5,  "aVF": 6,      # augmented limb leads
    "V1":  7,  "V2":  8,  "V3":  9,      # chest leads
    "V4": 10,  "V5": 11,  "V6": 12,
}

assert LABEL == {lead: i + 1 for i, lead in enumerate(LEADS)}, "LABEL must follow LEADS order"
N_CLASSES = len(LABEL) + 1               # 13: background + 12 leads

#renaming the leads
ECGPREP_LEADS = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
RENAME = {"DI": "I", "DII": "II", "DIII": "III", "AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

#conversions
VOLTAGE = 10.0                      # mm per mV, pmecg's default
CANVAS_H = 2274                     #height of the full one

# Physical distance between two lead baselines, per layout. Same numbers the digitiser
# uses (digitize/Digitize.py ROW_dist_MM), so the two can never drift apart. They come
# from one page budget of 180 mm: 12 x 15 = 6 x 30 = 4 x 45 = 180.
ROW_dist_MM = {"1x12": 15.0, "2x6": 30.0, "4x3+1": 45.0}

TEMPLATES = list(ROW_dist_MM)       # adding a layout = adding one line above

#colour palette for the leads
PALETTE = [(255, 255, 255),                                       # 0 background
           (228, 26, 28), (55, 126, 184), (77, 175, 74),          # I, II, III
           (152, 78, 163), (255, 127, 0), (166, 206, 27),         # aVR, aVL, aVF
           (166, 86, 40), (247, 129, 191), (0, 206, 209),         # V1, V2, V3
           (120, 120, 120), (0, 0, 255), (255, 0, 255)]           # V4, V5, V6


def make_pair(df, template, name, prefix=""):
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

    Image.fromarray(img).save(f"{OUT}/{prefix}imgs/{name}.png")   # the input the model sees
    save_labels(labels, f"{OUT}/{prefix}labels/{name}.png")       # palette PNG: values 0..12
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



def build(prefix, first, n, seed):
        """Records first..first+n-1 -> {prefix}imgs/, {prefix}labels/, {prefix}index.csv."""

        #make the necissary imgs and labels directories
        os.makedirs(f"{OUT}/{prefix}imgs", exist_ok=True)
        os.makedirs(f"{OUT}/{prefix}labels", exist_ok=True)

        #make seed to reproduce
        rng = random.Random(seed)

        #lets go through all the data in the h5 preprocessed file. Both files stay open
        #for the whole loop: the index gets one line per record as that record is written.
        with h5py.File(H5, "r") as f, open(f"{OUT}/{prefix}index.csv", "w") as index:
            index.write("name,record,template\n")         # header, once

            for i in range(first, first + n):
                #chooses one tempalte
                template = rng.choice(TEMPLATES)

                signal = f["tracings"][i]                                           #(4096, 12): since we have 21k, 4096, 12
                lead_names = [RENAME.get(lead, lead) for lead in ECGPREP_LEADS]
                df = pd.DataFrame(signal, columns=lead_names)                       #make a pd dataframe out of it so we can handle it better

                #already here we make the imgs and labels
                name = f"example_{i}"                                               # give it a name also
                labels = make_pair(df, template, name, prefix)                      # writes {prefix}imgs/{name}.png and {prefix}labels/{name}.png
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

        print(f"wrote {n} {prefix or 'train '}pairs to {OUT}")


if __name__ == "__main__":
        N_TRAIN, N_TEST = 500, 50
        build("", FIRST_RECORD, N_TRAIN, SEED)                           # records 600-1099
        build("test_", FIRST_RECORD + N_TRAIN, N_TEST, SEED + 1)         # records 1100-1149, never trained on

