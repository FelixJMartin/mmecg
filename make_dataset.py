"""
Build a mixed-layout ECG segmentation set: 1000 records, layout template drawn
at random from every pmecg built-in, images fully styled (grid, calibration,
lead labels, separators) so the model sees what real printouts look like.

Masks are plotted trace-only from the same configuration. Both figures are
saved WITHOUT bbox_inches="tight" -- tight cropping depends on what is drawn,
so labels on the image and no labels on the mask would crop differently and
break pixel alignment.
"""
import os
import random

import matplotlib.pyplot as plt   # <-- only real issue
import h5py
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

fs = 400
NEON_GREEN = (57, 255, 20)
N = 500         # training records: h5 rows 0..499
N_TEST = 50     # test records: h5 rows 500..549
SEED = 0

TEMPLATES = ['1x12', '2x6', '4x3+1']

# Per-template row spacing, so every layout renders at the SAME canvas height.
# pmecg sizes a figure as height ~ (n_rows - 1) * row_distance * voltage, so at a
# single row_distance each template got its own height (1535/2244/4370px) and the
# image height alone gave away the layout label -- a shortcut that doesn't exist on
# real scans, where the paper is fixed and the layout varies within it. Fewer rows
# -> wider spacing, same page, exactly like real ECG paper.
# NOTE: pmecg snaps row_distance * voltage to a multiple of 5mm, so these land at
# 2274/2244/2274 -- the residual 29px is padded out below (CANVAS_H).
ROW_DISTANCE = {'1x12': 1.5, '2x6': 3.0, '4x3+1': 4.5}
CANVAS_H = 2274  # tallest of the three; everything is bottom-padded up to this


def pad_bottom(arr, height, fill):
    '''Pad arr's bottom edge up to `height` rows with `fill`. Padding (not resizing)
    on purpose: rescaling would change the mm-per-pixel grid calibration that the
    digitize pipeline's ScaleFromPixels depends on.'''
    pad = height - arr.shape[0]
    assert pad >= 0, f'image is {arr.shape[0]}px, taller than CANVAS_H={height}'
    if pad == 0:
        return arr
    return np.concatenate([arr, np.full((pad, *arr.shape[1:]), fill, dtype=arr.dtype)])

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
ecgprep_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
rename = {'DI': 'I', 'DII': 'II', 'DIII': 'III', 'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}


def generate_pair(df, template, name, img_dir, mask_dir):
    configuration = pmecg.template_factory(template, df, leads_map=None)  # layout: which lead goes where

    # --- training image: fully styled, realistic printout ---
    img_path = f"{img_dir}/{name}.png"
    fig_img = pmecg.ECGPlotter(row_distance=ROW_DISTANCE[template]).plot(
        df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_img.savefig(img_path, dpi=300)
    # force greyscale, then pad to the common canvas (white = image background)
    img_grey = np.asarray(Image.open(img_path).convert("L"))
    Image.fromarray(pad_bottom(img_grey, CANVAS_H, 255)).save(img_path)

    # --- mask: same layout, trace-only (everything else switched off) ---
    mask_plotter = pmecg.ECGPlotter(
        row_distance=ROW_DISTANCE[template],  # must match fig_img exactly, or mask/image misalign
        grid_mode=None, show_calibration=False,
        show_leads_labels=False, show_separators=False, show_time_axis=False,
    )
    mask_path = f"{mask_dir}/{name}_mask.png"
    fig_mask = mask_plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_mask.savefig(mask_path, dpi=300)

    # recolor: non-white pixels -> neon green on black (binary trace mask)
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    trace = (mask_rgb != 255).any(axis=-1)
    mask_img = np.zeros((*trace.shape, 3), dtype=np.uint8)
    mask_img[trace] = NEON_GREEN
    Image.fromarray(pad_bottom(mask_img, CANVAS_H, 0)).save(mask_path)  # black = mask background

    plt.close(fig_img)
    plt.close(fig_mask)
    return template

def main(img_dir, mask_dir, index_csv, start, stop, seed):
    os.makedirs(img_dir, exist_ok=True); os.makedirs(mask_dir, exist_ok=True)
    rng = random.Random(seed)
    with open(index_csv, "w") as idx:
            idx.write("name,record,template\n")  # CSV header

            for i in range(start, stop):  # loop over this split's h5 record indices
                template = rng.choice(TEMPLATES)  # pick a random layout for this record

                # row i of the h5's "tracings" dataset -> (4096 samples, 12 leads),
                # renamed from ecgprep's DI/DII/... convention to pmecg's I/II/... convention
                df = pd.DataFrame(f["tracings"][i], columns=[rename.get(l, l) for l in ecgprep_leads])

                generate_pair(df, template, f"example_{i}", img_dir, mask_dir)  # render + save (image, mask) pair

                idx.write(f"example_{i},{i},{template}\n")  # log this record's name/index/template to the CSV

                if i % 50 == 0:
                    print(i, template, flush=True)  # progress heartbeat every 50 records


if __name__ == "__main__":
    # Train and test are the SAME generator over disjoint h5 record ranges -- the test
    # split gets its own seed so its template draw is independent of the training one.
    main("unet-src/data/imgs", "unet-src/data/masks",
         "unet-src/data/dataset_index.csv", 0, N, SEED)
    main("unet-src/data/test_imgs", "unet-src/data/test_masks",
         "unet-src/data/test_index.csv", N, N + N_TEST, SEED + 1)
