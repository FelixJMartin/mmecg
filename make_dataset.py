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
N = 1000        # training records: h5 rows 0..999
N_TEST = 100    # test records: h5 rows 1000..1099
SEED = 0

TEMPLATES = ['1x12', '2x6', '4x3+1']

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
ecgprep_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
rename = {'DI': 'I', 'DII': 'II', 'DIII': 'III', 'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}


def generate_pair(df, template, name, img_dir, mask_dir):
    configuration = pmecg.template_factory(template, df, leads_map=None)  # layout: which lead goes where

    # --- training image: fully styled, realistic printout ---
    img_path = f"{img_dir}/{name}.png"
    fig_img = pmecg.ECGPlotter().plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_img.savefig(img_path, dpi=300)
    Image.open(img_path).convert("L").save(img_path)  # force greyscale

    # --- mask: same layout, trace-only (everything else switched off) ---
    mask_plotter = pmecg.ECGPlotter(
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
    Image.fromarray(mask_img).save(mask_path)

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
    main("unet-src/data/mixed_imgs", "unet-src/data/mixed_masks",
         "unet-src/data/dataset_index.csv", 0, N, SEED)
    main("unet-src/data/mixed_test_imgs", "unet-src/data/mixed_test_masks",
         "unet-src/data/test_index.csv", N, N + N_TEST, SEED + 1)
