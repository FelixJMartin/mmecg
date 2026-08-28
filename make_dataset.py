"""
Sole dataset-generation script for this project (replaces the old make_mask.py,
which only ever produced single-template "1x3" images).

Build a mixed-layout ECG segmentation set: N records, layout template drawn at
random per record from TEMPLATES, images fully styled (grid, calibration,
lead labels, separators) so the model sees what real printouts look like.

Masks are plotted trace-only from the same configuration. Both figures are
saved WITHOUT bbox_inches="tight" -- tight cropping depends on what is drawn,
so labels on the image and no labels on the mask would crop differently and
break pixel alignment.

Output goes to unet-src/data/mixed_imgs + mixed_masks (train pool; train.py
points there and does its own random train/val split at runtime) plus
unet-src/data/mixed_test_imgs + mixed_test_masks (held-out test set, records
disjoint from the train pool, for evaluating the trained model afterwards).
"""
import os
import random

import h5py
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

fs = 400
NEON_GREEN = (57, 255, 20)
N = 1000       # training-pool examples, h5 records [0, N)
N_TEST = 100   # held-out test examples, h5 records [N, N + N_TEST) -- disjoint from train
SEED = 0

# Restricted to the 3 layouts we actually want examples of (was every pmecg
# built-in template) -- rng.choice below picks one of these per record.
TEMPLATES = ['1x12', '2x6', '4x3+1']

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
# Same lead-naming/rename dance as make_mask.py: the h5 stores ecgprep's
# DI/DII/... convention, pmecg wants standard I/II/aVR/... names.
ecgprep_leads = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
rename = {'DI': 'I', 'DII': 'II', 'DIII': 'III', 'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF'}


def generate_pair(df, template, name, img_dir, mask_dir):
    '''One record -> one (image, mask) pair, both laid out with the same template
    so the trace positions line up pixel-for-pixel between the two.'''

    # template_factory turns a template name ("1x12", "2x6", ...) into a concrete
    # layout configuration (which leads go where, in what grid) for this record's df.
    configuration = pmecg.template_factory(template, df, leads_map=None)

    # Training image: default ECGPlotter() draws everything (grid, calibration
    # pulse, lead labels, separators) -- a realistic-looking printout.
    img_path = f"{img_dir}/{name}.png"
    fig_img = pmecg.ECGPlotter().plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_img.savefig(img_path, dpi=300)
    Image.open(img_path).convert("L").save(img_path)  # L = greyscale

    # Mask source: same configuration/layout, but everything except the bare
    # trace lines is switched off, so only the ECG curves get drawn.
    mask_plotter = pmecg.ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        show_separators=False,
        show_time_axis=False,
    )
    mask_path = f"{mask_dir}/{name}_mask.png"
    fig_mask = mask_plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_mask.savefig(mask_path, dpi=300)

    # Recolor: anything not pure white is trace -> neon green on black.
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    trace = (mask_rgb != 255).any(axis=-1)
    mask_img = np.zeros((*trace.shape, 3), dtype=np.uint8)
    mask_img[trace] = NEON_GREEN
    Image.fromarray(mask_img).save(mask_path)

    # matplotlib keeps every created figure in memory until closed -- with 1000+
    # figures per run that leaks memory, so close both once they're saved to disk.
    import matplotlib.pyplot as plt
    plt.close(fig_img)
    plt.close(fig_mask)
    return template


def generate_set(records, img_dir, mask_dir, index_path, seed):
    '''Generate one (image, mask) pair per record in `records`, logging each one
    (h5 row + chosen template) to `index_path` so results trace back later.'''
    os.makedirs(img_dir, exist_ok=True); os.makedirs(mask_dir, exist_ok=True)
    # Seeded RNG -> which template each record gets is reproducible across reruns.
    rng = random.Random(seed)
    with open(index_path, "w") as idx:
        idx.write("name,record,template\n")
        for i in records:
            # Pick a random layout for this record, independent of the previous one.
            template = rng.choice(TEMPLATES)
            # Row i of the h5's "tracings" dataset -> a (4096 samples, 12 leads) DataFrame.
            df = pd.DataFrame(f["tracings"][i], columns=[rename.get(l, l) for l in ecgprep_leads])
            generate_pair(df, template, f"example_{i}", img_dir, mask_dir)
            idx.write(f"example_{i},{i},{template}\n")
            if i % 50 == 0:
                print(i, template, flush=True)  # progress heartbeat


def main(img_dir="unet-src/data/mixed_imgs", mask_dir="unet-src/data/mixed_masks"):
    # Train pool: records [0, N).
    generate_set(range(N), img_dir, mask_dir, "dataset_index.csv", SEED)


def main_test(img_dir="unet-src/data/mixed_test_imgs", mask_dir="unet-src/data/mixed_test_masks"):
    # Held-out test set: records [N, N + N_TEST), never seen by main() -> safe
    # to evaluate the trained model on without any train/val leakage.
    generate_set(range(N, N + N_TEST), img_dir, mask_dir, "test_index.csv", SEED + 1)


if __name__ == "__main__":
    main()
    main_test()
