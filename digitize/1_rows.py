"""
Turns a predicted 13-class mask into one line per lead: a page row (y) for every page column,
NaN where the lead isn't drawn. Saved as csv/lines.csv, which 2_digitize.py turns into mV and seconds.
"""

import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# Paths -----------------------------
HERE = Path(__file__).parent                                        # the digitize folder
TEST = HERE.parent / "test-set"                                     # U-net seg/test-set

sys.path.insert(0, str(HERE.parent / "ecg-preprocessing"))          # where vectorize_single_lead lives
from ecgprep.img_helpers import vectorize_single_lead

mask_path = TEST / "test_fold10" / "pred_masks" / "example_56.png"   # the model's prediction
label_path = TEST / "test_labels" / "example_8.png"                 # the true mask, for testing later
img_path = TEST / "test_imgs" / f"{mask_path.stem}.png"             # the page that belongs to the mask

LEADS = ["bg", "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
OUT = HERE.parent / "Predictions" / "digitized" / mask_path.stem    # .../digitized/example_8
OUT.mkdir(parents=True, exist_ok=True)



def lead_boxes(mask, leads, margin=200):
    """Generous [top, left, bottom, right] box per lead, built so single strays can't stretch it."""
    boxes = {}
    for name in leads:
        rows, cols = np.where(mask == LEADS.index(name))            # all pixels of this lead
        if len(rows) == 0:
            continue
        centre = int(np.median(rows))                               # the row the lead mostly sits on (II -> rhythm strip)
        left, right = np.percentile(cols, [1, 99]).astype(int)      # ignore the 1% most extreme columns (strays)
        boxes[name] = [max(centre - margin, 0), left,               # top, left
                       min(centre + margin, mask.shape[0]), right + 1]   # bottom, right
    return boxes


# load -----------------------------
mask = np.array(Image.open(mask_path))                              # (H, W), pixel value = class id 0..12
img = np.array(Image.open(img_path).convert("L"))                   # the page, grayscale, same size as the mask
H, W = mask.shape

boxes = lead_boxes(mask, LEADS[1:])                                 # generous box per lead
print(f"the boxes are {boxes}")

# Plot ----------------------------
plt.imshow(mask, cmap="tab20")                                      # mask, one colour per class
for name, (top, left, bottom, right) in boxes.items():
    plt.gca().add_patch(Rectangle((left, top), right - left, bottom - top,
                                  fill=False, edgecolor="red"))     # Rectangle wants (x, y), width, height
    plt.text(left, top, name, color="red")                          # lead name in the corner
plt.savefig(OUT / f"{mask_path.stem}_boxes.png", dpi=120)
plt.show()



# make the boxes for vectorise -------------
lines = {}
for name in LEADS[1:]:                                              # "I", "II", ..., "V6"
    if name not in boxes:                                           # lead not predicted on this page
        continue

    # 1. this lead only: its box, and inside the box only its own pixels -----------------------------
    t, l, b, r = boxes[name]
    only_this = mask == LEADS.index(name)                           # all other leads -> False

    # 2. one y per column (darkness-weighted centre), back to page rows -----------------------------
    h_from_box_bottom = vectorize_single_lead(img[t:b, l:r], only_this[t:b, l:r])
    y = np.full(W, np.nan)                                          # full page width, NaN outside the box
    y[l:r] = b - h_from_box_bottom                                  # height above box bottom -> page row
    
    lines[name] = y

    # 3. look at it: full page size, only this lead -----------------------------
    plt.figure(figsize=(14, 14 * H / W))                            # same shape as the mask
    plt.plot(np.arange(W), y, lw=1)                                 # the lead's line, in page pixels
    plt.xlim(0, W); plt.ylim(H, 0)                                  # whole page; y flipped like an image
    plt.title(f"{name} ({mask_path.stem})")
    plt.savefig(OUT / "individual" / f"{name}_page.png", dpi=120)
    # plt.show()


(OUT / "csv").mkdir(exist_ok=True)
table = pd.DataFrame(lines)                                         # one row per page column, one column per lead
table.index.name = "x"                                              # label the page-column column in the file
table.to_csv(OUT / "csv" / "lines.csv", float_format="%.2f")        # all leads' y, read by 2_digitize.py
