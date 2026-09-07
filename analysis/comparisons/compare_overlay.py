import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

TRUTH_COLOR = np.array([30, 60, 220])  # clear blue
PRED_COLOR = np.array([220, 30, 30])  # red
AGREE_COLOR = np.array([160, 40, 200])  # purple - both agree here
TRACE_ALPHA = 220


def _diff_layer(truth_mask, pred_mask):
    """RGBA layer: blue where only truth has ink, red where only prediction
    does, purple where both agree - makes misalignment/disagreement visible
    instead of one color simply hiding the other."""
    h, w = truth_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)

    only_truth = truth_mask & ~pred_mask
    only_pred = pred_mask & ~truth_mask
    both = truth_mask & pred_mask

    rgb[only_truth] = TRUTH_COLOR
    rgb[only_pred] = PRED_COLOR
    rgb[both] = AGREE_COLOR
    alpha[only_truth | only_pred | both] = TRACE_ALPHA

    layer = np.dstack([rgb, alpha])
    return Image.fromarray(layer, "RGBA")


def _grid_background(img_rgb):
    """The grid/white background only, with the original black trace removed."""
    near_black = img_rgb.sum(axis=-1) < 200
    bg = img_rgb.copy()
    bg[near_black] = 255
    return Image.fromarray(bg).convert("RGBA")


def overlay_comparison(image_path, truth_mask_path, predicted_mask_path, out_path):
    """Overlay the predicted mask on the ground-truth mask, over the page's grid.

    predicted_mask_path is the RAW predicted mask (neon-on-black), not a pmecg
    replot - the replot pipeline has a rendering-side drift, but the raw mask is
    already pixel-aligned with truth (verified directly).

    Truth comes from the MASK, not from the ink in the source image: the image
    also contains lead labels and calibration pulses, which the mask deliberately
    excludes. Taking truth from the image would paint those blue ("truth only")
    and read as model failures when ignoring them is correct behaviour.
    """
    page_img = Image.open(image_path).convert("RGB")
    truth_img = Image.open(truth_mask_path).convert("RGB")
    pred_img = Image.open(predicted_mask_path).convert("RGB")
    if page_img.size != pred_img.size:
        pred_img = pred_img.resize(page_img.size)

    truth_rgb = np.asarray(page_img)                 # grid background comes from the page
    truth_mask = np.asarray(truth_img).any(axis=-1)  # neon-on-black -> any non-black
    pred_mask = np.asarray(pred_img).any(axis=-1)

    composite = _grid_background(truth_rgb)  # raw mask has no grid of its own
    composite = Image.alpha_composite(composite, _diff_layer(truth_mask, pred_mask))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.imshow(composite)
    ax.axis("off")
    legend_handles = [
        mpatches.Patch(color=TRUTH_COLOR / 255, label="Truth only"),
        mpatches.Patch(color=PRED_COLOR / 255, label="Predicted only"),
        mpatches.Patch(color=AGREE_COLOR / 255, label="Both agree"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Overlay a predicted mask against ground truth.")
    ap.add_argument("record", nargs="?", default="example_549", help="e.g. example_549")
    ap.add_argument("--pred", default=None, help="predicted mask (default: Predictions/raw_v2/<record>_epoch9.png)")
    ap.add_argument("--img-dir", default="unet-src/data/test_imgs")
    ap.add_argument("--mask-dir", default="unet-src/data/test_masks")
    ap.add_argument("--out", default=None, help="default: comparisons/<record>_overlay.png")
    args = ap.parse_args()

    pred = args.pred or f"Predictions/raw_v2/{args.record}_epoch9.png"
    out = args.out or f"analysis/comparisons/{args.record}_overlay.png"
    overlay_comparison(f"{args.img_dir}/{args.record}.png",
                       f"{args.mask_dir}/{args.record}_mask.png",
                       pred, out)
    print(f"wrote {out}")
