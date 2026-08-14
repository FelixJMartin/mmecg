import glob
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


def overlay_comparison(truth_path, predicted_mask_path, out_path):
    """predicted_mask_path is the RAW predicted mask (neon-on-black), not a
    pmecg replot - the replot pipeline has a rendering-side drift, but the raw
    mask is already pixel-aligned with truth (verified directly)."""
    truth_img = Image.open(truth_path).convert("RGB")
    pred_img = Image.open(predicted_mask_path).convert("RGB")
    if truth_img.size != pred_img.size:
        pred_img = pred_img.resize(truth_img.size)

    truth_rgb = np.asarray(truth_img)
    pred_rgb = np.asarray(pred_img)

    truth_mask = truth_rgb.sum(axis=-1) < 200  # near-black ink
    pred_mask = pred_rgb.any(axis=-1)  # any non-black (neon) foreground

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
    for pred_path in sorted(glob.glob("Predictions/raw/*_epoch5.png")):
        name = os.path.basename(pred_path).split("_epoch")[0]
        truth_path = f"unet-src/data/test_imgs/{name}.png"
        out_path = f"comparisons/{name}_overlay.png"
        overlay_comparison(truth_path, pred_path, out_path)
        print(f"wrote {out_path}")
