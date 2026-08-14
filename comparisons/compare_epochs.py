import glob
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

TRUTH_COLOR = np.array([180, 180, 180])  # light grey reference line
EPOCH_COLORS = {
    1: np.array([235, 200, 40]),  # yellow
    2: np.array([110, 200, 60]),  # green
    3: np.array([40, 190, 180]),  # teal
    4: np.array([60, 110, 230]),  # blue
    5: np.array([170, 60, 210]),  # purple
}
EPOCH_ALPHA = 230
TRUTH_ALPHA = 130


def _mask_layer(mask, color, alpha):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    a = np.zeros((h, w), dtype=np.uint8)
    rgb[mask] = color
    a[mask] = alpha
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def _grid_background(img_rgb):
    near_black = img_rgb.sum(axis=-1) < 200
    bg = img_rgb.copy()
    bg[near_black] = 255
    return Image.fromarray(bg).convert("RGBA")


def multi_epoch_overlay(truth_path, epoch_mask_paths, out_path):
    """epoch_mask_paths: dict {epoch_number: raw_predicted_mask_path}.

    Shows each epoch's DISAGREEMENT with truth (XOR), not its full mask -
    plotting full masks stacked in order hides everything under whichever
    layer is drawn last. Early epochs should show big, spread-out colored
    error clouds; later epochs shrink toward nothing as they converge.
    """
    truth_img = Image.open(truth_path).convert("RGB")
    truth_rgb = np.asarray(truth_img)
    truth_mask = truth_rgb.sum(axis=-1) < 200

    composite = _grid_background(truth_rgb)
    composite = Image.alpha_composite(composite, _mask_layer(truth_mask, TRUTH_COLOR, TRUTH_ALPHA))
    for epoch in sorted(epoch_mask_paths):  # early (biggest diff) drawn first, later (smallest) on top
        pred_img = Image.open(epoch_mask_paths[epoch]).convert("RGB")
        if pred_img.size != truth_img.size:
            pred_img = pred_img.resize(truth_img.size)
        pred_mask = np.asarray(pred_img).any(axis=-1)
        diff_mask = truth_mask ^ pred_mask  # where this epoch disagrees with truth
        composite = Image.alpha_composite(composite, _mask_layer(diff_mask, EPOCH_COLORS[epoch], EPOCH_ALPHA))

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.imshow(composite)
    ax.axis("off")
    handles = [mpatches.Patch(color=TRUTH_COLOR / 255, label="Truth")]
    handles += [mpatches.Patch(color=EPOCH_COLORS[e] / 255, label=f"Epoch {e}") for e in sorted(epoch_mask_paths)]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    names = sorted({
        os.path.basename(p).split("_epoch")[0]
        for p in glob.glob("Predictions/raw/example_103*_epoch*.png")
    })
    for name in names:
        epoch_paths = {
            int(p.split("_epoch")[1].replace(".png", "")): p
            for p in glob.glob(f"Predictions/raw/{name}_epoch*.png")
        }
        truth_path = f"unet-src/data/test_imgs/{name}.png"
        out_path = f"comparisons/{name}_epochs.png"
        multi_epoch_overlay(truth_path, epoch_paths, out_path)
        print(f"wrote {out_path}")
