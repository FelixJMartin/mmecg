import glob
import os

from PIL import Image


def overlay_comparison(truth_path, predicted_path, out_path, truth_alpha=128):
    truth = Image.open(truth_path).convert("RGBA")
    predicted = Image.open(predicted_path).convert("RGBA")
    if truth.size != predicted.size:
        truth = truth.resize(predicted.size)

    truth.putalpha(truth_alpha)  # truth see-through, predicted stays fully opaque underneath
    composite = Image.alpha_composite(predicted, truth)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    composite.save(out_path)


if __name__ == "__main__":
    for pred_path in sorted(glob.glob("test_predictions_replotted/*_epoch5.png")):
        name = os.path.basename(pred_path).split("_epoch")[0]
        truth_path = f"unet-src/data/test_imgs/{name}.png"
        out_path = f"comparisons/{name}_overlay.png"
        overlay_comparison(truth_path, pred_path, out_path)
        print(f"wrote {out_path}")
