"""
Cut a raw prediction mask into per-row pieces, using the row-boundary minima
already found by raw_prediction_density.py (Predictions/replotted/*_row_boundaries.csv).
One piece per lead row -- vectorize() can then be run on each piece separately,
without needing the full tlbr_boxes layout up front.

n boundaries split the image into n+1 pieces (a cut is a line BETWEEN two
pieces, not a piece itself) -- e.g. "1x12"'s 11 boundaries -> 12 pieces, one
per lead.
"""
import csv
import os

from PIL import Image


def load_boundaries(csv_path):
    with open(csv_path) as f:
        return [int(row["row_boundary"]) for row in csv.DictReader(f)]


def cut_pieces(pred_path, boundaries_csv, out_dir):
    name = os.path.splitext(os.path.basename(pred_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(pred_path)
    w, h = img.size
    boundaries = load_boundaries(boundaries_csv)
    cuts = [0] + boundaries + [h]  # len(boundaries) cuts -> len(boundaries)+1 pieces

    for i in range(len(cuts) - 1):
        top, bottom = cuts[i], cuts[i + 1]
        piece = img.crop((0, top, w, bottom))
        out_path = f"{out_dir}/{name}_piece{i + 1}.png"
        piece.save(out_path)
        print(f"saved {out_path}  (rows {top}:{bottom})")


if __name__ == "__main__":
    cut_pieces(
        "Predictions/raw/example_1000_epoch10.png",
        "Predictions/replotted/example_1000_epoch10_row_boundaries.csv",
        "Predictions/pieces",
    )
