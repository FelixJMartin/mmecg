"""
Run a trained U-Net checkpoint on ECG pages. Two modes:

python predict.py -m runs/name/checkpoints/checkpoint_epoch40.pth -i page1.jpg page2.png
    predict single pages; coloured masks go to Predictions/raw/<name>.png

python predict.py -m runs/name/checkpoints/checkpoint_epoch40.pth --score [N]
    predict the test set (first N images, or all), score each against its true mask,
    print a table and write <save-dir>/score_report.csv (read by analysis/plot_test_metrics.py)
"""
#syntaxes for usage
import argparse
import csv
import logging
from pathlib import Path

#packages
import numpy as np
import torch
from PIL import Image

#own code
from unet import UNet
from utils.data_loading import image_to_array, IDX_TO_TEMPLATE

#paths
test_img, test_mask, test_index = Path('data/test_imgs'), Path('data/test_labels'), Path('data/test_index.csv')
raw_dir = Path('Predictions/raw')              # single-page mode output

# ---------------------------------------------------------------- fixed settings, must match train.py
n_classes = 13            # background + 12 leads
n_layouts = 4             # 1x12, 2x6, 4x3+1, 2x6+1

# one colour per class, same as dataset_build/make_multiclass.py (copied: importing it pulls in h5py, pmecg, ...)
palette = [(255, 255, 255),                                       # 0 background
           (228, 26, 28), (55, 126, 184), (77, 175, 74),          # I, II, III
           (152, 78, 163), (255, 127, 0), (166, 206, 27),         # aVR, aVL, aVF
           (166, 86, 40), (247, 129, 191), (0, 206, 209),         # V1, V2, V3
           (120, 120, 120), (0, 0, 255), (255, 0, 255)]           # V4, V5, V6


# ---------------------------------------------------------------- step helpers
def predict_page(net, image, device):
    """One page in -> class mask (H, W) with values 0..12, and the layout as (name, confidence)."""
    # PIL image -> tensor (1, 3, H, W), values 0..1; the 1 is a fake batch of one page
    x = torch.from_numpy(image_to_array(image)).unsqueeze(0).to(device)

    #  forward pass, no gradients needed
    with torch.no_grad():
        logits, layout_logits = net(x)

    # every pixel gets the class with the highest score: (1, 13, H, W) -> (H, W)
    mask = logits.argmax(dim=1)[0].cpu().numpy()

    # 4. layout: softmax, not just argmax, so we also get a confidence.
    probs = layout_logits[0].softmax(dim=0).cpu()
    idx = int(probs.argmax())
    return mask, (IDX_TO_TEMPLATE[idx], float(probs[idx]))


def overlap_scores(pred, true):
    """Dice, precision and recall for two boolean masks."""
    hit = (pred & true).sum()
    dice = 2 * hit / max(pred.sum() + true.sum(), 1)
    precision = hit / max(pred.sum(), 1)       
    recall = hit / max(true.sum(), 1)          
    return dice, precision, recall


def lead_dice(pred, true):
    """Mean dice over the leads that appear in pred or true: a pixel only counts if ink AND lead are right."""
    scores = []
    for c in range(1, n_classes):              # 1..12, skip background
        if (true == c).any() or (pred == c).any():
            scores.append(overlap_scores(pred == c, true == c)[0])
    return float(np.mean(scores)) if scores else float('nan')


def save_mask(mask, path):
    """Write class ids as a palette PNG: same numbers, but a viewer shows each lead in its own colour."""
    out = Image.fromarray(mask.astype(np.uint8), mode='P')
    out.putpalette([channel for colour in palette for channel in colour])
    out.save(path)


# ---------------------------------------------------------------- score mode
def score_test_set(net, device, limit, out_dir, tag):
    """Predict every test page, score it against its true mask, print a table and write score_report.csv.

    Precision and recall sit next to dice because dice alone hides whether the model over- or under-predicts.
    """
    # 1. page / layout / persona of every test image
    with open(test_index) as f:
        info = {r['name']: r for r in csv.DictReader(f)}

    # 2. which images: sorted by (length, name) so example_9 comes before example_10
    names = sorted((p.stem for p in test_img.iterdir()), key=lambda n: (len(n), n))[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'record':24s} {'true':7s} {'pred':7s} {'conf':>6s} {'dice':>7s} {'prec':>7s} {'rec':>7s} {'lead':>7s}")
    rows = []
    for name in names:
        # 3a. predict
        image = Image.open(next(test_img.glob(name + '.*'))).convert('RGB')      # .jpg or .png
        pred, (pred_layout, confidence) = predict_page(net, image, device)

        # 3b. score against the truth (palette PNG, pixel = class id 0..12)
        true = np.asarray(Image.open(test_mask / f'{name}.png'))
        dice, precision, recall = overlap_scores(pred > 0, true > 0)   # any lead vs background
        ldice = lead_dice(pred, true)                                   # right ink AND right lead

        # 3c. save the mask; tag = checkpoint name, so masks from different runs never collide
        save_mask(pred, out_dir / f'{name}_{tag}.png')

        # 3d. one report row + one printed line
        tags = info.get(name, {})
        true_layout = tags.get('template', '?')
        rows.append({'record': name, 'page': tags.get('page', '?'), 'persona': tags.get('persona', '?'),
                     'true_layout': true_layout, 'pred_layout': pred_layout, 'confidence': round(confidence, 4),
                     'dice': round(dice, 4), 'precision': round(precision, 4), 'recall': round(recall, 4),
                     'lead_dice': round(ldice, 4), 'layout_correct': int(pred_layout == true_layout)})
        print(f"{name:24s} {true_layout:7s} {pred_layout:7s} {confidence:6.3f} "
              f"{dice:7.4f} {precision:7.4f} {recall:7.4f} {ldice:7.4f}  "
              f"{'OK' if pred_layout == true_layout else 'MISS'}")

    # 4. the mean over all images, printed only (the plot scripts compute their own means)
    mean = {k: np.mean([r[k] for r in rows]) for k in ('dice', 'precision', 'recall', 'lead_dice', 'layout_correct')}
    print(f"\nMEAN over {len(rows)}: dice={mean['dice']:.4f}  precision={mean['precision']:.4f}  "
          f"recall={mean['recall']:.4f}  lead_dice={mean['lead_dice']:.4f}  layout_acc={mean['layout_correct']:.4f}")

    # 5. the report lands next to the masks, so a prediction folder describes itself
    path = out_dir / 'score_report.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path}')


def get_args():
    """Only what changes between runs."""
    parser = argparse.ArgumentParser(description='Predict lead masks + layout for ECG pages')
    parser.add_argument('--model', '-m', required=True, help='checkpoint (.pth) from train.py')
    parser.add_argument('--input', '-i', nargs='+', help='single-page mode: image files')
    parser.add_argument('--score', type=int, nargs='?', const=0, default=None,
                        help='score mode: the first N test images (no value = all)')
    parser.add_argument('--save-dir', default='Predictions/test', help='score mode: where masks + score_report.csv go')
    return parser.parse_args()


if __name__ == '__main__':

    #take all the args
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. build the network exactly as train.py did, then load the trained weights
    net = UNet(n_channels=3, n_classes=n_classes, n_layouts=n_layouts)
    state = torch.load(args.model, map_location=device)
    state.pop('mask_values', None)             # train.py stores it in the checkpoint, but it isn't a weight
    net.load_state_dict(state)
    net.to(device)
    net.eval()                                 # BatchNorm uses its learned statistics
    logging.info(f'Loaded {args.model} on {device}')

    # 2a. score mode
    if args.score is not None:
        tag = Path(args.model).stem.replace('checkpoint_', '')         # checkpoint_epoch40 -> epoch40
        score_test_set(net, device, limit=args.score or None, out_dir=Path(args.save_dir), tag=tag)

    # 2b. single-page mode
    elif args.input:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for path in args.input:
            image = Image.open(path).convert('RGB')                    # phone photos may be RGBA / greyscale
            mask, (layout, confidence) = predict_page(net, image, device)
            out_path = raw_dir / f'{Path(path).stem}.png'
            save_mask(mask, out_path)
            logging.info(f'{path}: layout {layout} (confidence {confidence:.3f}), mask saved to {out_path}')

    else:
        raise SystemExit('give either --input/-i or --score')
