import argparse
import csv
import logging
import os
import sys
from os import listdir
from os.path import join, splitext

# Imports
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset, IDX_TO_TEMPLATE
from unet import UNet

# Predict image using model, net= model, full img, device=cpu, threshold
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval() #eval mode


    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)) # reloads images to (C, H, W).
    img = img.unsqueeze(0)                                                                       # unsqueeze(0) fakes a batch of size 1.
    img = img.to(device=device, dtype=torch.float32)

    layout = None
    with torch.no_grad():                       #disable gradient tracking 
        output = net(img)                       #producing raw logits of shape (1, n_classes, h_scaled, w_scaled)
        if getattr(net, 'n_layouts', None):
            # net returns (segmentation, layout logits) once the layout head is enabled
            output, cls_logits = output
            probs = F.softmax(cls_logits.cpu()[0], dim=0)
            idx = int(probs.argmax())
            # softmax, not just argmax: the confidence is worth having downstream, where a
            # confidently-wrong layout would silently mis-slice every lead row.
            layout = (IDX_TO_TEMPLATE[idx], float(probs[idx]))
        output = output.cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear') #Resizes the output mask back up to the original image size.
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

        mask = mask[0]          # drop batch dim: (1, C, H, W) -> (C, H, W)
        mask = mask.long()      # cast bool/index values to integers
        mask = mask.squeeze()   # drop any leftover size-1 dims: (1, H, W) -> (H, W)
        mask = mask.numpy()     # convert tensor -> plain numpy array
        return mask, layout


def score_dataset(net, device, img_dir, mask_dir, index_csv, limit=None,
                  scale=0.5, out_dir=None, mask_suffix='_mask', score_csv=None, tag='pred'):
    """Predict every test image, score it against its ground-truth mask, print a table.

    Reports the numbers that matter together: segmentation quality (dice, and
    precision/recall separately -- dice alone hides whether the model over- or
    under-predicts) and layout accuracy with its softmax confidence.
    """
    truth = {}
    if index_csv and os.path.exists(index_csv):
        with open(index_csv) as f:
            truth = {row['name']: row['template'] for row in csv.DictReader(f)}

    names = sorted((splitext(f)[0] for f in listdir(img_dir) if not f.startswith('.')),
                   key=lambda n: (len(n), n))          # example_9 before example_10
    if limit:
        names = names[:limit]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"{'record':12s} {'true':7s} {'pred':7s} {'conf':>6s} "
          f"{'dice':>7s} {'prec':>7s} {'rec':>7s}")
    rows, records = [], []
    for name in names:
        img = Image.open(join(img_dir, name + '.png'))
        pred, layout = predict_img(net=net, full_img=img, scale_factor=scale, device=device)
        pred = pred.astype(bool)

        gt_path = join(mask_dir, name + mask_suffix + '.png')
        gt = (np.asarray(Image.open(gt_path).convert('RGB')) != 0).any(-1)
        inter = (pred & gt).sum()
        dice = 2 * inter / max(pred.sum() + gt.sum(), 1)
        prec = inter / max(pred.sum(), 1)
        rec = inter / max(gt.sum(), 1)

        if out_dir:
            out = np.zeros((*pred.shape, 3), np.uint8)
            out[pred] = (57, 255, 20)                  # neon on black, same as make_dataset.py
            # tag = the checkpoint that produced this, so masks from different
            # checkpoints/ablations never collide or become unattributable
            Image.fromarray(out).save(join(out_dir, f'{name}_{tag}.png'))

        t = truth.get(name, '?')
        pl, conf = layout if layout else ('n/a', float('nan'))
        ok = 'OK ' if pl == t else 'MISS'
        print(f"{name:12s} {t:7s} {pl:7s} {conf:6.3f} {dice:7.4f} {prec:7.4f} {rec:7.4f}  {ok}")
        rows.append((dice, prec, rec, pl == t))
        records.append({'record': name, 'true_layout': t, 'pred_layout': pl,
                        'confidence': round(conf, 4), 'dice': round(float(dice), 4),
                        'precision': round(float(prec), 4), 'recall': round(float(rec), 4),
                        'layout_correct': int(pl == t)})

    if rows:
        m = [float(np.mean([r[i] for r in rows])) for i in range(4)]
        print()
        print(f"MEAN over {len(rows)}: dice={m[0]:.4f}  precision={m[1]:.4f}  "
              f"recall={m[2]:.4f}  layout_acc={m[3]:.3f}")

        # Default the report next to the saved masks, so predictions and their
        # labels/scores end up in one folder rather than drifting apart.
        path = score_csv or (join(out_dir, 'score_report.csv') if out_dir else 'score_report.csv')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
            w.writerow({'record': 'MEAN', 'true_layout': '', 'pred_layout': '',
                        'confidence': '', 'dice': round(m[0], 4),
                        'precision': round(m[1], 4), 'recall': round(m[2], 4),
                        'layout_correct': round(m[3], 4)})
        print(f'wrote {path}')
    return rows


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--score', type=int, nargs='?', const=0, default=None,
                        help='Score mode: evaluate N test images against their masks and print '
                             'a dice/precision/recall/layout table (no value = all images)')
    parser.add_argument('--img-dir', default='data/test_imgs', help='Score mode: image directory')
    parser.add_argument('--mask-dir', default='data/test_masks', help='Score mode: mask directory')
    parser.add_argument('--index-csv', default='data/test_index.csv', help='Score mode: layout labels')
    parser.add_argument('--save-dir', default=None, help='Score mode: also write predicted masks here')
    parser.add_argument('--score-csv', default=None,
                        help='Score mode: path for the per-record CSV report '
                             '(default: <save-dir>/score_report.csv)')
    parser.add_argument('--layouts', type=int, default=None,
                        help='Set to 3 for a checkpoint trained with the layout head. '
                             'Must match training, or the state_dict will not load.')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        name = os.path.splitext(os.path.basename(fn))[0]
        return f'Predictions/raw/{name}.png'

    if args.output:
        return args.output
    os.makedirs('Predictions/raw', exist_ok=True)
    return list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray) -> Image.Image:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask.astype(bool)] = (57, 255, 20)  # neon green, matches ground-truth mask color
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args) if args.input else []   # not used in --score mode

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear,
               n_layouts=args.layouts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop('mask_values', None)  # not a real weight, strip before loading
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    if args.score is not None:
        score_dataset(net, device, args.img_dir, args.mask_dir, args.index_csv,
                      limit=args.score or None, scale=args.scale, out_dir=args.save_dir,
                      score_csv=args.score_csv,
                      tag=splitext(os.path.basename(args.model))[0].replace('checkpoint_', ''))
        sys.exit(0)

    assert args.input, 'either --input/-i or --score is required'

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask, layout = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        if layout is not None:
            logging.info(f'Predicted layout: {layout[0]}  (confidence {layout[1]:.3f})')

        out_filename = out_files[i]
        result = mask_to_image(mask)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')
