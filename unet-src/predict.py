import argparse
import logging
import os

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


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
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
    out_files = get_output_filenames(args)

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
