"""Serves the ECG training pairs to train.py (and helpers to predict.py / evaluate.py).


Whole pages, as they are: no resizing, no cropping, no padding. 3328 x 2496 are both multiples of 16,
which the U-Net needs (it halves the image 4 times), so every page goes straight in.

One sample = a dict of tensors:
    'image'      float (3, H, W), values 0..1
    'mask'       long  (H, W), class ids 0..12
    'template'   long  scalar, layout index (only if index_csv was given)
"""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

N_CLASSES = 13                      # background + 12 leads

# layout template -> index for the layout head. FROZEN: reordering silently reinterprets every
# checkpoint trained before. Only append. Must match layout_keys.csv in the project root.
TEMPLATE_TO_IDX = {'1x12': 0, '2x6': 1, '4x3+1': 2, '2x6+1': 3}
IDX_TO_TEMPLATE = {v: k for k, v in TEMPLATE_TO_IDX.items()}    # {0: '1x12', 1: '2x6', ...}


# ---------------------------------------------------------------- small helpers
def image_to_array(pil_img):
    """PIL image -> float32 array (3, H, W), values 0..1: the format the network takes."""
    rgb = np.asarray(pil_img.convert('RGB'), dtype=np.float32)    # (H, W, 3), 0..255
    return rgb.transpose(2, 0, 1) / 255                            # channels first, like PyTorch wants


def mask_to_array(pil_mask):
    """Palette PNG -> int64 array (H, W) of class ids. The pixel values already are the classes."""
    return np.asarray(pil_mask, dtype=np.int64)


# ---------------------------------------------------------------- the dataset
class BasicDataset(Dataset):
    def __init__(self, images_dir, mask_dir, index_csv=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)

        # 1. one sample per image file; its name is the file name without extension
        self.ids = sorted(p.stem for p in self.images_dir.iterdir() if p.is_file())
        if not self.ids:
            raise RuntimeError(f'No images found in {images_dir}')

        # 2. mask value = class id, so the mapping is fixed (train.py stores it in every checkpoint)
        self.mask_values = list(range(N_CLASSES))

        # 3. optional: each sample's layout, for the layout head (train.py passes index_csv only with --layouts)
        self.templates = None
        if index_csv is not None:
            with open(index_csv) as f:
                layout_of = {row['name']: row['template'] for row in csv.DictReader(f)}
            self.templates = {name: TEMPLATE_TO_IDX[layout_of[name]] for name in self.ids}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        One whole page: image + label, exactly as on disk.
        """

        # 1. which sample: its name, e.g. "ecg02898_photo_clean"
        name = self.ids[idx]

        # 2. where its two files are: the photo is a .jpg, the label a .png (from dataset_build/augment.py)
        img_path = self.images_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # 4. check they belong together, and fit the U-Net (it halves the image 4 times -> multiple of 16)
        assert img.size == mask.size, f"{name}: image {img.size} and label {mask.size} differ in size"
        assert img.width % 16 == 0 and img.height % 16 == 0, \
            f"{name}: {img.size} is not a multiple of 16 -- rebuild the data with dataset_build/augment.py"

        # 5. image -> numbers the network takes: float (3, H, W), values 0..1
        image = torch.from_numpy(image_to_array(img))

        # 6. label -> one class id per pixel: long (H, W), va
        label = torch.from_numpy(mask_to_array(mask))

        # 7. pack into the dict the training loop expects (name: so validation can log per image)
        sample = {"image": image, "mask": label, "name": name}

        # 8. with the layout head: also the page's layout, as an index (CrossEntropyLoss wants an index)
        if self.templates is not None:
            sample["template"] = torch.tensor(self.templates[name])


        return sample


# ---------------------------------------------------------------- class balance, for the loss weights
def class_pixel_counts(dataset, indices, n_classes=N_CLASSES):
    """How many pixels of each class the given samples contain, read from the label PNGs only.

    train.py turns these into class weights (background is ~98 % of all pixels). Only the labels
    are opened: loading the full images too ran a cluster job out of memory before training began.
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for i in indices:
        mask = mask_to_array(Image.open(dataset.mask_dir / f"{dataset.ids[i]}.png"))
        counts += np.bincount(mask.ravel(), minlength=n_classes)
    return counts