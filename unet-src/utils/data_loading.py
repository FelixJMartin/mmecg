import logging
import random
import numpy as np
import torch
from collections import defaultdict
from PIL import Image
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset, Sampler, Subset
from tqdm import tqdm


# make_mask.py only ever writes .png files, so no .npy/.pt handling needed here.
def load_image(filename):
    return Image.open(filename)

# Returns the distinct pixel values (or RGB tuples) found in one mask file.
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix  # lets mask filenames differ from image filenames, e.g. img001_mask.png

        # Sample IDs = image filenames without extension. Each id is later paired with
        # id + mask_suffix + <ext> to find the matching mask.
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples, scanning masks for unique values')
        # One-time upfront pass over every mask (parallelized) to find every distinct
        # label value across the whole dataset before training starts.
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        # Merge the per-file unique values into one sorted list: this becomes the
        # value -> class-index mapping used by preprocess() below.
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # NEAREST for masks so resizing never invents new label values between classes;
        # BICUBIC for images since smooth interpolation is fine (and looks better) there.
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            # Remap each raw pixel value/color to its class index (0, 1, 2, ...)
            # using the position of that value in mask_values.
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            # Move channel axis to the front (H, W, C) -> (C, H, W) as expected by PyTorch;
            # grayscale images get a leading channel dim of 1 instead.
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # Normalize 0-255 images to 0-1; leave already-normalized/float images alone.
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        # .copy() avoids "negative stride" errors torch.as_tensor can raise on some
        # numpy views (e.g. after transpose); .contiguous() ensures a clean memory layout.
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class GroupedBatchSampler(Sampler):
    '''Only ever batches together examples whose on-disk image is the same pixel
    size. The mixed-layout dataset (1x12/2x6/4x3+1 templates) has a different
    height per template, and the default collate can't torch.stack tensors of
    different shapes -- so a random batch mixing templates would crash.'''

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        # `dataset` may be a Subset (e.g. from random_split) wrapping a BasicDataset;
        # unwrap it so we can read .ids/.images_dir, but keep batches indexed the
        # way DataLoader expects them: local indices into whatever was passed in.
        if isinstance(dataset, Subset):
            base, subset_indices = dataset.dataset, dataset.indices
        else:
            base, subset_indices = dataset, range(len(dataset))

        groups = defaultdict(list)
        for local_idx, global_idx in enumerate(subset_indices):
            img_file = next(base.images_dir.glob(base.ids[global_idx] + '.*'))
            with Image.open(img_file) as im:
                groups[im.size].append(local_idx)
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        batches = []
        for indices in self.groups:
            indices = indices.copy()
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return sum(len(g) // self.batch_size for g in self.groups)
        return sum(-(-len(g) // self.batch_size) for g in self.groups)
