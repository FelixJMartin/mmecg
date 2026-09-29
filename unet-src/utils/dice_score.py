import torch
import torch.nn.functional as F
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def dice_loss_fg(logits: Tensor, target: Tensor, n_classes: int, eps: float = 1e-5):
    # BUGFIX (2026-09-23): multiclass_dice_coeff/dice_loss (above) flatten batch AND
    # class together into one pooled Dice number. With 13 classes where background is
    # ~98% of pixels (confirmed: 7.42M/7.55M px in a sample mask) and the 12 lead
    # classes are 1-2px traces, pooling lets a near-perfect background channel drown out
    # total failure on every real class -- this collapsed segmentation entirely in job
    # 17603025. Per nnU-Net / Krones et al. (ECG-Digitiser, PhysioNet 2024 winner): Dice
    # per class (batch dice, dims=(0,2,3)), background excluded, THEN averaged.
    p = logits.softmax(1).float()
    t = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2 * inter + eps) / (denom + eps).clamp_min(1e-8)
    return 1 - dice[1:].mean()             # [1:] drops background
