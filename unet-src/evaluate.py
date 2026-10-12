"""
Validation metrics, used by train.py once per epoch on the validation pages.

Every metric is computed per image first (image_metrics); the overall numbers and the per-persona /
per-layout groups are then the same pooling of those images (pool_metrics). One way, used everywhere.

    dice              how well each lead's predicted shape overlaps the truth (background excluded)
    trace_recall      of the true trace pixels, how many were called trace at all
    trace_precision   of the predicted trace pixels, how many really are trace
    lead_error        of the pixels both agree are trace, how many got the WRONG lead
                      -- the failure that swaps two leads in the digitised output
    column_coverage   per lead and image column the truth draws, how often the prediction has a pixel
                      there too. A missing column is a gap in the digitised signal.
    centre_rmse_mv    per column, the predicted trace centre minus the true one, in mV: the closest a
    centre_mae_mv     mask alone gets to the digitisation error (rmse punishes big misses more)
"""
#packages
import math

import torch
from tqdm import tqdm

#own code
from utils.data_loading import IDX_TO_TEMPLATE

# ---------------------------------------------------------------- fixed settings
px_per_mm = 300 / 25.4 * 0.77
mm_per_mv = 10.0
mv_per_px = 1 / px_per_mm / mm_per_mv

lead_names = ['background', 'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# the raw counts every image stores, so any group of images can be pooled exactly
count_keys = ['true_px', 'pred_px', 'both_px', 'wrong_lead_px',
              'col_true', 'col_hit', 'n_cols', 'sq_err_px', 'abs_err_px']
nan = float('nan')


# ---------------------------------------------------------------- small helpers
def ratio(a, b):
    """a / b, or NaN if there is nothing to divide by."""
    return a / b if b else nan


def mean_ignoring_nan(values):
    """Mean of the values that aren't NaN; NaN if none are left."""
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else nan


def ratios(c):
    """The rate metrics from the raw counts (one image's, or a group's summed counts)."""
    return {
        'trace_recall': ratio(c['both_px'], c['true_px']),
        'trace_precision': ratio(c['both_px'], c['pred_px']),
        'lead_error': ratio(c['wrong_lead_px'], c['both_px']),
        'column_coverage': ratio(c['col_hit'], c['col_true']),
        'centre_rmse_mv': ratio(c['sq_err_px'], c['n_cols']) ** 0.5 * mv_per_px,
        'centre_mae_mv': ratio(c['abs_err_px'], c['n_cols']) * mv_per_px,
    }


# ---------------------------------------------------------------- one image
def image_metrics(true, pred, n_classes):
    """All metrics for one image. true, pred: (H, W) class ids on the same device.

    Returns the raw counts too (pixels, columns, squared errors), so any group of images can be
    pooled exactly later on, instead of averaging ratios.
    """
    row = {}

    # 1. pixel level: trace found, trace invented, right trace with the wrong lead
    true_ink = true > 0
    pred_ink = pred > 0
    agree = true_ink & pred_ink
    row['true_px'] = int(true_ink.sum())
    row['pred_px'] = int(pred_ink.sum())
    row['both_px'] = int(agree.sum())
    row['wrong_lead_px'] = int((agree & (true != pred)).sum())

    # 2. per lead: dice, and per image column the trace centre (the digitiser's view)
    row_index = torch.arange(true.shape[0], device=true.device, dtype=torch.float32).view(-1, 1)   # (H, 1)
    for key in count_keys[4:]:                                          # the column counts start at 0
        row[key] = 0
    lead_dice = []

    for c in range(1, n_classes):
        lead = lead_names[c]
        t = true == c
        p = pred == c
        n_t = int(t.sum())
        n_p = int(p.sum())

        # 2a. dice: NaN when this lead isn't on the page at all, so it can't count as a 0
        dice = 2 * int((t & p).sum()) / (n_t + n_p) if n_t else nan
        lead_dice.append(dice)

        # 2b. columns: where does the truth draw this lead, and is the prediction there too?
        per_col_t = t.sum(0)                                            # (W,) ink pixels per column
        per_col_p = p.sum(0)
        has_t = per_col_t > 0
        both = has_t & (per_col_p > 0)

        # 2c. trace centre = mean ink row per column; error only where both have ink, in px
        centre_t = (t * row_index).sum(0) / per_col_t.clamp(min=1)
        centre_p = (p * row_index).sum(0) / per_col_p.clamp(min=1)
        err = (centre_t - centre_p)[both].abs()

        # 2d. this lead's own numbers (they end up in training_log_per_image.csv)
        row[f'dice_{lead}'] = dice
        row[f'coverage_{lead}'] = ratio(int(both.sum()), int(has_t.sum()))
        row[f'rmse_mv_{lead}'] = float((err ** 2).mean()) ** 0.5 * mv_per_px if err.numel() else nan
        row[f'n_cols_{lead}'] = err.numel()

        # 2e. add to this image's column counts
        row['col_true'] += int(has_t.sum())
        row['col_hit'] += int(both.sum())
        row['n_cols'] += err.numel()
        row['sq_err_px'] += float((err ** 2).sum())
        row['abs_err_px'] += float(err.sum())

    # 3. this image's headline numbers
    row['dice'] = mean_ignoring_nan(lead_dice)
    row.update(ratios(row))
    return row


# ---------------------------------------------------------------- many images
def pool_metrics(rows):
    """Overall metrics for a group of images: counts are summed, then turned into rates.

    Dice is the exception (there is no count to sum): the mean of the images' dice, and per lead the
    mean over the images that contain that lead.
    """
    # 1. sum every raw count over the images
    counts = {}
    for key in count_keys:
        counts[key] = sum(r[key] for r in rows)

    # 2. dice: mean over images, overall and per lead (background is not scored)
    dice_per_class = [nan]
    for lead in lead_names[1:]:
        dice_per_class.append(mean_ignoring_nan([r[f'dice_{lead}'] for r in rows]))

    return {'n_images': len(rows),
            'dice': mean_ignoring_nan([r['dice'] for r in rows]),
            'dice_per_class': dice_per_class,
            **ratios(counts)}


# ---------------------------------------------------------------- the validation pass
@torch.inference_mode()                         # no gradients: much less memory on whole pages
def evaluate(net, dataloader, device, amp):
    """Run the model over the validation pages.

    Returns the pooled metrics, the layout accuracy + confusion matrix, and 'per_image': one metrics row per image.
    """
    net.eval()                                  # BatchNorm uses its learned statistics
    rows = []
    confusion = torch.zeros(net.n_layouts, net.n_layouts, dtype=torch.long)    # rows = true, columns = predicted

    with torch.autocast(device.type, enabled=amp):
        for batch in tqdm(dataloader, desc='Validation', unit='batch', leave=False):
            # 1. predict
            images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true = batch['mask'].to(device=device, dtype=torch.long)
            logits, layout_logits = net(images)
            pred = logits.argmax(dim=1)                                 # (B, H, W) class ids
            pred_layout = layout_logits.argmax(dim=1).tolist()          # (B,) layout ids

            # 2. layout guess: count right/wrong per (true, predicted) pair
            for t, p in zip(batch['template'].tolist(), pred_layout):
                confusion[t][p] += 1

            # 3. metrics per image
            for b in range(len(pred)):
                row = {'name': batch['name'][b], 'pred_layout': IDX_TO_TEMPLATE[pred_layout[b]]}
                row.update(image_metrics(true[b], pred[b], net.n_classes))
                rows.append(row)

    net.train()                                 # back to training mode for the next epoch
    return {**pool_metrics(rows),
            'layout_acc': int(confusion.trace()) / int(confusion.sum()),
            'layout_confusion': confusion,
            'per_image': rows}


# ---------------------------------------------------------------- printing
def format_confusion(confusion):
    """The layout confusion matrix as a readable text table (rows = true, columns = predicted)."""
    names = [IDX_TO_TEMPLATE[i] for i in range(confusion.shape[0])]
    w = max(len(n) for n in names) + 1
    lines = [' ' * (w + 7) + 'predicted',
             ' ' * (w + 6) + ' '.join(f'{n:>7s}' for n in names)]
    for i, n in enumerate(names):
        tag = 'true ' if i == 0 else ' ' * 5
        lines.append(f'{tag}{n:<{w}s} ' + ' '.join(f'{int(v):7d}' for v in confusion[i]))
    return '\n'.join(lines)


def format_per_class(dice_per_class):
    """One entry per lead, so a lead the model never finds cannot hide behind the mean."""
    return 'per-lead dice: ' + '  '.join(f'{lead_names[c]}={d:.3f}' for c, d in enumerate(dice_per_class) if c > 0)
