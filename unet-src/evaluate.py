import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_coeff

# Paper geometry, so a pixel error can be reported in mV instead of pixels. pmecg renders
# at 300 dpi but matplotlib's default margins leave the axes 77% of the figure, so the page
# is 300/25.4*0.77 px/mm. At 10 mm/mV that is 0.011 mV per pixel at full scale; training
# downscales by img_scale, which makes each pixel worth proportionally more.
PX_PER_MM_FULL = 300 / 25.4 * 0.77
MM_PER_MV = 10.0

LEAD_NAMES = ['background', 'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, img_scale: float = 1.0):
    '''Validation metrics as a dict.

    Dice says how much of the trace was found. The rest say whether the output WILL
    DIGITISE, which is what we actually care about:

      trace_recall     of the true trace pixels, how many were called trace at all
      trace_precision  of the predicted trace pixels, how many really are trace
                       (grid lines or lead labels segmented as signal land here)
      lead_error       of the pixels both agree are trace, how many got the WRONG lead id
                       -- the failure that swaps two leads in the digitised output
      column_coverage  per (lead, column) the truth draws, how often the prediction has at
                       least one pixel there. A missing column is a NaN in the signal.
      centre_rmse_mv   the digitisation error itself: per column, vertical centre of the
                       predicted lead minus the truth's, converted to mV. Same quantity
                       the digitiser reads off, so it is the closest thing to
                       RMSE-against-the-h5 that a mask alone can give.
    '''
    net.eval()
    num_val_batches = len(dataloader)
    n_classes = net.n_classes
    dice_score = 0
    class_dice = torch.zeros(n_classes, dtype=torch.float64)
    class_batches = torch.zeros(n_classes, dtype=torch.float64)

    true_trace = pred_trace = both_trace = wrong_lead = 0     # pixel tallies
    col_true = col_hit = n_cols = 0                           # per-column tallies
    sq_err_px = abs_err_px = 0.0

    cls_correct = cls_total = 0
    # rows = true layout, cols = predicted. Accuracy alone can hide a class the model
    # never predicts at all (74% "accuracy" is reachable while missing 4x3+1 entirely).
    confusion = torch.zeros(net.n_layouts, net.n_layouts, dtype=torch.long) if getattr(net, 'n_layouts', None) else None

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            out = net(image)
            if getattr(net, 'n_layouts', None):
                mask_pred, cls_pred = out
                if 'template' in batch:
                    true_t = batch['template'].to(device=device)
                    pred_t = cls_pred.argmax(dim=1)
                    cls_correct += (pred_t == true_t).sum().item()
                    cls_total += true_t.numel()
                    for t_i, p_i in zip(true_t.cpu().tolist(), pred_t.cpu().tolist()):
                        confusion[t_i][p_i] += 1
            else:
                mask_pred = out

            # PadView (full-page eval, see utils/data_loading.py) pads the bottom/right
            # edges to a multiple of val_pad_to so the forward pass runs in one shot --
            # crop both the prediction and the truth back to the real page before scoring,
            # or the padding (arbitrary content, not real page) pollutes every metric below.
            if 'orig_size' in batch:
                h, w = batch['orig_size'][0].tolist()
                assert (batch['orig_size'] == batch['orig_size'][0]).all(), \
                    'PadView batch has mixed orig_size -- evaluate() only crops by the first'
                mask_pred = mask_pred[:, :, :h, :w]
                mask_true = mask_true[:, :h, :w]

            if n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                hard = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(hard, mask_true, reduce_batch_first=False)
                continue

            assert mask_true.min() >= 0 and mask_true.max() < n_classes, 'True mask indices should be in [0, n_classes['
            hard = mask_pred.argmax(dim=1)                                   # (B, H, W) class ids
            true_1h = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            pred_1h = F.one_hot(hard, n_classes).permute(0, 3, 1, 2).float()

            # per-class dice, so one lead being invisible cannot hide behind the mean.
            # val['dice'] is this batch's mean over classes actually present, NOT the old
            # multiclass_dice_coeff(pred_1h[:, 1:], true_1h[:, 1:]) -- that pooled every
            # foreground class's pixels into one Dice number, which has the same
            # background-style dilution bug among the 12 lead classes (see
            # dice_loss_fg in utils/dice_score.py). Kept consistent with what the
            # training loss now optimizes.
            batch_class_dice = []
            for c in range(1, n_classes):
                if true_1h[:, c].sum() > 0:
                    d = dice_coeff(pred_1h[:, c], true_1h[:, c], reduce_batch_first=False).item()
                    class_dice[c] += d
                    class_batches[c] += 1
                    batch_class_dice.append(d)
            if batch_class_dice:
                dice_score += sum(batch_class_dice) / len(batch_class_dice)

            # ---- pixel level: found, spurious, or right ink with the wrong lead id
            t_ink, p_ink = mask_true > 0, hard > 0
            true_trace += t_ink.sum().item()
            pred_trace += p_ink.sum().item()
            agree = t_ink & p_ink
            both_trace += agree.sum().item()
            wrong_lead += (mask_true[agree] != hard[agree]).sum().item()

            # ---- column level: coverage and vertical centre error, the digitiser's view
            rows = torch.arange(mask_true.shape[1], device=device, dtype=torch.float32).view(1, -1, 1)
            for c in range(1, n_classes):
                t_c, p_c = (mask_true == c), (hard == c)
                cnt_t, cnt_p = t_c.sum(1), p_c.sum(1)                        # (B, W) ink pixels per column
                has_t, has_p = cnt_t > 0, cnt_p > 0
                col_true += has_t.sum().item()
                col_hit += (has_t & has_p).sum().item()

                both = has_t & has_p
                if not both.any():
                    continue
                centre_t = (t_c * rows).sum(1) / cnt_t.clamp(min=1)          # mean ink row per column
                centre_p = (p_c * rows).sum(1) / cnt_p.clamp(min=1)
                err = (centre_t - centre_p)[both].abs()
                sq_err_px += (err ** 2).sum().item()
                abs_err_px += err.sum().item()
                n_cols += err.numel()

    net.train()
    mv_per_px = 1.0 / (PX_PER_MM_FULL * img_scale) / MM_PER_MV               # px -> mm -> mV
    return {
        'dice': float(dice_score / max(num_val_batches, 1)),
        'dice_per_class': [float(class_dice[c] / class_batches[c]) if class_batches[c] else float('nan')
                           for c in range(n_classes)],
        'trace_recall': both_trace / true_trace if true_trace else float('nan'),
        'trace_precision': both_trace / pred_trace if pred_trace else float('nan'),
        'lead_error': wrong_lead / both_trace if both_trace else float('nan'),
        'column_coverage': col_hit / col_true if col_true else float('nan'),
        'centre_rmse_mv': (sq_err_px / n_cols) ** 0.5 * mv_per_px if n_cols else float('nan'),
        'centre_mae_mv': abs_err_px / n_cols * mv_per_px if n_cols else float('nan'),
        'centre_mse_mv2': sq_err_px / n_cols * mv_per_px ** 2 if n_cols else float('nan'),
        'layout_acc': cls_correct / cls_total if cls_total else None,
        'layout_confusion': confusion if cls_total else None,
    }


def format_confusion(confusion):
    '''Render the layout confusion matrix as a readable text table.'''
    from utils.data_loading import IDX_TO_TEMPLATE
    names = [IDX_TO_TEMPLATE[i] for i in range(confusion.shape[0])]
    w = max(len(n) for n in names) + 1
    lines = [' ' * (w + 7) + 'predicted',
             ' ' * (w + 6) + ' '.join(f'{n:>7s}' for n in names)]
    for i, n in enumerate(names):
        tag = 'true ' if i == 0 else ' ' * 5
        lines.append(f'{tag}{n:<{w}s} ' + ' '.join(f'{int(v):7d}' for v in confusion[i]))
    return chr(10).join(lines)


def format_per_class(dice_per_class):
    '''One entry per lead, so a lead the model never finds cannot hide behind the mean.'''
    parts = [f'{LEAD_NAMES[c] if c < len(LEAD_NAMES) else c}={d:.3f}'
             for c, d in enumerate(dice_per_class) if c > 0]
    return 'per-lead dice: ' + '  '.join(parts)
