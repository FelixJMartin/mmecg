import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    '''Returns (dice, layout_accuracy). layout_accuracy is None when the net has
    no layout head, so segmentation-only runs behave exactly as before.'''
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    cls_correct = cls_total = 0
    # rows = true layout, cols = predicted. Accuracy alone can hide a class the model
    # never predicts at all (74% "accuracy" is reachable while missing 4x3+1 entirely).
    confusion = torch.zeros(net.n_layouts, net.n_layouts, dtype=torch.long) if getattr(net, 'n_layouts', None) else None

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask (+ layout, when the head exists)
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

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    accuracy = cls_correct / cls_total if cls_total else None
    return dice_score / max(num_val_batches, 1), accuracy, (confusion if cls_total else None)


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
