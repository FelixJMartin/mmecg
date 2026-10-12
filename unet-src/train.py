"""
Train the U-Net on whole ECG pages: 13 classes (background + 12 leads) + optional layout head.

python train.py -l 3e-4 --epochs 40 -b 4 --run-dir runs/name

Every epoch: train once on every training page, validate once (evaluate.py), append to the logs:
    training_log.csv            losses + validation metrics
    training_log_per_lead.csv   validation dice per lead
    training_log_per_image.csv  every validation image's metrics, tagged with page / layout / persona
    training_log_per_group.csv  the same pooled per persona and per layout

"""
#syntaxes for usage
import argparse
import csv
import logging
from pathlib import Path

#torch packages
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

#metrics
from evaluate import evaluate, pool_metrics, format_confusion, format_per_class, lead_names
from unet import UNet
from utils.data_loading import BasicDataset, class_pixel_counts
from utils.dice_score import dice_loss_fg

#paths
data = Path('./data')
dir_img, dir_mask, index_csv = data / 'imgs', data / 'labels', data / 'index.csv'

#colomns for later
group_cols = ['n_images', 'dice', 'trace_recall', 'trace_precision', 'lead_error', 'column_coverage', 'centre_rmse_mv']
log_cols = ['dice', 'trace_recall', 'trace_precision', 'lead_error', 'column_coverage', 'centre_rmse_mv', 'centre_mae_mv', 'layout_acc']


# ---------------------------------------------------------------- fixed settings (change here if ever needed)
n_classes = 13            # background + 12 leads
n_layouts = 4             # 1x12, 2x6, 4x3+1, 2x6+1 -- None = no layout head
amp = True                # mixed precision: about half the GPU memory
workers = 8               # processes loading the next pages while the GPU works
grad_max = 1.0
val_percent = 0.10


# ---------------------------------------------------------------- step helpers
def read_index(index_csv):
    """name -> {page, template, persona} from index.csv, to tag every validation image."""
    with open(index_csv) as f:
        return {r['name']: {k: r.get(k, '') for k in ('page', 'template', 'persona')} for r in csv.DictReader(f)}


def split_by_page(dataset, val_percent):
    """
    Train / validation split by ECG page, not by image.
    Every persona of a page lands on the same side, so validation pages are never seen in training.
    """
    # 1. find every unique page: "ecg02898_scan_clean" -> "ecg02898"
    pages = set()
    for name in dataset.ids:
        page = name.split('_')[0]
        pages.add(page)

    # 2. put them in a fixed order, then shuffle with a fixed seed -> same split on every run
    pages = sorted(pages)
    rng = random.Random(0)
    rng.shuffle(pages)

    # 3. the first val_percent of the shuffled pages become validation pages
    n_val_pages = int(len(pages) * val_percent)
    val_pages = set(pages[:n_val_pages])

    # 4. send every image to the side its page went to (store positions, that's what Subset needs)
    train_idx = []
    val_idx = []
    for i, name in enumerate(dataset.ids):
        page = name.split('_')[0]
        if page in val_pages:
            val_idx.append(i)
        else:
            train_idx.append(i)

    # 5. print what happened
    n_train_pages = len(pages) - n_val_pages
    logging.info(f'Split by page: {n_train_pages} train pages ({len(train_idx)} images), '
                 f'{n_val_pages} val pages ({len(val_idx)} images)')

    return train_idx, val_idx


def class_weights(dataset, train_idx, n_classes, device):
    """
    Loss weight per class, so lead pixels count more than background pixels.
    """
    # 1. how many pixels of each class the training labels contain
    counts = torch.as_tensor(class_pixel_counts(dataset, train_idx, n_classes), dtype=torch.float64)

    # 2. rarer class -> bigger weight: 1 / sqrt(count)  (clamp: a class with no pixels doesn't divide by 0)
    #    sqrt softens it: background isn't ignored, leads aren't weighted thousands of times higher
    weights = 1 / counts.clamp(min=1).sqrt()

    # 3. scale so the weights average 1, which keeps the loss in its usual range
    weights = weights / weights.mean()

    logging.info('Class weights: ' + ', '.join(f'{lead_names[i]}={w:.3f}' for i, w in enumerate(weights.tolist())))
    return weights.float().to(device)


def make_optimizer(model, lr, epochs):
    """
    AdamW + cosine learning-rate decay.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-5)

    # cosine: the lr starts at `lr` and smoothly drops toward 0 by the last epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    return optimizer, scheduler


def append_row(path, row, header):
    """Append one row to a CSV; write the header first if the file is new."""
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        if new:
            csv.writer(f).writerow(header)
        csv.writer(f).writerow(row)

def write_epoch_logs(out_dir, epoch, losses, val, image_info):
    """Append this epoch to the four training_log*.csv files."""
    # 1. overall: losses + validation metrics
    row = [epoch, *losses.values()] + [val[c] for c in log_cols]
    header = ['epoch', *losses.keys()] + log_cols
    append_row(out_dir / 'training_log.csv', row, header)

    # 2. dice per lead
    append_row(out_dir / 'training_log_per_lead.csv', [epoch, *val['dice_per_class'][1:]],
               header=['epoch'] + [f'dice_{lead}' for lead in lead_names[1:]])

    # 3. every validation image, tagged with its page / layout / persona
    rows = [{'epoch': epoch, **image_info.get(r['name'], {}), **r} for r in val['per_image']]
    fields = ['epoch', 'name', 'page', 'template', 'persona', 'pred_layout']
    fields += [k for k in rows[0] if k not in fields]
    path = out_dir / 'training_log_per_image.csv'
    new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)

 # 4. the same images pooled per persona and per layout
    for key in ('persona', 'template'):

        # 4a. sort the validation images into buckets, e.g. {'scan_clean': [...], 'photo_dark': [...]}
        groups = {}
        for r in rows:
            group = r.get(key) or '?'          # this image's persona (or layout); '?' if the tag is missing
            if group not in groups:
                groups[group] = []             # first image of this group: start an empty list
            groups[group].append(r)

        # 4b. one set of metrics per bucket -> one CSV row + one printed line
        for group in sorted(groups):
            members = groups[group]
            m = pool_metrics(members)          # combine all images of this bucket into one set of metrics

            row = [epoch, key, group] + [m[c] for c in group_cols]
            header = ['epoch', 'group_by', 'group'] + group_cols
            append_row(out_dir / 'training_log_per_group.csv', row, header)

            logging.info(f'  {key:<8} {group:<15} n={m["n_images"]:<4d} dice={m["dice"]:.4f} '
                         f'lead_err={m["lead_error"]:.3%} centre_rmse={m["centre_rmse_mv"]:.3f}mV')


# ---------------------------------------------------------------- training
def train_model(model, device, epochs=5, batch_size=1, learning_rate=1e-5, val_percent=0.1, amp=False, num_workers=None, run_dir=None):

    #where to store all
    out_dir = Path(run_dir)                          
    out_dir.mkdir(parents=True, exist_ok=True)       

    # data: whole pages
    dataset = BasicDataset(dir_img, dir_mask, index_csv=index_csv)
    image_info = read_index(index_csv)

    #splitting
    train_idx, val_idx = split_by_page(dataset, val_percent)

    #dataloaders, make them work with the network
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # 2 losses: weighted pixel cross-entropy + dice on the leads only; layout cross-entropy for the head
    criterion = nn.CrossEntropyLoss(weight=class_weights(dataset, train_idx, model.n_classes, device))
    layout_criterion = nn.CrossEntropyLoss() #loss for layout

    # 3. optimizer, learning-rate schedule, mixed precision
    optimizer, scheduler = make_optimizer(model, learning_rate, epochs)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    autocast_device = device.type if device.type != 'mps' else 'cpu'

    logging.info(f'Training: {epochs} epochs, batch {batch_size}, lr {learning_rate}, '
                 f'amp {amp}, {len(train_idx)} train / {len(val_idx)} val images, device {device.type}')

    for epoch in range(1, epochs + 1):
        # 4. one pass over the training pages
        model.train()
        totals = {'loss': 0.0, 'seg_ce': 0.0, 'seg_dice': 0.0, 'cls_ce': 0.0}

        # - loss: the total, which is what the model actually trains on
        # - seg_ce: the pixel cross-entropy part
        # - seg_dice: the dice part
        # - cls_ce: the layout-head part

        #start the bar off during training
        with tqdm(total=len(train_idx), desc=f'Epoch {epoch}/{epochs}', unit='img') as bar:
            
            #for each batch in training.
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true = batch['mask'].to(device=device, dtype=torch.long)

                #autocast makes them float 16 instead to save memory
                with torch.autocast(autocast_device, enabled=amp):
                    out = model(images)
                    logits, layout_logits = out
                    seg_ce = criterion(logits, true)                               # every pixel, weighted
                    seg_dice = dice_loss_fg(logits, true, model.n_classes)         # each lead's shape
                    cls_ce = layout_criterion(layout_logits, batch['template'].to(device))
                    loss = seg_ce + seg_dice + cls_ce       

                #the updating taking place, needs some adjustements since amp
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                for name, value in (('loss', loss), ('seg_ce', seg_ce), ('seg_dice', seg_dice), ('cls_ce', cls_ce)):
                    totals[name] += value.item()
                bar.update(images.shape[0])
                bar.set_postfix(loss=loss.item())

        # validate once, on whole validation pages
        val = evaluate(model, val_loader, device, amp)
        losses = {name: total / len(train_loader) for name, total in totals.items()}

        # 6. next learning rate
        scheduler.step()

        # 7. logs + checkpoint
        write_epoch_logs(out_dir, epoch, losses, val, image_info)

        #printing
        logging.info(f'epoch {epoch}: loss={losses["loss"]:.4f} dice={val["dice"]:.4f} '   
                     f'recall={val["trace_recall"]:.3f} precision={val["trace_precision"]:.3f} '
                     f'lead_err={val["lead_error"]:.3%} centre_rmse={val["centre_rmse_mv"]:.3f}mV '
                     f'layout_acc={val["layout_acc"]:.3f}')
        logging.info(format_per_class(val['dice_per_class']))
        logging.info('layout confusion matrix:\n' + format_confusion(val['layout_confusion']))

        #saving
        (out_dir / 'checkpoints').mkdir(exist_ok=True)
        state = model.state_dict()
        state['mask_values'] = dataset.mask_values         # predict.py expects it in the checkpoint
        torch.save(state, out_dir / 'checkpoints' / f'checkpoint_epoch{epoch}.pth')


def get_args():
    """Only what changes between runs."""

    parser = argparse.ArgumentParser(description='Train the U-Net on whole ECG pages')
    parser.add_argument('--epochs', '-e', type=int, default=40)
    parser.add_argument('--learning-rate', '-l', type=float, default=3e-4, dest='lr')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='whole pages per step')
    parser.add_argument('--run-dir', type=str, default='runs/test', help='where checkpoints/ and the logs go')
    parser.add_argument('--load', type=str, default=None, help='continue from this checkpoint (.pth)')
    return parser.parse_args()


if __name__ == '__main__':
    
    #take all the args
    args = get_args()

    #this is how to display the training + cuda
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3 input channels, this is the arguments that setup then!
    model = UNet(n_channels=3, n_classes=n_classes, n_layouts=n_layouts)

    #if we have a previous loaded model we can: 
    if args.load:
        state = torch.load(args.load, map_location=device)
        state.pop('mask_values', None)
        model.load_state_dict(state)
        logging.info(f'Loaded {args.load}')

    model = model.to(device=device, memory_format=torch.channels_last)

    #train the model!!!!
    train_model(model, device, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                val_percent=val_percent, amp=amp, num_workers=workers, run_dir=args.run_dir)
