import argparse
import csv
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate, format_confusion, format_per_class, LEAD_NAMES
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, CropView, PadView
from utils.dice_score import dice_loss, dice_loss_fg

# Built by Dataset_build/make_multiclass.py: labels are class indices 0=background,
# 1..12 = lead I..V6 (a palette PNG, so a viewer shows colours but the pixel values
# stay the class ids). Train with --classes 13.
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/labels/')
dir_index = Path('./data/index.csv')           # name -> layout template, for the layout head
dir_checkpoint = Path('./checkpoints/')        # default when --run-dir is not passed



def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        num_workers: int = None,
        weight_decay: float = None,
        momentum: float = None,
        gradient_clipping: float = None,
        run_dir: str = None,
        crop_size=None,
        fg_oversample: float = 1 / 3,
        val_pad_to: int = 16,
        optimizer_name: str = 'rmsprop',
):
    # Outputs default to cwd-relative paths (dir_checkpoint / training_log*.csv);
    # --run-dir redirects all three under one directory instead.
    checkpoint_dir = Path(run_dir) / 'checkpoints' if run_dir else dir_checkpoint
    log_path = Path(run_dir) / 'training_log.csv' if run_dir else Path('training_log.csv')
    per_lead_path = Path(run_dir) / 'training_log_per_lead.csv' if run_dir else Path('training_log_per_lead.csv')
    if run_dir:
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale,
                                 index_csv=dir_index if model.n_layouts else None)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale,
                               index_csv=dir_index if model.n_layouts else None)

    # Guard: --layouts asked for the head, so the labels must actually be present.
    # Without this a mis-wired dataset silently trains with no classification signal.
    assert not model.n_layouts or dataset.templates is not None,         'layout head enabled but dataset has no template labels (index_csv not applied)'

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # On Windows each DataLoader worker is a full process copy; with full-page ECG images
    # os.cpu_count() workers can exhaust RAM. 0 = load in the main process.
    workers = os.cpu_count() if num_workers is None else num_workers
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    # crop_size=None keeps the legacy whole-image --scale resize (backward compatible).
    # crop_size set: CropView draws full-resolution training crops (random, or
    # fg_oversample-biased on a present class), and PadView gives eval the whole page
    # in one padded-to-val_pad_to forward pass -- see utils/data_loading.py. --scale
    # is meaningless for CropView's crops (always full-res) but evaluate.py's mV
    # conversion still reads it, so pass --scale 1.0 alongside --crop-size.
    if crop_size:
        train_loader = DataLoader(CropView(train_set, crop_size, fg_oversample), shuffle=True, **loader_args)
        val_loader = DataLoader(PadView(val_set, val_pad_to), shuffle=False, drop_last=True, **loader_args)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # BUGFIX (2026-09-22): class-imbalance collapse. Job 17603025 (13-class + layout
    # head, unweighted CE, dice computed over ALL 13 classes incl. background) fell to
    # predicting background on every pixel by epoch 2 and never recovered -- val
    # trace_recall=0.0, trace_precision=nan (zero predicted trace pixels at all) from
    # epoch 2 through 18. Root cause: background covers ~all of every image, the other
    # 12 classes are 1-2px trace lines, so "always background" is a trivial low-CE
    # minimum, and multiclass_dice_coeff pools all classes+batch into ONE dice number --
    # a near-perfect background channel drowns out total failure on the 12 real classes.
    # evaluate.py's own val['dice'] metric already excludes background (utils/data_loading.py
    # via evaluate.py:82, `pred_1h[:, 1:]`) -- the training loss just never matched that
    # convention. Two fixes below: weighted CE so rare lead classes aren't free to ignore,
    # and excluding background from the training dice loss to match evaluate.py.
    class_weights = None
    if model.n_classes > 1:
        counts = torch.zeros(model.n_classes, dtype=torch.float64)
        for batch in DataLoader(train_set, batch_size=8, num_workers=workers):
            counts += torch.bincount(batch['mask'].flatten(), minlength=model.n_classes).double()
        # TUNED (2026-09-22): raw inverse frequency (job 17604711, 20 epochs) fixed the
        # collapse but overcorrected -- trace_recall pinned at ~1.0 the whole run while
        # trace_precision only reached 0.18, i.e. the model learned to flag almost
        # everything as trace instead of almost nothing. sqrt softens the weight spread
        # (background 0.078 -> ~0.28, lead classes ~40-78 -> ~6-9 pre-renormalisation) so
        # rare classes are still upweighted but not enough to make over-predicting them
        # nearly free. Re-normalised after the sqrt so weights still average to 1.
        class_weights = (counts.sum() / (model.n_classes * counts.clamp(min=1))).sqrt()
        class_weights = (class_weights / class_weights.mean()).float().to(device)
        logging.info('Class weights (sqrt inverse pixel frequency): ' +
                     ', '.join(f'{i}={w:.3f}' for i, w in enumerate(class_weights.tolist())))

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    #
    # TUNED (2026-09-23): the template's RMSprop lr=1e-5 + ReduceLROnPlateau was far too
    # slow -- job 17604711's seg_ce/centre_rmse_mv basically flatlined after epoch ~7 of
    # 20. Krones et al. (ECG-Digitiser) use SGD-nesterov with a poly LR decay (standard
    # nnU-Net recipe); AdamW+cosine offered as the other common alternative. rmsprop
    # stays the default so old invocations are unaffected; pass --optimizer sgd (or
    # adamw) to opt in. Per-optimizer defaults below only apply when the CLI leaves
    # weight_decay/momentum/gradient_clipping at None (i.e. not explicitly overridden).
    opt_defaults = {
        'rmsprop': dict(weight_decay=1e-8, momentum=0.999, gradient_clipping=1.0),
        'sgd':     dict(weight_decay=3e-5, momentum=0.99,  gradient_clipping=12.0),
        'adamw':   dict(weight_decay=3e-5, momentum=None,  gradient_clipping=1.0),
    }[optimizer_name]
    weight_decay = opt_defaults['weight_decay'] if weight_decay is None else weight_decay
    momentum = opt_defaults['momentum'] if momentum is None else momentum
    gradient_clipping = opt_defaults['gradient_clipping'] if gradient_clipping is None else gradient_clipping

    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              nesterov=True, weight_decay=weight_decay)
        # nnU-Net's poly decay: (1 - epoch/max_epochs)**0.9, stepped once per epoch (not
        # keyed on validation Dice like ReduceLROnPlateau above).
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / max(epochs, 1)) ** 0.9)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f'Unknown --optimizer {optimizer_name!r}')
    per_epoch_schedule = optimizer_name in ('sgd', 'adamw')
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # Layout is 3 mutually exclusive templates -> CrossEntropy, not BCE.
    cls_criterion = nn.CrossEntropyLoss()
    global_step = 0

    # One row per epoch. The losses say the model is training; the metrics after 'dice' say
    # whether the mask is usable downstream -- centre_rmse_mv is the digitisation error in
    # mV, the number the whole pipeline is judged on. Per-lead dice goes to its own file so
    # this one stays a fixed-width table.
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'seg_ce', 'seg_dice', 'cls_ce',
                                'dice', 'trace_recall', 'trace_precision', 'lead_error',
                                'column_coverage', 'centre_rmse_mv', 'centre_mae_mv',
                                'centre_mse_mv2', 'layout_acc'])
    with open(per_lead_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch'] + [f'dice_{name}' for name in LEAD_NAMES[1:model.n_classes]])

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_seg_ce = epoch_seg_dice = epoch_cls_ce = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                true_templates = batch['template'].to(device=device) if 'template' in batch else None

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    out = model(images)
                    masks_pred, cls_pred = out if model.n_layouts else (out, None)
                    # Kept as three named terms so each can be logged separately: the
                    # layout CE is ~100x the segmentation terms, so a single total hides
                    # whether segmentation is converging at all.
                    if model.n_classes == 1:
                        seg_ce = criterion(masks_pred.squeeze(1), true_masks.float())
                        seg_dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(),
                                             multiclass=False)
                    else:
                        seg_ce = criterion(masks_pred, true_masks)
                        # Per-class Dice, background excluded, THEN averaged -- see the
                        # dice_loss_fg BUGFIX comment in utils/dice_score.py. Replaces the
                        # earlier [:, 1:]-on-pooled-dice fix, which still diluted rare
                        # classes together instead of averaging each one separately.
                        seg_dice = dice_loss_fg(masks_pred, true_masks, model.n_classes)
                    loss = seg_ce + seg_dice
                    cls_ce = None
                    if cls_pred is not None:
                        # Added un-weighted: the head is detached from the encoder, so this
                        # term only ever updates the head's own 3k params. It cannot move dice.
                        cls_ce = cls_criterion(cls_pred, true_templates)
                        loss = loss + cls_ce

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_seg_ce += seg_ce.item()
                epoch_seg_dice += seg_dice.item()
                epoch_cls_ce += cls_ce.item() if cls_ce is not None else 0.0
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # BUGFIX (2026-09-23): pre-existing (not from this session's other
                        # changes) -- wandb.Histogram(...) calls np.histogram, which raises
                        # ValueError on a perfectly constant tensor (zero-width bin range,
                        # e.g. an unused/not-yet-updated bias). Job 17616389 (first run with
                        # --optimizer sgd, more aggressive lr=1e-2) hit this at step 20 and
                        # crashed training entirely over a debug histogram, with
                        # WANDB_MODE=disabled anyway. Guarded per-tensor so one degenerate
                        # weight/grad only drops that one histogram, not the whole run.
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            try:
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                            except ValueError:
                                pass

                        val = evaluate(model, val_loader, device, amp, img_scale)
                        val_score, val_acc = val['dice'], val['layout_acc']
                        if not per_epoch_schedule:
                            scheduler.step(val_score)   # ReduceLROnPlateau: keyed on dice, mid-epoch

                        logging.info('Validation Dice {:.4f}  centre RMSE {:.4f} mV  lead err {:.3%}{}'.format(
                            val_score, val['centre_rmse_mv'], val['lead_error'],
                            '' if val_acc is None else f'  |  layout acc: {val_acc:.4f}'))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'trace recall': val['trace_recall'],
                                'trace precision': val['trace_precision'],
                                'lead error': val['lead_error'],
                                'column coverage': val['column_coverage'],
                                'centre RMSE mV': val['centre_rmse_mv'],
                                'centre MSE mV2': val['centre_mse_mv2'],
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if per_epoch_schedule:
            scheduler.step()   # poly (sgd) / cosine (adamw): stepped once per epoch, not on dice

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_seg_ce, avg_seg_dice = epoch_seg_ce / n_batches, epoch_seg_dice / n_batches
        avg_cls_ce = epoch_cls_ce / n_batches
        val = evaluate(model, val_loader, device, amp, img_scale)
        epoch_acc, epoch_cm = val['layout_acc'], val['layout_confusion']
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, avg_seg_ce, avg_seg_dice, avg_cls_ce,
                                    val['dice'], val['trace_recall'], val['trace_precision'],
                                    val['lead_error'], val['column_coverage'],
                                    val['centre_rmse_mv'], val['centre_mae_mv'],
                                    val['centre_mse_mv2'], epoch_acc])
        with open(per_lead_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch] + val['dice_per_class'][1:])
        logging.info(f'epoch {epoch}: loss={avg_loss:.5f} dice={val["dice"]:.4f} '
                     f'recall={val["trace_recall"]:.3f} precision={val["trace_precision"]:.3f} '
                     f'lead_err={val["lead_error"]:.3%} coverage={val["column_coverage"]:.3f} '
                     f'centre_rmse={val["centre_rmse_mv"]:.4f}mV '
                     f'layout_acc={"n/a" if epoch_acc is None else f"{epoch_acc:.4f}"}')
        if model.n_classes > 1:
            logging.info(format_per_class(val['dice_per_class']))
        if epoch_cm is not None:
            logging.info('layout confusion matrix:' + chr(10) + format_confusion(epoch_cm))

        if save_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--workers', type=int, default=None,
                        help='DataLoader workers (default: os.cpu_count(); use 0 on low-RAM machines)')
    parser.add_argument('--layouts', type=int, default=None,
                        help='Number of layout templates (3) to enable the detached layout head. '
                             'Omit for segmentation only.')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Directory for checkpoints/ and training_log*.csv. '
                             'Omit to write them next to train.py (cwd-relative), as before.')
    parser.add_argument('--crop-size', type=int, nargs=2, default=None, metavar=('H', 'W'),
                        help='Train on full-resolution HxW crops instead of a whole-image '
                             '--scale resize (which mangles 1-2px traces via NEAREST '
                             'downsampling). Pass --scale 1.0 alongside this -- crops are '
                             'always full-res regardless of --scale. Omit to keep the old '
                             'whole-image behaviour. e.g. --crop-size 1024 1280')
    parser.add_argument('--fg-oversample', type=float, default=1 / 3,
                        help='Fraction of training crops force-centred on a random pixel of '
                             'a randomly-chosen present lead class, rather than a uniform-'
                             'random window -- otherwise a crop can easily miss a thin, '
                             'localised trace entirely. Only used with --crop-size.')
    parser.add_argument('--val-pad-to', type=int, default=16,
                        help='Validation runs the WHOLE page through in one forward pass '
                             '(no sliding-window), padded to a multiple of this so the '
                             "network's up/downsampling stays aligned. Only used with "
                             '--crop-size.')
    parser.add_argument('--optimizer', choices=['rmsprop', 'sgd', 'adamw'], default='rmsprop',
                        help="'rmsprop' (default) is the template's original RMSprop + "
                             "ReduceLROnPlateau -- kept as the default so old invocations "
                             "are unaffected. 'sgd' is nnU-Net's nesterov-momentum + poly-LR "
                             "decay recipe; 'adamw' is AdamW + cosine annealing. Both are "
                             'meant to replace the slow RMSprop lr=1e-5 default -- pass a '
                             'higher --learning-rate with them (e.g. 1e-2 for sgd, 3e-4 for '
                             'adamw).')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=1 for grayscale ECG plot images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear,
                 n_layouts=args.layouts)   # detach_head=True by default
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.workers,
            run_dir=args.run_dir,
            crop_size=args.crop_size,
            fg_oversample=args.fg_oversample,
            val_pad_to=args.val_pad_to,
            optimizer_name=args.optimizer
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            num_workers=args.workers,
            run_dir=args.run_dir,
            crop_size=args.crop_size,
            fg_oversample=args.fg_oversample,
            val_pad_to=args.val_pad_to,
            optimizer_name=args.optimizer
        )
