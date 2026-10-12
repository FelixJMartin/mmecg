# Handover notes — ECG U-Net, RGB "personas" dataset (written 2026-10-07)

## What this is
A U-Net that segments paper-ECG images per lead: every pixel -> one of 13 classes
(0 = background, 1..12 = leads I, II, III, aVR, aVL, aVF, V1..V6), plus an optional
layout classification head. The masks are later digitised to mV (not part of this folder).

Training data is synthetic: PTB-XL signals rendered as printed ECGs with `pmecg`, then
augmented into realistic phone photos and scans. The aim is to work on real photos/scans.

## Folder layout
```
unet-src/
  train.py  evaluate.py  predict.py  unet/  utils/      training + inference
  build_dataset.sh                                       builds data/ (step 1 + 2 below)
  requirements.txt
  dataset_build/
    export_records.py   (already run locally) picked 1000 PTB-XL records, folds 1-9 only
    make_multiclass.py  step 1: records -> clean pages      -> data/pages/
    augment.py          step 2: pages -> 6 personas each    -> data/imgs, data/labels, data/index.csv
    assets/
      ptb_records.h5 / .csv   the 1000 records (signals float32 (N,4096,12) + metadata)
      wrinkles/  backgrounds/  frames/arc_frame.png  handwriting_bank/   augmentation inputs
  data/              created by build_dataset.sh (~17 GB for 6000 pairs)
```

## Run it
```bash
pip install -r requirements.txt
bash build_dataset.sh            # 500 pages -> 3000 pairs (~8.5 GB); PAGES=1000 for all 6000 (~17 GB)
                                 # uses $SLURM_CPUS_PER_TASK cores (default 8)
                                 # ~3 s/page (step 1) + ~15 s/page (step 2) per core -> ~10 min on 16 cores for 500
# Storage: on Arrhenius, /home/felixm has ~27 GB free in total -- start with 500 pages; data/pages/ can be
# deleted after step 2 (training doesn't read it).
python train.py --classes 13 --layouts 4 --crop-size 1024 1280 --scale 1.0 --amp \
                --optimizer adamw -l 3e-4 --epochs 40 --batch-size 4 --run-dir runs/rgb_personas
```
Flags are an example; `--classes 13 --layouts 4` and `--crop-size ... --scale 1.0` are the important ones
(`--scale 0.5` would halve the 1-2 px traces). Preview one page's personas: `python dataset_build/augment.py --preview ecg00018`.

## The data
- Pages (step 1): 4 layouts, picked uniformly: `1x12`, `2x6`, `2x6+1`, `4x3+1` (`+1` = lead II rhythm strip).
  Paper tinted grey to yellowish; ~70 % have the machine's printed header (date, patient line, intervals).
  Pages are cropped to the grid (natural size, no padding). Labels: palette PNG, values = class ids.
  `check_leads()` asserts every lead the layout should show is labelled.
- Personas (step 2), each page -> 6 images, every one 3318 x 2488 RGB JPEG + its own label:
  | persona | steps |
  |---|---|
  | scan_clean | scan_frame (ARC "scanned ECG" page, white-balanced, <=1 deg skew), light noise |
  | scan_marked | handwriting, scan_frame, light noise, redaction |
  | photo_clean | background (desk photo, page 85-95 % of frame, <=2 deg tilt), noise |
  | photo_folded | handwriting, creases, background, shadow, noise |
  | photo_wrinkled | handwriting, wrinkles, creases, background, shadow, noise |
  | photo_redacted | handwriting, creases, background, shadow, noise, redaction (black box, label 0 under it) |
- Every step changes image and label together; geometric steps use NEAREST for labels (no invented class ids).
- Seeds are per page, so output doesn't depend on `--workers`.
- `data/index.csv`: `name, page, template, persona`; train.py reads `name -> template` for the layout head.

## Changes made today (vs the previous greyscale model)
- Input is RGB: `UNet(n_channels=3)` in train.py and predict.py; data_loading converts images to RGB.
- New layout `2x6+1`: `TEMPLATE_TO_IDX = {'1x12': 0, '2x6': 1, '4x3+1': 2, '2x6+1': 3}` (append-only), `--layouts 4`.
- predict.py reads .jpg too (score mode globs `name.*`).
- requirements: numpy>=1.24, matplotlib>=3.7, pandas>=2.0 (pmecg needs them) + h5py, pmecg, opencv-python-headless, torchvision.
- Smoke-tested locally: dataset build on 8 records (4 workers) and a few CPU training steps with
  `--layouts 4 --crop-size 256 256` (3 channels, 13 mask values, 4 layouts loaded, loss computed). Not trained further.

## Watch out for
1. **Old checkpoints don't load**: the previous `best.pth` is 1-channel / 3 layouts. Train from scratch
   (or inflate the first conv `w.repeat(1,3,1,1)/3` and extend the layout head if warm-starting).
2. **Validation is optimistic**: train.py splits images randomly, so personas of the same page land in both
   train and val. For an honest number split by `page` (data/index.csv) or score on PTB-XL fold 10.
3. **Fold 10 = test set**: never train on it. export_records.py only takes folds 1-9.
4. **utils/data_loading.py must contain `CropView` / `PadView`** (train.py imports them). The local copy had lost
   them once; it was restored from git commit 5ea0665. If the cluster has a newer data_loading.py, merge
   rather than overwrite: today's only changes there are `TEMPLATE_TO_IDX` (+ '2x6+1') and `.convert('RGB')`
   on the image in `__getitem__` and `load_raw`.
5. Full-page validation (`PadView`) at 3318 x 2488 needs a lot of GPU memory; use `--amp`.
6. `layout_keys.csv` lives one level up (not uploaded); data_loading only cross-checks it when present.
