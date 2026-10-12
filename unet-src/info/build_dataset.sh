#!/bin/bash
# Build the training set on the cluster: PAGES records from assets/ptb_records.* -> PAGES pages -> 6 x PAGES photos/scans.
#
#   bash info/build_dataset.sh              # 500 pages -> 3000 pairs; uses $SLURM_CPUS_PER_TASK cores, or 8
#   PAGES=1000 bash info/build_dataset.sh   # all exported records -> 6000 pairs
#
# Then train on whole pages (3328 x 2496, no cropping), e.g.:
#   python train.py --classes 13 --layouts 4 --amp --batch-size 1 \
#                   --optimizer adamw -l 3e-4 --epochs 60 --run-dir runs/whole_pages
set -e
cd "$(dirname "$0")/.."                                # unet-src/ (this script lives in unet-src/info/)
N=${SLURM_CPUS_PER_TASK:-8}
PAGES=${PAGES:-500}

python dataset_build/make_multiclass.py --n "$PAGES" --workers "$N"   # -> data/pages/   (~3 s per page per core)
python dataset_build/augment.py --workers "$N"           # -> data/imgs, data/labels, data/index.csv (~15 s per page per core)
