"""
Pick the PTB-XL records the cluster builds its dataset from, and copy just those.

Run ONCE, locally (the full ptb_preprocessed.h5 is 8.5 GB and stays here):

python unet-src/dataset_build/export_records.py            # 1000 training records (folds 1-9) -> ptb_records.*
python unet-src/dataset_build/export_records.py --n 2000
python unet-src/dataset_build/export_records.py --test --n 5    # 5 TEST records (fold 10) -> ptb_test.*

"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                         # .../unet-src/dataset_build
PTB = HERE.parents[1] / "ptb-xl"                               # .../U-net seg/ptb-xl (local only)
ASSETS = HERE / "assets"

if __name__ == "__main__":
    
    #take the arguments as how many and which seed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="number of records")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="take fold 10 (the test set) instead of folds 1-9")
    args = parser.parse_args()
    out_name = "ptb_test" if args.test else "ptb_records"

    # 1. metadata; row i here = row i of the full h5 (checked against the old index.csv)
    db = pd.read_csv(PTB / "ptbxl_database.csv")
    db["record"] = np.arange(len(db)) # 1,2....3


    # 2a. keep only the training folds: PTB-XL fold 10 is the test set, never train on it
    #     (or, with --test, ONLY fold 10: records the model has never seen)
    TEST_FOLD = 10
    is_test = db["strat_fold"] == TEST_FOLD
    pool = db[is_test] if args.test else db[~is_test]

    # 2b. pick n random records from those (random_state fixes which ones: same pick every run)
    pick = pool.sample(n=args.n, random_state=args.seed)

    # 2c. sort by row number: h5py reads rows fastest in increasing order
    pick = pick.sort_values("record")

    # 3. copy their signals and metadata
    ASSETS.mkdir(exist_ok=True)
    with h5py.File(PTB / "ptb_preprocessed.h5", "r") as f:
        tracings = f["tracings"][pick.record.to_numpy()].astype(np.float32)   # float64 -> float32: half the size
    
    with h5py.File(ASSETS / f"{out_name}.h5", "w") as f:
        f.create_dataset("tracings", data=tracings)

    cols = ["record", "ecg_id", "strat_fold", "age", "sex", "recording_date", "device", "report"]
    pick[cols].to_csv(ASSETS / f"{out_name}.csv", index=False)

    print(f"wrote {len(pick)} records (folds {sorted(pick.strat_fold.unique())}) to {ASSETS}")
