"""

Step 1 of 2: render clean 13-class ECG pages (image + per-lead label mask) from PTB-XL records.


"""
import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent                     # .../unet-src/dataset_build
ASSETS = HERE / "assets"
H5 = ASSETS / "ptb_records.h5"                             # signals, from export_records.py
DB = ASSETS / "ptb_records.csv"                            # metadata, row i = h5 row i
OUT = HERE.parent / "data" / "pages"                       # .../unet-src/data/pages

# ---------------------------------------------------------------- what to build
DPI = 300                           

# ---------------------------------------------------------------- leads and class ids
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LABEL = {
    "I":   1,  "II":  2,  "III": 3,      # limb leads
    "aVR": 4,  "aVL": 5,  "aVF": 6,      # augmented limb leads
    "V1":  7,  "V2":  8,  "V3":  9,      # chest leads
    "V4": 10,  "V5": 11,  "V6": 12,
}

assert LABEL == {lead: i + 1 for i, lead in enumerate(LEADS)}, "LABEL must follow LEADS order"
N_CLASSES = len(LABEL) + 1               # 13: background + 12 leads

# ---------------------------------------------------------------- page geometry
VOLTAGE = 10.0                      

# layouts and their row spacing in mm. Must match TEMPLATE_TO_IDX in utils/data_loading.py
ROW_dist_MM = {"1x12": 15.0, "2x6": 30.0, "2x6+1": 30.0, "4x3+1": 45.0}   
TEMPLATES = list(ROW_dist_MM)

#colour palette for the leads
PALETTE = [(255, 255, 255),                                       # 0 background
           (228, 26, 28), (55, 126, 184), (77, 175, 74),          # I, II, III
           (152, 78, 163), (255, 127, 0), (166, 206, 27),         # aVR, aVL, aVF
           (166, 86, 40), (247, 129, 191), (0, 206, 209),         # V1, V2, V3
           (120, 120, 120), (0, 0, 255), (255, 0, 255)]           # V4, V5, V6


#background tint for the images
PAPERS = ["#f2f2f2", "#e6e6e3", "#efeee6", "#f5f1e0", "#f3ecd2", "#efe4c0"]


# ---------------------------------------------------------------- printed header (machine text above the traces)

# share of pages with a header; the rest stay bare
HEADER_P = 0.7                                                   
HOSPITALS = ["Hospital Central", "ARC Clinical", "St. Mary's", "Klinikum Nord", "Cardiology Dept.", None]
FILTERS = ["0.05-150 Hz", "0.5-40 Hz", "0.05-100 Hz 50Hz", None]


# ---------------------------------------------------------------- functions 

def make_header(meta, rng):
    """
    pmecg header text for one record: real PTB-XL age/sex/date/device, report as patient line.

    """

    #some random info at the top
    info = pmecg.ECGInformation(
        hospital=rng.choice(HOSPITALS),
        patient_name=f"ID {rng.randint(1, 9999):04d}  {meta['report'][:16]}",   
                                                                                 
        age=None if pd.isna(meta["age"]) else int(meta["age"]),
        sex={0: "M", 1: "F"}.get(meta["sex"]),                                   
        date=str(meta["recording_date"]),
        machine_model=" ".join(str(meta["device"]).split()),                    
        filter=rng.choice(FILTERS),
    )

    #some random stats
    stats = pmecg.ECGStats(
        bpm=rng.randint(50, 110),
        rr_interval_ms=rng.randint(550, 1200),
        pr_interval_ms=rng.randint(120, 200),
        qrs_duration_ms=rng.randint(70, 120),
        qt_interval_ms=rng.randint(340, 440),
        qtc_interval_ms=rng.randint(380, 460),
        qrs_axis_deg=rng.randint(-30, 90),
    )

    # retunrns ecg info at random and some stats at random
    return info, stats                                                          


def make_pair(df, template, name, rng, meta, out):
    """
    Make a pair of lead mask and images
    
    """
    # e.g. "#f3ecd2"
    paper = rng.choice(PAPERS)                                   

    # ... and whether it gets the machine's printed header 
    header = rng.random() < HEADER_P
    info, stats = make_header(meta, rng) if header else (None, None)

    # Natural size, no padding: augment.py's background() puts it on a fixed-size desk photo.
    # (H, W, 3), grid edge to edge
    img = render(df, template, paper=paper, header=header, info=info, stats=stats)   

    #now lets make the labels also
    labels = np.zeros(img.shape[:2], dtype=np.uint8)  # (H, W), 0 everywhere = background

    #iterate over leads to turn one on
    for lead in LEADS:

        other_leads = [name for name in LEADS if name != lead]
        only = df.copy()
        only[other_leads] = np.nan                 

        # Same geometry as `img`, so the pixels line up one to one. Stays on white paper.
        trace = render(only, template, trace_only=True, header=header)  
        drawn = (trace < 255).any(axis=2)               

        # Stamp those pixels with this lead's class id. Where two leads cross, the later one overwrites the earlier: one pixel can only carry one label.
        labels[drawn] = LABEL[lead]


    # RGB PNG, the input the model sees
    Image.fromarray(img).save(out / "imgs" / f"{name}.png")

    # palette PNG: values 0..12
    save_labels(labels, out / "labels" / f"{name}.png")

    #all three needed for the build record. 
    return labels, paper, header


def render(df, template, trace_only=False, paper="white", header=False, info=None, stats=None):
    """One figure -> RGB array (H, W, 3), background in the `paper` colour.

    trace_only=True switches off everything except the signal (grid, calibration pulse,
    lead labels, separators, time axis), which is what the label masks are built from.
    """

    #set config acc to template and df
    configuration = pmecg.template_factory(template, df, leads_map=None)

    # how far apart the rows sit, in mV: pmecg wants mV, ROW_dist_MM is in mm
    # (15 mm at 10 mm/mV -> 1.5 mV, 30 -> 3.0, 45 -> 4.5)
    dist = ROW_dist_MM[template] / VOLTAGE

    #depending on trace only true or not says what to save
    if trace_only:
        # labels are built from this: signal only, nothing else on the page
        plotter = pmecg.ECGPlotter(row_distance=dist,
                                   grid_mode=None,
                                   show_calibration=False,
                                   show_leads_labels=False,
                                   show_separators=False,
                                   show_time_axis=False,
                                   print_information=header)  
    else:
        
        # the printout the model sees: grid, calibration pulse, lead labels, separators
        plotter = pmecg.ECGPlotter(row_distance=dist, print_information=header)

    #we plot the df on that config
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=400, show=False, information=info, stats=stats)

    if trace_only:
        for text in list(fig.axes[0].texts):
            text.remove()
    fig.patch.set_facecolor(paper)          # area around the axes
    fig.axes[0].set_facecolor(paper)        # the plot area itself
    fig.set_dpi(DPI)                                     
    fig.canvas.draw()                                     

    #drop some of the data, need to look at
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # (H, W, 4) RGBA -> drop alpha


    # Crop to the plot area = where the grid is. pmecg leaves a plain margin around it, but real have to the edge. should i keep this? 
    x0, y0, x1, y1 = np.round(fig.axes[0].get_window_extent().extents).astype(int)   
    H = rgb.shape[0]
    rgb = rgb[H - y1:H - y0, x0:x1]
    plt.close(fig)                                        

    # (H, W, 3) RGB, paper colour where there's no ink
    return rgb.copy()              


def save_labels(labels, path):
    """
    Write the class-index mask as a palette PNG: same numbers, but a viewer shows colours. mask but colour.
    """
    out = Image.fromarray(labels, mode="P")
    out.putpalette([channel for colour in PALETTE for channel in colour] + [0] * (256 - len(PALETTE)) * 3)
    out.save(path)


def build_record(job):
    """
    One record -> one page pair on disk. Runs in a worker process; returns its index.csv row.
    """

    #unpack the input
    signal, meta, out = job                     # out = folder to write to (passed along: workers can't see main's variables)
    #make a random seed per record
    rng = random.Random(int(meta["record"]))   # own seed per record: same page whatever the worker split
    #choose random templates
    template = rng.choice(TEMPLATES)

    #make a dataframe of the signal from unpack
    df = pd.DataFrame(signal, columns=LEADS)     

    # PTB-XL ecg_id: traceable back to the record
    name = f"ecg{int(meta['ecg_id']):05d}"                         
    labels, paper, header = make_pair(df, template, name, rng, meta, out)

    #quick check
    assert len(np.unique(labels)) == 13, f"{name}: a lead is missing from the label"

    #this is the things that is returned, and will make the csv file
    return f"{name},{int(meta['record'])},{int(meta['ecg_id'])},{template},{paper},{int(header)}\n"


if __name__ == "__main__":
    
    #parse these arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="only the first N records (default: all)")      #how many records
    parser.add_argument("--workers", type=int, default=1, help="CPU processes rendering in parallel")       #fast
    parser.add_argument("--test", action="store_true", help="test records (ptb_test.*) -> data/test_pages")
    args = parser.parse_args()

    # training or test set: which records to read, where the pages go
    if args.test:
        h5, db_csv, out = ASSETS / "ptb_test.h5", ASSETS / "ptb_test.csv", HERE.parent / "data" / "test_pages"
    else:
        h5, db_csv, out = H5, DB, OUT

    # where to save them
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)


    #read all the h5 files until the argument n
    db = pd.read_csv(db_csv)[:args.n]

    # (N, 4096, 12), ~200 MB for 1000 records, take all the tracings
    with h5py.File(h5, "r") as f:
        tracings = f["tracings"][:len(db)]

    #[(signal_0, meta_0, out), (signal_1, meta_1, out), …]
    jobs = [(tracings[k], db.iloc[k].to_dict(), out) for k in range(len(db))]


    #logging the process.
    with Pool(args.workers) as pool, open(out / "index.csv", "w") as index:
        index.write("name,record,ecg_id,template,paper,header\n")
        for k, row in enumerate(pool.imap(build_record, jobs), 1):
            index.write(row)
            print(f"{k}/{len(jobs)}  {row.split(',')[0]}  {row.split(',')[3]}", flush=True)

    print(f"wrote {len(jobs)} pages to {out}")
