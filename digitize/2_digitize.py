"""
Turns the per-lead pixel lines from rows.py into a 12-lead signal in mV.

Old version (row pieces + equal-block splitting) lives in old/Digitize.py.
"""

# imports
import numpy as np
from pathlib import Path
import pandas as pd

# Paths -----------------------------
HERE = Path(__file__).parent                                          # the digitize folder
PAGE = HERE.parent / "Predictions" / "digitized" / "example_56" / "csv"   # the folder 1_rows.py wrote to
                                                    

# constants: paper conventions and layouts
mm_to_mv = 0.1   #(10 mm per mV)
mm_to_s  = 0.04  #(25 mm per second)
ROW_DIST_MM = {"1x12": 15, "2x6": 30, "4x3+1": 45}   # distance between two row baselines which leads sit on which row, per layout

table = pd.read_csv(PAGE / "lines.csv", index_col=0)                  # one row per page column, one column per lead
lines = {name: table[name].to_numpy() for name in table}              # {"I": y, "II": y, ..., "V6": y}
print(list(lines))

# 1. row baselines: median y of each row's leads, straight-line fit across rows -> zero line per row + px per mm
info = {}                                                  # name -> its box and zero line
for name, y in lines.items():
    cols = np.where(~np.isnan(y))[0]                       # columns where this lead has ink
    if len(cols) == 0:
        continue                                           # lead not found on this page

    info[name] = {
        "x_start": cols[0],                                # leftmost column  -> where its time starts
        "x_end":   cols[-1],                               # rightmost column -> where its time ends
        "y_top":   np.nanmin(y),                           # highest point on the page (smallest row number)
        "y_bottom": np.nanmax(y),                          # lowest point
        "zero":    np.nanmedian(y),                        # baseline = 0 mV: where the trace spends most time
    }
    print(f"{name:>3}: x {cols[0]}-{cols[-1]}  y {np.nanmin(y):.0f}-{np.nanmax(y):.0f}  zero {np.nanmedian(y):.1f}")


# 2. pixels -> mV -----------------------------
PX_PER_MM = 9.1                                            # ponytail: synthetic pages only; real photos get it from the grid

mv = {}                                                    

for name, y in lines.items():
    if name not in info:                                   # lead had no ink on this page (skipped in step 1)
        continue

    zero = info[name]["zero"]                              # this lead's baseline, in pixels (0 mV sits here)

    height_px = zero - y                                   # pixels ABOVE the baseline (rows grow downward, so zero - y)
    height_mm = height_px / PX_PER_MM                      # pixels -> millimetres of paper
    height_mv = height_mm * mm_to_mv                       # millimetres -> mV  (10 mm = 1 mV)

    mv[name] = height_mv

    print(f"{name:>3}: lowest {np.nanmin(height_mv):+.2f} mV   highest {np.nanmax(height_mv):+.2f} mV")

# 3. columns -> seconds -----------------------------
page_x_start = min(info[name]["x_start"] for name in info)
print("traces start at column", page_x_start)

columns = np.arange(len(next(iter(lines.values()))))      # every page column: 0, 1, 2, ..., W-1

dist_px = columns - page_x_start                           # pixels to the right of the start
dist_mm = dist_px / PX_PER_MM                              # pixels -> millimetres of paper
t = dist_mm * mm_to_s                                      # millimetres -> seconds  (25 mm = 1 s, so 1 mm = 0.04 s)

for name in info:
    t_from = t[info[name]["x_start"]]                      # time of the lead's first column
    t_to = t[info[name]["x_end"]]                          # time of its last column
    print(f"{name:>3}: {t_from:.2f} s  to  {t_to:.2f} s")


# 4. one table: 12 leads on the h5's time grid -----------------------------
FS = 400                                                   # h5 truth: 400 samples per second
N = 4096                                                   # h5 truth: 4096 samples (10.24 s)
H5_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

t_grid = np.arange(N) / FS                                 # 0, 0.0025, 0.005, ... 10.2375 s

table = pd.DataFrame(np.nan, index=t_grid, columns=H5_ORDER)   # all NaN to start: 4096 rows x 12 leads

for name in mv:
    has_ink = ~np.isnan(mv[name])                          # columns where this lead was read
    t_lead = t[has_ink]                                    # their times
    v_lead = mv[name][has_ink]                             # their voltages

    inside = (t_grid >= t_lead[0]) & (t_grid <= t_lead[-1])     # grid times within this lead's window
    table.loc[inside, name] = np.interp(t_grid[inside], t_lead, v_lead)   # value at each grid time


# 5. save -----------------------------
table.index.name = "t_s"                                   # label the time column in the file
table.to_csv(PAGE / "digitized.csv", float_format="%.4f")  # one row per sample: time + 12 leads in mV
print("saved", PAGE / "digitized.csv", table.shape)