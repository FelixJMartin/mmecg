import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmecg
from PIL import Image

LEADS = ["I", "II", "V2"]  # top-to-bottom row order for the "1x3" template
DURATION_S = 4096 / 400  # matches the original preprocessing (new_len / new_freq)


def replot(pred_path, out_dir):
    name = os.path.splitext(os.path.basename(pred_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    mask = np.asarray(Image.open(pred_path).convert("RGB"))
    trace = mask.any(axis=-1)  # True where predicted foreground (non-black)

    h, w = trace.shape
    band_h = h // len(LEADS)

    signal = {}
    for i, lead in enumerate(LEADS):
        band = trace[i * band_h:(i + 1) * band_h, :]
        centroid = np.full(w, np.nan)
        for x in range(w):
            ys = np.where(band[:, x])[0]
            if len(ys):
                centroid[x] = ys.mean()

        s = pd.Series(centroid)
        predicted = s.notna()
        s = s.rolling(5, center=True, min_periods=1).mean()  # smooth pixel-jitter
        s[~predicted] = np.nan  # keep true gaps as gaps, don't let smoothing bridge them
        baseline = s.median()
        voltage = (baseline - s) / s.std() * 0.5  # rough, unitless-ish mV scale
        signal[lead] = voltage.to_numpy()

    df = pd.DataFrame(signal)
    fs = w / DURATION_S

    configuration = pmecg.template_factory("1x3", df, leads_map=None)
    plotter = pmecg.ECGPlotter(show_calibration=False, show_leads_labels=False, show_separators=False)
    fig = plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    plot_path = f"{out_dir}/{name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Row-density histogram: count trace (black, non-grid) pixels per row, so the
    # 3 leads show up as dense bands separated by near-empty valleys -> useful
    # later for splitting the plot into 3 separate per-lead crops.
    plot_rgb = np.asarray(Image.open(plot_path).convert("RGB"))
    is_trace = plot_rgb.sum(axis=-1) < 200  # near-black; excludes white bg and pink grid
    row_density = is_trace.sum(axis=1)

    fig2, (ax_img, ax_hist) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [10, 1]}, sharey=True
    )
    ax_img.imshow(plot_rgb, aspect="auto")
    ax_img.axis("off")
    ax_hist.barh(np.arange(len(row_density)), row_density, color="black", height=1)
    ax_hist.set_xlabel("trace px / row")
    fig2.tight_layout()
    # tight_layout can give ax_hist a different box height than ax_img (it has an
    # xlabel/ticks eating vertical space that ax_img doesn't) -> force them equal
    # so row 0 and row h line up exactly top-to-bottom between the two panels.
    pos_img = ax_img.get_position()
    pos_hist = ax_hist.get_position()
    ax_hist.set_position([pos_hist.x0, pos_img.y0, pos_hist.width, pos_img.height])
    fig2.savefig(f"{out_dir}/{name}_density.png", dpi=300)
    plt.close(fig2)


if __name__ == "__main__":
    for pred_path in sorted(glob.glob("test_predictions/*.png")):
        replot(pred_path, "test_predictions_replotted")
