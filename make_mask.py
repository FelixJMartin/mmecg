import numpy as np
import pandas as pd
import pmecg
import h5py
from PIL import Image

fs = 400
NEON_GREEN = (57, 255, 20)

f = h5py.File("ptb-xl/ptb_preprocessed.h5", "r")
ecgprep_leads = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
rename = {'DI':'I','DII':'II','DIII':'III','AVR':'aVR','AVL':'aVL','AVF':'aVF'}


def generate_pair(df, template, name, img_dir="unet-src/data/imgs", mask_dir="unet-src/data/masks"):
    '''
    Full styled plot -> training image. Calibration pulse, lead labels, and
    separators are off: none of them have a matching label in the mask, so
    leaving them in only teaches the model false positives on non-trace shapes.
    '''

    # make training image and save in data/img, just clean no labels. 
    configuration = pmecg.template_factory(template, df, leads_map=None)
    image_plotter = pmecg.ECGPlotter(
        show_calibration=False,
        show_leads_labels=False,
        show_separators=False,
    )
    img_path = f"{img_dir}/{name}.png"
    fig_img = image_plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_img.savefig(img_path, dpi=300, bbox_inches="tight")
    #L for greyscale image.
    Image.open(img_path).convert("L").save(img_path)  

    # Bare trace only, no grid/labels/calibration -> source for the mask
    mask_plotter = pmecg.ECGPlotter(
        grid_mode=None,
        show_calibration=False,
        show_leads_labels=False,
        show_separators=False,
        show_time_axis=False,
    )
    mask_path = f"{mask_dir}/{name}_mask.png"
    fig_mask = mask_plotter.plot(df, configuration=configuration, sampling_frequency=fs, show=False)
    fig_mask.savefig(mask_path, dpi=300, bbox_inches="tight")

    # Anything not white is the ECG trace line -> neon-on-black binary mask
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    trace = (mask_rgb != 255).any(axis=-1)
    mask_img = np.zeros((*trace.shape, 3), dtype=np.uint8)
    mask_img[trace] = NEON_GREEN
    Image.fromarray(mask_img).save(mask_path)




for i in range(0, 1000):
    df = pd.DataFrame(f["tracings"][i], columns=[rename.get(l, l) for l in ecgprep_leads])
    generate_pair(df, "1x3", f"example_{i}")

for i in range(1000, 1050):
    df = pd.DataFrame(f["tracings"][i], columns=[rename.get(l, l) for l in ecgprep_leads])
    generate_pair(df, "1x3", f"example_{i}", img_dir="unet-src/data/test_imgs", mask_dir="unet-src/data/test_masks")
