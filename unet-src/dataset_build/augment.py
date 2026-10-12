"""Step 2 of 2: turn the clean pages into realistic phone photos and scans, one augmentation per function.

Each page goes through 6 personas (see PERSONAS) -> 6 training pairs, each with its own label
(same truth, moved/hidden the same way as the image). Step 1 is make_multiclass.py.


Order follows a real photo: things on the paper -> paper on a desk (or in the scanner) -> the camera -> anonymisation.
https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator
Wrinkle textures are copied from the kit; handwriting strokes come from make_handwriting_bank.py.

"""
import argparse
import io
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from make_multiclass import save_labels                    

# -------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent                     # .../unet-src/dataset_build
ASSETS = HERE / "assets"
SRC = HERE.parent / "data" / "pages"                       # clean pages from make_multiclass.py
OUT = HERE.parent / "data"                                 # imgs/, labels/, index.csv for train.py
TEXTURES = ASSETS / "wrinkles"                             # crumpled-paper photos from the kit's wrinkles-dataset
BACKGROUNDS = ASSETS / "backgrounds"                       # your own desk/table photos
HANDWRITING = ASSETS / "handwriting_bank"                  # pen strokes from the kit's handwriting RNN
FRAME = ASSETS / "frames" / "arc_frame.png"                # the ARC "scanned ECG" page some real files come in


# -------------------------------------------------------- settings

FRAME_BOX = (0.06, 0.01, 1.0, 0.99)                        # its white content area: top, left, bottom, right (fractions)
PHOTO_H, PHOTO_W = 2496, 3328                              # every augmented photo: ~4:3 like a phone camera, and both
                                                           # multiples of 16 (the U-Net halves 4x), so never any padding
INKS = [(25, 40, 140), (30, 30, 30), (160, 25, 25)]        # blue ballpoint, black, red pen
SEED = 2

#random things later
rng = np.random.default_rng(SEED)                          # re-seeded per page in make_personas()


# ---------------------------------------------------------------- load and look
def load(name, src=SRC):
    img = np.array(Image.open(src / "imgs" / f"{name}.png").convert("RGB"))   # (H, W, 3)
    lbl = np.array(Image.open(src / "labels" / f"{name}.png"))               # (H, W), class ids 0..12
    return img, lbl


def show(img, lbl, title):
    #two plots next to eachoter
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(img)
    ax[0].set_title(title)
    ax[1].imshow(img)
    ax[1].imshow(np.ma.masked_equal(lbl, 0), cmap="tab20", interpolation="nearest")   # label on top, background hidden
    ax[1].set_title("label on top")
    for a in ax:
        a.axis("off")
    plt.show()



# ---------------------------------------------------------------- 1. handwritten notes
def handwriting(img, lbl, low=1, high=5):
    # Notes from the kit's handwriting RNN, in a random pen colour, size and tilt, anywhere on the page.
    # Label unchanged: notes are background, and a trace under the ink is still that lead.
    page = Image.fromarray(img)
    h, w = lbl.shape
    bank = sorted(HANDWRITING.glob("*.npy"))

    for _ in range(rng.integers(low, high + 1)):                         # 1..5 notes per page
        pts = np.load(rng.choice(bank))                                  # (N, 3): x, y (y down, like an image), pen_up
        ink = INKS[rng.integers(len(INKS))]
        alpha = int(255 * rng.uniform(0.7, 0.95))                        # pen ink is never fully opaque
        pen = int(rng.integers(3, 7))                                    # stroke width in px

        # scale so the writing is ~4..9 mm tall at 300 dpi
        xy = pts[:, :2] - pts[:, :2].min(axis=0)
        xy = xy * rng.uniform(50, 110) / max(xy[:, 1].max(), 1) + 10     # + 10 px margin

        # draw stroke by stroke on a transparent layer, so it can be tilted before going on the page
        layer = Image.new("RGBA", (int(xy[:, 0].max()) + 20, int(xy[:, 1].max()) + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        ends = np.flatnonzero(pts[:, 2] == 1)                            # last point of each stroke
        for start, end in zip(np.r_[0, ends[:-1] + 1], ends + 1):
            stroke = [tuple(p) for p in xy[start:end]]
            if len(stroke) > 1:
                draw.line(stroke, fill=ink + (alpha,), width=pen, joint="curve")
        layer = layer.rotate(rng.uniform(-15, 15), expand=True, resample=Image.BICUBIC)

        x = rng.integers(0, max(1, w - layer.width))
        y = rng.integers(0, max(1, h - layer.height))
        page.paste(layer, (int(x), int(y)), layer)                       # layer's alpha = how much ink shows

    return np.array(page), lbl


# ---------------------------------------------------------------- 2. wrinkles
def wrinkles(img, lbl, patch=250):
    # One random 250 px patch of a crumpled-paper photo, stretched over the page and overlay-blended
    
    path = rng.choice(sorted(TEXTURES.glob("*.jpg")))
    tex = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255     # grey, 0..1
    s = min(patch, *tex.shape)                                                # a few textures are small
    y = rng.integers(0, tex.shape[0] - s + 1)
    x = rng.integers(0, tex.shape[1] - s + 1)
    tex = tex[y:y + s, x:x + s]

    # stretch over the page, shift brightness so the mean is 0.4 (kit's number; higher = lighter)
    h, w = lbl.shape
    tex = np.array(Image.fromarray(tex).resize((w, h), Image.BILINEAR))
    tex = (tex - (tex.mean() - 0.4))[..., None]                               # (H, W, 1): same on R, G, B

    # overlay blend: dark texture darkens the page, bright texture (> 0.6) lightens it
    page = img.astype(np.float32) / 255
    out = np.where(tex > 0.6, 1 - 2 * (1 - page) * (1 - tex), 2 * page * tex)
    return (255 * out).clip(0, 255).astype(np.uint8), lbl


# ---------------------------------------------------------------- 3. creases
def creases(img, lbl, n_horizontal=2, n_vertical=3, max_angle=3, width=25, low=0.05, high=0.15):
    # A real fold is a soft ridge: one side catches the light, the other is in shadow.
    h, w = lbl.shape
    yy, xx = np.ogrid[0:h, 0:w]                                    # pixel coordinates, broadcast
    folds = np.ones((h, w), np.float32)                            # 1 = untouched paper
    tilt = np.tan(np.deg2rad(rng.uniform(-max_angle, max_angle)))  # whole sheet slightly skewed
    norm = np.sqrt(1 + tilt ** 2)

    def fold(dist):
        # dist = signed distance to the fold line in px. Derivative-of-gaussian:
        # +strength on one side, -strength on the other, back to 1 a few widths away.
        z = dist / width
        strength = rng.uniform(low, high) * rng.choice([-1, 1])  # which side is lit is random
        return strength * z * np.exp(0.5 - 0.5 * z ** 2)        # peak = +-strength at z = +-1

    for i in range(n_horizontal):
        y0 = (i + 1) * h / (n_horizontal + 1) + rng.normal(0, 0.02 * h)   # roughly evenly spaced, hand-folded
        folds += fold((yy - y0 - tilt * xx) / norm)
    
    for i in range(n_vertical):
        x0 = (i + 1) * w / (n_vertical + 1) + rng.normal(0, 0.02 * w)
        folds += fold((xx - x0 + tilt * yy) / norm)

    return np.clip(img * folds[..., None], 0, 255).astype(np.uint8), lbl



# ---------------------------------------------------------------- 4. background
def background(img, lbl, low=0.6, high=0.95, shift=0.05, max_tilt=2.0):     # page fills 60-95 %: real photos vary in scale
    # The page lying flat on a desk, roughly in the middle and up to ~2 degrees crooked; desk photo
    # everywhere else (label 0). Pages come in different sizes; the result is always PHOTO_H x PHOTO_W.
    h, w = PHOTO_H, PHOTO_W

    # a random desk photo: landscape, maybe mirrored, cropped to 4:3 and resized to the photo size
    path = rng.choice(sorted(p for p in BACKGROUNDS.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png")))
    bg = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if bg.height > bg.width:
        bg = bg.rotate(90, expand=True)                                 # your photos are portrait
    if rng.random() < 0.5:
        bg = ImageOps.mirror(bg)
    cw = int(min(bg.width, bg.height * w / h) * rng.uniform(0.95, 1.0))  # use 95..100 % of the desk photo
    ch = int(cw * h / w)
    x0 = rng.integers(0, bg.width - cw + 1)
    y0 = rng.integers(0, bg.height - ch + 1)
    bg = np.array(bg.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BILINEAR))

    # a couple of degrees crooked: the corners this opens up are transparent (alpha 0) and label 0
    deg = rng.uniform(-max_tilt, max_tilt)
    rgba = Image.fromarray(img).convert("RGBA").rotate(deg, resample=Image.BILINEAR, expand=True)
    lbl = np.array(Image.fromarray(lbl).rotate(deg, resample=Image.NEAREST, expand=True, fillcolor=0))

    # shrink the page, keeping its shape, so it fills low..high of the photo (NEAREST: no invented class ids)
    fit = rng.uniform(low, high) * min(w / lbl.shape[1], h / lbl.shape[0])
    pw, ph = int(lbl.shape[1] * fit), int(lbl.shape[0] * fit)
    page = np.array(rgba.resize((pw, ph), Image.BICUBIC)).astype(np.float32)
    small = np.array(Image.fromarray(lbl).resize((pw, ph), Image.NEAREST))

    # roughly centred, nudged up to +-shift, kept inside the frame; desk shows where the page isn't
    x = int(np.clip((w - pw) / 2 + rng.uniform(-shift, shift) * w, 0, w - pw))
    y = int(np.clip((h - ph) / 2 + rng.uniform(-shift, shift) * h, 0, h - ph))
    out, out_lbl = bg.copy(), np.zeros((h, w), np.uint8)
    paper = page[..., 3:] / 255                                        # 1 on the page, 0 in the opened corners
    out[y:y + ph, x:x + pw] = (page[..., :3] * paper + out[y:y + ph, x:x + pw] * (1 - paper)).astype(np.uint8)
    out_lbl[y:y + ph, x:x + pw] = small
    return out, out_lbl



# ---------------------------------------------------------------- 4b. scan frame (instead of background)
def scan_frame(img, lbl, low=0.92, high=0.99, max_skew=1.0):
    # A flatbed scan inside the ARC "scanned ECG" page: under the header bar, flat, at most ~1 degree
    # crooked, no desk. Result is PHOTO_H x PHOTO_W like background(); label 0 outside the page.
    h, w = PHOTO_H, PHOTO_W
    out = np.array(Image.open(FRAME).convert("RGB").resize((w, h), Image.BICUBIC))
    top, left, bottom, right = int(FRAME_BOX[0] * h), int(FRAME_BOX[1] * w), int(FRAME_BOX[2] * h), int(FRAME_BOX[3] * w)
    bh, bw = bottom - top, right - left                                  # the white area the page goes in

    # scanners white-balance: whatever colour the paper is, it comes out near-white
    paper = np.median(img.reshape(-1, 3), axis=0)                      # most pixels are paper -> its colour
    img = np.clip(img * (rng.uniform(245, 255) / paper), 0, 255).astype(np.uint8)

    # page slightly crooked on the glass; scanner lid = white around it, label 0
    deg = rng.uniform(-max_skew, max_skew)
    img = np.array(Image.fromarray(img).rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)))
    lbl = np.array(Image.fromarray(lbl).rotate(deg, resample=Image.NEAREST, expand=True, fillcolor=0))

    # shrink to fit the white area, keeping its shape; centred left-right, near the top like the real scans
    fit = rng.uniform(low, high) * min(bw / lbl.shape[1], bh / lbl.shape[0])
    pw, ph = int(lbl.shape[1] * fit), int(lbl.shape[0] * fit)
    page = np.array(Image.fromarray(img).resize((pw, ph), Image.BICUBIC))
    small = np.array(Image.fromarray(lbl).resize((pw, ph), Image.NEAREST))
    x = left + (bw - pw) // 2
    y = top + int(rng.uniform(0, 0.3) * (bh - ph))

    out_lbl = np.zeros((h, w), np.uint8)
    out[y:y + ph, x:x + pw] = page
    out_lbl[y:y + ph, x:x + pw] = small
    return out, out_lbl



# ---------------------------------------------------------------- 5. perspective (not used: real photos are fairly straight)
def perspective(img, lbl, tilt=0.06):
    # The phone held at an angle: the whole photo (page and desk) warped by moving its 4 corners up to 6 %.
    h, w = lbl.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])                 # top-left, top-right, bottom-right, bottom-left
    dst = src + rng.uniform(-tilt, tilt, (4, 2)) * [w, h]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))       # 3x3: old pixel -> new pixel

    # gaps at the edges: image repeats its edge pixels (the desk), label = 0
    img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lbl = cv2.warpPerspective(lbl, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, lbl


# ---------------------------------------------------------------- 6. shadow (not in the kit)
def shadow(img, lbl, low=0.2, high=0.5, softness=0.1):
    # A soft dark band across the photo from a random direction (a hand, the phone, a lamp).
    h, w = lbl.shape
    yy, xx = np.ogrid[0:h, 0:w]
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (xx / w - 0.5) * np.cos(angle) + (yy / h - 0.5) * np.sin(angle)   # -0.7..0.7 along that direction
    dark = 1 / (1 + np.exp(-(ramp - rng.uniform(-0.3, 0.3)) / softness))     # 0 lit .. 1 shadow, soft edge
    return np.clip(img * (1 - rng.uniform(low, high) * dark)[..., None], 0, 255).astype(np.uint8), lbl


# ---------------------------------------------------------------- 7. noise
def noise(img, lbl, sigma=10):
    # Camera sensor grain, same on R, G, B, like the kit's AdditiveGaussianNoise.
    grain = rng.normal(0, sigma, img.shape[:2])[..., None]
    return np.clip(img + grain, 0, 255).astype(np.uint8), lbl


# ---------------------------------------------------------------- 7b. quality (image only)
def quality(img, lbl, low=1000, high=PHOTO_W):
    # Real files are often small, compressed JPEGs that get enlarged (the harddrive PDFs hold images
    # 1024-4032 px wide; B003 is 1024). So: shrink to a random width, save as a rough JPEG, enlarge
    # back to the photo size. high=PHOTO_W means sometimes no shrinking at all. The label stays sharp:
    # the trace is still exactly where it was, the network has to find it in the blurrier image.
    h, w = lbl.shape
    small_w = int(rng.uniform(low, high))
    small = Image.fromarray(img).resize((small_w, round(h * small_w / w)), Image.BILINEAR)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=int(rng.integers(50, 90)))    # 50 = blocky, 90 = nearly clean
    return np.array(Image.open(buffer).convert("RGB").resize((w, h), Image.BICUBIC)), lbl



# ---------------------------------------------------------------- 8. redaction
def redaction(img, lbl, low_w=0.15, high_w=0.35, low_h=0.03, high_h=0.08):
    # The patient's name/ID blacked out on the finished photo: straight on the image, pure black,
    # in the header band above the first trace row, over the page. Under the box the trace is gone: label 0.
    h, w = lbl.shape
    bw = int(rng.uniform(low_w, high_w) * w)
    bh = int(rng.uniform(low_h, high_h) * h)

    rows = np.flatnonzero((lbl > 0).any(axis=1))             # photo rows with a trace
    cols = np.flatnonzero((lbl > 0).any(axis=0))             # photo columns with a trace = the page's width
    top = rows[0] if len(rows) else h // 4                   # first trace row = just below the header
    left, right = (cols[0], cols[-1]) if len(cols) else (0, w)
    y = rng.integers(max(0, top - 3 * bh), max(1, top))      # box ends somewhere above or just into it
    x = rng.integers(left, max(left + 1, right - bw))

    img, lbl = img.copy(), lbl.copy()
    img[y:y + bh, x:x + bw] = 0
    lbl[y:y + bh, x:x + bw] = 0
    return img, lbl


# ================================================================ personas: one kind of real image each
# Fixed list of steps per persona; how strong each step is stays random. Add a persona = add a line.
PERSONAS = {
    # quality last: the finished file is what gets shrunk and compressed
    "scan_clean":     [scan_frame, partial(noise, sigma=4), quality],
    "scan_marked":    [handwriting, scan_frame, partial(noise, sigma=4), redaction, quality],
    "photo_clean":    [background, noise, quality],          # no perspective(): real photos are fairly straight,
    "photo_folded":   [handwriting, creases, background, shadow, noise, quality],   # background() tilts up to ~2 deg
    "photo_wrinkled": [handwriting, wrinkles, creases, background, shadow, noise, quality],
    "photo_redacted": [handwriting, creases, background, shadow, noise, redaction, quality],
}


def make_personas(name, src=SRC):
    """One clean page -> {persona: (img, lbl)}. Re-seeds rng from the page name: same output whatever the worker split."""
    global rng
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
    img0, lbl0 = load(name, src)
    results = {}
    for persona, steps in PERSONAS.items():
        img, lbl = img0, lbl0
        for step in steps:
            img, lbl = step(img, lbl)
        results[persona] = (img, lbl)
    return results


def augment_page(page):
    """Worker: one page -> 6 pairs on disk; returns their index.csv rows."""
    name, template, src, img_dir, lbl_dir = page     # folders passed along: workers can't see main's variables
    rows = []
    for persona, (img, lbl) in make_personas(name, src).items():
        out_name = f"{name}_{persona}"
        # image as JPEG (real files are JPEGs: random quality adds realistic compression), label as palette PNG
        Image.fromarray(img).save(img_dir / f"{out_name}.jpg", quality=int(rng.integers(75, 96)))
        save_labels(lbl, lbl_dir / f"{out_name}.png")
        rows.append(f"{out_name},{name},{template},{persona}\n")
    return rows


# ---------------------------------------------------------------- every page x every persona -> data/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="CPU processes augmenting in parallel")
    parser.add_argument("--preview", metavar="PAGE", help="plot this page's personas instead of building")
    parser.add_argument("--test", action="store_true", help="data/test_pages -> data/test_imgs, test_labels, test_index.csv")
    args = parser.parse_args()

    # training or test set: where the clean pages come from, where the pairs go (test = what predict.py --score reads)
    data = HERE.parent / "data"
    if args.test:
        src, img_dir, lbl_dir, index_csv = data / "test_pages", data / "test_imgs", data / "test_labels", data / "test_index.csv"
    else:
        src, img_dir, lbl_dir, index_csv = SRC, OUT / "imgs", OUT / "labels", OUT / "index.csv"

    if args.preview:
        for persona, (img, lbl) in make_personas(args.preview, src).items():
            show(img, lbl, f"{args.preview}  {persona}")
        raise SystemExit

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    pages = [(name, template, src, img_dir, lbl_dir)
             for name, template in pd.read_csv(src / "index.csv")[["name", "template"]].itertuples(index=False, name=None)]

    with Pool(args.workers) as pool, open(index_csv, "w") as index:
        index.write("name,page,template,persona\n")
        for k, rows in enumerate(pool.imap(augment_page, pages), 1):
            index.writelines(rows)
            print(f"{k}/{len(pages)}  {rows[0].split(',')[1]}", flush=True)

    print(f"wrote {len(pages) * len(PERSONAS)} pairs to {img_dir} + {lbl_dir}")
