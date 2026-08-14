from ecgprep.img_helpers import vectorize_single_lead, signal_to_ecg, bounding_boxes, ScaleFromPixels, vectorize
from ecgprep.read_ecg import read_ecg
from ecgprep.preprocess import preprocess_ecg
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


PLOT = True


# Test vectorize_single_lead
def test_vectorize_single_lead():
    ecg_from_signal, freq_signal, lead_names = read_ecg('ptbxl/00001_hr', format='wfdb')

    lead_src_img = cv2.imread('img/ptbxl-individual-leads/long DII.png')
    mask_src_img = cv2.imread('img/ptbxl-individual-leads-mask/long DII.png')

    lead_img = cv2.cvtColor(lead_src_img, cv2.COLOR_RGB2GRAY)
    mask_img = cv2.cvtColor(mask_src_img, cv2.COLOR_RGB2GRAY)>240
    signal_from_image = vectorize_single_lead(lead_img, mask_img)

    assert signal_from_image .shape[0] == lead_img.shape[1]


    ecg_from_image, freq_img_extracted = signal_to_ecg(signal_from_image, 4, 4)


    ecg_signal, sample_frequency, lead_names = preprocess_ecg(ecg_from_signal, freq_signal, lead_names, new_freq=freq_img_extracted, remove_baseline=True)

    # remove the mean
    ecg_1 = ecg_from_image - np.mean(ecg_from_image)
    ecg_2 = ecg_signal[1, :] - np.mean(ecg_signal[1, :])

    # make both same shape
    l1 = ecg_1.shape
    l2 = ecg_2.shape
    l = min(l1, l2)[0]
    ecg_1 = ecg_1[:l]
    ecg_2 = ecg_2[:l]

    # compute mean_absolute error:
    error = np.mean(np.abs(ecg_1 - ecg_2))
    assert error < 0.05

    if PLOT:
        plt.plot(ecg_1)
        plt.plot(ecg_2)
        plt.show()


def test_bounding_boxes():
    # Read image
    img = cv2.imread("img/ptbxl.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    border = 10
    height, width = img.shape
    tlbr_boxes = bounding_boxes(height - 2 * border, width - 2 * border, h_offset=border, w_offset=border)

    if PLOT:
        # Overlay boxes (assuming tlbr_boxes are (top, left, bottom, right))
        pil_img = Image.fromarray(img)
        overlay = pil_img.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for name, (t, l, b, r) in zip(tlbr_boxes.keys(), tlbr_boxes.values()):
            draw.rectangle([l, t, r, b], outline=(255, 0, 0), width=2)
            draw.text((l+5, t+ 5), name, fill="red")
        overlay.show()

def test_vectorize():
    # Read image
    img = cv2.imread("img/ptbxl.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get basic dimensions
    border = 10
    height, width = img.shape

    # Get mask
    img_clean = cv2.imread("img/ptbxl-clean.png")
    img_clean  = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)
    mask = img_clean < 240


    # Compute bounding boxes
    tlbr_boxes = bounding_boxes(height - 2 * border, width - 2 * border, h_offset=border, w_offset=border)

    # Get scaler
    scale_from_pixels = ScaleFromPixels(height - 2 * border, width - 2 * border)

    # Put all together
    ecg, sample_rate, leads = vectorize(img, mask, tlbr_boxes, scale_from_pixels)

    print(ecg, sample_rate, leads)



if  __name__ == '__main__':
    pass

