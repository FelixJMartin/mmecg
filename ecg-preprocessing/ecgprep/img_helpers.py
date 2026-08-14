import numpy as np

# Estimate number of courser grid
n_grid_h = 24
n_grid_w = 50

# convert corser grid to mm
grid_to_mm = 5

# convert corser grid to mm
mm_to_mv = 0.5 / 5
mm_to_s= 0.2 / 5

# Start
duration_short = 12.5
duration_long = 50
n_rows = 4
n_cols = 4


def bounding_boxes(height, width, n_rows = 4, n_cols = 4, lead_positions=None, h_offset=0, w_offset=0):
    if lead_positions is None:
        lead_positions = {
            'DI': (0, 0),
            'DII': (1, 0),
            'DIII': (2, 0),
            'AVR': (0, 1),
            'AVL': (1, 1),
            'AVF': (2, 1),
            'V1': (0, 2),
            'V2': (1, 2),
            'V3': (2, 2),
            'V4': (0, 3),
            'V5': (1, 3),
            'V6': (2, 3),
            'long DII': (3, 0)
        }
    tlbr_boxes = {}

    for lead_name in lead_positions.keys():
        # Save leads using cv2
        positions = lead_positions[lead_name]
        block_h = height // n_rows
        block_w = width // n_cols
        h_start = positions[0] * block_h + h_offset
        h_end = height - 1 + h_offset if 'long' in lead_name else h_start + block_h
        w_start = positions[1] * block_w + w_offset
        w_end = width - 1 + w_offset if 'long' in lead_name else w_start + block_w

        tlbr_boxes[lead_name] = [h_start, w_start, h_end, w_end]

    return tlbr_boxes


class ScaleFromPixels():
    def __init__(self, height, width):
        self.pixels_per_mm_h = height / (n_grid_h * grid_to_mm)
        self.pixels_per_mm_w = width / (n_grid_w * grid_to_mm)

    def __call__(self, signal):
        y = signal * mm_to_mv / self.pixels_per_mm_h
        sample_frequency = self.pixels_per_mm_w / (mm_to_s)
        return y, sample_frequency

def signal_to_ecg(signal, pixels_per_mm_h, pixels_per_mm_w):
    y = signal * mm_to_mv / pixels_per_mm_h
    sample_frequency = pixels_per_mm_w / (mm_to_s)
    return y, sample_frequency

def vectorize_single_lead(lead_img, mask):
    H, W = mask.shape[0], mask.shape[1]
    signal_pixels = np.zeros(W)
    for i in range(W):
        ys = np.nonzero(mask[:, i])[0]  # y-locations of the signal
        if len(ys) > 0:
            v = 255 - lead_img[ys, i]
            signal_pixels[i] = H - np.mean(ys[v==max(v)])  # take mean pixel coordinate
        else:
            signal_pixels[i] = np.nan
    # might be good to do something smarter to potentially remove some artifacts

    return signal_pixels


def vectorize(img, mask, tlbr_boxes, scale_from_pixels):
    # Find big box
    BB = np.array(list(tlbr_boxes.values()))
    start_w_pixel = min(BB[:, 1])
    end_w_pixel = max(BB[:, 3])

    # Define signal matrix
    X = np.zeros((len(tlbr_boxes),  end_w_pixel  - start_w_pixel))
    for i, (t, l, b, r) in enumerate(tlbr_boxes.values()):
        X[i, l-start_w_pixel:r-start_w_pixel] = vectorize_single_lead(img[t:b, l:r],  mask[t:b, l:r] )
        i+=1

    ecg, sample_rate = scale_from_pixels(X)
    leads = tlbr_boxes.keys()

    return ecg, sample_rate, leads
