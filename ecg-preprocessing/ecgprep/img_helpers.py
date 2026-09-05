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

    #for the full width
    for i in range(W):
        #walk all rows, and just that colomn and see what the y value is. 
        ys = np.nonzero(mask[:, i])[0]  # y-locations of the signal
        if len(ys) > 0:
            v = 255 - lead_img[ys, i] #darkness score
            signal_pixels[i] = H - np.mean(ys[v==max(v)])  # take mean pixel coordinate 
        # put it to nan if not larger than 0
        else:
            signal_pixels[i] = np.nan
    # might be good to do something smarter to potentially remove some artifacts

    return signal_pixels


def vectorize(img, mask, tlbr_boxes, scale_from_pixels):
    '''
    Inputs:
      img              - full-page grayscale pixel array (H, W), NOT cropped to one lead.
      mask             - same shape as img, nonzero wherever a pixel is predicted/labeled trace.
      tlbr_boxes       - {lead_name: [top, left, bottom, right]} pixel box per lead,
                          e.g. from bounding_boxes(). Boxes can start/end at different
                          columns (a staggered layout), which is why step 1 below has
                          to work out the combined horizontal span across all of them.

                          Example shapes, all for a 300 (H) x 400 (W) page:

                          1 lead, spanning the whole page (e.g. the vectorize_playground.py sandbox):
                            {"lead": [0, 0, 300, 400]}

                          2 leads, stacked vertically (each half the height, full width):
                            {"I":  [0,   0, 150, 400],
                             "II": [150, 0, 300, 400]}

                          12 leads, a 3-row x 4-col grid (each box a quarter-width, third-height
                          slice; this is what bounding_boxes()'s default layout produces):
                            {"I":   [0,   0,   100, 100], "aVR": [0,   100, 100, 200], "V1": [0,   200, 100, 300], "V4": [0,   300, 100, 400],
                             "II":  [100, 0,   200, 100], "aVL": [100, 100, 200, 200], "V2": [100, 200, 200, 300], "V5": [100, 300, 200, 400],
                             "III": [200, 0,   300, 100], "aVF": [200, 100, 300, 200], "V3": [200, 200, 300, 300], "V6": [200, 300, 300, 400]}
                          In all three of these, every box's left/right lines up column-wise across
                          rows, so start_w_pixel/end_w_pixel just come out to 0/400. A rhythm-strip
                          box (like bounding_boxes()'s "long DII", spanning the full width in one
                          extra row) is the case that's genuinely staggered relative to the others.
      scale_from_pixels- callable: raw pixel-position matrix -> (signal_in_real_units, sample_rate).

    Output:
      ecg        - (num_leads, total_width) array, one row per lead, in real units
                   (whatever scale_from_pixels returns, e.g. mV). NaN where no trace
                   pixel was found in mask for that column.
      sample_rate- scalar, from scale_from_pixels.
      leads      - lead names, in the same row order as ecg.
    '''


    # Step 1: work out the overall horizontal span the output needs to cover.
   
    BB = np.array(list(tlbr_boxes.values()))  
    # list all values in the tlbr boxes we inputed. 
    # 3×4 grid example, [[0,200,100,300], ..., [200,300,300,400]]
    # list turn them into a running full list instead. 
    # array makes a 12x4 grid, 12 tblr combos, and they are across 4 colomns then. 

    start_w_pixel = min(BB[:, 1])  # leftmost column across all boxes
    end_w_pixel = max(BB[:, 3])    # rightmost column across all[] boxes

    # Step 2: allocate the output matrix -- one row per lead, one column per pixel
    # of that combined horizontal span.
    # makes the 12 x 400 matrix that will recieve values
    X = np.zeros((len(tlbr_boxes),  end_w_pixel  - start_w_pixel))

    # Step 3: fill in one lead (one row of X) at a time.
    for i, (t, l, b, r) in enumerate(tlbr_boxes.values()):
        # so this becomes I, II and realted tlbr box

        # Crop img/mask down to just this lead's box, extract one height-value per
        # column (vectorize_single_lead, the per-pixel-column logic), and write it
        # into X at this lead's correct horizontal offset (l-start_w_pixel is where
        # this box's left edge falls within the combined span from step 1).
        X[i, l-start_w_pixel:r-start_w_pixel] = vectorize_single_lead(img[t:b, l:r],  mask[t:b, l:r] )
        

    # Step 4: convert the whole matrix from raw pixel positions into real units
    # (e.g. mV) and derive the sample rate, using the page's known physical
    # calibration (grid size, paper speed) baked into scale_from_pixels.
    ecg, sample_rate = scale_from_pixels(X)
    leads = tlbr_boxes.keys()  # same order as the rows of ecg/X

    return ecg, sample_rate, leads
