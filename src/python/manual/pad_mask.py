import sys
import numpy as np
import pandas as pd
from tqdm import trange
from skimage.transform import resize

def main():
    npy_name = sys.argv[1]
    csv_name = sys.argv[2]
    size_pad = int(sys.argv[3])

    npy_data = np.load(npy_name)
    csv_data = pd.read_csv(csv_name)
    n = npy_data.shape[0]

    padded_output = np.zeros(shape=(n, size_pad, size_pad, 3), dtype=npy_data.dtype)
    center = size_pad // 2
    for i in trange(n):
        w, h = csv_data.loc[i, ["Height", "Width"]] 
        h, w = int(h), int(w)
        
        original_rgb = resize(npy_data[i, :, :, :], (w, h, 3), preserve_range=True).astype("uint8")
        # top left pixel 
        tl_x = center - w // 2
        tl_y = center - h // 2
        if w <= size_pad and h <= size_pad:
            padded_output[i, tl_x:tl_x+w, tl_y:tl_y+h, :] = original_rgb
        elif w > size_pad and h <= size_pad:
            padded_output[i, :, tl_y:tl_y+h, :] = original_rgb[0:size_pad,:]
        elif w <= size_pad and h > size_pad:
            padded_output[i, tl_x:tl_x+w, :, :] = original_rgb[:, 0:size_pad]
        elif w > size_pad and h > size_pad:
            padded_output[i, :, :, :] = original_rgb[0:size_pad, 0:size_pad]

    np.save(npy_name.replace(".npy", "_paddedmask.npy"), padded_output)

if __name__ == "__main__":
    main()
