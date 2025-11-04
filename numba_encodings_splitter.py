#%%
import h5py
import numpy as np
from numba import njit

#%% Load everything at once
with h5py.File('/mounts/data/analyses/bawad2/testData/Images.h5', 'r') as hf:
    temp_img = hf['IMAGE'][:]  # Read the full dataset into memory (fast if fits in RAM)

print(f"Loaded dataset with shape: {temp_img.shape}")  # e.g. (N, 8, H, W, C)

#%% Numba-accelerated rearrangement
@njit(parallel=True)
def rearrange_images(img):
    encs = [img[:, i, :, :, :] for i in range(img.shape[1])]
    # Equivalent of your specific rearrangement pattern
    img1 = np.concatenate((encs[0], encs[5], encs[2], encs[7]), axis=1)
    img2 = np.concatenate((encs[4], encs[1], encs[6], encs[3]), axis=1)
    return img1, img2

img1, img2 = rearrange_images(temp_img)
print(img1.shape, img2.shape)

#%% Write output
with h5py.File('/mounts/data/analyses/bawad2/testData/Images1.h5', 'w') as hf:
    hf.create_dataset("IMAGE", data=img1)
