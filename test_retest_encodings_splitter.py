#%%
import h5py

#%%
with h5py.File('testData/Images.h5', 'r') as f:
    ds_tot = ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]
    img = [[] for _ in range(8)]
    for name in ds_tot:
        if name in f:
            ds = f[name]
            print(f"Dimensions of dataset: '{name}': {ds.shape}")
        img[0]
#%%
with h5py.File('testData/Images.h5', 'r') as f:
    img1 = []; img2 = []
    ds_names = ['IMAGE', 'IMAGE_MAG', 'IMAGE_PHASE']
    ds0 = ds_names[0]
    if ds0 in f:
        ds = f[ds0]
        shape = ds.shape[1]
        for i in range(shape):
            data_chunk = ds[i:min(i + 41, 8)] # 41 comes from 348 slices / 8 encodings = 41 slices/encoding
            halfoutput_h5 = f"split_{img1 if i == 0 or i == 5 or i == 2 or i ==7 else img2}{i}"
            with h5py.File(halfoutput_h5, 'r') as fhalfout:
                fhalfout.create_dataset('testData/Images.h5', data = data_chunk)
            print(f"Saved output {halfoutput_h5} with shape {data_chunk.shape}")
        #for i in range(8):
        #    img[i].append(ds0[:[i]:::])
#%%
