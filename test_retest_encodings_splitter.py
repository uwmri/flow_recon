#%%
import h5py
import numpy as np

#%% Print out shape of .h5 file
with h5py.File('/mounts/data/analyses/bawad2/testData/Images.h5', 'r') as f:
    ds_tot = ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]
    img = [[] for _ in range(8)]
    for name in ds_tot:
        if name in f:
            ds = f[name]
            print(f"Dimensions of dataset: '{name}': {ds.shape}")
        img[0]

        #for i in range(8):
        #    img[i].append(ds0[:[i]:::])
#%% Splitting the files into their own components
with h5py.File('/mounts/data/analyses/bawad2/testData/Images.h5', 'r') as hf:
    temp_img = hf['IMAGE']
    # temp_mag = hf['IMAGE_MAG']
    # temp_phase = hf['IMAGE_PHASE']
    
    encs=[]
    for i in range(temp_img.shape[1]):
        encs = np.array(temp_img[:, i:i+1, :, :, :])
        print(f"encs[{i}] shape: {temp_img[:, i:i+1, :, :, :].shape}")
#%%
enc1 = encs[0]; enc6 = encs[5]; enc3 = encs[2]; enc8 = encs[7]
enc5 = encs[4]; enc2 = encs[1]; enc7 = encs[6]; enc4 = encs[3]
img1 = np.concatenate((enc1, enc6, enc3, enc8), axis=1)
img2 = np.concatenate((enc5, enc2, enc7, enc4), axis=1)
print(np.array(img1).shape)
print(np.array(img2).shape)
#%%
with h5py.File('/mounts/data/analyses/bawad2/testData/Images1.h5', 'w'):  
    hf.create_dataset("IMAGE", data=img1)
#%%
