# %% 
import os
import glob
import csv
import ctypes
import h5py
import argparse
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
import sigpy as sp
import sigpy.mri as mri
import json
from read_scan_archive import *


# Try to recon data with MRI structure data (pts, arms)
archive_filename_scan= '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/ScanArchive_608WIMRMR2_20240403_152702561.h5'
skope_path= '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data'
demod = -250
gate_delay=200
MRI_Raw = load_ScanArchive(archive_filename_scan, gate_delay, demod,skope_path)

# %%
new_coord = np.zeros((3, 1868, 2000, 2), dtype=np.float32)
new_kw = np.zeros((3, 1868, 2000), dtype=np.float32)
new_ksp = np.zeros((48, 3, 1868, 2000), dtype=np.complex64)

Num_Encodings = 3
Num_Coils = 48

for encode in range(Num_Encodings):
     
    s = f"KX_E{encode}"
    new_coord[encode,:,:,0] = MRI_Raw['Kdata'][s]

    s = f"KY_E{encode}"
    new_coord[encode,:,:,1] = MRI_Raw['Kdata'][s]

    s = f"KW_E{encode}"
    new_kw[encode,:,:] = MRI_Raw['Kdata'][s]

    for coil in range(Num_Coils):
        s = f"KData_E{encode}_C{coil}"
        new_ksp[coil, encode,:,:] = MRI_Raw['Kdata'][s]
        

print(new_ksp.shape)
print(new_coord.shape)
print(new_kw.shape)

new_ksp = np.moveaxis(new_ksp, 0, -1)
new_ksp = np.moveaxis(new_ksp, 2, 0)
print(new_ksp.shape)

new_coord = np.moveaxis(new_coord, 2, 0)
print(new_coord.shape)

new_kw = np.moveaxis(new_kw, 1, -1)
print(new_kw.shape)



# %%

res = [3,320,320]
sos_combined = np.zeros(res, dtype=np.float32) 

try:
    device = sp.Device(0)
except:
    device = sp.cpu_device

new_coord = sp.to_device(new_coord, device)
new_kw = sp.to_device(new_kw, device)
new_ksp = sp.to_device(new_ksp, device)

for enc in range(new_ksp.shape[1]):
    images = []
    for coil in range(new_ksp.shape[-1]):
        #print(coil)
        kdata_temp = new_ksp[:,enc,:,coil]
        xp = sp.get_device(kdata_temp).xp

        image = sp.nufft_adjoint(kdata_temp[:,:]*new_kw[enc,:,:], new_coord[:,enc,:,:], oshape=[320, 320])
        images.append( sp.to_device(image))

    images = np.stack(images,0)
    sos = np.sqrt(np.sum(np.abs(images)**2, axis=0))     


    sos_combined[enc,...] = sos 

directory = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/'
recon_name = f'{directory}/vol_centered_test1_ScanArchive_function2.h5'

with h5py.File(recon_name, 'w') as hf:
    hf.create_dataset("sos", data=np.abs(sos_combined))


# %%
