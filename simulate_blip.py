#! /usr/bin/env python
import os
import h5py
import numpy as np
import math
import shutil


# Display HDF5 file structure
def scan_h5(path, tab_step=2):
    print('DISPLAYING HDF5 DATA HIERARCHY')
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group):
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)


# Load raw k-space data (MRI_Raw.h5)
# NOTE: Raw k-space is output from C++ pcvipr_recon_binary with "-export_kdata" flag
def load_mri_raw(directory, file):
    filename = os.path.join(directory, file)
    scan_h5(filename)

    with h5py.File(filename, 'r') as hf:
        num_encs = 0  #get number of encodes
        while f'KW_E{num_encs}' in hf['Kdata']:
            num_encs += 1
        num_coils = 0 #get number of coils
        while f'KData_E0_C{num_coils}' in hf['Kdata']:  # find number of coils
            num_coils += 1

        kcoord = []  # kcoord: [encode, dimension(x,y,z,w,t), projections, coords]
        ksp = []  # ksp --> [encode, coils, projections, data]
        for e in range(num_encs):
            print(f'Encode: {e}')
            kx = np.array(hf['Kdata'][f'KX_E{e}'])
            ky = np.array(hf['Kdata'][f'KY_E{e}'])
            kz = np.array(hf['Kdata'][f'KZ_E{e}'])
            kw = np.array(hf['Kdata'][f'KW_E{e}'])
            kt = np.array(hf['Kdata'][f'KT_E{e}'])
            kcoord.append(np.stack((kx, ky, kz, kw, kt)))
            del kx, ky, kz, kw, kt

            kdata = []
            for c in range(num_coils):
                print(f'Coil: {c}')
                k = hf['Kdata'][f'KData_E{e}_C{c}']
                kdata.append(k['real'] + 1j * k['imag'])
            kdata = np.stack(kdata, 0)
            ksp.append(kdata)

        kcoord = np.squeeze(kcoord)
        ksp = np.squeeze(ksp)
        return kcoord, ksp


# Compile Data in Separate Folder
# NOTE: Time-averaged recon is first required with "/export/home/groberts/local/bin/sms_2dpc_recon" to get k-space data
sms_dir = ('/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2019_LIFE_PWV/patients/'
           'life_life00045_12154_2021-10-21-h08/sms_simulation/')

# Copy Slice 1 (aortic arch 2DPC)
directory1 = ('/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2019_LIFE_PWV/patients/'
              'life_life00045_12154_2021-10-21-h08/12154_00006_pwv-radial_AAo/TA_recon/')
src = os.path.join(directory1, 'MRI_Raw.h5')  # original file
dest = os.path.join(sms_dir, 'MRI_Raw_S1.h5')  # create new slice identifier
if not(os.path.exists(dest)):
    shutil.copyfile(src, dest)  # copy file to new directory

# Copy Slice 2 (abdominal aorta 2DPC)
directory2 = ('/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2019_LIFE_PWV/patients/'
              'life_life00045_12154_2021-10-21-h08/12154_00009_pwv-radial_AbdAo/TA_recon/')
src = os.path.join(directory2, 'MRI_Raw.h5')
dest = os.path.join(sms_dir, 'MRI_Raw_S2.h5')
if not(os.path.exists(dest)):
    shutil.copyfile(src, dest)  # copy file to new directory

# Create Dummy HDF5s For Appending (see end of code)
shutil.copyfile(dest, os.path.join(sms_dir, 'MRI_Raw_S2Blip.h5'))  # copy file to new directory
shutil.copyfile(dest, os.path.join(sms_dir, 'MRI_Raw_SMS.h5'))  # copy file to new directory
shutil.copyfile(dest, os.path.join(sms_dir, 'MRI_Raw_SMSin.h5'))  # copy file to new directory

# Load K-Space from Slice 1
kcoord1, ksp1 = load_mri_raw(sms_dir, 'MRI_Raw_S1.h5')  # load raw k-space data (see function above)
sample = 0  # first sample along spoke
encode = 0  # first encode (should be same for both)
Kx = kcoord1[encode, 0, :, sample]  # get x-coordinate of single data point
Ky = kcoord1[encode, 1, :, sample]  # get y-coordinate of single data point
angles1 = (180/math.pi)*np.arctan(Ky/Kx) + 180  # get projection angle --> golden angle = 111.2

# Load K-Space from Slice 2
kcoord2, ksp2 = load_mri_raw(sms_dir,'MRI_Raw_S2.h5')
sample = 0
encode = 0
Kx = kcoord2[encode, 0, :, sample]
Ky = kcoord2[encode, 1, :, sample]
angles2 = (180/math.pi)*np.arctan(Ky/Kx) + 180  # golden angle = 111.2

print(angles2-angles1)  # Check if angles are approximately the same

# Simulate Phase Blip
num_slices = 2  # number of slices to simulate
phase = (2*math.pi)/num_slices  # phase blip required

encodes = ksp1.shape[0]  # number of encodes (for 2DPC, should be 2)
coils = ksp1.shape[1]  # number of coils
projs = ksp1.shape[2]  # number of projections
ksp2_blip = ksp2
for e in range(encodes):
    for c in range(coils):
        for p in range(projs):
            euler = np.complex(math.cos(p*phase), math.sin(p*phase))  # e^i*theta
            ksp2_blip[e, c, p, :] = euler * ksp2[e, c, p, :]  # simulate slice-direction phase blip

# Rewrite Raw K-Space Data
print('SAVING SLICE 2 WITH PHASE BLIP')
filename = os.path.join(sms_dir, 'MRI_Raw_S2Blip.h5')
f = h5py.File(filename, 'a')  # append data to MRI_Raw_S2Blip (copy of MRI_Raw_S2 to not overwrite)
for e in range(encodes):
    print(f'Writing Encode: {e}')
    for c in range(coils):
        print(f'Writing Coil: {c}')
        k = f['Kdata'][f'KData_E{e}_C{c}']
        temp = ksp2_blip[e, c, :, :]  # get projection from blipped k-space (not SMS, just slice 2)
        temp = temp[None, :]  # add singleton dim
        k['real'] = np.real(temp)  # rewrite data into real channel of HDF5
        k['imag'] = np.imag(temp)  # rewrite data into imaginary channel of HDF5
        f.require_dataset(f'Kdata/KData_E{e}_C{c}', shape=k.shape, dtype=k.dtype)  # overwrite data here

print('SAVING SLICE1+SLICE2 (no phase blip, aliased images)')
filename = os.path.join(sms_dir, 'MRI_Raw_SMS.h5')
f = h5py.File(filename, 'a')  # append data to MRI_Raw_SMS (copy of MRI_Raw_S2 to not overwrite)
for e in range(encodes):
    print(f'Writing Encode: {e}')
    for c in range(coils):
        print(f'Writing Coil: {c}')
        k = f['Kdata'][f'KData_E{e}_C{c}']
        temp = ksp1[e, c, :, :] + ksp2_blip[e, c, :, :]  # add k-space data from both slices (SMS simulation)
        temp = temp[None, :]  # add singleton dim
        k['real'] = np.real(temp)  # rewrite data into real channel of HDF5
        k['imag'] = np.imag(temp)  # rewrite data into imaginary channel of HDF5
        f.require_dataset(f'Kdata/KData_E{e}_C{c}', shape=k.shape, dtype=k.dtype)  # overwrite data here

print('SAVING SLICE1+SLICE2 (with phase blip on slice 2)')
filename = os.path.join(sms_dir, 'MRI_Raw_SMSin.h5')
f = h5py.File(filename, 'a')  # append data to MRI_Raw_SMS (copy of MRI_Raw_S2 to not overwrite)
for e in range(encodes):
    print(f'Writing Encode: {e}')
    for c in range(coils):
        print(f'Writing Coil: {c}')
        k = f['Kdata'][f'KData_E{e}_C{c}']
        temp = ksp1[e, c, :, :] + ksp2_blip[e, c, :, :]  # add k-space data from both slices (SMS simulation)
        temp = temp[None, :]  # add singleton dim
        k['real'] = np.real(temp)  # rewrite data into real channel of HDF5
        k['imag'] = np.imag(temp)  # rewrite data into imaginary channel of HDF5
        f.require_dataset(f'Kdata/KData_E{e}_C{c}', shape=k.shape, dtype=k.dtype)  # overwrite data here



