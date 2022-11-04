import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load raw k-space data (MRI_Raw.h5)
# NOTE: Raw k-space is output from C++ pcvipr_recon_binary with "-export_kdata" flag
def load_mri_raw(directory, file):
    filename = os.path.join(directory, file)
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

#Tk().withdraw()
#sms_dir = askopenfilename()

#sms_dir = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06024_2022-11-03/06024_00004_b1_p1_ff0p50_ga/SMS_2DPC/"
# sms_dir = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06024_2022-11-03/06024_00009_b1_p0_ff0p46_seq/SMS_2DPC/"
# sms_dir = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06024_2022-11-03/06024_00013_b1_p0_ff0p50_ga/SMS_2DPC/"
# sms_dir = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06024_2022-11-03/06024_00010_b1_p0_ff0p50_seq/SMS_2DPC/"
sms_dir = "/data/data_mrcv2/99_GSR/SMS_Testing/LIFEVOLUNTEER_06041_2022-11-04/06041_00007_pwv-radial_SMS/SMS_2DPC/"

# Show image space and k-space for all imagesets
lists = ['InPhase.h5', 'OutPhase.h5']
titles = ['SLICE1 (IN PHASE)', 'SLICE2 (OUT OF PHASE)']


numPhases = len(lists)
encode = 1
frame = 0

fig, axs = plt.subplots(nrows=numPhases, ncols=2, figsize=(12, 8))
fig.suptitle('Encode' + str(encode))
cols = ['Magnitude', 'Phase']
for ax, col in zip(axs[0], cols):
    ax.set_title(col)
for ax, row in zip(axs[:,0], lists):
    ax.set_ylabel(row, rotation=90, size="large")

for i in range(numPhases):
    filename = os.path.join(sms_dir, lists[i])
    f = h5py.File(filename, 'r')
    mags = f['IMAGE_MAG']
    phases = f['IMAGE_PHASE']
    encodes = mags.shape[1]

    mag = mags[frame, encode, :, :]
    phase = phases[frame, encode, :, :]
    axs[i, 0].imshow(mag, cmap='gray', aspect="auto")
    axs[i, 0].tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
    axs[i, 1].imshow(phase, cmap='gray', aspect="auto")
    axs[i, 1].tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

fig.tight_layout()
plt.show()


# kcoord, ksp = load_mri_raw(sms_dir, 'MRI_Raw.h5')
#
# encode = 0
# Kx = kcoord[encode, 0, :, :]
# Ky = kcoord[encode, 1, :, :]
#
# sample = 0
# Kxd = Kx[:, sample]
# Kyd = Ky[:, sample]
# angles = (180/math.pi)*np.arctan(Kyd/Kxd) + 180
# plt.plot(angles)
# plt.show()
#
# numproj = Kx.shape[0]
# for p in range(numproj):
#     Kxa = Kx[p, :]
#     Kya = Ky[p, :]
#     plt.scatter(Kxa, Kya, s=0.1, marker='o')
#     plt.show()
