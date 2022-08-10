import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import math

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


sms_dir = '/data/users/groberts/SMS_Testing/Exam5450/SMS_MB2/'

# Show image space and k-space for all imagesets
lists = ['InPhase.h5', 'OutPhase.h5']
titles = ['SLICE1 (IN PHASE)', 'SLICE2 (OUT OF PHASE)']
frame = 0
for i in range(len(lists)):
    filename = os.path.join(sms_dir, lists[i])
    f = h5py.File(filename, 'r')
    mags = f['IMAGE_MAG']
    mag1 = mags[:, 0, :, :]
    mag2 = mags[:, 1, :, :]
    phases = f['IMAGE_PHASE']
    phase1 = phases[:, 0, :, :]
    phase2 = phases[:, 1, :, :]

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(titles[i])
    axs[0, 0].imshow(mag1[frame, :, :], cmap='gray')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(mag2[frame, :, :], cmap='gray')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(phase1[frame, :, :], cmap='gray')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(phase2[frame, :, :], cmap='gray')
    axs[1, 1].axis('off')

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
