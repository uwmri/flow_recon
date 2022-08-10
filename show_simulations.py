import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

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


sms_dir = ('/data/data_mrcv/45_DATA_HUMANS/CHEST/STUDIES/2019_LIFE_PWV/patients/'
           'life_life00045_12154_2021-10-21-h08/sms_simulation/')

# Show image space and k-space for all imagesets
lists = ['FullRecon_S1.h5', 'FullRecon_S2.h5', 'FullRecon_S2Blip.h5', 'FullRecon_SMS.h5', 'FullRecon_SMSin.h5', 'FullRecon_SMSout.h5']
titles = ['SLICE1', 'SLICE2', 'SLICE2 BLIPPED', 'SLICE1 + SLICE2', 'SLICE1 (IN PHASE)', 'SLICE2 (OUT OF PHASE)']
frame = 7
for i in range(len(lists)):
    filename = os.path.join(sms_dir, lists[i])
    f = h5py.File(filename, 'r')
    img1 = np.mean(f['IMAGE_MAG'], 1)
    phases = f['IMAGE_PHASE']
    img2 = phases[:, 1, :, :] - phases[:, 0, :, :]
    IMG1 = np.fft.fftshift(np.fft.fft2(img1))
    IMG2 = np.fft.fftshift(np.fft.fft2(img2))

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(titles[i])
    axs[0, 0].imshow(img1[frame, :, :], cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Magnitude')

    axs[0, 1].imshow(img2[frame, :, :], cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Phase')

    axs[1, 0].imshow(np.log(abs(IMG1[frame, :, :])+1), cmap='gray')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(np.angle(IMG2[frame, :, :]), cmap='gray')
    axs[1, 1].axis('off')

    fig.tight_layout()
    plt.show()

# Show trajectories and phase patterns
kcoord, ksp = load_mri_raw(sms_dir, 'MRI_Raw_S2.h5')
encode = 0
proj = 1
Kx = kcoord[encode, 0, 0:250, :]
Ky = kcoord[encode, 1, 0:250, :]
Kxa = np.ravel(Kx)
Kya = np.ravel(Ky)
Ca = np.ones(Kxa.shape)
plt.scatter(Kxa, Kya, s=0.1, c=Ca, marker='o')  # sampling pattern
plt.show()

C2 = np.ones(Kx.shape)
for i in range(Kx.shape[0]):
    if i % 2:
        C2[i,:] = -C2[i,:]
Ca2 = np.ravel(C2)
plt.scatter(Kxa, Kya, s=0.1, c=Ca2, marker='o')  # phase blip pattern
plt.show()
plt.scatter(Kxa, Kya, s=0.1, c=-Ca2, marker='o')  # phase conjugate
plt.show()
