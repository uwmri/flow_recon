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

# %%
# Try to recon data with MRI structure data (pts, arms)
archive_filename_scan= '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/ScanArchive_608WIMRMR2_20240403_152702561.h5'
skope_path= '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data'
demod = -250
gate_delay=200
#MRI_Raw = load_ScanArchive(archive_filename_scan, gate_delay, demod,skope_path)
MRI_Raw = load_ScanArchive(archive_filename_scan, gate_delay, demod, skope_path, compress_coils=-1, max_encodes=None)


# %%
new_coord = np.stack(MRI_Raw.coords, axis=0)
new_kw = np.stack(MRI_Raw.dcf, axis=0)
new_ksp = np.stack(MRI_Raw.kdata, axis=0)

ecg = np.stack(MRI_Raw.ecg, axis=0)
time = np.stack(MRI_Raw.time, axis=0)


print(new_coord.shape)
print(new_ksp.shape)
print(new_kw.shape)
print(ecg.shape)
print(time.shape)

# %%
print(ecg[:,:,:,0])
np.savetxt("ecgoutput.csv", ecg[:,0,:,0], delimiter=",")
np.savetxt("timeoutput.csv", time[:,0,:,0], delimiter=",")


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

for enc in range(new_ksp.shape[0]):
    images = []
    for coil in range(new_ksp.shape[1]):
        #print(coil)
        kdata_temp = new_ksp[enc,coil,0, ...]
        xp = sp.get_device(kdata_temp).xp

        image = sp.nufft_adjoint(kdata_temp[:,:]*new_kw[enc,0,:,:], new_coord[enc,0,:,:,:], oshape=[320, 320])
        images.append( sp.to_device(image))

    images = np.stack(images,0)
    sos = np.sqrt(np.sum(np.abs(images)**2, axis=0))     


    sos_combined[enc,...] = sos 

directory = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/'
recon_name = f'{directory}/vol_centered_test1_ScanArchive_mriraw.h5'

with h5py.File(recon_name, 'w') as hf:
    hf.create_dataset("sos", data=np.abs(sos_combined))
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




# %% Applying High Frequency Filtering to kspace

# Load Image
image_data = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/Time0005_500tf_500iter.h5'   
#image_data = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/VOL01/free_breathing/dv/rawdata_01711_00006_Spiral_Dual_Venc_8-75/Cardiac0075_30tf.h5'
with h5py.File(image_data, 'r') as hf:
    complex_data = hf['IMAGE']


    complex_img = np.stack(complex_data)
    print(complex_img.shape)

    # plot mag and phase
    mag = np.abs(complex_img)
    phase = np.angle(complex_img * np.conj(complex_img[:,0:1,:,:]))


    frame_index = 14  # You can change this depending on the frame you want to view
    coil_index = 1   # You can change this depending on the coil/channel you want to view

    # Plot magnitude
    plt.figure(figsize=(12, 6))

    # Plot magnitude image
    plt.subplot(1, 2, 1)
    plt.imshow(mag[frame_index, coil_index, :, :], cmap='gray', aspect='auto')
    plt.title(f'Magnitude (Frame {frame_index}, Coil {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Plot phase image
    plt.subplot(1, 2, 2)
    plt.imshow(phase[frame_index, coil_index, :, :], cmap='gray', aspect='auto')  # 'twilight' colormap for phase
    plt.title(f'Phase (Frame {frame_index}, Coil {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

    # Perform 2D FFT (for 2D images) or 3D FFT (for 3D data)
    kspace = np.fft.fftn(complex_img, axes=(-2, -1))  # For 2D image (axes are last two dimensions)

    # Shift the zero-frequency component to the center of k-space
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))

    # You can plot k-space magnitude for visualization (optional)
    kspace_mag = np.abs(kspace)
    kspace_phase = np.angle(kspace)  # Phase of k-space data


    # Plot k-space magnitude (just for visualization, not usually done in MRI directly)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.log(1 + kspace_mag[frame_index, coil_index, :, :]), cmap='gray', aspect='auto')  # Log scale for better contrast
    plt.title(f'K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    plt.colorbar()
    plt.axis('off')
    plt.show()


    #plt.figure(figsize=(6, 6))
    #plt.imshow((kspace_phase[frame_index, coil_index, :, :]), cmap='twilight', aspect='auto')  # Example colormap
    #plt.title(f'K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    #plt.colorbar()
    #plt.axis('off')
    #plt.show()

    # Apply inverse FFT on the last two axes (axis=-2 and axis=-1)
    image_space = np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1))

    # You can extract the magnitude and/or phase from the complex image
    magnitude_image = np.abs(image_space)
    phase_image = np.angle(image_space * np.conj(image_space[:,0:1,:,:]))

    # Plot the magnitude image (first frame, first coil)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(image_space[frame_index, coil_index, :, :]), cmap='gray')
    plt.title('Reconstructed Image (Magnitude')
    plt.colorbar()
    plt.axis('off')

    # Optionally, plot the phase image (first frame, first coil)
    plt.subplot(1, 2, 2)
    plt.imshow(phase_image[frame_index, coil_index, :, :] - phase[frame_index, coil_index, :, :], cmap='gray')
    plt.title('Reconstructed Image (Phase)')
    plt.colorbar()
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

    #fh0=(0.5+atan(200*(1-rk/max(kx(:))))/pi);
 
    # Assuming you have kx, ky, and rk arrays already
    # Example: Create kx and ky for demonstration (Replace with your actual data)
    kx = np.linspace(-1, 1, 320)  # Example kx axis (replace with actual data)
    ky = np.linspace(-1, 1, 320)  # Example ky axis (replace with actual data)
    kx, ky = np.meshgrid(kx, ky)  # Create a meshgrid (replace with actual mesh)

    # Calculate radial distance rk
    rk = np.sqrt(kx**2 + ky**2)

    # Normalize rk by the maximum of kx (assuming this is what you're intending)
    kx_max = np.max(kx)

    # Apply the filter (using numpy for vectorized operations)
    fh0 = (0.5 + np.arctan(100 * (1 - rk / kx_max)) / np.pi)

    # Add new dimensions for broadcasting
    fh0_broadcasted = fh0[np.newaxis, np.newaxis, :, :]  # Shape becomes (1, 1, 320, 320)

    # Broadcast it to match kspace shape (30, 3, 320, 320)
    fh0_broadcasted = fh0_broadcasted * np.ones((500, 3, 1, 1))  # Shape becomes (30, 3, 320, 320)

    # Apply the filter to the k-space data
    filtered_kspace = kspace * fh0_broadcasted

    filtered_kspace_mag = np.abs(filtered_kspace)

    kspace_min = np.min(np.log(kspace_mag[frame_index, coil_index, :, :]))
    kspace_max = np.max(np.log(kspace_mag[frame_index, coil_index, :, :]))

    # Plot the magnitude image (first frame, first coil)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(kspace_mag[frame_index, coil_index, :, :]), cmap='gray', aspect='auto',vmin=kspace_min, vmax=kspace_max)
    plt.title(f'K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Optionally, plot the phase image (first frame, first coil)
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(filtered_kspace_mag[frame_index, coil_index, :, :]), cmap='gray', aspect='auto',vmin=kspace_min, vmax=kspace_max)
    plt.title(f'Filtered K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

    image_space_filtered = np.fft.ifft2(np.fft.ifftshift(filtered_kspace, axes=(-2, -1)), axes=(-2, -1))
    magnitude_image_filtered = np.abs(image_space_filtered)
    phase_image_filtered = np.angle(image_space_filtered * np.conj(image_space_filtered[:,0:1,:,:]))


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(magnitude_image_filtered[frame_index, coil_index, :, :], cmap='gray', aspect='auto')
    plt.title(f'K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Optionally, plot the phase image (first frame, first coil)
    plt.subplot(1, 2, 2)
    plt.imshow(phase_image_filtered[frame_index, coil_index, :, :], cmap='gray', aspect='auto')
    plt.title(f'Filtered K-space Magnitude (Frame {frame_index}, Encode {coil_index})')
    plt.colorbar()
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()


image_out = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data_TEST3/testing_filter/time_filterW100_200.h5'   
with h5py.File(image_out, 'w') as hf2:
    hf2.create_dataset("IMAGE_MAG", data=magnitude_image_filtered)
    hf2.create_dataset("IMAGE_PHASE_DIFFERENCE", data=phase_image_filtered)

# %%
