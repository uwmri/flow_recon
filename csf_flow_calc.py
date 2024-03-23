import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import cupy
import time
import math
import sys
from skimage.restoration import unwrap_phase


from mri_raw import *
from multi_scale_low_rank_recon import *
from llr_recon import *
from svt import *
import numba as nb
import os
import scipy.ndimage as ndimage
from registration_tools import *

research_flow_vencfree_bins = 4
research_flow_vencfree_power = 0.5
research_flow_max_venc = 800
research_flow_min_venc = 50

# calculation might be wrong need fix
#temp_scale = np.power( (np.arange(research_flow_vencfree_bins)-1.)/(research_flow_vencfree_bins - 2.0), research_flow_vencfree_power)
#vencs = research_flow_max_venc - temp_scale*(research_flow_max_venc - research_flow_min_venc)
#vencs[0] = 1000000000
#print(vencs)

# hard code Vencs at the moment 
vencs = [1000000000, 1000, 390, 50]

# ~~~ translation of the seperate matlab "lap3" function code ~~~#
def lap3(in_, dir, mod, real_flag=0):
    #  Args:
    #     in: 3D input array
    #     dir: forward or inverse transform (1 or -1)
    #     mod: Laplaican kernel in frequency space
    #     real_flag: restrict output to real (doesn't really matter, but this
    #                lowers the memory load)
    #  Returns:
    #     out: output matrix
    [sx, sy, sz] = np.array(in_.shape)
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(in_)))

    if (dir == 1):
        K = K * mod
    elif (dir == -1):
        mod[ sx // 2, sy // 2, sz // 2] = 1
        K = K / mod
    else:
        print('ERROR')

    if real_flag > 0:
        out = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K))))
    else:
        out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))
    return out


# ~~~ translation of the seperate matlab "unwrap3D" function code ~~~#
def unwrap_3D(phi_w, real_flag=1):  # need to do the unwrap_4D function
    #  Args:
    #     phi_w: Wrapped input array (-pi to pi)
    #     ts: Scales the temporal data to spatial dimensions
    #     real_flag: restrcit laplacians to real (doesn't really matter, but
    #     this lowers the memory load)
    #  Returns:
    #     nr: integer array containing the NUMBER of wraps per voxel
    #         (note that this is not the actual unwrapped data)
    phi_w_size = np.array(phi_w.shape)
    # [X, Y, Z] = np.meshgrid((range((int(-phi_w_size[0] / 2)), (int(phi_w_size[0] / 2)))),
    #                         (range((int(-phi_w_size[1] / 2)), (int(phi_w_size[1] / 2)))),
    #                         (range((int(-phi_w_size[2] / 2)), (int(phi_w_size[2] / 2)))), indexing='ij')
    #
    ndim = phi_w.shape
    X, Y, Z = np.mgrid[-ndim[0] // 2:ndim[0] // 2:,
                 -ndim[1] // 2:ndim[1] // 2:,
                 -ndim[2] // 2:ndim[2] // 2:]


    mod = 2 * np.cos(math.pi * X / phi_w_size[0]) + 2 * np.cos(math.pi * Y / phi_w_size[1]) + 2 * np.cos(
        math.pi * Z / phi_w_size[2]) - 6
    # print(mod)

    lap_phiw = lap3(phi_w, 1, mod, real_flag)
    lap_phi = (np.cos(phi_w) * lap3(np.sin(phi_w), 1, mod, real_flag)) - (
                np.sin(phi_w) * lap3(np.cos(phi_w), 1, mod, real_flag))
    ilap_phidiff = (lap3(lap_phi - lap_phiw, -1, mod, real_flag))
    nr = np.int8(np.round((ilap_phidiff / 2) / math.pi))
    return nr

def get_image( m0, v, vencs):
    image = []
    for encode, venc in enumerate(vencs):
        image.append(m0[encode]*np.exp(1j*math.pi*v/venc))
    image =np.stack(image, 0)
    return image

def error_func(im1, im2, vtest, vencs):

    error = np.zeros(im1[0].shape)
    for i in range(len(vencs)):
        error += np.abs(im1[i] - im2[i])

    return error

def background_phase_correct(image_in, mag_thresh=0.15, fit_order=2):

    # Average time frames
    magnitude_avg = np.abs(np.mean(image_in, 0))
    phase_avg = np.angle(np.mean(image_in, 0))

    # Threshold
    max_mag = np.max(magnitude_avg)

    # Get the number of coeficients
    py,px = np.meshgrid(range(fit_order+1),range(fit_order+1))
    idx = np.where( (px+py) <= fit_order )
    px = px[idx]
    py = py[idx]
    N = len(px)

    #print('Polynomial fitting with %d variables' % (N,))
    AhA = np.zeros((N, N), dtype=np.float64)
    AhBx= np.zeros((N, 1), dtype=np.float64)

    # Now gather terms (Nt x Nz x Ny x Nx x 3 )
    y, x = np.meshgrid(np.linspace(-1, 1, magnitude_avg.shape[0]),
                          np.linspace(-1, 1, magnitude_avg.shape[1]),
                          indexing='ij')

    # Grab array
    mask = (magnitude_avg > (mag_thresh * max_mag))


    # Subselect values
    x_slice = x[mask].flatten()
    y_slice = y[mask].flatten()
    phase_slice = phase_avg[mask].flatten()

    for ii in range(N):
        for jj in range(N):
            AhA[ii, jj] = np.sum((x_slice ** px[ii] * y_slice ** py[ii] ) *
                                 (x_slice ** px[jj] * y_slice ** py[jj] ) )

    for ii in range(N):
        phi = np.power(x_slice, px[ii]) * np.power(y_slice, py[ii])
        AhBx[ii] = np.sum(phase_slice * phi)

    polyfit_x = np.linalg.solve(AhA, AhBx)

    # Now Subtract
    background_phase = np.zeros(phase_avg.shape, phase_avg.dtype)

    #print("Subtract")
    for ii in range(N):
        phi = np.power(x, px[ii]) * np.power(y, py[ii])
        background_phase += polyfit_x[ii]*phi

    #Expand and subtract)
    return background_phase


with h5py.File(r'Time03_blk8_500_stride_3arms.h5') as hf:
    image = np.array(hf['IMAGE'])

# Arrange into [encode, frame, xres, yres]
image = np.swapaxes(image,0,1)
image *= np.conj(image[0])

image_sub = image.copy()

bf = []
vz = []
angio = []
for i in range(image_sub.shape[0]):
    background_phase = background_phase_correct(image_sub[i], mag_thresh=0.08, fit_order=3)
    image_sub[i] *= np.exp(-1j*background_phase)
    bf.append(background_phase)

    phase = unwrap_3D(np.angle(image_sub[i]))* 2 * np.pi  + np.angle(image_sub[i])
    #phase = np.unwrap(np.angle(image_sub[i]), axis=0)
    #phase = unwrap_phase(np.angle(image_sub[i]))
    #phase = np.angle(image_sub[i])
    vz.append(vencs[i] * phase /math.pi)

    vmag = np.abs(vencs[i] * phase /math.pi)
    mag  = np.abs(image_sub[i])
    angio.append(mag*np.sin(math.pi/2.0*vmag/vencs[i]))

max_angio = np.max(np.array(angio), axis=None)
norm_angio = (np.array(angio) / max_angio) * 100

bf = np.stack(bf,0)

with h5py.File(r'RTvelocities.h5','w') as hf:
    hf.create_dataset('image_abs', data=np.abs(image_sub))
    hf.create_dataset('image_angle', data=np.angle(image_sub))

    #hf.create_dataset('image_noBGC_abs', data=np.abs(image))
    #hf.create_dataset('image_noBGC_angle', data=np.angle(image))

    hf.create_dataset('back_phase', data=bf)
    hf.create_dataset('vz', data=vz)
    hf.create_dataset('angio', data=angio)
