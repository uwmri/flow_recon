import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import cupy
import time
import math
import sys
import fnmatch
import hashlib
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
# vencs = [1000000000, 800, 500, 70]

# Laplacian based phase unwrapping (time working, need to verify the spatial only version)
def unwrap_2D_time(phase_w):

    # inpute phase range must be from (-pi to pi)
    logger = logging.getLogger('Laplacian Unwrap')

    ts = 2.0  # scales temporal data to spatial dimentions
    # real_flag = 1 # restrict laplacians to real (lowers memory load)
    phase_w = np.moveaxis(phase_w, 0, -1)  # x y t

    ndim = phase_w.shape

    # create grid
    X, Y, T = np.mgrid[-ndim[0] // 2:ndim[0] // 2:,
                 -ndim[1] // 2:ndim[1] // 2:,
                 -ndim[2] // 2:ndim[2] // 2:]

    # get mod
    mod = 2.0 * np.cos(np.pi * X / ndim[0]) + 2.0 * np.cos(np.pi * Y / ndim[1]) + ts * np.cos(np.pi * T / ndim[2]) - 6.0 - ts

    X = None
    Y = None
    T = None

    logger.info('Laplacian')
    print('Forward')
    lap_phase_w = lap2t(phase_w, 1, mod)
    lap_phase = np.cos(phase_w) * lap2t(np.sin(phase_w), 1, mod) - np.sin(phase_w) * lap2t(np.cos(phase_w), 1, mod)

    logger.info('Inverse Laplacian')
    print('Backwards')
    ilap_phasediff = lap2t(lap_phase - lap_phase_w, -1, mod)
    n_u4 = np.int8(np.real(np.ndarray.round(ilap_phasediff / 2 / np.pi)))

    phase_w = np.moveaxis(phase_w, -1, 0)  # t x y 
    n_u4 = np.moveaxis(n_u4, -1, 0)

    return n_u4


def lap2t(phase_w, direction, mod):
    ndim = phase_w.shape
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phase_w)))

    if direction == 1:
        K *= mod

    elif direction == -1:
        mod[ndim[0] // 2, ndim[1] // 2, ndim[2] // 2] = 1
        K /= mod

    else:
        print("ERROR")

    out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out

# Laplacian based phase unwrapping (time working, need to verify the spatial only version)
def unwrap_2D_space(phase_w):

    # inpute phase range must be from (-pi to pi)
    logger = logging.getLogger('Laplacian Unwrap')
    # real_flag = 1 # restrict laplacians to real (lowers memory load)

    ndim = phase_w.shape

    # create grid
    X, Y = np.mgrid[-ndim[0] // 2:ndim[0] // 2:,
                 -ndim[1] // 2:ndim[1] // 2:]

    # get mod
    mod = 2.0 * np.cos(np.pi * X / ndim[0]) + 2.0 * np.cos(np.pi * Y / ndim[1]) - 6.0

    X = None
    Y = None

    logger.info('Laplacian')
    print('Forward')
    lap_phase_w = lap2space(phase_w, 1, mod)
    lap_phase = np.cos(phase_w) * lap2space(np.sin(phase_w), 1, mod) - np.sin(phase_w) * lap2space(np.cos(phase_w), 1, mod)

    logger.info('Inverse Laplacian')
    print('Backwards')
    ilap_phasediff = lap2space(lap_phase - lap_phase_w, -1, mod)
    n_u4 = np.int8(np.real(np.ndarray.round(ilap_phasediff / 2 / np.pi)))

    return n_u4


def lap2space(phase_w, direction, mod):
    ndim = phase_w.shape
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phase_w)))

    if direction == 1:
        K *= mod

    elif direction == -1:
        mod[ndim[0] // 2, ndim[1] // 2] = 1
        K /= mod

    else:
        print("ERROR")

    out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out


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

def background_phase_correct(image_in, mag_thresh=0.15, fit_order=2, ):

    # Average time frames
    magnitude_avg = np.abs(np.mean(image_in, 0))
    phase_avg = np.angle(np.mean(image_in, 0))

    # Create angio to exclude vessels
    #vmag = np.abs(venc_in * phase_avg /math.pi)
    #angio = magnitude_avg*np.sin(math.pi/2.0*vmag/venc_in)
    #angio_thres = 0.20
    #max_angio = np.max(angio)
    #mask_angio = (angio > (angio_thres * max_angio))

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
    #combined_mask = mask & (mask_angio == False)

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

class recon_file:
    def __init__(self):
        self.full_hash = None
        self.short_hash = None
        self.filename = None
        self.folder = None
        self.extension = None

    def __str__(self):
        name = f'[ File:{os.path.join(self.folder,self.filename)} Hash:{self.short_hash} ]'
        return name

    def fullname(self):
        return(os.path.join(self.folder, self.filename))

def find(patterns, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    if os.path.islink(os.path.join(root, name)) == False:
                        f = recon_file()
                        f.filename = name
                        f.folder = root
                        #f.short_hash = short_md5(os.path.join(root, name))
                        f.extension = os.path.splitext(name)[1]
                        result.append(f)
    return result

def dual_venc_vest( phase_high, mag_high, im_low, venc_high, venc_low):
    # fix high venc issue
    velocity_low = np.angle(im_low) * venc_low / np.pi
    velocity_high = phase_high * venc_high / np.pi

    # Unwrap 
    velocity_low_unwrap = velocity_low + 2.0*venc_low*np.round((velocity_high - velocity_low)/(2.0*venc_low))

    # Single step unwrap with Laplacian to correct remaining aliasing errors.
    # The range needs to be [-pi, pi] otherwise Laplacian seems to fail;
    # therefore, results from velocity_low_unwrap might not be in the expected input domain 

    #phase_low_unwrap = velocity_low_unwrap / venc_low * np.pi 
    #phase_low_unwrap2 = unwrap_2D_time(phase_low_unwrap)* 2 * np.pi  + phase_low_unwrap
    #velocity_low_unwrap2 = phase_low_unwrap2 * venc_low / np.pi

    # Weighted average of low and high VENC. The low venc has been unwrapped, the high venc went thru single step unwrap
    weight_high = mag_high**2 / venc_high**2
    weight_low = np.abs(im_low)**2 / venc_low**2

    velocity_combined = (weight_low*velocity_low_unwrap + weight_high*velocity_high)/(weight_high + weight_low)

    return velocity_low_unwrap, velocity_combined


if __name__ == '__main__':

    print('Finding Cardiac and Time recons', flush=True)
    files_to_compare = find(['Cardiac*.h5','TTTime*.h5'], os.getcwd())

    for idx, f in enumerate(files_to_compare):
        
        full_filename = f.fullname()
        filename = os.path.basename(full_filename)
        folder = f.folder

        output_full_filename = os.path.join(folder, f'uVel_{filename}')

        #print(full_filename)
        #print(filename)
        print(folder)

        os.chdir(folder)
        os.system(f'ls {filename}')

        with h5py.File(full_filename) as hf:
            image = np.array(hf['IMAGE'])
        num_encodes = image.shape[1]

        if num_encodes == 4:
            vencs = [1000000000, 1000, 500, 80]
        elif num_encodes == 3:
            vencs = [1000000000, 750, 80]
        elif num_encodes == 2:
            user_input = input("Enter a value for the single Venc (mm/s) (i.e. Vencs: 80, 750, 1000): ")
            vencs = [1000000000, int(user_input)]
        else:
            raise ValueError("Invalid number of encodes. Expected 2, 3, or 4.")
        
        print(f' Number of encodes = {num_encodes}')
        print(f'Vencs = {vencs}')

        # Arrange into [encode, frame, xres, yres]
        image = np.swapaxes(image,0,1)

        if num_encodes == 3:
            MAG_combined = np.sqrt(np.abs(image[0])**2 + np.abs(image[1])**2)

        if num_encodes == 4:
            MAG_combined = np.sqrt(np.abs(image[0])**2 + np.abs(image[1])**2 + np.abs(image[2])**2)

        # Subtract reference or combined balanced encodes
        image *= np.conj(image[0]) 
        #image *= np.exp(1j*math.pi/4.0) # to remove aliasing in arteries of case 13, this works for SV75, but need to do BGP
        image_sub = image.copy()

        bf = []
        vz = []
        angio = []

        for i in range(image_sub.shape[0]):
            background_phase = background_phase_correct(image_sub[i], mag_thresh=0.08, fit_order=2, )
            image_sub[i] *= np.exp(-1j*background_phase)
            bf.append(background_phase)

            #Single step phase unwrapping using Laplacian (do not use on CSF, there seems to be some residuals errors for CSF in DV and MV)
            if vencs[i] > 8000000:
                print(f'Venc = {vencs[i]}')                
                phase = unwrap_2D_time(np.angle(image_sub[i]))* 2 * np.pi  + np.angle(image_sub[i])
            else:
                phase = np.angle(image_sub[i]*np.exp(1j*math.pi/3.5)) - math.pi/3.5    # this value solves SV75 ICA1 case13

            #math.pi/3.5    # this value solves SV75, DV ICA1 case7 (both repeats)

            vz.append(vencs[i] * phase /math.pi)
            vmag = np.abs(vencs[i] * phase /math.pi)
            
            # for dv and mv scans lets use the reference encode for angio for the high and middle venc scans (not for CSF)
            if (num_encodes == 3 and i < 2) or (num_encodes == 4 and i < 3):
                mag  = np.copy(MAG_combined)           
            else:
                # reference single venc scans (less important how we do angio) 
                mag  = np.abs(image_sub[i])

            # The Mag images look good, lets try making an angio more magntidue weighted. use mag^2 instead of mag 
            raw_angio = mag*mag*np.sin(math.pi/2.0*vmag/vencs[i])
            #print(f'shape of angio is {raw_angio.shape}')
            norm_angio = np.zeros_like(raw_angio)

            for j in range(image_sub.shape[1]):
                agio_tframe = raw_angio[j]
                norm_angio[j] = agio_tframe / np.max(agio_tframe)

            angio.append(norm_angio)
            #angio.append(mag*np.sin(math.pi/2.0*vmag/vencs[i]))

        #max_angio = np.max(np.array(angio), axis=None)
        #norm_angio = (np.array(angio) / max_angio) * 100
        bf = np.stack(bf,0)

        # Multi-Venc unwrapping (BGC already done)
        if num_encodes == 4:
            vencs = [1000000000, 1000, 500, 80]

            image_high = image_sub[1]
            image_low = image_sub[2]

            # unwrap high venc using single step Laplacian
            phase_high = unwrap_2D_time(np.angle(image_high))* 2 * np.pi  + np.angle(image_high)
            mag_high = np.abs(image_high)
            #image_high_unwrap = np.abs(image_high)*np.exp(1j*phase_high)

            # unwrap low venc using high venc and combine
            velocity_low_unwrap, velocity_combined = dual_venc_vest(phase_high, mag_high, image_low, vencs[1], vencs[2])

            vz.append(velocity_low_unwrap)
            vz.append(velocity_combined)

            #vmag = np.abs(velocity_low_unwrap)
            #mag  = np.abs(image_low)

            # The Mag images look good, lets try making an angio more magntidue weighted. use mag^2 instead of mag. 
            #This angio mignt not be better than the native velocities one,scaled by the corresponding magnitude image.
            #medium_venc = 500 # lets try the right one
            #raw_angio = mag*mag*np.sin(math.pi/2.0*vmag/medium_venc)
            #print(f'shape of angio is {raw_angio.shape}')
            #norm_angio = np.zeros_like(raw_angio)

            #for j in range(image_sub.shape[1]):
            #    agio_tframe = raw_angio[j]
            #    norm_angio[j] = agio_tframe / np.max(agio_tframe)

            #angio.append(norm_angio)

        with h5py.File(output_full_filename,'w') as hf:
            hf.create_dataset('image_abs', data=np.abs(image_sub))
            #hf.create_dataset('image_noBGC_abs', data=np.abs(image))
            #hf.create_dataset('image_angle', data=np.angle(image_sub))
            #hf.create_dataset('image_noBGC_angle', data=np.angle(image))
            hf.create_dataset('back_phase', data=bf)
            hf.create_dataset('vz', data=vz)
            hf.create_dataset('angio', data=angio)