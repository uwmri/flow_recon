#! /usr/bin/env python
import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import cupy
import time
import math

from mri_raw import *
from multi_scale_low_rank_recon import *
from llr_recon import *
from svt import *
import numba as nb
#import torch as torch
import os
import scipy.ndimage as ndimage
import torch
from gpu_ops import *
from readout_regrid import *

class MRI_Raw:
    Num_Encodings = 0
    Num_Coils = 0
    trajectory_type = None
    dft_needed = None
    Num_Frames = None
    coords = None
    time = None
    ecg = None
    prep = None
    resp = None
    dcf = None
    kdata = None
    frame = None
    target_image_size = [256, 256, 64]


def resample_arc(input, coord, oshape=None, oversamp=2, width=7):
    """Adjoint non-uniform Fast Fourier Transform.

    Args:

    Returns:
        array: signal domain array with shape specified by oshape.

    See Also:
        :func:`sigpy.nufft.nufft`

    """

    # Get the image dimensions
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    if oshape is None:
        oshape = list(input.shape[:-coord.ndim + 1]) + estimate_shape(coord)
    else:
        oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp)
    output = interp.gridding(input, coord, os_shape,
                             kernel='kaiser_bessel', width=width, param=beta)
    output /= width**ndim

    # IFFT
    output = ifft(output, axes=range(-ndim, 0), norm=None)

    # Crop
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:])**0.5

    # Apodize
    _apodize(output, ndim, oversamp, width, beta)

    return output


def radial3d_regrid(coord_regrid, dcf, kdata, new_dk=0.5):
    """Data resampling operator to deal with variable density oversampling

    Args:
        coord_regrid (array): Coordinates to regrid [..., npts, ndim]
        dcf (array): Density compensation [..., npts]
        kdata(array): Input k-space data [batch_size, ..., npts]
        new_dk(float):

    Returns:
        output (tuple): Tuple of coord, dcf, kdata resampled to the new dy.

    """
    logger = logging.getLogger('radial3d_regrid')
    logger.info(f'Coord shape = {coord_regrid.shape}')
    logger.info(f'Kdata shape = {kdata.shape}')
    logger.info(f'DCF shape = {dcf.shape}')

    width = 11
    oversamp = 1.0
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5))**2 - 0.8)**0.5
    logger.info(f'Using Beta {beta} {oversamp} {width}')

    # Get input shape
    dcf_input_shape = list(dcf.shape)
    coord_input_shape = list(coord_regrid.shape)
    kdata_input_shape = list(kdata.shape)

    # Reshape to be flat (readout x xres )
    ndim = coord_regrid.shape[-1]
    base_shape = coord_regrid.shape[:-1]
    xres = base_shape[-1]
    num_coils = kdata.shape[0]

    coord_regrid = np.reshape(coord_regrid,[-1, xres, ndim])
    dcf = np.reshape(dcf, [-1, xres])
    kdata = np.reshape(kdata, [-1, ] + list(dcf.shape))

    logger.info(f'New Coord shape = {coord_regrid.shape}')
    logger.info(f'New Kdata shape = {kdata.shape}')
    logger.info(f'New DCF shape = {dcf.shape}')

    # Estimate the arc length using discrete differences
    dk = np.diff(coord_regrid, axis=-2)   # Delta kspace along readout axis
    im_old_shape = np.array(dk.shape)
    im_old_shape[-2] = 1
    dk = np.concatenate((dk, np.zeros(im_old_shape)), axis=-2)

    dk_mag = np.sqrt(np.sum(dk**2, axis=-1)) # magnitude
    arc_length = np.sum(dk_mag, axis=-1) #to get arc length

    max_arc_length = np.max(arc_length)
    npts = ceil( max_arc_length / new_dk)
    logger.info(f'Max arc length = {max_arc_length}')
    logger.info(f'New pts = {npts}, old pts = {coord_regrid.shape[-2]}')

    # Now setup resample
    arc_length =np.cumsum(dk_mag, axis=-1) #to get arc length
    target_shape = np.array(arc_length.shape)
    target_shape[:] = 1
    target_shape[-2] = npts
    arc_sample = np.reshape(np.linspace(0, npts-1, npts), target_shape)*np.max(arc_length, -2, keepdims=True)/max_arc_length

    arc_sample = arc_length / max_arc_length * npts

    logger.info(f'Arc length shape {arc_length.shape}')
    logger.info(f'Arc sample shape {arc_sample.shape}')
    logger.info(f'Max arc sample = {np.max(arc_sample)}')
    logger.info(f'Min arc sample = {np.max(arc_sample)}')

    # Get the DCF for resampling
    est_grad = np.diff(arc_sample, axis=-1)
    arc_density = 1./(est_grad + 1e-3*np.max(est_grad))
    end_line = np.expand_dims(arc_density[...,-1], axis=-1)
    arc_density = np.concatenate([arc_density, end_line], axis=-1)
    arc_density /= np.max(arc_density)

    logger.info(f'Arc density shape {arc_density.shape}')

    plt_verbose = False
    if plt_verbose:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(arc_sample[0])
        plt.show()

        plt.figure()
        plt.plot(arc_density[0])
        plt.show()

    # This is a gridding of ones to estimate the density
    resampling_dcf = readout_gridding(np.ones_like(dcf), arc_sample, arc_density, (npts,), kernel="kaiser_bessel", width=width, param=beta)

    # Estimate the average DCF
    resampled_dcf = readout_gridding(dcf, arc_sample, arc_density, (npts,), kernel="kaiser_bessel", width=width, param=beta)
    resampled_dcf /= resampling_dcf

    # Resample for each coord dimension
    resampled_coord = []
    for dim in range(coord_regrid.shape[-1]):
        coord_t = coord_regrid[..., dim]

        # Regrid the data
        output = readout_gridding(coord_t, arc_sample, arc_density, (npts,), kernel="kaiser_bessel", width=width, param=beta)
        output /= resampling_dcf
        resampled_coord.append(output) 
    resampled_coord = np.stack(resampled_coord, axis=-1)

    if plt_verbose:
        plt.figure()
        plt.plot((resampled_coord[0]))
        plt.show()

        plt.figure()
        plt.plot((resampled_dcf[0]))
        plt.show()
    
        plt.figure()
        plt.plot((coord_t[0,:]))
        plt.show()
      
    # Grid to Cartesian Using
    resampled_kdata = []
    for coil in range(kdata.shape[0]):
        logger.info(f'Resample coil {coil}')

        kdata_t = kdata[coil]
        
        output = readout_gridding(kdata_t, arc_sample, arc_density, (npts,), kernel="kaiser_bessel", width=width, param=beta)
        output /= resampling_dcf
        
        resampled_kdata.append(output)
    resampled_kdata = np.stack(resampled_kdata, axis=0)

    dcf_output_shape = dcf_input_shape
    kdata_output_shape = kdata_input_shape
    coord_output_shape = coord_input_shape

    dcf_output_shape[-1] = npts
    kdata_output_shape[-1] = npts
    coord_output_shape[-2] = npts

    resampled_kdata = np.reshape(resampled_kdata, kdata_output_shape)
    resampled_dcf = np.reshape(resampled_dcf, dcf_output_shape)
    resampled_coord = np.reshape(resampled_coord, coord_output_shape)

    resampled_dcf = np.sum(resampled_coord**2, axis=-1)

    logger.info(f'resampled_coord.shape {resampled_coord.shape}')
    logger.info(f'resampled_kdata.shape {resampled_kdata.shape}')
    logger.info(f'resampled_dcf.shape {resampled_dcf.shape}')
    
    return resampled_coord, resampled_dcf, resampled_kdata 

def pca_coil_compression(kdata=None, axis=0, target_channels=None):
    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f'Compressing to {target_channels} channels, along axis {axis}')
    logger.info(f'Initial  size = {kdata_cc.shape} ')

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f'Old channels =  {old_channels} ')

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1 - 0.05])

    # Create a subsampled array
    kcc = np.zeros((old_channels, np.sum(mask)), dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[..., c]
        kcc[c, :] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e], -1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e] = np.matmul(u, kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e], -1, axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata, axis, -1)
        kdata = np.expand_dims(kdata, -1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata, axis=-1)
        kdata = kdata[..., :target_channels]

        # Put back
        kdata = np.moveaxis(kdata, -1, axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata


def get_smaps(mri_rawdata=None, args=None, smap_type='jsense', device=None, thresh_maps=True, log_dir=''):
    logger = logging.getLogger('Get sensitivity maps')

    # Set to GPU
    if device is None:
        device = sp.Device(0)

    op_device = device
    store_device = sp.cpu_device

    # Reference for shortcut
    coord = mri_rawdata.coords[0]
    dcf = mri_rawdata.dcf[0]
    kdata = mri_rawdata.kdata[0]

    if smap_type == 'espirit':

        # Low resolution images
        res = 64
        lpf = np.sum(coord ** 2, axis=-1)
        lpf = np.exp(-lpf / (2.0 * res * res))

        img_shape = sp.estimate_shape(coord)
        ksp = xp.ones([mri_rawdata.Num_Coils] + img_shape, dtype=xp.complex64)

        for c in range(mri_rawdata.Num_Coils):
            logger.info(f'Reconstructing  coil {c}')
            kxp = sp.get_device(kdata[c]).xp
            ksp_t = kxp.copy(kdata[c])
            ksp_t *= kxp.squeeze(dcf)
            ksp_t *= kxp.squeeze(lpf)
            ksp_t = array_to_gpu(ksp_t, device=device)
            coord_t = array_to_gpu(coord, device=device)

            ksp[c] = sp.nufft_adjoint(ksp_t, coord_t, img_shape)

        # Put onto CPU due to memory issues in ESPiRIT
        ksp = sp.to_device(ksp, sp.cpu_device)

        # Espirit Cal
        smaps = sp.mri.app.EspiritCalib(ksp, calib_width=24, thresh=0.02, kernel_width=6, crop=0.0, max_iter=100,
                                        device=sp.cpu_device, show_pbar=True).run()

    elif smap_type == 'walsh':

        logger = logging.getLogger('walsh')

        # Get a composite image
        img_shape = sp.estimate_shape(coord)
        image = xp.zeros([mri_rawdata.Num_Coils] + img_shape, dtype=xp.complex64)

        for e in range(mri_rawdata.Num_Encodings):
            kr = sp.get_device(mri_rawdata.coords[e]).xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
            kr = array_to_gpu(kr, device)
            lpf = xp.exp(-kr / (2 * (128. ** 2)))

            for c in range(mri_rawdata.Num_Coils):
                logger.info(f'Reconstructing encode, coil {e} , {c} ')
                ksp = array_to_gpu(mri_rawdata.kdata[e][c, ...], device)
                ksp *= array_to_gpu(mri_rawdata.dcf[e], device)
                ksp *= lpf
                coords_temp = sp.to_device(mri_rawdata.coords[e], device)
                image[c] += sp.nufft_adjoint(ksp, coords_temp, img_shape)

        # Need on CPU for reliable SVD
        image = sp.to_device(image)

        logger.info(f'Image size {image.shape}')

        # Block operator
        if len(image.shape) == 4:
            block_shape = [8, 8, 8]
            block_stride = [8, 8, 8]
        else:
            block_shape = [8, 8]
            block_stride = [8, 8]

        B = sp.linop.ArrayToBlocks(image.shape, block_shape, block_stride)

        # Grab blocks
        blocked_image = B*image
        logger.info(f'Blocked images size {blocked_image.shape}')

        # Reshape to blocks x pixels x coils
        blocked_image = np.moveaxis(blocked_image, 0, -1)  # First axis is coil
        old_shape = blocked_image.shape
        new_shape = (-1, np.prod(block_shape), blocked_image.shape[-1])
        blocked_image = np.reshape(blocked_image, new_shape)

        # Now do svd
        for bi in range(blocked_image.shape[0]):
            block = blocked_image[bi]

            [u, s, vh] = np.linalg.svd(block, full_matrices=False)
            s = (vh[0,:])
            s*= np.conj(s[0])/np.abs(s[0])

            # set phase to first coil
            coil0 = np.conj(block[:,0])
            coil0 /= np.abs(coil0) * np.prod(block_shape)


            blocked_image[bi] = coil0[:,np.newaxis]*s[np.newaxis,:]

        # Reshape back
        blocked_image = np.reshape(blocked_image, old_shape)
        blocked_image = np.moveaxis(blocked_image, -1, 0)  # First axis is coil

        smaps = B.H*blocked_image

    elif smap_type == "lowres":
        # Get a composite image
        img_shape = sp.estimate_shape(coord)
        image = store_device.xp.zeros([mri_rawdata.Num_Coils] + img_shape, dtype=store_device.xp.complex64)

        #for e in range(mri_rawdata.Num_Encodings):
        for e in range(1):
            xp = sp.get_device(mri_rawdata.coords[e]).xp
            kr = xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
            lpf = xp.exp(-kr / (2 * (32. ** 2)))
            lpf *= mri_rawdata.dcf[e]
            lpf = array_to_gpu(lpf, device)

            for c in range(mri_rawdata.Num_Coils):
                logger.info(f'Reconstructing encode, coil {e} , {c} ')
                ksp = array_to_gpu(mri_rawdata.kdata[e][c, ...], op_device)
                ksp *= lpf
                coords_temp = array_to_gpu(mri_rawdata.coords[e], op_device)
                image_temp = sp.nufft_adjoint(ksp, coords_temp, img_shape)
                image[c] += sp.to_device(image_temp, store_device)

        xp = store_device.xp
        sos = xp.sqrt(xp.sum(xp.abs(image) **2, axis=0))
        sos = xp.expand_dims(sos, axis=0)
        sos = sos + xp.max(sos)*1e-5

        smaps = image / sos

        print(f'Smaps Device = {sp.get_device(smaps)}')
    else:

        dcf = array_to_gpu(dcf, device)
        coord = array_to_gpu(coord, device)
        kdata = array_to_gpu(kdata, device)

        smaps = mr.app.JsenseRecon(kdata,
                                   coord=coord,
                                   weights=dcf,
                                   mps_ker_width=args.mps_ker_width,
                                   ksp_calib_width=args.ksp_calib_width,
                                   lamda=args.jsense_lamda,
                                   device=device,
                                   max_iter=args.jsense_max_iter,
                                   max_inner_iter=args.jsense_max_inner_iter).run()

        # Get a composite image
        img_shape = sp.estimate_shape(coord)
        image = 0
        for e in range(mri_rawdata.Num_Encodings):
            xp = sp.get_device(mri_rawdata.coords[e]).xp
            kr = xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
            kr = array_to_gpu(kr, device)
            lpf = xp.exp(-kr / (2 * (16. ** 2)))

            for c in range(mri_rawdata.Num_Coils):
                logger.info(f'Reconstructing encode, coil {e} , {c} ')
                ksp = array_to_gpu(mri_rawdata.kdata[e][c, ...], device)
                ksp *= array_to_gpu(mri_rawdata.dcf[e], device)
                ksp *= lpf
                coords_temp = array_to_gpu(mri_rawdata.coords[e], device)
                image += xp.abs(sp.nufft_adjoint(ksp, coords_temp, img_shape)) ** 2

        if thresh_maps:
            image = xp.sqrt(image)  # sos image
            image = array_to_gpu(image, sp.get_device(smaps))

            xp = sp.get_device(smaps).xp

            # Threshold
            image = xp.abs(image)
            image /= xp.max(image)
            thresh = 0.015
            # print(thresh)
            mask = image > thresh

            mask = sp.to_device(mask, sp.cpu_device)
            if len(mask.shape) == 3:
                zz, xx, yy = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
                rad = zz ** 2 + xx ** 2 + yy ** 2
            else:
                xx, yy = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
                rad = xx ** 2 + yy ** 2

            smap_mask = rad < 1.0
            # print(smap_mask)
            mask = ndimage.morphology.binary_dilation(mask, smap_mask)
            mask = np.array(mask, dtype=np.float32)
            mask = sp.to_device(mask, sp.get_device(smaps))

            # print(image)
            # print(image.shape)
            # print(smaps.shape)
            smaps = mask * smaps

    smaps_cpu = sp.to_device(smaps, store_device)
    # if thresh_maps:
    #     mask_cpu = sp.to_device(mask, sp.cpu_device)

    # Export to file
    out_name = os.path.join(log_dir,'SenseMaps.h5')
    logger.info('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        # hf.create_dataset("IMAGE", data=np.abs(image_cpu))
        hf.create_dataset("SMAPS", data=np.abs(smaps_cpu))
        # if thresh_maps:
        #     hf.create_dataset("MASK", data=np.abs(mask_cpu))

    return smaps


def sos_recon(mri_rawdata=None, device=None):
    # Set to GPU
    if device is None:
        device = sp.Device(0)

    xp = device.xp
    logger = logging.getLogger('SOS Recon')
    img_shape = sp.estimate_shape(mri_rawdata.coords)

    img = 0
    for c in range(mri_rawdata.Num_Coils):
        logger.info(f'Reconstructing coil {c}')
        ksp = array_to_gpu(mri_rawdata.kdata[c, ...], device)
        ksp *= array_to_gpu(mri_rawdata.dcf, device)
        coords_temp = array_to_gpu(mri_rawdata.coords, device)

        img += xp.abs(sp.nufft_adjoint(ksp, coords_temp, img_shape)) ** 2

    img = xp.sqrt(img)

    img = sp.to_device(img, sp.cpu_device)

    return img


def pils_recon(mri_rawdata=None, smaps=None, device=None):
    # Set to GPU
    if device is None:
        device = sp.Device(0)

    # Reference for shortcut
    coord = mri_rawdata.coords
    dcf = mri_rawdata.dcf
    kdata = mri_rawdata.kdata

    with device:
        pils = sp.mri.app.SenseRecon(kdata, mps=smaps, weights=dcf, coord=coord, device=device, max_iter=1,
                                     coil_batch_size=1)
        img = pils.run()

    img = sp.to_device(img, sp.cpu_device)

    return (img)


def crop_kspace(mri_rawdata=None, crop_factor=2, crop_type='radius'):
    logger = logging.getLogger('Recon images')

    # Get initial shape
    if isinstance(mri_rawdata.coords, list):
        img_shape = sp.estimate_shape(mri_rawdata.coords[0])
    else:
        img_shape = sp.estimate_shape(mri_rawdata.coords)

    # Crop kspace
    img_shape_new = np.floor(np.array(img_shape) / crop_factor)
    logger.info(f'New shape = {img_shape_new}, Old shape = {img_shape}')

    for e in range(len(mri_rawdata.coords)):

        # Find values where kspace is within bounds (square crop)
        if crop_type == 'radius':

            radius = np.sum(np.array(mri_rawdata.coords[e]) ** 2, -1)

            idx = np.argwhere(np.logical_and.reduce([radius < (img_shape_new[0] / 2) ** 2]))
        else:
            idx = np.argwhere(np.logical_and.reduce([np.abs(mri_rawdata.coords[e][..., 0]) < img_shape_new[0] / 2,
                                                     np.abs(mri_rawdata.coords[e][..., 1]) < img_shape_new[1] / 2,
                                                     np.abs(mri_rawdata.coords[e][..., 2]) < img_shape_new[2] / 2]))

        # Now crop
        mri_rawdata.coords[e] = mri_rawdata.coords[e][idx[:, 0], :]
        mri_rawdata.dcf[e] = mri_rawdata.dcf[e][idx[:, 0]]
        mri_rawdata.kdata[e] = mri_rawdata.kdata[e][:, idx[:, 0]]
        mri_rawdata.time[e] = mri_rawdata.time[e][idx[:, 0]]
        mri_rawdata.ecg[e] = mri_rawdata.ecg[e][idx[:, 0]]
        mri_rawdata.prep[e] = mri_rawdata.prep[e][idx[:, 0]]
        mri_rawdata.resp[e] = mri_rawdata.resp[e][idx[:, 0]]

        logger.info(f'New shape = {sp.estimate_shape(mri_rawdata.coords[e])}')


def get_gate_bins( gate_signal, gate_type, num_frames, discrete_gates=False, prep_disdaqs=0):
    logger = logging.getLogger('Get Gate bins')

    #print(gate_signal)
    #print(gate_signal[0].dtype)

    # Loop over all encodes
    t_min = np.min([np.min(gate) for gate in gate_signal])
    t_max = np.max([np.max(gate) for gate in gate_signal])

    if gate_type == 'ecg':
        logger.info('Using median ECG value for tmax')
        median_rr = np.mean([np.median(gate) for gate in gate_signal])
        median_rr = 2.0 * (median_rr - t_min) + t_min
        t_max = median_rr
        logger.info(f'Median RR = {median_rr}')

        # Check the range
        sum_within = np.sum([np.sum(gate < t_max) for gate in gate_signal])
        sum_total = np.sum([gate.size for gate in gate_signal])
        within_rr = 100.0 * sum_within / sum_total
        logger.info(f'ECG, {within_rr} percent within RR')
    elif gate_type == 'resp':
        # Outlier rejection
        q05 = np.mean([np.quantile(gate, 0.05) for gate in gate_signal])
        q95 = np.mean([np.quantile(gate, 0.95) for gate in gate_signal])

        # Linear fit
        t_max = q95 + (q95 - q05) / 0.9 * 0.05
        t_min = q05 + (q95 - q05) / 0.9 * -0.05
    elif gate_type == 'prep':
        # Skip a number of projections
        t_min = np.min([np.min(gate) for gate in gate_signal]) + prep_disdaqs


    if discrete_gates:
        t_min -= 0.5
        t_max += 0.5
    else:
        # Pad so bins are inclusive
        t_min -= 1e-6
        t_max += 1e-6

    logger.info(f'Max time = {t_max}')
    logger.info(f'Min time = {t_min}')

    delta_time = (t_max - t_min) / num_frames
    logger.info(f'Delta = {delta_time}')

    return t_min, t_max, delta_time


def gate_kspace2d(mri_raw=None, num_frames=[10, 10], gate_type=['time', 'prep'],
                  discrete_gates=[False, False], ecg_delay=300e-3, prep_disdaqs=0):
    logger = logging.getLogger('Gate k-space 2D')

    # Assume the input is a list

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils = mri_raw.Num_Coils
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []
    mri_rawG.ecg = []
    mri_rawG.prep = []
    mri_rawG.resp = []

    gate_signals = {
        'ecg': mri_raw.ecg,
        'time': mri_raw.time,
        'prep': mri_raw.prep,
        'resp': mri_raw.resp
    }
    gate_signal0 = gate_signals.get(gate_type[0], f'Cannot interpret gate signal {gate_type}')
    gate_signal1 = gate_signals.get(gate_type[1], f'Cannot interpret gate signal {gate_type}')

    # For ECG, delay the waveform
    if gate_type == 'ecg':
       time = mri_raw.time

       for e in range(mri_raw.Num_Encodings):
           time_encode = time[e].flatten()
           ecg_encode = gate_signal[e].flatten()

            #Sort the data by time
           idx = np.argsort(time_encode)
           idx_inverse = idx.argsort()

           # Estimate the delay
           if e == 0:
               print(f'Time max {time_encode.max()}')
               print(f'Time size {time_encode.size}')
               print(f'Time ecg delay {ecg_delay}')
               
               ecg_shift = int(ecg_delay / time_encode.max() * time_encode.size)
               print(f'Shifting by {ecg_shift}')

           #Using circular shift for now. This should be fixed
           ecg_sorted = ecg_encode[idx]
           ecg_shifted = np.roll( ecg_sorted, -ecg_shift)
           gate_signal[e] = np.reshape(ecg_shifted[idx_inverse], time[e].shape)


    print(f'Gating off of {gate_type}')

    t_min0, t_max0, delta_time0 = get_gate_bins(gate_signal0, gate_type[0], num_frames[0], discrete_gates[0])
    t_min1, t_max1, delta_time1 = get_gate_bins(gate_signal1, gate_type[1], num_frames[1], discrete_gates[1])


    points_per_bin = []
    count = 0
    for t0 in range(num_frames[0]):

        t_start0 = t_min0 + delta_time0 * t0
        t_stop0 = t_start0 + delta_time0

        for t1 in range(num_frames[1]):
            t_start1 = t_min1 + delta_time1 * t1
            t_stop1 = t_start1 + delta_time1

            for e in range(mri_raw.Num_Encodings):


                # Find index where value is held
                idx = np.logical_and.reduce([
                    np.abs(gate_signal0[e]) >= t_start0,
                    np.abs(gate_signal0[e]) < t_stop0,
                    np.abs(gate_signal1[e]) >= t_start1,
                    np.abs(gate_signal1[e]) < t_stop1])

                current_points = np.sum(idx)

                # Gate the data
                points_per_bin.append(current_points)

                logger.info(f'Frame {t0} [{t_start0} to {t_stop0} ] | [{t_start1} to {t_stop1} ]  {e}, Points = {current_points}')

                new_kdata = []
                for coil in range(mri_raw.kdata[e].shape[0]):
                    old_kdata = mri_raw.kdata[e][coil]
                    new_kdata.append(old_kdata[idx])
                mri_rawG.kdata.append(np.stack(new_kdata, axis=0))

                new_coords = []
                for dim in range(mri_raw.coords[e].shape[-1]):
                    old_coords = mri_raw.coords[e][...,dim]
                    new_coords.append(old_coords[idx])
                mri_rawG.coords.append(np.stack(new_coords, axis=-1))

                mri_rawG.dcf.append(mri_raw.dcf[e][idx])
                mri_rawG.time.append(mri_raw.time[e][idx])
                mri_rawG.resp.append(mri_raw.resp[e][idx])
                mri_rawG.prep.append(mri_raw.prep[e][idx])
                mri_rawG.ecg.append(mri_raw.ecg[e][idx])

                count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(
        f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    mri_rawG.Num_Frames = num_frames

    return (mri_rawG)

def gate_kspace(mri_raw=None, num_frames=10, gate_type='time', discrete_gates=False, ecg_delay=300e-3):
    logger = logging.getLogger('Gate k-space')

    # Assume the input is a list

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils = mri_raw.Num_Coils
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []
    mri_rawG.ecg = []
    mri_rawG.prep = []
    mri_rawG.resp = []

    gate_signals = {
        'ecg': mri_raw.ecg,
        'time': mri_raw.time,
        'prep': mri_raw.prep,
        'resp': mri_raw.resp
    }
    gate_signal = gate_signals.get(gate_type, f'Cannot interpret gate signal {gate_type}')

    # For ECG, delay the waveform
    if gate_type == 'ecg':
       time = mri_raw.time

       for e in range(mri_raw.Num_Encodings):
           time_encode = time[e].flatten()
           ecg_encode = gate_signal[e].flatten()

            #Sort the data by time
           idx = np.argsort(time_encode)
           idx_inverse = idx.argsort()

           # Estimate the delay
           if e == 0:
               print(f'Time max {time_encode.max()}')
               print(f'Time size {time_encode.size}')
               print(f'Time ecg delay {ecg_delay}')
               
               ecg_shift = int(ecg_delay / time_encode.max() * time_encode.size)
               print(f'Shifting by {ecg_shift}')

           #Using circular shift for now. This should be fixed
           ecg_sorted = ecg_encode[idx]
           ecg_shifted = np.roll( ecg_sorted, -ecg_shift)
           gate_signal[e] = np.reshape(ecg_shifted[idx_inverse], time[e].shape)

    print(f'Gating off of {gate_type}')

    t_min, t_max, delta_time = get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates)

    points_per_bin = []
    count = 0
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # # Find index where value is held
            # idx = np.argwhere(np.logical_and.reduce([
            #     np.abs(gate_signal[e]) >= t_start,
            #     np.abs(gate_signal[e]) < t_stop]))
            
            idx = np.logical_and.reduce([
                np.abs(gate_signal[e]) >= t_start,
                np.abs(gate_signal[e]) < t_stop])
            current_points = np.sum(idx)

            post_gate = gate_signal[e][idx]
            #print(f'Post gate min = {np.min(post_gate)}')
            #print(f'Post gate max = {np.max(post_gate)}')
            #print(f'Size of gate = {gate_signal[e].shape}')

            ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            # Gate the data
            points_per_bin.append(current_points)

            #print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
            logger.info(f'Frame {t} [{t_start} to {t_stop} ] | {e}, Points = {current_points}')

            #print(gate_signal[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(mri_raw.coords[e].shape)
            #print(idx.shape)

            # mri_rawG.coords.append(mri_raw.coords[e][idx[:, 0], :])
            # mri_rawG.dcf.append(mri_raw.dcf[e][idx[:, 0]])
            # mri_rawG.kdata.append(mri_raw.kdata[e][:, idx[:, 0]])
            # mri_rawG.time.append(mri_raw.time[e][idx[:, 0]])
            # mri_rawG.resp.append(mri_raw.resp[e][idx[:, 0]])
            # mri_rawG.prep.append(mri_raw.prep[e][idx[:, 0]])
            # mri_rawG.ecg.append(mri_raw.ecg[e][idx[:, 0]])

            new_kdata = []
            for coil in range(mri_raw.kdata[e].shape[0]):
                old_kdata = mri_raw.kdata[e][coil]
                new_kdata.append(old_kdata[idx])
            mri_rawG.kdata.append(np.stack(new_kdata, axis=0))

            new_coords = []
            for dim in range(mri_raw.coords[e].shape[-1]):
                old_coords = mri_raw.coords[e][...,dim]
                new_coords.append(old_coords[idx])
            mri_rawG.coords.append(np.stack(new_coords, axis=-1))

            mri_rawG.dcf.append(mri_raw.dcf[e][idx])
            mri_rawG.time.append(mri_raw.time[e][idx])
            mri_rawG.resp.append(mri_raw.resp[e][idx])
            mri_rawG.prep.append(mri_raw.prep[e][idx])
            mri_rawG.ecg.append(mri_raw.ecg[e][idx])

            #print(f'ECG Time before = {np.min(mri_raw.ecg[e])} {np.max(mri_raw.ecg[e])}')
            #print(f'ECG Time after = {np.min(mri_rawG.ecg[-1])} {np.max(mri_rawG.ecg[-1])}')

            ecg = mri_raw.ecg[e][idx]
            #print(f'Post ecg min = {np.min(ecg)}')
            #print(f'Post ecg max = {np.max(ecg)}')
            #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


            #mri_rawG.kdata.append(mri_raw.kdata[e][idx_kdata])
            #mri_rawG.coords.append(mri_raw.coords[e][idx_coord])
            
            count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(
        f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    mri_rawG.Num_Frames = num_frames

    return (mri_rawG)

@nb.jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def bounded_medfilt(signal, window):

  filtered = np.empty_like(signal)
  for i in range(len(signal)):
    start =  max(0, i - window)
    stop = start + 2*window + 1

    if stop > (len(signal)-1):
      start = len(signal)-1 - (2*window+1)
      stop = len(signal)-1

    if i % 10000 == 0:
        print(start)
    filtered[i] = np.median(signal[start:stop])

  filtered = signal - filtered

  return filtered 


def median_filter_resp(time, resp, window):
  
  # Get the sort index
  idx_sort = np.argsort(time)

  # Sort the data
  time_sorted = time[idx_sort]
  resp_sorted = resp[idx_sort]

  # Get an index to unsort the data
  idx_unsort = np.empty_like(idx_sort)
  idx_unsort[idx_sort] = np.arange(idx_sort.size)

  # Median filtered
  from scipy.signal import medfilt
  resp_filtered = bounded_medfilt(resp_sorted,window // 2)

  # Unsort back into as acquired data
  resp_unsorted = resp_filtered[idx_unsort] 
  
  return resp_unsorted


def resp_gate(mri_raw=None, efficiency=0.5, filter_resp=True):
    logger = logging.getLogger('Resp Gate k-space')

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils = mri_raw.Num_Coils
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type
    mri_rawG.Num_Frames = mri_raw.Num_Frames
    mri_rawG.Num_Encodings = mri_raw.Num_Encodings

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []
    mri_rawG.ecg = []
    mri_rawG.prep = []
    mri_rawG.resp = []

    points_per_bin = []
    count = 0
    for e in range(mri_raw.Num_Encodings):

        # Grab the time and respiratory waveforms
        time = mri_raw.time[e].flatten()
        resp = mri_raw.resp[e].flatten()

        # Estimate the TR
        dt = np.max(time) / len(time)
        resp_filter_width = int(10 / dt)

        logger.info(f'Estimated TR = {dt} based on {np.max(time)} s acquisition with {len(time)} points')
        logger.info(f'Using a filter window of {resp_filter_width}')
                
        # Filter the respiratory signal
        # filter_resp = median_filter_resp(time, resp, resp_filter_width)
        filter_resp = resp

        # Resp sorted to get threshold value
        sorted_resp = np.sort(filter_resp)
        resp_thresh = sorted_resp[int(len(filter_resp)*efficiency)]

        logger.info(f'Resp threshold = {resp_thresh}, based on {efficiency}')

        # Find index where value is held
        idx = mri_raw.resp[e] < resp_thresh
        current_points = np.sum(idx)

        # Gate the data
        points_per_bin.append(current_points)

        logger.info(f'Encode {e}, Points = {current_points}')

        new_kdata = []
        for coil in range(mri_raw.kdata[e].shape[0]):
            old_kdata = mri_raw.kdata[e][coil]
            new_kdata.append(old_kdata[idx])
        mri_rawG.kdata.append(np.stack(new_kdata, axis=0))

        new_coords = []
        for dim in range(mri_raw.coords[e].shape[-1]):
            old_coords = mri_raw.coords[e][...,dim]
            new_coords.append(old_coords[idx])
        mri_rawG.coords.append(np.stack(new_coords, axis=-1))

        mri_rawG.dcf.append(mri_raw.dcf[e][idx])
        mri_rawG.time.append(mri_raw.time[e][idx])
        mri_rawG.resp.append(mri_raw.resp[e][idx])
        mri_rawG.prep.append(mri_raw.prep[e][idx])
        mri_rawG.ecg.append(mri_raw.ecg[e][idx])

        count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    return (mri_rawG)


def load_MRI_raw(h5_filename=None, max_coils=None, max_encodes=None, compress_coils=False):
    with h5py.File(h5_filename, 'r') as hf:

        try:
            Num_Encodings = np.squeeze(hf['Kdata'].attrs['Num_Encodings'])
            Num_Coils = np.squeeze(hf['Kdata'].attrs['Num_Coils'])
            Num_Frames = np.squeeze(hf['Kdata'].attrs['Num_Frames'])

            trajectory_type = [np.squeeze(hf['Kdata'].attrs['trajectory_typeX']),
                               np.squeeze(hf['Kdata'].attrs['trajectory_typeY']),
                               np.squeeze(hf['Kdata'].attrs['trajectory_typeZ'])]

            dft_needed = [np.squeeze(hf['Kdata'].attrs['dft_neededX']), np.squeeze(hf['Kdata'].attrs['dft_neededY']),
                          np.squeeze(hf['Kdata'].attrs['dft_neededZ'])]

            logging.info(f'Frames {Num_Frames}')
            logging.info(f'Coils {Num_Coils}')
            logging.info(f'Encodings {Num_Encodings}')
            logging.info(f'Trajectory Type {trajectory_type}')
            logging.info(f'DFT Needed {dft_needed}')

        except Exception:
            logging.info('Missing header data')
            pass

        if max_coils is not None:
            Num_Coils = min(max_coils, Num_Coils)

        if max_encodes is not None:
            Num_Encodings = min(max_encodes, Num_Encodings)

        # Get the MRI Raw structure setup
        mri_raw = MRI_Raw()
        mri_raw.Num_Coils = int(Num_Coils)
        mri_raw.Num_Encodings = int(Num_Encodings)
        mri_raw.Num_Frames = int(Num_Frames)
        mri_raw.dft_needed = tuple(dft_needed)
        mri_raw.trajectory_type = tuple(trajectory_type)

        # List array
        mri_raw.coords = []
        mri_raw.dcf = []
        mri_raw.kdata = []
        mri_raw.time = []
        mri_raw.prep = []
        mri_raw.ecg = []
        mri_raw.resp = []

        for encode in range(Num_Encodings*Num_Frames):

            logging.info(f'Loading encode {encode}')

            # Get the coordinates
            coord = []
            for i in ['Z', 'Y', 'X']:
                logging.info(f'Loading {i} coord.')

                kcoord = np.array(hf['Kdata'][f'K{i}_E{encode}'])

                # Check range to distinguish 2D from 3D
                if i == 'Z':
                    krange = np.max(kcoord) - np.min(kcoord)
                    if krange < 1e-3:
                        continue
                coord.append(np.array(hf['Kdata'][f'K{i}_E{encode}']))
            coord = np.stack(coord, axis=-1)

            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])

            # Get k-space
            ksp = []
            for c in range(Num_Coils):
                logging.info(f'Loading kspace, coil {c + 1} / {Num_Coils}.')

                k = hf['Kdata'][f'KData_E{encode}_C{c}']
                try:
                    ksp.append(np.array(k['real'] + 1j * k['imag']))
                except:
                    ksp.append(k)
            ksp = np.stack(ksp, axis=0)


            # Regrid the readout to reduce oversampling
            # coord, dcf, ksp = radial3d_regrid(coord, dcf, ksp)

            # Load time data
            try:
                time_readout = np.array(hf['Gating']['time'])
            except Exception:
                time_readout = np.array(hf['Gating'][f'TIME_E{encode}'])

            try:
                ecg_readout = np.array(hf['Gating']['ecg'])
            except Exception:
                ecg_readout = np.array(hf['Gating'][f'ECG_E{encode}'])

            '''
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist( ecg_readout.flatten(), bins=100)
            plt.show()
            '''

            try:
                prep_readout = np.array(hf['Gating']['prep'])
            except Exception:
                prep_readout = np.array(hf['Gating'][f'PREP_E{encode}'])

            try:
                resp_readout = np.array(hf['Gating']['resp'])
            except Exception:
                resp_readout = np.array(hf['Gating'][f'RESP_E{encode}'])

            if resp_readout.size != dcf.size:

                # This assigns the same time to each point in the readout
                time_readout = np.expand_dims(time_readout, -1)
                ecg_readout = np.expand_dims(ecg_readout, -1)
                resp_readout = np.expand_dims(resp_readout, -1)
                prep_readout = np.expand_dims(prep_readout, -1)

                time = np.tile(time_readout, (1, 1, dcf.shape[2]))
                resp = np.tile(resp_readout, (1, 1, dcf.shape[2]))
                ecg = np.tile(ecg_readout, (1, 1, dcf.shape[2]))
                prep = np.tile(prep_readout, (1, 1, dcf.shape[2]))

                #prep = prep.flatten()
                #resp = resp.flatten()
                #ecg = ecg.flatten()
                #dcf = dcf.flatten()
                #time = time.flatten()
                print(f'Min/max = {np.min(time)} {np.max(time)}')

            else:
                print('No more flattening')
                #time = time_readout.flatten()
                #resp = resp_readout.flatten()
                #ecg = ecg_readout.flatten()
                #prep = prep_readout.flatten()
                #dcf = dcf.flatten()

            # Append to list and flatten
            mri_raw.dcf.append(dcf.flatten())

            ksp2 = []
            for e in range(Num_Coils):
                ksp2.append(ksp[e].flatten())
            ksp2 = np.stack(ksp2, axis=0)

            coords2 = []
            for dim in range(coord.shape[-1]):
                coords2.append(coord[...,dim].flatten())
            coords2 = np.stack(coords2, axis=-1)

            mri_raw.coords.append(coords2)

            mri_raw.kdata.append(ksp2)
            mri_raw.time.append(time.flatten())
            mri_raw.prep.append(prep.flatten())
            mri_raw.ecg.append(ecg.flatten())
            mri_raw.resp.append(resp.flatten())

            # Log the data
            logging.info(f'MRI coords {mri_raw.coords[encode].shape}')
            logging.info(f'MRI dcf {mri_raw.dcf[encode].shape}')
            logging.info(f'MRI kdata {mri_raw.kdata[encode].shape}')
            logging.info(f'MRI time {mri_raw.time[encode].shape}')
            logging.info(f'MRI ecg {mri_raw.ecg[encode].shape}')
            logging.info(f'MRI resp {mri_raw.resp[encode].shape}')
            logging.info(f'MRI prep {mri_raw.prep[encode].shape}')

        '''
        try:
            noise = hf['Kdata']['Noise']['real'] + 1j * hf['Kdata']['Noise']['imag']

            logging.info('Whitening ksp.')
            cov = mr.util.get_cov(noise)
            ksp = mr.util.whiten(ksp, cov)
        except Exception:
            ksp /= np.abs(ksp).max()
            logging.info('No noise data.')
            pass
        '''

        # Scale k-space to max 1
        kdata_max = [np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(f'Max kdata {kdata_max}')
        kdata_max = np.max(np.array(kdata_max))
        for ksp in mri_raw.kdata:
            ksp /= kdata_max

        kdata_max = [np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(f'Max kdata {kdata_max}')

        if compress_coils:
            # Compress Coils
            if 18 < Num_Coils <= 32:
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=20)
                mri_raw.Num_Coils = 20

            if Num_Coils > 32:
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=20)
                mri_raw.Num_Coils = 20

        return mri_raw



def save_MRI_raw(mri_raw, h5_filename=None):
    import os
    try:
        os.remove(h5_filename)
    except OSError:
        pass

    with h5py.File(h5_filename, 'w') as hf:

        grp = hf.create_group("Kdata")
        gate_grp = hf.create_group("Gating")

        grp.attrs['Num_Encodings'] = mri_raw.Num_Encodings
        grp.attrs['Num_Coils'] = mri_raw.Num_Coils
        grp.attrs['Num_Frames'] = 1 if mri_raw.Num_Frames is None else mri_raw.Num_Frames

        grp.attrs['trajectory_typeX'] = mri_raw.trajectory_type[2]
        grp.attrs['trajectory_typeY'] = mri_raw.trajectory_type[1]
        grp.attrs['trajectory_typeZ'] = mri_raw.trajectory_type[0]

        grp.attrs['dft_neededX'] = mri_raw.dft_needed[2]
        grp.attrs['dft_neededY'] = mri_raw.dft_needed[1]
        grp.attrs['dft_neededZ'] = mri_raw.dft_needed[0]

        for encode in range(len(mri_raw.kdata)):

            logging.info(f'Writing encode {encode}')

            # Write the coordinates
            grp.create_dataset(f'KX_E{encode}',data=mri_raw.coords[encode][...,2])
            grp.create_dataset(f'KY_E{encode}',data=mri_raw.coords[encode][...,1])
            grp.create_dataset(f'KZ_E{encode}',data=mri_raw.coords[encode][...,0])
            grp.create_dataset(f'KW_E{encode}',data=mri_raw.dcf[encode])

            # Write the gating
            gate_grp.create_dataset(f'TIME_E{encode}', data=mri_raw.time[encode])
            gate_grp.create_dataset(f'ECG_E{encode}', data=mri_raw.ecg[encode])
            gate_grp.create_dataset(f'PREP_E{encode}', data=mri_raw.prep[encode])
            gate_grp.create_dataset(f'RESP_E{encode}', data=mri_raw.resp[encode])

            # Write k-space
            for c in range(mri_raw.Num_Coils):
                logging.info(f'Writing kspace, coil {c + 1} / {mri_raw.Num_Coils}.')

                grp.create_dataset(f'KData_E{encode}_C{c}', data=mri_raw.kdata[encode][c])




def autofov(mri_raw=None, device=None,
            thresh=0.05, scale=1, oversample=2.0, square=True, block_size=8, logdir=None):
    logger = logging.getLogger('autofov')

    # Set to GPU
    if device is None:
        device = sp.Device(0)
    logger.info(f'Device = {device}')
    xp = device.xp

    with device:
        # Put on GPU
        coord = oversample * mri_raw.coords[0]
        dcf = mri_raw.dcf[0]

        # Low resolution filter
        res = 64
        lpf = np.sum(coord ** oversample, axis=-1)
        lpf = np.exp(-lpf / (2.0 * res * res))

        # Get reconstructed size
        img_shape = sp.estimate_shape(coord)
        img_shape = [int(min(i, 64)) for i in img_shape]
        images = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
        kdata = mri_raw.kdata

        sos = xp.zeros(img_shape, dtype=xp.float32)

        logger.info(f'Kdata shape = {kdata[0].shape}')
        logger.info(f'Images shape = {images.shape}')

        coord_gpu = array_to_gpu(coord, device=device)  # coord needs to be push to device in new sigpy version
        
        for c in range(mri_raw.Num_Coils):
            logger.info(f'Reconstructing  coil {c}')
            ksp_t = np.copy(kdata[0][c, ...])
            ksp_t *= np.squeeze(dcf)
            ksp_t *= np.squeeze(lpf)
            ksp_t = array_to_gpu(ksp_t, device=device)
            
            sos += xp.square(xp.abs(sp.nufft_adjoint(ksp_t, coord_gpu, img_shape)))

        # Multiply by SOS of low resolution maps
        sos = xp.sqrt(sos)

    sos = sp.to_device(sos)

    ndim = len(sos.shape)

    # Spherical mask
    if len(sos.shape) == 3:
        zz, xx, yy = np.meshgrid(np.linspace(-1, 1, sos.shape[0]),
                                 np.linspace(-1, 1, sos.shape[1]),
                                 np.linspace(-1, 1, sos.shape[2]), indexing='ij')
        rad = zz ** 2 + xx ** 2 + yy ** 2
    elif len(sos.shape) == 2:
        xx, yy = np.meshgrid(np.linspace(-1, 1, sos.shape[0]),
                             np.linspace(-1, 1, sos.shape[1]), indexing='ij')
        rad = xx ** 2 + yy ** 2

    idx = (rad >= 1.0)
    sos[idx] = 0.0

    # Export to file
    out_name = os.path.join(logdir, 'AutoFOV.h5')
    logger.info('Saving autofov to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=np.abs(sos))

    boxc = sos > thresh * sos.max()
    boxc_idx = np.nonzero(boxc)
    boxc_center = np.array(img_shape) // 2
    boxc_shape = np.array([2*int(oversample * max(c - min(boxc_idx[i]), max(boxc_idx[i]) - c) * scale)
                           for i, c in zip(range(3), boxc_center)])

    # Get the Box size
    logger.info(f'Box center: {boxc_center}')
    logger.info(f'Box shape: {boxc_shape}')
    

    #  Due to double FOV scale by 2
    target_recon_scale = boxc_shape / img_shape
    logger.info(f'Target recon scale: {target_recon_scale}')

    # Scale to new FOV
    target_recon_size = sp.estimate_shape(coord) * target_recon_scale  / oversample
    if square:
        target_recon_size[:] = np.max(target_recon_size)

    # Round for blocks and FFT
    target_recon_size = block_size * np.ceil(target_recon_size / block_size)

    # Get the actual scale without rounding
    ndim = coord.shape[-1]

    with sp.get_device(coord):
        img_scale = [(oversample * target_recon_size[i] / (coord[..., i].max() - coord[..., i].min())) for i in range(ndim)]

    # fix precision errors in x dir
    for i in range(ndim):
        round_img_scale = round(img_scale[i], 6)
        if round_img_scale - img_scale[i] < 0:
            round_img_scale += 0.000001
        img_scale[i] = round_img_scale

    logger.info(f'Target recon size: {target_recon_size}')
    logger.info(f'Kspace Scale: {img_scale}')

    for e in range(len(mri_raw.coords)):
        mri_raw.coords[e] *= img_scale

    new_img_shape = sp.estimate_shape(mri_raw.coords[0])
    print(sp.estimate_shape(mri_raw.coords[0]))


    #mri_raw.time[0] = np.arange(len(mri_raw.time[0]))
    # print(sp.estimate_shape(mri_raw.coords[1]))
    # print(sp.estimate_shape(mri_raw.coords[2]))
    # print(sp.estimate_shape(mri_raw.coords[3]))
    # print(sp.estimate_shape(mri_raw.coords[4]))

    logger.info('Image shape: {}'.format(new_img_shape))
