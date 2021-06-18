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

    xp = device.xp

    with device:

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
                ksp_t = sp.to_device(ksp_t, device=device)
                coord_t = sp.to_device(coord, device=device)

                ksp[c] = sp.nufft_adjoint(ksp_t, coord_t, img_shape)

            # Put onto CPU due to memory issues in ESPiRIT
            ksp = sp.to_device(ksp, sp.cpu_device)

            # Espirit Cal
            smaps = sp.mri.app.EspiritCalib(ksp, calib_width=24, thresh=0.02, kernel_width=6, crop=0.0, max_iter=100,
                                            device=sp.cpu_device, show_pbar=True).run()

        elif smap_type == 'walsh':

            # Get a composite image
            img_shape = sp.estimate_shape(coord)
            image = xp.zeros([mri_rawdata.Num_Coils] + img_shape, dtype=xp.complex64)

            for e in range(mri_rawdata.Num_Encodings):
                kr = sp.get_device(mri_rawdata.coords[e]).xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
                kr = sp.to_device(kr, device)
                lpf = xp.exp(-kr / (2 * (128. ** 2)))

                for c in range(mri_rawdata.Num_Coils):
                    logger.info(f'Reconstructing encode, coil {e} , {c} ')
                    ksp = sp.to_device(mri_rawdata.kdata[e][c, ...], device)
                    ksp *= sp.to_device(mri_rawdata.dcf[e], device)
                    ksp *= lpf
                    coords_temp = sp.to_device(mri_rawdata.coords[e], device)
                    image[c] += sp.nufft_adjoint(ksp, coords_temp, img_shape)

            # Need on CPU for reliable SVD
            image = sp.to_device(image)

            # Block operator
            block_shape = [16, 16]
            B = sp.linop.ArrayToBlocks(image.shape, block_shape, [1,1])

            # Grab blocks
            blocked_image = B*image

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
            blocked_ksp_calib_widthimage = np.reshape(blocked_image, old_shape)
            blocked_image = np.moveaxis(blocked_image, -1, 0)  # First axis is coil

            smaps = B.H*blocked_image
        elif smap_type == "lowres":
            # Get a composite image
            img_shape = sp.estimate_shape(coord)
            image = xp.zeros([mri_rawdata.Num_Coils] + img_shape, dtype=xp.complex64)

            for e in range(mri_rawdata.Num_Encodings):
                kr = sp.get_device(mri_rawdata.coords[e]).xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
                kr = sp.to_device(kr, device)
                lpf = xp.exp(-kr / (2 * (32. ** 2)))

                for c in range(mri_rawdata.Num_Coils):
                    logger.info(f'Reconstructing encode, coil {e} , {c} ')
                    ksp = sp.to_device(mri_rawdata.kdata[e][c, ...], device)
                    ksp *= sp.to_device(mri_rawdata.dcf[e], device)
                    ksp *= lpf
                    coords_temp = sp.to_device(mri_rawdata.coords[e], device)
                    image[c] += sp.nufft_adjoint(ksp, coords_temp, img_shape)

            # Need on CPU for reliable SVD
            image = sp.to_device(image)

            sos = np.sqrt( np.sum( np.abs(image) **2, axis=0))
            sos = np.expand_dims(sos, axis=0)
            sos = sos + np.max(sos)*1e-5

            smaps = image / sos

        else:

            dcf = sp.to_device(dcf, device)
            coord = sp.to_device(coord, device)
            kdata = sp.to_device(kdata, device)

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
                kr = sp.get_device(mri_rawdata.coords[e]).xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
                kr = sp.to_device(kr, device)
                lpf = xp.exp(-kr / (2 * (16. ** 2)))

                for c in range(mri_rawdata.Num_Coils):
                    logger.info(f'Reconstructing encode, coil {e} , {c} ')
                    ksp = sp.to_device(mri_rawdata.kdata[e][c, ...], device)
                    ksp *= sp.to_device(mri_rawdata.dcf[e], device)
                    ksp *= lpf
                    coords_temp = sp.to_device(mri_rawdata.coords[e], device)
                    image += xp.abs(sp.nufft_adjoint(ksp, coords_temp, img_shape)) ** 2

            if thresh_maps:
                image = xp.sqrt(image)  # sos image
                image = sp.to_device(image, sp.get_device(smaps))

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

    smaps_cpu = sp.to_device(smaps, sp.cpu_device)
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
        ksp = sp.to_device(mri_rawdata.kdata[c, ...], device)
        ksp *= sp.to_device(mri_rawdata.dcf, device)
        coords_temp = sp.to_device(mri_rawdata.coords, device)

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


def get_gate_bins( gate_signal, gate_type, num_frames, discrete_gates=False):
    logger = logging.getLogger('Get Gate bins')

    print(gate_signal)
    print(gate_signal[0].dtype)

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

def gate_kspace(mri_raw=None, num_frames=10, gate_type='time', discrete_gates=False):
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

    print(f'Gating off of {gate_type}')

    t_min, t_max, delta_time = get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates)

    points_per_bin = []
    count = 0
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # Find index where value is held
            idx = np.argwhere(np.logical_and.reduce([
                np.abs(gate_signal[e]) >= t_start,
                np.abs(gate_signal[e]) < t_stop]))
            current_points = len(idx)

            # Gate the data
            points_per_bin.append(current_points)

            print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
            logger.info(f'Frame {t} [{t_start} to {t_stop} ] | {e}, Points = {current_points}')

            print(gate_signal[e].shape)
            print(mri_raw.coords[e].shape)

            mri_rawG.coords.append(mri_raw.coords[e][idx[:, 0], :])
            mri_rawG.dcf.append(mri_raw.dcf[e][idx[:, 0]])
            mri_rawG.kdata.append(mri_raw.kdata[e][:, idx[:, 0]])
            mri_rawG.time.append(mri_raw.time[e][idx[:, 0]])
            mri_rawG.resp.append(mri_raw.resp[e][idx[:, 0]])
            mri_rawG.prep.append(mri_raw.prep[e][idx[:, 0]])
            mri_rawG.ecg.append(mri_raw.ecg[e][idx[:, 0]])

            count += 1

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')
    logger.info(
        f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
    logger.info(f'Standard deviation = {np.std(points_per_bin)}')

    mri_rawG.Num_Frames = num_frames

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
                coord.append(np.array(hf['Kdata'][f'K{i}_E{encode}']).flatten())
            coord = np.stack(coord, axis=-1)
            print(coord.shape)

            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])

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

                prep = prep.flatten()
                resp = resp.flatten()
                ecg = ecg.flatten()
                dcf = dcf.flatten()
                time = time.flatten()
                print(f'Min/max = {np.min(time)} {np.max(time)}')

            else:
                time = time_readout.flatten()
                resp = resp_readout.flatten()
                ecg = ecg_readout.flatten()
                prep = prep_readout.flatten()
                dcf = dcf.flatten()

            # Get k-space
            ksp = []
            for c in range(Num_Coils):
                logging.info(f'Loading kspace, coil {c + 1} / {Num_Coils}.')

                k = hf['Kdata'][f'KData_E{encode}_C{c}']
                try:
                    ksp.append(np.array(k['real'] + 1j * k['imag']).flatten())
                except:
                    ksp.append(k)
            ksp = np.stack(ksp, axis=0)

            # Append to list
            mri_raw.coords.append(coord)
            mri_raw.dcf.append(dcf)
            mri_raw.kdata.append(ksp)
            mri_raw.time.append(time)
            mri_raw.prep.append(prep)
            mri_raw.ecg.append(ecg)
            mri_raw.resp.append(resp)

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
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=28)
                mri_raw.Num_Coils = 28

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



def load_MRI_raw_ou(h5_filename=None, max_coils=None, compress_coils=False):


    proj = 688*12
    prep = np.arange(688*12) % 688
    idx = np.where( (prep < 320) & (prep%2==0) )

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

        # Get the MRI Raw structure setup
        mri_raw = MRI_Raw()
        mri_raw.Num_Coils = int(Num_Coils)
        mri_raw.Num_Encodings = int(Num_Encodings*Num_Frames)
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

        for encode in range(Num_Encodings):

            logging.info(f'Loading encode {encode}')

            # Get the coordinates
            coord = []
            for i in ['Z', 'Y', 'X']:
                logging.info(f'Loading {i} coord.')

                kcoord = np.array(hf['Kdata'][f'K{i}_E{encode}'])[:,idx,...]

                # Check range to distinguish 2D from 3D
                if i == 'Z':
                    krange = np.max(kcoord) - np.min(kcoord)
                    if krange < 1e-3:
                        continue
                coord.append(kcoord.flatten())
            coord = np.stack(coord, axis=-1)

            ##Rotate the image
            #coordR =  math.cos(math.pi*37/180)*coord[...,0] + math.sin(math.pi*37/180)*coord[...,1]
            #coordI = -math.sin(math.pi*37/180)*coord[...,0] + math.cos(math.pi*37/180)*coord[...,1]
            #coord = np.stack([coordR, coordI], axis=-1)

            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])[:,idx,...]
            #dcf = np.ones_like(dcf)

            # Load time data
            try:
                time_readout = np.array(hf['Gating']['time'])[:,idx,...]
            except Exception:
                time_readout = np.array(hf['Gating'][f'TIME_E{encode}'])[:,idx,...]

            temp = np.array(hf['Gating'][f'TIME_E{encode}'])

            print(temp.shape)

            try:
                ecg_readout = np.array(hf['Gating']['ecg'])[:,idx,...]
            except Exception:
                ecg_readout = np.array(hf['Gating'][f'ECG_E{encode}'])[:,idx,...]

            '''
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist( ecg_readout.flatten(), bins=100)
            plt.show()
            '''

            try:
                prep_readout = np.array(hf['Gating']['prep'])[:,idx,...]
            except Exception:
                prep_readout = np.array(hf['Gating'][f'PREP_E{encode}'])[:,idx,...]

            try:
                resp_readout = np.array(hf['Gating']['resp'])[:,idx,...]
            except Exception:
                resp_readout = np.array(hf['Gating'][f'RESP_E{encode}'])[:,idx,...]

            # This assigns the same time to each point in the readout
            time_readout = np.expand_dims(time_readout, -1)
            ecg_readout = np.expand_dims(ecg_readout, -1)
            resp_readout = np.expand_dims(resp_readout, -1)
            prep_readout = np.expand_dims(prep_readout, -1)

            print(f'DCF shape = {dcf.shape}')
            print(f'Time shape = {time_readout.shape}')
            time = np.tile(time_readout, (1, 1, dcf.shape[-1]))
            resp = np.tile(resp_readout, (1, 1, dcf.shape[-1]))
            ecg = np.tile(ecg_readout, (1, 1, dcf.shape[-1]))
            prep = np.tile(prep_readout, (1, 1, dcf.shape[-1]))

            print(f'Time shape = {time.shape}')

            prep = prep.flatten()
            resp = resp.flatten()
            ecg = ecg.flatten()
            dcf = dcf.flatten()
            time = time.flatten()

            # Get k-space
            ksp = []
            mri_raw.Num_Coils -= 3

            for c in range(Num_Coils):

                if c==9 or c==13 or c==14:
                    continue

                logging.info(f'Loading kspace, coil {c + 1} / {Num_Coils}.')

                k = hf['Kdata'][f'KData_E{encode}_C{c}']
                ksp.append(np.array(k['real'] + 1j * k['imag'])[:,idx,...].flatten())
            ksp = np.stack(ksp, axis=0)

            # Append to list
            mri_raw.coords.append(coord)
            mri_raw.dcf.append(dcf)
            mri_raw.kdata.append(ksp)
            mri_raw.time.append(time)
            mri_raw.prep.append(prep)
            mri_raw.ecg.append(ecg)
            mri_raw.resp.append(resp)

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
                mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=28)
                mri_raw.Num_Coils = 28

        return mri_raw

def autofov(mri_raw=None, device=None,
            thresh=0.05, scale=1, oversample=2.0, square=True):
    logger = logging.getLogger('autofov')

    # Set to GPU
    if device is None:
        device = sp.Device(0)
    logger.info(f'Device = {device}')
    xp = device.xp

    with device:
        # Put on GPU
        coord = 2.0 * mri_raw.coords[0]
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

        coord_gpu = sp.to_device(coord, device=device)  # coord needs to be push to device in new sigpy version

        for c in range(mri_raw.Num_Coils):
            logger.info(f'Reconstructing  coil {c}')
            ksp_t = np.copy(kdata[0][c, ...])
            ksp_t *= np.squeeze(dcf)
            ksp_t *= np.squeeze(lpf)
            ksp_t = sp.to_device(ksp_t, device=device)

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
    out_name = 'AutoFOV.h5'
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
    boxc_shape = np.array([int(2 * max(c - min(boxc_idx[i]), max(boxc_idx[i]) - c) * scale)
                           for i, c in zip(range(3), boxc_center)])

    #  Due to double FOV scale by 2
    target_recon_scale = boxc_shape / img_shape
    logger.info(f'Target recon scale: {target_recon_scale}')

    # Scale to new FOV
    target_recon_size = sp.estimate_shape(coord) * target_recon_scale
    if square:
        target_recon_size[:] = np.max(target_recon_size)

    # Round to 16 for blocks and FFT
    target_recon_size = 16 * np.ceil(target_recon_size / 16)

    # Get the actual scale without rounding
    ndim = coord.shape[-1]

    with sp.get_device(coord):
        img_scale = [(2.0 * target_recon_size[i] / (coord[..., i].max() - coord[..., i].min())) for i in range(ndim)]

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
