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
import torch as torch
import os
import scipy.ndimage as ndimage

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--thresh', type=float, default=0.12)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
    parser.add_argument('--mps_ker_width', type=int, default=16)
    parser.add_argument('--ksp_calib_width', type=int, default=24)
    parser.add_argument('--lamda', type=float, default=0.000005)
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--jsense_max_iter', type=int, default=30)
    parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--compress_coils', type=int, default=20)
    parser.add_argument('--crop_factor', type=int, default=1.25)

    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)')
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')

    args = parser.parse_args()

    # For tracking memory
    mempool = cupy.get_default_memory_pool()

    # Put up a file selector if the file is not specified
    if args.filename is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()
        args.filename = askopenfilename()

    if args.out_folder is None:
        args.out_folder = os.path.dirname(os.path.abspath(args.filename))

    # Save to Folder
    logger.info(f'Saving to {args.out_folder}')

    # Load Data
    logger.info(f'Load MRI from {args.filename}')
    mri_raw = load_MRI_raw(h5_filename=args.filename)

    # Compress Coils
    if mri_raw.Num_Coils > args.compress_coils:
        mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=args.compress_coils)
        mri_raw.Num_Coils = args.compress_coils

    num_enc = mri_raw.Num_Encodings
    crop_kspace(mri_rawdata=mri_raw, crop_factor=args.crop_factor)

    # Reconstruct an low res image and get the field of view
    logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale)

    # Get sensitivity maps
    logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    smaps = get_smaps(mri_rawdata=mri_raw, args=args)

    # Take complex diff
    mri_raw.kdata[0] -= mri_raw.kdata[1]
    mri_raw.Num_Encodings -= 1
    num_enc = mri_raw.Num_Encodings

    # Gate k-space
    mri_raw = gate_kspace(mri_raw=mri_raw, num_frames=args.frames, gate_type='time')  # control num of time frames

    # Reconstruct the image
    recon_type = 'llr'
    # recon_type = 'multiscale'

    if recon_type == 'multiscale':
        print(f'mri_raw.kdata.shape = {mri_raw.kdata.shape}')
        print(f'mri_raw.dcf.shape = {mri_raw.dcf.shape}')
        print(f'mri_raw.coords.shape = {mri_raw.coords.shape}')

        comm = sp.Communicator()

        lrimg = MultiScaleLowRankRecon(mri_raw.kdata, coord=mri_raw.coords, dcf=mri_raw.dcf, mps=smaps,
                                       sgw=None,
                                       blk_widths=[64, 128, 256],
                                       T=None,
                                       lamda=args.lamda,
                                       max_epoch=args.epochs,
                                       device=sp.Device(args.device), comm=comm).run()

        out_name = os.path.join(args.out_folder, 'MPNrageLRObject.h5')
        lrimg.save(out_name)

        # generate some slices
        logger.info('Generating slices for export')
        lrimg.use_device(sp.Device(args.device))
        Sz = lrimg[:, lrimg.shape[1] // 2, :, :]
        Sy = lrimg[:, :, lrimg.shape[2] // 2, :]
        Sx = lrimg[:, :, :, lrimg.shape[3] // 2]

        mipx = []
        mipy = []
        mipz = []
        for t in range(lrimg.shape[0]):
            img_t = lrimg[t]
            img_t = np.squeeze(np.abs(img_t))
            mipz.append(np.amax(img_t, (0,)))
            mipy.append(np.amax(img_t, (1,)))
            mipx.append(np.amax(img_t, (2,)))

        out_name = os.path.join(args.out_folder, 'ExampleSlices.h5')
        logger.info('Saving images to ' + out_name)
        try:
            os.remove(out_name)
        except OSError:
            pass
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset('Sz', data=np.abs(Sz))
            hf.create_dataset('Sy', data=np.abs(Sy))
            hf.create_dataset('Sx', data=np.abs(Sx))
            hf.create_dataset('mipx', data=np.abs(mipx))
            hf.create_dataset('mipy', data=np.abs(mipy))
            hf.create_dataset('mipz', data=np.abs(mipz))

        out_name = os.path.join(args.out_folder, 'FullImages.h5')
        logger.info('Saving images to ' + out_name)
        try:
            os.remove(out_name)
        except OSError:
            pass
        with h5py.File(out_name, 'w') as hf:
            for t in range(lrimg.shape[0]):
                img_t = lrimg[t]
                hf.create_dataset(f'Image{t:05}', data=np.squeeze(np.abs(img_t)))

    else:
        logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        img = BatchedSenseRecon(mri_raw.kdata, mps=smaps, weights=mri_raw.dcf, coord=mri_raw.coords,
                                device=sp.Device(args.device), lamda=args.lamda, num_enc=num_enc,
                                coil_batch_size=1, max_iter=args.max_iter, fast_maxeig=True).run()

        img = sp.to_device(img, sp.cpu_device)

        # Copy back to make easy
        smaps = sp.to_device(smaps, sp.cpu_device)
        smaps_mag = np.abs(smaps)

        img = sp.to_device(img, sp.cpu_device)
        img_mag = np.abs(img)
        img_phase = np.angle(img)

        # Export to file
        out_name = os.path.join(args.out_folder, 'FullRecon.h5')
        logger.info('Saving images to ' + out_name)
        try:
            os.remove(out_name)
        except OSError:
            pass
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset("IMAGE", data=img)
            hf.create_dataset("IMAGE_MAG", data=img_mag)
            hf.create_dataset("IMAGE_PHASE", data=img_phase)
            hf.create_dataset("SMAPS", data=smaps_mag)
