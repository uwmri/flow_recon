#! /usr/bin/env python

import os
import time
import math
import logging

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import numba as nb
import h5py
import cupy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from mri_raw import *
from llr_recon import *
from registration_tools import *
from svt import *
from gpu_ops import *


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    # PARSE COMMAND LINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
    parser.add_argument('--frames2', type=int, default=1, help='Number of time frames')

    parser.add_argument('--lamda', type=float, default=0.0001)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--smap_type', type=str, default='lowres', help='Sensitvity type: lowres, walsh, espirit')

    parser.add_argument('--krad_cutoff', type=float, default=999990)
    parser.add_argument('--max_encodes', type=int, default=None)
    parser.add_argument('--coil_batch_size', type=int, default=1)

    parser.add_argument('--gate_type', type=str, default='time')
    parser.add_argument('--gate_type2', type=str, default='prep')
    parser.add_argument('--prep_disdaqs', type=int, default=0)
    parser.add_argument('--crop_factor', type=float, default=1.0)
    parser.add_argument('--recon_type', type=str, default='sense')

    parser.add_argument('--discrete_gates', dest='discrete_gates', action='store_true', default=False)
    parser.add_argument('--discrete_gates2', dest='discrete_gates2', action='store_true', default=False)
    parser.add_argument('--resp_gate', dest='resp_gate', action='store_true', default=False)
    parser.add_argument('--compress_coils', dest='compress_coils', action='store_true', default=False)

    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='FullRecon.h5')

    parser.add_argument('--sms_phase', type=int, default=0)  # phase flip (-1=-pi/2, 0=0, 1=pi/2)

    # START RECON
    args = parser.parse_args()
    mempool = cupy.get_default_memory_pool()  # For tracking memory

    # Put up a file selector if the file is not specified
    #if args.filename is None:
    #    from tkinter import Tk
    #    from tkinter.filedialog import askopenfilename
    #    Tk().withdraw()
    #    args.filename = askopenfilename()
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/05771_00005_SMS_b1_p0_ff16/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/05771_00006_SMS_b1_p0_ff4/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/05771_00007_SMS_b1_p0_ff2/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/05771_00008_SMS_b1_p0_ff1/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/05771_00009_SMS_b1_p0_ff0p5/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05918_2022-10-20/05918_00004_SMS_shim/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00013_SMS_b1_p0_ff0p25/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00014_SMS_b1_p0_ff0p33/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00015_SMS_b1_p0_ff0p5/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00016_SMS_b1_p0_ff0p66/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00017_SMS_b1_p0_ff1/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00018_SMS_b1_p0_ff1p33/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00019_SMS_b1_p0_ff2/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00020_SMS_b1_p0_ff3/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05939_2022-10-24/p0_seq/05939_00021_SMS_b1_p0_ff4/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06018_2022-11-02/06018_00003_SMS_b1_p1_ff0p5/SMS_2DPC/MRI_Raw.h5"
    #args.filename = "/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_06018_2022-11-02/06018_00008_SMS_b1_p0_ff0p50/SMS_2DPC/MRI_Raw.h5"

    # Save to input raw data folder
    if args.out_folder is None:
        args.out_folder = os.path.dirname(args.filename)

    # Save to Folder
    logger.info(f'Saving to {args.out_folder}')

    # Load Data
    logger.info(f'Load MRI from {args.filename}')
    mri_raw = load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils,
                           max_encodes=args.max_encodes, sms_phase=args.sms_phase)
    print(f'Min/max = {np.max(mri_raw.time[0])} {np.max(mri_raw.time[0])}')

    # Resample
    # radial3d_regrid(mri_raw)

    num_enc = mri_raw.Num_Encodings
    if args.crop_factor > 1.0:
        crop_kspace(mri_rawdata=mri_raw, crop_factor=args.crop_factor)  # 2.5 (320/128)

    # Perform respiratory gating 
    if args.resp_gate:
        mri_raw = resp_gate(mri_raw)

    # Get sensitivity maps
    logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    if mri_raw.Num_Coils == 1:
        img_shape = sp.estimate_shape(mri_raw.coords[0])
        xp = sp.Device(args.device).xp
        smaps = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
    else:
        smaps = get_smaps(mri_rawdata=mri_raw, args=args, thresh_maps=False,
                          smap_type=args.smap_type, log_dir=args.out_folder)

    # Put the maps on the GPU
    smaps = array_to_gpu(smaps, sp.Device(args.device))

    # Gate k-space
    if args.frames > 1:
        if args.frames2 > 1:
            mri_raw = gate_kspace2d(mri_raw=mri_raw, num_frames=[args.frames, args.frames2],
                                    gate_type=[args.gate_type, args.gate_type2],
                                    discrete_gates=[args.discrete_gates, args.discrete_gates2],
                                    prep_disdaqs=args.prep_disdaqs)
        else:
            mri_raw = gate_kspace(mri_raw=mri_raw,
                                  num_frames=args.frames,
                                  gate_type=args.gate_type,
                                  discrete_gates=args.discrete_gates)

    if args.recon_type == 'sense':
        img = []
        for i in range(len(mri_raw.kdata)):
            logger.info(f'Sense Recon : Frame {i}')

            kdata = array_to_gpu(mri_raw.kdata[i], args.device)
            dcf = array_to_gpu(mri_raw.dcf[i], args.device)
            coord = array_to_gpu(mri_raw.coords[i], args.device)

            print(f'Smaps device = {sp.get_device(smaps)}')
            print(f'Kdata = device = {sp.get_device(kdata)}')
            print(f'DCF device = {sp.get_device(dcf)}')
            print(f'Coord device = {sp.get_device(coord)}')

            sense = sp.mri.app.SenseRecon(kdata, smaps, lamda=0, weights=dcf, coord=coord, max_iter=args.max_iter,
                                          coil_batch_size=args.coil_batch_size, device=args.device)
            print('Run Sense')
            img.append(sp.to_device(sense.run(), sp.cpu_device))
    elif args.recon_type == 'pils':
        logger.info('PILS Recon')
        img = []
        for i in range(len(mri_raw.kdata)):
            t = time.time()
            logger.info(f'Frame {i} of {len(mri_raw.kdata)}')

            kdata = array_to_gpu(mri_raw.kdata[i], args.device)
            dcf = array_to_gpu(mri_raw.dcf[i], args.device)
            coord = array_to_gpu(mri_raw.coords[i], args.device)

            # Low resolution images
            xp = sp.get_device(coord).xp
            res = args.krad_cutoff
            lpf = xp.sum(coord ** 2, axis=-1)
            lpf = xp.exp(-lpf / (2.0 * res * res))
            dcf = dcf * lpf

            E = sp.mri.linop.Sense(mps=smaps, coord=coord, weights=dcf ** 2, coil_batch_size=args.coil_batch_size)
            Eh = E.H

            img.append(sp.to_device(Eh * kdata))
            logger.info(f'Frame {i} took {time.time()-t}')
    elif args.recon_type == 'gridded':
        import matplotlib.pyplot as plt
        logger.info('Gridded Recon')
        img = []
        smapsCPU = sp.to_device(smaps, sp.cpu_device)
        for i in range(len(mri_raw.kdata)):
            kdata = array_to_gpu(mri_raw.kdata[i], args.device)
            dcf = array_to_gpu(mri_raw.dcf[i], args.device)
            coord = array_to_gpu(mri_raw.coords[i], args.device)

            xp = sp.Device(args.device).xp
            img_shape = sp.estimate_shape(coord)

            imgsc = np.zeros([mri_raw.Num_Coils, img_shape[0], img_shape[1]])
            imgsc = array_to_gpu(imgsc)
            imgs = 0
            for c in range(mri_raw.Num_Coils):
                logger.info(f'Reconstructing coil {c}')
                ksp = kdata[c, :]
                ksp *= dcf
                #imgs += xp.abs(sp.nufft_adjoint(ksp, coord, img_shape)) ** 2
                imgs += sp.nufft_adjoint(ksp, coord, img_shape)
                imgsc[c,:,:] = xp.abs(sp.nufft_adjoint(ksp, coord, img_shape))
            imgsc = sp.to_device(imgsc, sp.cpu_device)
            #for c in range(mri_raw.Num_Coils):
            #    plt.figure(c + 1)
            #    plt.imshow(imgsc[c, :, :])

            #img.append(sp.to_device(xp.sqrt(imgs), sp.cpu_device))
            img.append(sp.to_device(imgs, sp.cpu_device))
    else:
        print('Specify either "sense" or "pils" recon')

    # Copy to CPU and reshape
    img = np.stack(img, axis=0)
    img = sp.to_device(img, sp.cpu_device)
    img = np.reshape(img, (args.frames*args.frames2, -1) + img.shape[1:])
    logger.info(f'Image shape {img.shape}')

    img_mag = np.abs(img)
    img_phase = np.angle(img)

    # Export to file
    out_name = os.path.join(args.out_folder, args.out_filename)
    logger.info('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=img)
        hf.create_dataset("IMAGE_MAG", data=img_mag)
        hf.create_dataset("IMAGE_PHASE", data=img_phase)
