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
from flow_processing import *
from svt import *
import numba as nb
#import torch as torch
import os
import scipy.ndimage as ndimage
from registration_tools import *
from gpu_ops import * 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--frames',type=int, default=1, help='Number of time frames')
    parser.add_argument('--frames2', type=int, default=1, help='Number of time frames')

    parser.set_defaults(autofov=False)
    parser.add_argument('--autofov', dest='autofov', action='store_true')
    
    parser.set_defaults(thresh_maps=False)
    parser.add_argument('--thresh_maps', dest='thresh_maps', action='store_true')


    parser.add_argument('--reset_dens', dest='reset_dens', action='store_true')
    parser.set_defaults(reset_dens=False)

    parser.add_argument('--mps_ker_width', type=int, default=16)
    parser.add_argument('--ksp_calib_width', type=int, default=32)
    parser.add_argument('--lamda', type=float, default=0.0001)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--jsense_max_iter', type=int, default=30)
    parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
    parser.add_argument('--jsense_lamda', type=float, default=0.0)
    parser.add_argument('--smap_type', type=str, default='jsense', help='Sensitivity type jsense, lowres, walsh, espirit')

    parser.add_argument('--krad_cutoff', type=float, default=999990)
    parser.add_argument('--max_encodes', type=int, default=None)
    parser.add_argument('--coil_batch_size', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gate_type', type=str, default='time')  # recon type
    parser.add_argument('--gate_type2', type=str, default='prep')  # recon type
    parser.add_argument('--prep_disdaqs', type=int, default=0)
    parser.add_argument('--crop_factor', type=float, default=1.0)
    parser.add_argument('--recon_type', type=str, default='llr')
    parser.add_argument('--llr_block_width', type=int, default=32)
    parser.add_argument('--data_oversampling', type=float, default=2.0)

    parser.set_defaults(discrete_gates=False)
    parser.add_argument('--discrete_gates', dest='discrete_gates', action='store_true')

    parser.set_defaults(discrete_gates2=False)
    parser.add_argument('--discrete_gates2', dest='discrete_gates2', action='store_true')

    parser.set_defaults(resp_gate=False)
    parser.add_argument('--resp_gate', dest='resp_gate', action='store_true')
    parser.add_argument('--resp_efficiency', type=float, default=0.5)
    parser.add_argument('--resp_sign', type=int, help='flip the respiratory waveform', default=1)
    parser.add_argument('--resp_filter_window', type=float, default=10)
    parser.add_argument('--time_range', type=str, help='specify ranges as start1-end1,start2-end2,... in seconds (Ex: 50-100,220-340)', default=None)

    parser.add_argument('--fast_maxeig', dest='fast_maxeig', action='store_true')
    parser.set_defaults(fast_maxeig=False)
    parser.add_argument('--test_run', dest='test_run', action='store_true')
    parser.set_defaults(test_run=False)
    parser.add_argument('--compress_coils', type=int, dest='compress_coils', default=-1, help='Number of coils to compress to')

    parser.set_defaults(strided_gate=False)
    parser.add_argument('--strided_gate', dest='strided_gate', action='store_true')
    parser.add_argument('--shots_per_frame', type=int, default=2)


    # SMS Reconstruction
    parser.add_argument('--sms_factor', type=int, default=1)  # sms factor
    parser.add_argument('--sms_slice', type=int, default=0)  # sms slice, ordered bottom (0) to top (sms_factor - 1)
    parser.add_argument('--venc', type=int, default=1500) # mm/s
    parser.add_argument('--unwrap_lap', dest='unwrap_lap', action='store_true')

    # Flow Processing
    parser.add_argument('--flow_processing', dest='flow_processing', action='store_true', default=False)

    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)')
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='FullRecon.h5')

    # Debugging / mslr mag and example images
    parser.add_argument('--example_images', dest='example_images', action='store_true')
    parser.set_defaults(example_images=False)

    args = parser.parse_args()

    # For tracking memory
    mempool = cupy.get_default_memory_pool()

    # Put up a file selector if the file is not specified
    if args.filename is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        Tk().withdraw()
        args.filename = askopenfilename()

    # Save to input raw data folder
    if args.out_folder is None:
        args.out_folder = os.path.dirname(args.filename)

    # Save to Folder    logger.info(f'Saving to {args.out_folder}')

    # Load Data
    logger.info(f'Load MRI from {args.filename}')
    if args.test_run:
        mri_raw = load_MRI_raw(h5_filename=args.filename, max_coils=2, max_encodes=args.max_encodes, sms_factor=args.sms_factor, sms_slice=args.sms_slice)
    else:
        mri_raw = load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils, max_encodes=args.max_encodes, sms_factor=args.sms_factor, sms_slice=args.sms_slice)

    # Resample
    # radial3d_regrid(mri_raw)

    if args.crop_factor > 1.0:
        crop_kspace(mri_rawdata=mri_raw, crop_factor=args.crop_factor)  # 2.5 (320/128)

    # Perform respiratory gating 
    if args.resp_gate:
        if args.time_range is not None:
            time_ranges = []
            ranges = args.time_range.split(',')
            for trange in ranges:
                time_ranges.append([float(x) for x in trange.split('-')])
        else:
            time_ranges = None
        mri_raw = resp_gate(mri_raw, efficiency=args.resp_efficiency, resp_sign=args.resp_sign, resp_filter_window=args.resp_filter_window, time_ranges=time_ranges)

    # Reconstruct an low res image and get the field of view
    logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    if args.recon_type == 'llr':
        autofov_block_size = args.llr_block_width
    else:
        autofov_block_size = 8

    if args.autofov:
        autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale, oversample=args.data_oversampling,
            square=False, block_size=autofov_block_size, logdir=args.out_folder)

    # Get sensitivity maps
    logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    if mri_raw.Num_Coils == 1:
        img_shape = sp.estimate_shape(mri_raw.coords[0])
        xp = sp.Device(args.device).xp
        smaps = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
    else:
        smaps = get_smaps(mri_rawdata=mri_raw, args=args, thresh_maps=args.thresh_maps, smap_type=args.smap_type, log_dir=args.out_folder)


    # Gate k-space
    #if args.frames > 1:
    #    if args.frames2 > 1:
    #        mri_raw = gate_kspace2d(mri_raw=mri_raw,
    #                              num_frames=[args.frames, args.frames2],
    #                              gate_type=[args.gate_type, args.gate_type2],
    #                              discrete_gates=[args.discrete_gates, args.discrete_gates2],
    #                              prep_disdaqs=args.prep_disdaqs)
    #    else:
    #        mri_raw = gate_kspace(mri_raw=mri_raw,
    #                              num_frames=args.frames,
    #                              gate_type=args.gate_type,
    #                              discrete_gates=args.discrete_gates)
    
    # For the spiral flow situation with interleaved encodings
    if args.strided_gate:
        logger.info(f'Strided gating for spiral with interleaved encodes')
        # Hardcoded frames per cardiac bin. 
        mri_raw = strided_encoding(mri_raw, stride=1, shots_per_frame=args.shots_per_frame)
        args.frames = mri_raw.Num_Frames
    else:
        if args.frames > 1:
            if args.frames2 > 1:
                mri_raw = gate_kspace2d(mri_raw=mri_raw,
                                        num_frames=[args.frames, args.frames2],
                                        gate_type=[args.gate_type, args.gate_type2],
                                        discrete_gates=[args.discrete_gates, args.discrete_gates2],
                                        prep_disdaqs=args.prep_disdaqs)
            else:
                mri_raw = gate_kspace(mri_raw=mri_raw,
                                    num_frames=args.frames,
                                    gate_type=args.gate_type,
                                    discrete_gates=args.discrete_gates)


    # Fake rotations
    if False:
        for i in range(mri_raw.Num_Frames*mri_raw.Num_Encodings):
            print(f'Frame {i} ')
            device = sp.get_device(mri_raw.coords[i])
            kdata = sp.to_device(mri_raw.kdata[i], device)
            dcf = sp.to_device(mri_raw.dcf[i], device)
            coord = sp.to_device(mri_raw.coords[i], device)

            psi = -float(i // mri_raw.Num_Encodings)*0.05
            phi = 0
            theta = float(i // mri_raw.Num_Encodings)*0.1
            print(f'Rotation = {theta} {phi} {psi}')

            tx = -float(i // mri_raw.Num_Encodings) * 0.01
            ty =  float(i // mri_raw.Num_Encodings) * 0.02
            tz = -float(i // mri_raw.Num_Encodings) * 0.005
            mri_raw.kdata[i] *= device.xp.exp(1j*2.0*math.pi*tx*mri_raw.coords[i][...,0])

            # Build Rotation matrix
            rot = build_rotation(theta, phi, psi)
            rot = sp.to_device( rot, device)

            coord_rot = coord
            coord_rot = device.xp.expand_dims( coord_rot, -1)
            coord_rot = device.xp.matmul(rot, coord_rot)
            coord_rot = device.xp.squeeze( coord_rot)

            mri_raw.coords[i] = coord_rot

    if args.reset_dens:
        for i in range(len(mri_raw.kdata)):
            mri_raw.dcf[i][:] = 1.0


    if True:
        for i in range(len(mri_raw.kdata)):
            mri_raw.kdata[i] = sp.to_device(mri_raw.kdata[i], sp.Device(args.device))
            mri_raw.coords[i] = sp.to_device(mri_raw.coords[i], sp.Device(args.device))
            mri_raw.dcf[i] = sp.to_device(mri_raw.dcf[i], sp.Device(args.device))

    # Put the maps on the GPU
    smaps = sp.to_device(smaps, sp.Device(args.device))

    # Reconstruct the image
    if args.recon_type == 'mslr':
        comm = sp.Communicator()
        #blk_widths = (128, 64, 48, 32, 24, 16)
        #blk_widths = (128, 96, 64, 48)
        #blk_widths = (512, 64)

        num_scales = 3
        blk_widths = []
        for scale in range(num_scales):
            blk_widths.append([im_size// (2**scale) for im_size in smaps.shape[1:]])
        
        # blk_widths = [
        #     [320, 320],
        #     [160, 160],
        # ]

        kdata = mri_raw.kdata
        coord = mri_raw.coords
        dcf = mri_raw.dcf

        mslr_recon = MultiScaleLowRankRecon(kdata, coord=coord, dcf=dcf, mps=smaps,
                           sgw=None,
                           blk_widths=blk_widths,
                           lamda=args.lamda,
                           max_epoch=args.epochs,
                           device=sp.Device(args.device),
                           out_iter_mon=True,
                           comm=comm,
                           alpha=0.8,
                           hanning_window=True,
                           log_dir=args.out_folder,
                           num_encodings=mri_raw.Num_Encodings)

        lrimg = mslr_recon.run()
        out_name = os.path.join(args.out_folder,'MSLRObject.h5')
        lrimg.save(out_name)

        print(lrimg.shape)
        #Sz = lrimg[..., lrimg.shape[-3] // 2, :, :]
        #Sy = lrimg[..., lrimg.shape[-2] // 2, :]
        #Sx = lrimg[..., lrimg.shape[-1] // 2]
        
        #img = lrimg[:, :, :, :, :]
        img = []
        for t in range(lrimg.total_images):
            img.append(sp.to_device(lrimg[t]))
        img = np.stack( img, axis=0)

        
        img = np.reshape(img, (args.frames, -1) + img.shape[1:])
        out_name = os.path.join(args.out_folder, 'FullRecon.h5')
        logger.info('Saving images to ' + out_name)
        try:
            os.remove(out_name)
            with h5py.File(out_name, 'w') as hf:
                hf.create_dataset("IMAGE", data=img)

        except OSError:
            pass

        if args.example_images:
        
            Sz = lrimg[:, :, lrimg.shape[-3] // 2, :, :]
            Sy = lrimg[:, :, :, lrimg.shape[-2] // 2, :]
            Sx = lrimg[:, :, :, :, lrimg.shape[-1] // 2]
        
            # generate some slices
            logger.info('Generating slices for export')
            lrimg.use_device(sp.Device(args.device))

            # Export into Mag
            out_name = os.path.join(args.out_folder, 'MagImages.h5')
            logger.info('Saving images to ' + out_name)
            try:
                os.remove(out_name)
            except OSError:
                pass

            with h5py.File(out_name, 'w') as hf:
                for t in range(lrimg.shape[0]):
                    Im = lrimg[t]
                    hf.create_dataset(f'Frame{t:04}', data=np.squeeze(np.abs(Im)))

            Im0 = lrimg[:, :, :]

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
                hf.create_dataset('Frame0', data=np.abs(Im0))
                hf.create_dataset('aFrame0', data=np.angle(Im0))
                pass

    elif args.recon_type == 'llr':
        logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        img = BatchedSenseRecon(mri_raw.kdata, mps=smaps, weights=mri_raw.dcf, coord=mri_raw.coords,
                                device=sp.Device(args.device), lamda=args.lamda, num_enc=mri_raw.Num_Encodings,
                                coil_batch_size=None, max_iter=args.max_iter, batched_iter=args.max_iter,
                                gate_type=args.gate_type, fast_maxeig=args.fast_maxeig,
                                block_width=args.llr_block_width, log_folder=args.out_folder,
                                composite_init=False
                                ).run()
        logger.info(f'Image shape {img.shape}')
    elif args.recon_type == 'sense':

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

            sense = sp.mri.app.SenseRecon(kdata, smaps, lamda=0, weights=dcf, coord=coord, max_iter=args.max_iter, coil_batch_size=args.coil_batch_size, device=args.device)
            #sense = sp.mri.app.L1WaveletRecon(kdata, smaps, lamda=1e-1, weights=dcf, coord=coord, max_iter=50, coil_batch_size=1, device=args.device)

            print('Run Sense')
            img.append(sp.to_device(sense.run(), sp.cpu_device))
    elif args.recon_type == 'pils':
        logger.info('PILS Recon')
        img = []

        import time
 
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

    else:
        print('Please input recon_type: llr, sense, pils, mslr')

    # Copy to CPU and reshape
    img = np.stack(img,axis=0)
    img = sp.to_device(img, sp.cpu_device)
    img = np.reshape(img, (args.frames*args.frames2, -1) + smaps.shape[1:])
    img = np.squeeze(img)
    logger.info(f'Image shape {img.shape}')

    img_mag = np.abs(img)
    img_phase = np.angle(img)
    img_phase_difference = np.angle( img * np.conj(np.expand_dims(img[:, 0, ...], axis=1)))
    
    smaps = sp.to_device(smaps, sp.cpu_device)
    smaps_mag = np.abs(smaps)

    if args.flow_processing:
        if mri_raw.Num_Encodings == 5:
            encoding = "5pt"
        elif mri_raw.Num_Encodings == 4:
            encoding = "4pt-referenced"
        elif mri_raw.Num_Encodings == 3:
            encoding = "3pt"
        elif mri_raw.Num_Encodings == 2:
            encoding = "2pt"

        print(f' encoding type is {encoding}')

        # Solve for Velocity
        mri_flow = MRI_4DFlow(encode_type=encoding, venc=args.venc, unwrap_lap=args.unwrap_lap)
        mri_flow.signal = np.moveaxis(img, 1, -1)
        mri_flow.solve_for_velocity()
        # mri_flow.update_angiogram()
        # mri_flow.background_phase_correct()
        mri_flow.update_angiogram()

    

    # Export to file
    out_name = os.path.join(args.out_folder, args.out_filename)
    logger.info('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        header_group = hf.create_group("HEADER")
        args_str = ";".join(f"{k}={v}" for k, v in vars(args).items() if v is not None)
        header_group.attrs["recon_params"] = args_str
        header_group.attrs["fovx"] = mri_raw.fovx
        header_group.attrs["fovy"] = mri_raw.fovy
        header_group.attrs["fovz"] = mri_raw.fovz
        header_group.attrs["matrixx"] = img.shape[2]
        header_group.attrs["matrixy"] = img.shape[3]
        if len(img.shape) > 4:
            header_group.attrs["matrixz"] = img.shape[4]
        else:
            header_group.attrs["matrixz"] = 1
        header_group.attrs["frames"] = args.frames
        if args.time_range is not None:
            scan_length = sum([x[1] - x[0] for x in time_ranges])
            header_group.attrs["time_res"] = scan_length/args.frames
        else:
            header_group.attrs["time_res"] = (np.max(mri_raw.time[0].flatten()))/args.frames
        
        if args.sms_factor > 1:
            header_group.attrs["sms_factor"] = args.sms_factor
            header_group.attrs["sms_slice"] = args.sms_slice
        
        hf.create_dataset("IMAGE_REAL", data=np.real(img))
        hf.create_dataset("IMAGE_IMAG", data=np.imag(img))
        hf.create_dataset("IMAGE_PHASE_DIFFERENCE", data=img_phase_difference)
        hf.create_dataset("SMAPS", data=smaps_mag)
        if args.flow_processing:
            header_group.attrs["venc"] = mri_flow.venc
            header_group.attrs["encoding_matrix"] = mri_flow.EncodingMatrix
            hf.create_dataset("VX", data=mri_flow.velocity_estimate[..., 0])
            hf.create_dataset("VY", data=mri_flow.velocity_estimate[..., 1])
            hf.create_dataset("VZ", data=mri_flow.velocity_estimate[..., 2])
            hf.create_dataset("CD", data=mri_flow.angiogram)
            hf.create_dataset("MAG", data=mri_flow.magnitude)









