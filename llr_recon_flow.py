# %%
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
from registration_tools import *
from read_scan_archive import *
from joblib import dump, load

import matplotlib.pyplot as plt
# %%
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
    parser.add_argument('--thresh_maps_val', type=float, default=0.08)


    parser.add_argument('--reset_dens', dest='reset_dens', action='store_true')
    parser.set_defaults(reset_dens=False)

    parser.add_argument('--mps_ker_width', type=int, default=16)
    parser.add_argument('--ksp_calib_width', type=int, default=32)
    parser.add_argument('--lamda', type=float, default=0.0001)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--jsense_max_iter', type=int, default=30)
    parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
    parser.add_argument('--jsense_lamda', type=float, default=0.0)
    parser.add_argument('--krad_cutoff', type=float, default=999990)
    parser.add_argument('--max_encodes', type=int, default=None)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gate_type', type=str, default='time')  # recon type
    parser.add_argument('--gate_type2', type=str, default='prep')  # recon type
    parser.add_argument('--prep_disdaqs', type=int, default=0)
    parser.add_argument('--crop_factor', type=float, default=1.0)
    parser.add_argument('--recon_type', type=str, default='llr')
    parser.add_argument('--llr_block_width',type=int, default=32)

    parser.set_defaults(discrete_gates=False)
    parser.add_argument('--discrete_gates', dest='discrete_gates', action='store_true')

    parser.set_defaults(discrete_gates2=False)
    parser.add_argument('--discrete_gates2', dest='discrete_gates2', action='store_true')

    parser.add_argument('--fast_maxeig', dest='fast_maxeig', action='store_true')
    parser.set_defaults(fast_maxeig=False)
    parser.add_argument('--test_run', dest='test_run', action='store_true')
    parser.set_defaults(test_run=False)
    parser.add_argument('--compress_coils', type=int, dest='compress_coils', default=-1, help='Number of coils to compress to')

    parser.set_defaults(strided_gate=False)
    parser.add_argument('--strided_gate', dest='strided_gate', action='store_true')
    parser.add_argument('--shots_per_frame', type=int, default=2)

    parser.add_argument('--demod', type=float, default=0.0)
    parser.add_argument('--gate_delay', type=float, default=0.0)
    parser.add_argument('--single_encode_gate', dest='single_encode_gate', action='store_true')
    parser.set_defaults(single_encode_gate=False)

    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5, ScanArch* (only for 2D PC spiral))')
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='FullRecon')

    parser.add_argument('--skope_path', type=str, help='path/filename to SKOPE data')
    
    parser.add_argument('--long_scan', dest='long_scan', action='store_true')
    parser.set_defaults(long_scan=False)
    parser.add_argument('--recon_split_factor', type=int, default=1)



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

    # Save to Folder
    logger.info(f'Saving to {args.out_folder}')

    # %%
    if args.long_scan:
        total_recons = args.recon_split_factor   

        print(f'Recon will be split into {total_recons} reconstructions')
        
        for rsplit in range(total_recons):
            # Load Data
            logger.info(f'Load MRI from {args.filename} (is a long scan)')
            if args.test_run:
                mri_raw = load_MRI_raw(h5_filename=args.filename, max_coils=2, max_encodes=args.max_encodes)
            else:
                if "ScanArchive" in args.filename:
                    logger.info(f'loading {args.filename} with demod = {args.demod} and gate delay = {args.gate_delay}')   
                    base_path = os.path.dirname(os.path.abspath(args.filename))
                    full_path = os.path.join(base_path, "MRI_raw_long.joblib")
                    #full_path = os.path.join(base_path, "MRI_raw_short.joblib")

                    if os.path.exists(full_path):
                        mri_raw = load(full_path)
                        print('MRI_raw_long loaded') # ~5min

                    else:
                        print('No MRI_raw_long.joblib file, loading directly from scan archive (~2 hours for 25min scan)')
                        mri_raw = load_ScanArchive_long_scan(os.path.abspath(args.filename), args.gate_delay, args.demod, args.skope_path, compress_coils=args.compress_coils, max_encodes=args.max_encodes) 
                else:
                    mri_raw = load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils, max_encodes=args.max_encodes)

            # Limitations: All data gets loaded (slow), some shots may be left out if division is not integer, but for now testing is fine
            #full_path = '/mounts/data/scratch/99_LARR/lfos/vol_rtflow_10434_2025-10-08_25minPCscan/10434_00004_realtime_flow_num_passes_19_26min/raw_data/MRI_raw_long.joblib'
            #mri_raw = load(full_path)

            total_shots = mri_raw.kdata.shape[3]
            recon_chunk = total_shots // total_recons

            idx_s = rsplit*recon_chunk
            idx_e = (rsplit + 1)*recon_chunk

            # Recon an image chunk (slow because we are loading all data first) lets try half 
            mri_raw.coords = mri_raw.coords[:,:,idx_s:idx_e,:,:] 
            mri_raw.dcf = mri_raw.dcf[:,:,idx_s:idx_e,:] 
            mri_raw.kdata = mri_raw.kdata[:,:,:,idx_s:idx_e,:] 
            mri_raw.time = mri_raw.time[:,:,idx_s:idx_e,:] 
            mri_raw.prep= mri_raw.prep[:,:,idx_s:idx_e,:] 
            mri_raw.ecg = mri_raw.ecg[:,:,idx_s:idx_e,:] 
            mri_raw.resp = mri_raw.resp[:,:,idx_s:idx_e,:] 

            logger.info(f'index start and index end {idx_s} {idx_e}, current recon index {rsplit}')
            logger.info(f'kspace data shape {mri_raw.kdata.shape}')
            logger.info(f'time data shape {mri_raw.time.shape}')
            logger.info(f'coords data shape {mri_raw.coords.shape}')
            logger.info(f'time range in this gap is min:{np.min(mri_raw.time)} and max :{np.max(mri_raw.time)} ')

            #coords data shape (3, 1, 2000, 3420, 2)
            #kspace data shape (3, 20, 1, 2000, 3420)
            # time data shape (3, 1, 2000, 3420)

            #%%
            # Compute k-space magnitude
            #ks_mag = np.sum(mri_raw.kdata, axis=1)        # Sum along axis 1
            #ks_mag = np.abs(np.squeeze(ks_mag))           # Shape: (3, 2000, 3420)
            #ks_coords = np.squeeze(mri_raw.coords)        # Shape: (3, 2000, 3420, 2)
            #ks_mag_flat = ks_mag.flatten()
            #ks_time = mri_raw.time.flatten()

            #plt.figure(figsize=(8, 6))
            #plt.plot(ks_time, np.log(ks_mag_flat + 1e-6), color='black')
            #plt.title('Time vs K-space Magnitude')
            #plt.xlabel('Time (s)')
            #plt.ylabel('Log Magnitude')
            #plt.grid(True)
            #plt.show()


            # Plot all shots together
            #plt.figure(figsize=(8, 8))

            # Loop through all shots and accumulate points
            #for shot in range(2000):
            #    coords_shot = ks_coords[0, shot]          # (3420, 2)
            #    kx = coords_shot[:, 0]
            #    ky = coords_shot[:, 1]
            #    mag_shot = ks_mag[0, shot]                # (3420,)

                # Plot in grayscale, log scale for better contrast
            #    plt.scatter(kx, ky, c=np.log(mag_shot + 1e-6), cmap='gray', s=0.5)

            #plt.title('Combined K-space Magnitude (All Shots)')
            #plt.axis('equal')
            #plt.axis('off')
            #plt.colorbar(label='Log Magnitude')
            #plt.show()
        

            #continue


        #%%

            # Resample
            # radial3d_regrid(mri_raw)

            # Shift
            # spatial_shift(mri_raw, [0, 30/220])

            if args.crop_factor > 1.0:
                crop_kspace(mri_rawdata=mri_raw, crop_factor=args.crop_factor)  # 2.5 (320/128)

            if args.autofov:
                # Reconstruct an low res image and get the field of view
                logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
                autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale, square=False)


            # Get sensitivity maps
            logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
            if mri_raw.Num_Coils == 1:
                img_shape = sp.estimate_shape(mri_raw.coords[0])
                xp = sp.Device(args.device).xp
                smaps = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
            else:
                smaps = get_smaps(mri_rawdata=mri_raw, args=args, thresh_maps=args.thresh_maps, thresh_maps_val=args.thresh_maps_val, smap_type='jsense', log_dir=args.out_folder)


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
                                            discrete_gates=args.discrete_gates,
                                            single_encode_gate=args.single_encode_gate)


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
                #blk_widths=(128, 64, 32)
                blk_widths=(128, 64, 32, 16, 8)
                blk_widths=(128, 64, 32, 16, 8, 4)
                blk_widths=(128, 96, 64, 48, 32, 24, 16, 8, 4)

                kdata = mri_raw.kdata
                coord = mri_raw.coords
                dcf = mri_raw.dcf

                lrimg = MultiScaleLowRankRecon(kdata, coord=coord, dcf=dcf, mps=smaps,
                                sgw=None,
                                blk_widths=blk_widths,
                                lamda=args.lamda,
                                max_epoch=args.epochs,
                                device=sp.Device(args.device),
                                out_iter_mon=True,
                                comm=comm,
                                log_dir=args.out_folder,
                                num_encodings=mri_raw.Num_Encodings).run()

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

            elif args.recon_type == 'sense':

                img = []
                for i in range(len(mri_raw.kdata)):
                    logger.info(f'Sense Recon : Frame {i}')

                    kdata = sp.to_device(mri_raw.kdata[i], args.device)
                    
                    dcf = sp.to_device(mri_raw.dcf[i], args.device)
                    #dcf = sp.to_device(np.ones_like(mri_raw.dcf[i]), args.device)

                    coord = sp.to_device(mri_raw.coords[i], args.device)

                    print(f'Smaps device = {sp.get_device(smaps)}')
                    print(f'Kdata = device = {sp.get_device(kdata)}')
                    print(f'DCF device = {sp.get_device(dcf)}')
                    print(f'Coord device = {sp.get_device(coord)}')

                    sense = sp.mri.app.SenseRecon(kdata, smaps, lamda=0, weights=dcf, coord=coord, max_iter=args.max_iter, coil_batch_size=1, device=args.device)
                    #sense = sp.mri.app.L1WaveletRecon(kdata, smaps, lamda=1e-1, weights=dcf, coord=coord, max_iter=50, coil_batch_size=1, device=args.device)

                    print('Run Sense')
                    img.append(sp.to_device(sense.run(), sp.cpu_device))
            elif args.recon_type == 'pils':
                logger.info('PILS Recon')
                img = []

                for i in range(len(mri_raw.kdata)):
                    logger.info(f'Frame {i} of {len(mri_raw.kdata)}')

                    kdata = sp.to_device(mri_raw.kdata[i], args.device)
                    dcf = sp.to_device(mri_raw.dcf[i], args.device)
                    coord = sp.to_device(mri_raw.coords[i], args.device)

                    # Low resolution images
                    xp = sp.get_device(coord).xp
                    res = args.krad_cutoff
                    lpf = xp.sum(coord ** 2, axis=-1)
                    lpf = xp.exp(-lpf / (2.0 * res * res))
                    dcf = dcf * lpf

                    E = sp.mri.linop.Sense(mps=smaps, coord=coord, weights=dcf ** 2, coil_batch_size=1)
                    Eh = E.H

                    img.append(sp.to_device(Eh * kdata))
            else:
                print('Please input recon_type (llr, sense, pils, mslr')

            # Copy to CPU and reshape
            img = np.stack(img,axis=0)
            img = sp.to_device(img, sp.cpu_device)
            img = np.reshape(img, (args.frames*args.frames2, -1) + smaps.shape[1:])
            logger.info(f'Image shape {img.shape}')

            img_mag = np.abs(img)
            img_phase = np.angle(img)

            img_phase_difference = np.angle( img * np.conj(np.expand_dims(img[:, 0, ...], axis=1)))

            # Export to file
            out_name = os.path.join(args.out_folder, f'{args.out_filename}_{rsplit}.h5')
            logger.info('Saving images to ' + out_name)
            try:
                os.remove(out_name)
            except OSError:
                pass
            with h5py.File(out_name, 'w') as hf:
                hf.create_dataset("IMAGE", data=img)
                hf.create_dataset("IMAGE_MAG", data=img_mag)
                hf.create_dataset("IMAGE_PHASE", data=img_phase)
                hf.create_dataset("IMAGE_PHASE_DIFFERENCE", data=img_phase_difference)
        
    else:
        # Load Data
        logger.info(f'Load MRI from {args.filename}')
        if args.test_run:
            mri_raw = load_MRI_raw(h5_filename=args.filename, max_coils=2, max_encodes=args.max_encodes)
        else:
            if "ScanArchive" in args.filename:
                logger.info(f'loading {args.filename} with demod = {args.demod} and gate delay = {args.gate_delay}')
                mri_raw = load_ScanArchive(os.path.abspath(args.filename), args.gate_delay, args.demod, args.skope_path, compress_coils=args.compress_coils, max_encodes=args.max_encodes) 
            else:
                mri_raw = load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils, max_encodes=args.max_encodes)

        # Resample
        # radial3d_regrid(mri_raw)

        # Shift
        # spatial_shift(mri_raw, [0, 30/220])

        if args.crop_factor > 1.0:
            crop_kspace(mri_rawdata=mri_raw, crop_factor=args.crop_factor)  # 2.5 (320/128)

        if args.autofov:
            # Reconstruct an low res image and get the field of view
            logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
            autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale, square=False)


        # Get sensitivity maps
        logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        if mri_raw.Num_Coils == 1:
            img_shape = sp.estimate_shape(mri_raw.coords[0])
            xp = sp.Device(args.device).xp
            smaps = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
        else:
            smaps = get_smaps(mri_rawdata=mri_raw, args=args, thresh_maps=args.thresh_maps, thresh_maps_val=args.thresh_maps_val, smap_type='jsense', log_dir=args.out_folder)


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
                                        discrete_gates=args.discrete_gates,
                                        single_encode_gate=args.single_encode_gate)


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
            #blk_widths=(128, 64, 32)
            blk_widths=(128, 64, 32, 16, 8)
            blk_widths=(128, 64, 32, 16, 8, 4)
            blk_widths=(128, 96, 64, 48, 32, 24, 16, 8, 4)

            kdata = mri_raw.kdata
            coord = mri_raw.coords
            dcf = mri_raw.dcf

            lrimg = MultiScaleLowRankRecon(kdata, coord=coord, dcf=dcf, mps=smaps,
                            sgw=None,
                            blk_widths=blk_widths,
                            lamda=args.lamda,
                            max_epoch=args.epochs,
                            device=sp.Device(args.device),
                            out_iter_mon=True,
                            comm=comm,
                            log_dir=args.out_folder,
                            num_encodings=mri_raw.Num_Encodings).run()

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

        elif args.recon_type == 'sense':

            img = []
            for i in range(len(mri_raw.kdata)):
                logger.info(f'Sense Recon : Frame {i}')

                kdata = sp.to_device(mri_raw.kdata[i], args.device)
                
                dcf = sp.to_device(mri_raw.dcf[i], args.device)
                #dcf = sp.to_device(np.ones_like(mri_raw.dcf[i]), args.device)

                coord = sp.to_device(mri_raw.coords[i], args.device)

                print(f'Smaps device = {sp.get_device(smaps)}')
                print(f'Kdata = device = {sp.get_device(kdata)}')
                print(f'DCF device = {sp.get_device(dcf)}')
                print(f'Coord device = {sp.get_device(coord)}')

                sense = sp.mri.app.SenseRecon(kdata, smaps, lamda=0, weights=dcf, coord=coord, max_iter=args.max_iter, coil_batch_size=1, device=args.device)
                #sense = sp.mri.app.L1WaveletRecon(kdata, smaps, lamda=1e-1, weights=dcf, coord=coord, max_iter=50, coil_batch_size=1, device=args.device)

                print('Run Sense')
                img.append(sp.to_device(sense.run(), sp.cpu_device))
        elif args.recon_type == 'pils':
            logger.info('PILS Recon')
            img = []

            for i in range(len(mri_raw.kdata)):
                logger.info(f'Frame {i} of {len(mri_raw.kdata)}')

                kdata = sp.to_device(mri_raw.kdata[i], args.device)
                dcf = sp.to_device(mri_raw.dcf[i], args.device)
                coord = sp.to_device(mri_raw.coords[i], args.device)

                # Low resolution images
                xp = sp.get_device(coord).xp
                res = args.krad_cutoff
                lpf = xp.sum(coord ** 2, axis=-1)
                lpf = xp.exp(-lpf / (2.0 * res * res))
                dcf = dcf * lpf

                E = sp.mri.linop.Sense(mps=smaps, coord=coord, weights=dcf ** 2, coil_batch_size=1)
                Eh = E.H

                img.append(sp.to_device(Eh * kdata))
        else:
            print('Please input recon_type (llr, sense, pils, mslr')

        # Copy to CPU and reshape
        img = np.stack(img,axis=0)
        img = sp.to_device(img, sp.cpu_device)
        img = np.reshape(img, (args.frames*args.frames2, -1) + smaps.shape[1:])
        logger.info(f'Image shape {img.shape}')

        img_mag = np.abs(img)
        img_phase = np.angle(img)

        img_phase_difference = np.angle( img * np.conj(np.expand_dims(img[:, 0, ...], axis=1)))

        # Export to file
        out_name = os.path.join(args.out_folder, f'{args.out_filename}.h5')
        logger.info('Saving images to ' + out_name)
        try:
            os.remove(out_name)
        except OSError:
            pass
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset("IMAGE", data=img)
            hf.create_dataset("IMAGE_MAG", data=img_mag)
            hf.create_dataset("IMAGE_PHASE", data=img_phase)
            hf.create_dataset("IMAGE_PHASE_DIFFERENCE", data=img_phase_difference)




