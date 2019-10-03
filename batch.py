#! /usr/bin/env python
from flow_processing import MRI_4DFlow
import llr_recon_flow
import logging
import cupy
import argparse
import sigpy as sp
import numpy as np
import h5py
import os
import shutil

if __name__ == '__main__':

    data_source = ['Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00270_v3\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00353_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00405_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00589_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00689_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00694_v3\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00719\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00721_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00754_v2\\MRI_Raw.h5',
                   'Z:\\99_LARR\\ADRC_PCVIPR\\raw_data\\adrc00767_v2\\MRI_Raw.h5']

    for i in data_source:

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('main')

        # Parse Command Line
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=int, default=0)
        parser.add_argument('--thresh', type=float, default=0.1)
        parser.add_argument('--scale', type=float, default=1.1)
        parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
        parser.add_argument('--mps_ker_width', type=int, default=16)
        parser.add_argument('--ksp_calib_width', type=int, default=32)
        parser.add_argument('--lamda', type=float, default=0.0001)
        parser.add_argument('--max_iter', type=int, default=200)
        parser.add_argument('--jsense_max_iter', type=int, default=10)
        parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
        parser.add_argument('--venc', type=float, default=80.0)


        # Input Output
        parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)', default=i)
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

        # Load Data
        logger.info(f'Load MRI from {args.filename}')

        mri_raw =  llr_recon_flow.load_MRI_raw(h5_filename=args.filename)
        num_enc = mri_raw.Num_Encodings
        llr_recon_flow.crop_kspace(mri_rawdata=mri_raw, crop_factor=2)

        # Reconstruct an low res image and get the field of view
        logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        llr_recon_flow.autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale)

        # Get sensitivity maps
        logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        smaps = llr_recon_flow.get_smaps(mri_rawdata=mri_raw, args=args)

        # Gate k-space
        mri_raw = llr_recon_flow.gate_kspace(mri_raw=mri_raw, num_frames=args.frames, gate_type='time') # control num of time frames

        # Reconstruct the image
        logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
        img = llr_recon_flow.BatchedSenseRecon(mri_raw.kdata, mps=smaps, weights=mri_raw.dcf, coord=mri_raw.coords,
                                device=sp.Device(args.device), lamda=args.lamda, num_enc=num_enc,
                                coil_batch_size=1, max_iter=args.max_iter).run()
        img = sp.to_device(img, sp.cpu_device)

        # Copy back to make easy
        smaps = sp.to_device(smaps, sp.cpu_device)
        smaps_mag = np.abs(smaps)

        img = sp.to_device(img, sp.cpu_device)
        img_mag = np.abs(img)
        img_phase = np.angle(img)

        # Export to file
        out_name = 'FullRecon.h5'
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

        with h5py.File('FullRecon.h5', 'r') as hf:
            temp = hf['IMAGE']
            print(temp.shape)
            # temp = temp['real'] + 1j*temp['imag']
            # temp = np.moveaxis(temp, -1, 0)
            # frames = int(temp.shape[0]/5)
            frames = int(temp.shape[0])
            num_encodes = int(temp.shape[1])
            print(f' num of frames =  {frames}')
            print(f' num of encodes = {num_encodes}')
            # temp = np.reshape(temp,newshape=(5, frames,temp.shape[1],temp.shape[2],temp.shape[3]))
            # temp = np.reshape(temp,newshape=(temp.shape[1], frames,temp.shape[2],temp.shape[3],temp.shape[4]))

            temp = np.moveaxis(temp, 1, -1)
            print(temp.shape)

        if num_encodes == 5:
            encoding = "5pt"
        elif num_encodes == 4:
            encoding = "4pt-referenced"
        elif num_encodes == 3:
            encoding = "3pt"

        print(f' encoding type is {encoding}')

        # Solve for Velocity
        mri_flow = MRI_4DFlow(encode_type=encoding, venc=args.venc)
        mri_flow.signal = temp
        mri_flow.solve_for_velocity()
        # mri_flow.update_angiogram()
        # mri_flow.background_phase_correct()
        mri_flow.update_angiogram()

        # Export to file
        out_name = 'Flow.h5'
        try:
            os.remove(out_name)
        except OSError:
            pass
        with h5py.File(out_name, 'w') as hf:
            hf.create_dataset("VX", data=mri_flow.velocity_estimate[..., 0])
            hf.create_dataset("VY", data=mri_flow.velocity_estimate[..., 1])
            hf.create_dataset("VZ", data=mri_flow.velocity_estimate[..., 2])

            hf.create_dataset("ANGIO", data=mri_flow.angiogram)
            hf.create_dataset("MAG", data=mri_flow.magnitude)

        outdir = i[32:]
        outdir = outdir[:-11]
        os.mkdir(outdir)
        shutil.move('AutoFOV.h5', outdir)
        shutil.move('Flow.h5', outdir)
        shutil.move('FullRecon.h5', outdir)
        shutil.move('ReconLog.h5', outdir)
        shutil.move('SenseMaps.h5', outdir)