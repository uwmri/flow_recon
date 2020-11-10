#match a pattern
#! /usr/bin/env python
import sys
sys.path.append('/export/home/larivera/CODE/llr_recon_py')  # UPDATE!
from flow_processing import MRI_4DFlow
import llr_recon_flow
import logging
import cupy
import argparse
import sigpy as sp
import numpy as np
import h5py
import shutil
import os, fnmatch, subprocess

import re
import os.path
from os import path

# gets dir tuple that have scan archives or pfiles
def find_scandata(pattern, path):
    result_path = []
    result_id = []
    for data in pattern:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, data):
                    #result.append(os.path.join(root, name))
                     result_path.append(os.path.join(root))
                     result_id.append(os.path.join(name))

    return result_path, result_id

y = os.getcwd() # get current path
scan_data = ['P*.7', 'P*.7.bz2','ScanArchive*.h5'] # files to get paths to

scan_path, scan_id = find_scandata(scan_data, y)

if __name__ == '__main__':

    print('number of scans found: ', len(scan_path))
    # go to each file and work

    for i in range(len(scan_path)):
        os.chdir(scan_path[i]) # enter dir
        print(scan_id[i])

        if path.exists("cardiac_hist.txt"):

        	print('cardiac_hist available')

        	if scan_id[i][-5:] == '7.bz2':  # for compress files

                    decompress_command = subprocess.run(["bzip2", "-dk", scan_id[i]])

                    pfile_name = scan_id[i][:8]

                    export_MRIdata = subprocess.run(["pcvipr_recon_binary", "-f", pfile_name, "-single_encode_tframe", "-threads", "32", "-gate_delay", "9", "-export_kdata"])

                    delete_pfile = subprocess.run(["rm", "-f", pfile_name])
 
        	else:
                    export_MRIdata = subprocess.run(["pcvipr_recon_binary", "-f", scan_id[i], "-single_encode_tframe", "-threads", "32", "-gate_delay", "9", "-export_kdata"])
        else:

            if scan_id[i][-5:] == '7.bz2':  # for compress files
                    decompress_command = subprocess.run(["bzip2", "-dk", scan_id[i]])

                    pfile_name = scan_id[i][:8]

                    gating_command = subprocess.run(["check_gating", "-f", pfile_name, "|", "tee", "cardiac_hist.txt"])

                    export_MRIdata = subprocess.run(["pcvipr_recon_binary", "-f", pfile_name, "-single_encode_tframe", "-threads", "32", "-gate_delay", "9", "-export_kdata"])

                    delete_pfile = subprocess.run(["rm", "-f", pfile_name])
            else:
                    gating_command = subprocess.run(["check_gating", "-f", scan_id[i], "|", "tee", "cardiac_hist.txt"])

                    export_MRIdata = subprocess.run(["pcvipr_recon_binary", "-f", scan_id[i], "-single_encode_tframe", "-threads", "32", "-gate_delay", "9", "-export_kdata"])


        # Check gating quality
        h = open('cardiac_hist.txt', 'r')
        content = h.readlines() # string per line
        content_str = content[-2]
        type_of_hist = content_str.split(' ', 1)[0] 

        if type_of_hist == "Expected": 
        	gating_confidence = re.findall('\d*\.?\d+',content[-1])
        	gating_hr = re.findall('\d*\.?\d+',content[-2])

        elif type_of_hist == "WARNING!!!":
        	gating_confidence = re.findall('\d*\.?\d+',content[-1])
        	gating_hr = re.findall('\d*\.?\d+',content[-3])
        else: #not type_of_hist: # some hist comes with error e.g. ecg=nan
        	gating_confidence = [0]
        	gating_hr = [0]

        print('hist confidence: ',gating_confidence,'%',' HR: ', gating_hr,'BPM')

        if float(gating_confidence[0]) > 85.0 and 35.0 <= float(gating_hr[0]) <= 180.0:
        	gating_qa_flag = [1]
        else:
        	gating_qa_flag = [0]

        print(gating_qa_flag)

        #check number of coils 

        k = open('data_header.txt', 'r')
        content2 = k.readlines() # string per line
        content_str2 = content2[1] # second line
        type_of_header = content_str2.split(' ', 1)[0]
        #print(type_of_header)

        content_str3 = content2[-1] # last line
        type_of_encode = content_str3.split(' ', 1)[0]
        #print(type_of_encode)


        # choose regularizers for llr (determined empirically)
        ecg_lamda = []
        time_lamda = []

        if type_of_header == "numrecv": 
            head_coil = re.findall('\d*\.?\d+',content2[1])

        if type_of_encode == "num_encodes":
        	encodes = re.findall('\d*\.?\d+',content2[-1])

        if float(head_coil[0]) == 32  and float(encodes[0]) == 4:
            ecg_lamda = 0.001
            time_lamda = 0.0001 

        if float(head_coil[0]) == 32  and float(encodes[0]) == 5:
            ecg_lamda = 0.001
            time_lamda = 0.0001

        if  float(head_coil[0]) >= 44 and float(encodes[0]) == 4:
            ecg_lamda = 0.001
            time_lamda = 0.0001

        print('time lamda =',time_lamda,' ecg lamda =',ecg_lamda)
        print('coil used', head_coil)
        print('encodes',encodes)

        #define ecg and time recons

        def ecg_recon(ecglamda):

	        logging.basicConfig(level=logging.INFO)
	        logger = logging.getLogger('main')

	        # Parse Command Line
	        parser = argparse.ArgumentParser()
	        parser.add_argument('--device', type=int, default=0)
	        parser.add_argument('--thresh', type=float, default=0.1)
	        parser.add_argument('--scale', type=float, default=1.0)
	        parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
	        parser.add_argument('--mps_ker_width', type=int, default=16)
	        parser.add_argument('--ksp_calib_width', type=int, default=32)
	        parser.add_argument('--lamda', type=float, default=ecglamda) #input
	        parser.add_argument('--max_iter', type=int, default=200)
	        parser.add_argument('--jsense_max_iter', type=int, default=30)
	        parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
	        parser.add_argument('--venc', type=float, default=80.0)
	        parser.add_argument('--gate_type', type=str, default='ecg', help='sets blocks size to 4 in svt')
	        parser.add_argument('--fast_maxeig', dest='fast_maxeig', action='store_true')
	        parser.set_defaults(fast_maxeig=False)
	        parser.add_argument('--test_run', dest='test_run', action='store_true')
	        parser.set_defaults(test_run=False)
	        parser.add_argument('--compress_coils', dest='compress_coils', action='store_true')
	        parser.set_defaults(compress_coils=False)


	        # Input Output
	        parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)', default= 'MRI_Raw.h5')
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
	        if args.test_run:
	            mri_raw = llr_recon_flow.load_MRI_raw(h5_filename=args.filename, max_coils=2)
	        else:
	            mri_raw = llr_recon_flow.load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils)

	        num_enc = mri_raw.Num_Encodings
	        llr_recon_flow.crop_kspace(mri_rawdata=mri_raw, crop_factor=2.5)

	        # Reconstruct an low res image and get the field of view
	        logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        llr_recon_flow.autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale, device=sp.Device(args.device))

	        # Get sensitivity maps
	        logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        smaps = llr_recon_flow.get_smaps(mri_rawdata=mri_raw, args=args, device=sp.Device(args.device))

	        # Gate k-space
	        mri_raw = llr_recon_flow.gate_kspace(mri_raw=mri_raw, num_frames=args.frames, gate_type=args.gate_type)

	        # Reconstruct the image
	        logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        img = llr_recon_flow.BatchedSenseRecon(mri_raw.kdata, mps=smaps, weights=mri_raw.dcf, coord=mri_raw.coords,
	                                device=sp.Device(args.device), lamda=args.lamda, num_enc=num_enc, batched_iter=args.max_iter,
	                                coil_batch_size=1, max_iter=args.max_iter, gate_type=args.gate_type,fast_maxeig=args.fast_maxeig).run()
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
	            hf.create_dataset("IMAGE", data=img, compression="lzf")
	            hf.create_dataset("IMAGE_MAG", data=img_mag, compression="lzf")
	            hf.create_dataset("IMAGE_PHASE", data=img_phase, compression="lzf")
	            hf.create_dataset("SMAPS", data=smaps_mag, compression="lzf")

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

	        # send data to folder
	        create_dir = subprocess.run(["mkdir", "ecg_recon"])
	        move_data = subprocess.run(["mv", "Flow.h5", "FullRecon.h5", "ReconLog.h5", "ecg_recon/"])

        def time_recon(timelamda):
        
	        logging.basicConfig(level=logging.INFO)
	        logger = logging.getLogger('main')

	        # Parse Command Line
	        parser = argparse.ArgumentParser()
	        parser.add_argument('--device', type=int, default=0)
	        parser.add_argument('--thresh', type=float, default=0.1)
	        parser.add_argument('--scale', type=float, default=1.0)    # originaly we tried 1.1
	        parser.add_argument('--frames', type=int, default=100, help='Number of time frames')
	        parser.add_argument('--mps_ker_width', type=int, default=16)
	        parser.add_argument('--ksp_calib_width', type=int, default=32)
	        parser.add_argument('--lamda', type=float, default=timelamda) # input
	        parser.add_argument('--max_iter', type=int, default=200)
	        parser.add_argument('--jsense_max_iter', type=int, default=30)
	        parser.add_argument('--jsense_max_inner_iter', type=int, default=10)
	        parser.add_argument('--venc', type=float, default=80.0)
	        parser.add_argument('--gate_type', type=str, default='time')
	        parser.add_argument('--fast_maxeig', dest='fast_maxeig', action='store_true')
	        parser.set_defaults(fast_maxeig=False)
	        parser.add_argument('--test_run', dest='test_run', action='store_true')
	        parser.set_defaults(test_run=False)
	        parser.add_argument('--compress_coils', dest='compress_coils', action='store_true')
	        parser.set_defaults(compress_coils=False)

	        # Input Output
	        parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)', default= 'MRI_Raw.h5')
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

	        if args.test_run:
	        	mri_raw = llr_recon_flow.load_MRI_raw(h5_filename=args.filename, max_coils=2)
	        else:
	        	mri_raw =  llr_recon_flow.load_MRI_raw(h5_filename=args.filename, compress_coils=args.compress_coils)

	        num_enc = mri_raw.Num_Encodings
	        llr_recon_flow.crop_kspace(mri_rawdata=mri_raw, crop_factor=2.5)

	        # Reconstruct an low res image and get the field of view
	        logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        llr_recon_flow.autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale, device=sp.Device(args.device))

	        # Get sensitivity maps
	        logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        smaps = llr_recon_flow.get_smaps(mri_rawdata=mri_raw, args=args, device=sp.Device(args.device))


	        # Gate k-space
	        mri_raw = llr_recon_flow.gate_kspace(mri_raw=mri_raw, num_frames=args.frames, gate_type=args.gate_type)

	        # Reconstruct the image
	        logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
	        img = llr_recon_flow.BatchedSenseRecon(mri_raw.kdata, mps=smaps, weights=mri_raw.dcf, coord=mri_raw.coords,
	                                device=sp.Device(args.device), lamda=args.lamda, num_enc=num_enc, batched_iter=args.max_iter,
	                                coil_batch_size=1, max_iter=args.max_iter, gate_type=args.gate_type,fast_maxeig=args.fast_maxeig).run()
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
	            hf.create_dataset("IMAGE", data=img, compression="lzf")
	            hf.create_dataset("IMAGE_MAG", data=img_mag, compression="lzf")
	            hf.create_dataset("IMAGE_PHASE", data=img_phase, compression="lzf")
	            hf.create_dataset("SMAPS", data=smaps_mag, compression="lzf")

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

	        # send data to folder
	        create_dir = subprocess.run(["mkdir", "time_recon"])
	        move_data = subprocess.run(["mv", "Flow.h5", "FullRecon.h5", "ReconLog.h5", "time_recon/"])

        #Run recons
        #if path.exists("time_recon"):
        #	print('time recon already done')
        #else:
	        # Do time-resolved recon
	    #    time_recon(time_lamda)


        # do cardiac-resolved recon if gating is good
        if gating_qa_flag == [1]:
            ecg_recon(ecg_lamda)
          
        #delete_MRIdata = subprocess.run(["rm", "-f", "MRI_Raw.h5"])
