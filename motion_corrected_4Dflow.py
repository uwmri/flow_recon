# ! /usr/bin/env python
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
import fnmatch
import time

def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

if __name__ == '__main__':

    # y = os.getcwd()  # get current path
    y = 'D:\\mc_flow\\DATA\\mc_adrc00574' # get data path
    y= '/mounts/data/analyses/larivera/projects/multivenc/test4dflow/radial_scan'
    y = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data'
    y = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data_TEST'
    print(y)
    # scan_data = ['P*.7', 'P*.7.bz2', 'ScanArchive*.h5','MRI_Raw.h5']  # files to get paths to
    #scan_data = 'MRI_Raw.h5'  # files to get paths to
    scan_data = 'ScanArchive_608WIMRMR2_20240403_152702561.h5'
    scan_paths = find_file(scan_data, y)
    print(scan_paths)
    scan_filename = scan_paths[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    #parser.add_argument('--filename', type=str, help='File to process')
    parser.add_argument('--filename', type=str, default=scan_filename, help='File to process')
    parser.add_argument('--motion_correction', dest='motion_correction', action='store_true')
    parser.add_argument('--no-motion_correction', dest='motion_correction', action='store_false')
    parser.set_defaults(motion_correction=True)
    parser.add_argument('--get_navigators', dest='get_navigators', action='store_true')
    parser.set_defaults(get_navigators=True)

    args = parser.parse_args()

    code_folder = os.path.dirname(os.path.realpath(__file__))
    print(f'Code folder {code_folder}')

    recon_script = os.path.join( code_folder, 'llr_recon_flow.py')
    motion_script = os.path.join( code_folder, 'rigid_correction.py')
    flow_script = os.path.join( code_folder, 'flow_processing.py')

    filename = os.path.realpath(args.filename) #'Q:/BBF/MRI_Raw.h5'
    base_folder = os.path.dirname(filename)
    print(f'Base folder {base_folder} and filename {os.path.join(base_folder, "MRI_Raw_Corrected.h5")}')

    file_nav = os.path.join( base_folder, 'Dynamic.h5')

    if args.motion_correction:
        if args.get_navigators:
            # Run the recon to get navigators
            #os.system(f'python {recon_script} --gate_type time --frames 256 --filename {filename} '
            #          f'--out_filename {file_nav} --out_folder {base_folder} --max_encodes 1 '
            #          f'--recon_type mslr --lamda 1e-8 --crop_factor 2--krad_cutoff 32 --compress_coils 20 --epochs 200')

            os.system(f'python {recon_script} --gate_type time --frames 2000 --filename {filename} '
                      f'--out_filename {file_nav} --out_folder {base_folder} '
                      f'--recon_type mslr --lamda 1e-10 --epochs 200 '
                      f'--demod -250 --gate_delay 200 --skope_path /mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data '
                      f'--strided_gate --shots_per_frame 1 --thresh_maps --thresh_maps_val 0')

            # Run the recon to get realtime spiral (maybe max eigen should be done across all encodes?)
            #os.system(f'python {recon_script} --gate_type ecg --frames 30 --filename {filename} '
            #          f'--out_filename {file_nav} --out_folder {base_folder} '
            #          f'--recon_type mslr --lamda 1e-8 --epochs 100 --single_encode_gate '
            #          f'--demod -250 --gate_delay 200 --skope_path /mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data '
            #          f'--thresh_maps --thresh_maps_val 0.025')
            
            #the recon seems to diverge if the smaps are threshold >0.05
            #without thresholding the smaps, recon converges, but the cardiac recon is quite flat even with the smallest lamda and many block sizes.
            #real time does not look great either for blood.
                    
        # Run the motion corrected
        #print('Motion Correcting Data')
        #os.system(f'python {motion_script} --file_nav {file_nav} --file_data {filename} --out_folder {base_folder} --out_filename MRI_Raw_Corrected.h5')
        #exit()

        # Now run a simple recon to verify the correction
        #os.system(f'python {recon_script} --filename {os.path.join(base_folder, "MRI_Raw_Corrected.h5")}'
        #          f' --gate_type ecg --frames 1'
        #          f' --recon_type sense '
        #          f' --compress_coils --thresh 0.15'
        #          f' --out_filename Corrected.h5')

        # Just llr
        os.system(f'python {recon_script} --filename {os.path.join(base_folder, "MRI_Raw_Corrected.h5")}'
                  f' --frames 20 --max_iter 100'
                  f' --gate_type ecg '
                  f' --recon_type llr --llr_block_width 4 --lamda 0.00001'
                  f' --compress_coils --thresh 0.15'
                  f' --out_filename Corrected.h5')

        # Flow processing
        os.system(f'python {flow_script} --filename {os.path.join(base_folder,"Corrected.h5")} '
                  f'--out_filename Flow.h5')

    # Reference recon without corrections
    #os.system(f'python {recon_script} --filename {filename}'
    #          f' --gate_type ecg --frames 1'
    #          f' --recon_type sense '
    #          f' --compress_coils --thresh 0.15'
    #          f' --out_filename NoCorrection.h5')

    # Run recon without corrections
    # Just llr
    #os.system(f'python {recon_script} --filename {filename}'
    #          f' --frames 20 --max_iter 100'
    #          f' --gate_type ecg --fast_maxeig'
    #          f' --recon_type llr --llr_block_width 4 --lamda 0.00001'
    #          f' --compress_coils --thresh 0.15'
    #          f' --out_filename Images.h5')
    # Just sense
    #os.system(f'python {recon_script} --filename {filename}'
    #          f' --frames 20 --max_iter 10'
    #          f' --gate_type ecg'
    #          f' --recon_type sense'
    #          f' --compress_coils --thresh 0.15'
    #          f' --out_filename Images.h5')

    # Just pils
    #os.system(f'python {recon_script} --filename {filename}'
    #          f' --frames 20'
    #          f' --gate_type ecg --recon_type pils'
    #          f' --compress_coils --thresh 0.15'
    #          f' --out_filename Images.h5')

    # Flow processing
    #os.system(f'python {flow_script} --filename {os.path.join(base_folder,"Images.h5")} '
    #          f'--out_filename FlowNoCorrection.h5')


