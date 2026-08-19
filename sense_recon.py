# ! /usr/bin/env python
import time
import fnmatch
import shutil
import os
import h5py
import numpy as np
import argparse
import cupy
import logging
import llr_recon_flow
from flow_processing import MRI_4DFlow
import sys
sys.path.append('/home/bxa033/Home/CODE/python_recon/flow_recon')  # UPDATE!

# gets dir tuple that have scan archives or pfiles


def find_scandata(pattern, path):
    result_path = []
    result_id = []
    for data in pattern:
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, data):
                    # result.append(os.path.join(root, name))
                    result_path.append(os.path.join(root))
                    result_id.append(os.path.join(name))

    return result_path, result_id

# def find_file(pattern, path):
#    result = []
#    for root, dirs, files in os.walk(path):
#        for name in files:
#            if fnmatch.fnmatch(name, pattern):
#                result.append(os.path.join(root, name))
#    return result

# def sizeof_fmt(num, suffix='B'):
#    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
#        if abs(num) < 1024.0:
#            return "%3.1f%s%s" % (num, unit, suffix)
#        num /= 1024.0
#    return "%.1f%s%s" % (num, 'Yi', suffix)


if __name__ == '__main__':

    y = os.getcwd()  # get current path
    # y = '/home/larivera/projects/mc_flow/impaired/adrc00302' # get data path
    print(y)
    # scan_data = ['P*.7', 'P*.7.bz2', 'ScanArchive*.h5','MRI_Raw.h5']  # files to get paths to
    # files to get paths to (previously generated)
    scan_data = ['MRI_Raw_Corrected.h5', 'MRI_Raw.h5']
    # scan_paths = find_file(scan_data, y)
    scan_paths, scan_id = find_scandata(scan_data, y)
    print(scan_paths)
    print(scan_id)

    print('number of scans found: ', len(scan_paths))

    for i in range(len(scan_paths)):

        # os.chdir(os.path.dirname(scan_paths[i])) # enter dir
        os.chdir(scan_paths[i])  # enter dir
        cwd = os.getcwd()
        print('Working on path and filename')
        print(cwd)
        print(scan_id[i])
        scan_filename = scan_id[i]

        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=int, default=0)
        # parser.add_argument('--filename', type=str, help='File to process')
        parser.add_argument('--filename', type=str,
                            default=scan_filename, help='File to process')
        parser.add_argument('--motion_correction',
                            dest='motion_correction', action='store_true')
        parser.add_argument('--no-motion_correction',
                            dest='motion_correction', action='store_false')
        parser.set_defaults(motion_correction=False)
        parser.add_argument('--get_motion_navigators',
                            dest='get_motion_navigators', action='store_true')
        parser.set_defaults(get_motion_navigators=True)
        args = parser.parse_args()

        code_folder = '/home/bxa033/Home/CODE/python_recon/flow_recon'
        print(f'Code folder {code_folder}')

        recon_script = os.path.join(code_folder, 'llr_recon_flow.py')
        motion_script = os.path.join(code_folder, 'rigid_correction.py')
        flow_script = os.path.join(code_folder, 'flow_processing.py')

        filename = os.path.realpath(args.filename)  # 'Q:/BBF/MRI_Raw.h5'
        base_folder = os.path.dirname(filename)
        print(
            f'Base folder {base_folder} and filename {os.path.join(base_folder, "MRI_Raw.h5")}')

        file_nav = os.path.join(base_folder, 'Dynamic.h5')

        # Just LLR
        os.system(f'python {recon_script} --filename {filename} '
                  f' --frames 20'
                  f' --gate_type ecg'
                  f' --recon_type sense'
                  f' --max_iter 10'
                  f' --compress_coils --thresh 0.15'
                  f' --out_filename Images.h5')

        # Flow processing
        os.system(f'python {flow_script} --filename {os.path.join(base_folder, "Images.h5")} '
                  f'--out_filename Flow.h5')
