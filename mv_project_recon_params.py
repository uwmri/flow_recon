#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import fnmatch
import hashlib
import sys


class ge_file:
    def __init__(self):
        self.full_hash = None
        self.short_hash = None
        self.filename = None
        self.folder = None
        self.extension = None

    def __str__(self):
        name = f'[ File:{os.path.join(self.folder,self.filename)} Hash:{self.short_hash} ]'
        return name

    def fullname(self):
        return (os.path.join(self.folder, self.filename))

    def update_short_hash(self):
        self.short_hash = short_md5(os.path.join(self.folder, self.filename))

    def update_hash(self):
        self.full_hash = full_md5(os.path.join(self.folder, self.filename))


def short_md5(fname, chunk_length=1048576):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        chunk = f.read(chunk_length)
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


def full_md5(fname, chunk_length=1048576):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_length), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                if os.path.islink(os.path.join(root, name)) == False:
                    f = ge_file()
                    f.filename = name
                    f.folder = root
                    # f.short_hash = short_md5(os.path.join(root, name))
                    f.extension = os.path.splitext(name)[1]
                    result.append(f)
    return result


def duplicate_set_filenames(file_list):
    uniq_list = []
    uniq_set = set()
    duplicate_set = set()
    for item in file_list:
        if item.filename not in uniq_set:
            uniq_list.append(item.filename)
            uniq_set.add(item.filename)
        elif item.filename not in duplicate_set:
            duplicate_set.add(item.filename)

    duplicate_files = []
    for item in file_list:
        if item.filename in duplicate_set:
            duplicate_files.append(item)

    print(f'Input {len(file_list)} files, {len(duplicate_files)} are duplicates')

    return duplicate_set, duplicate_files


def duplicate_set_short_hash(file_list):
    uniq_list = []
    uniq_set = set()
    duplicate_set = set()
    for item in file_list:
        if item.short_hash not in uniq_set:
            uniq_list.append(item.short_hash)
            uniq_set.add(item.short_hash)
        elif item.short_hash not in duplicate_set:
            duplicate_set.add(item.short_hash)

    duplicate_files = []
    for item in file_list:
        if item.short_hash in duplicate_set:
            duplicate_files.append(item)

    print(f'Input {len(file_list)} files, {len(duplicate_files)} are duplicates')

    return duplicate_set, duplicate_files


if __name__ == '__main__':

    # Returns list with file structure
    print('Finding Scan Archives', flush=True)
    #files_to_compare = find('MRI_Raw.h5', os.getcwd())
    files_to_compare = find('ScanArchive*.h5', os.getcwd())
   
    for idx, f in enumerate(files_to_compare):

        full_filename = f.fullname()
        folder = f.folder
        filename = os.path.basename(full_filename)

        #print(f'{full_filename} {folder}')
        print(f'{filename}')

        os.chdir(folder)

        # Read from best_demod.txt
        with open('best_demod.txt', 'r') as f:
            demod = float(f.read().strip())
            # Print the variable
            print("Demod value:", demod)

        # For cardiac recons:
        # We decided to use no CSmap masking and lamda = 0.001 (less flickering), may have to go up or down depending on case (use steps of 1e-3)
        # With CSmap masking we need a lamda ~= 0.0002, may have to go up or down depending on case (use steps of 2.5e-4)
        # jsense still produces best smaps (even if they are not great and have holes). B0 skope correction off otherwise large flickering artifacts
        # Filtering data makes kspace look cleaner, but minimal changes in image quality including it may remove some noise. Add it to the post-procesing.
        
        lamda = 0.001
        command = (
                    'python /home/larivera/CODE/RECON/python_recon/flow_recon/llr_recon_flow.py '
                    f'--filename {filename} '
                    '--thresh_maps '
                    '--thresh_maps_val 0 '
                    '--recon_type llr '
                    '--max_iter 200 '
                    '--llr_block_width 8 '
                    '--frames 20 '
                    '--gate_type ecg '
                    f'--lamda {lamda} '
                    f'--demod {demod} '
                    '--gate_delay 200 '
                    '--single_encode_gate '
                    '--skope_path /mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data '
                    '> recon.log'
                )
        
         #'--lamda 0.001 ' valued used for most recons
         #'--lamda 0.00025 ' value used for cases 3 and 6

        #os.system(command)
        #os.rename('FullRecon.h5', f'tf_20_Cardiac{lamda}.h5') # we have tested block width 8 mostly  (block 4 not necessary better)

        # for realtime recons: 
        # We decided to use no CSmap masking and lamda = 0.0025 (less flickering), may have to go up or down depending on case (use steps of 2e-3)
        # With CSmap masking we need a lamda ~= 0.0005, may have to go up or down depending on case (use steps of 2.5e-4)
        
        #lamda = 
        command = (
                    'python /home/larivera/CODE/RECON/python_recon/flow_recon/llr_recon_flow.py '
                    f'--filename {filename} '
                    '--thresh_maps '
                    '--thresh_maps_val 0 '
                    '--strided_gate '
                    '--shots_per_frame 4 '
                    '--recon_type llr '
                    '--max_iter 200 '
                    '--llr_block_width 8 '
                    '--frames 500 '
                    '--gate_type time '
                    f'--lamda {lamda} '
                    f'--demod {demod} '
                    '--gate_delay 200 '
                    '--skope_path /mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data '
                    '> recon.log'
                )
            #'--lamda 0.0025 ' value used for most recons
            #'--lamda 0.0008 ' value used for cases 3 and 6

        #os.system(command)
        #os.rename('FullRecon.h5', 'Time0025_500tf_200iter_noSmapmask.h5')
        #os.rename('FullRecon.h5', f'400tf_Time{lamda}.h5')


        # time average
        command = (
                    'python /home/larivera/CODE/RECON/python_recon/flow_recon/llr_recon_flow.py '
                    f'--filename {filename} '
                    '--thresh_maps '
                    '--thresh_maps_val 0 '
                    '--recon_type pils '
                    '--max_iter 200 '
                    '--llr_block_width 8 '
                    '--frames 1 '
                    '--gate_type ecg '
                    f'--lamda 0.01 '
                    f'--demod -250 '
                    '--gate_delay 200 '
                    '--single_encode_gate '
                    '--skope_path /mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data '
                    '> recon.log'
                )
        
         #'--lamda 0.001 ' valued used for most recons
         #'--lamda 0.00025 ' value used for cases 3 and 6

        os.system(command)
        os.rename('FullRecon.h5', 'withSkope_timeAvg_pils.h5') # we have tested block width 8 mostly  (block 4 not necessary better)



    