import os
import glob
import csv
import ctypes
import h5py
import argparse
import numpy as np
from numba import njit, prange
from matplotlib import pyplot as plt
import sigpy as sp
import sigpy.mri as mri
import json
import logging
import struct
from scipy.integrate import cumtrapz

# Local libs
from orc_support import * 

# Update GE path
import sys
orc_folder = os.getenv('ORC_PYTHON_SDKTOP', '/home/larivera/Software/Orchestra/orchestra-sdk-2.1-1.python')
sys.path.append(orc_folder)
import GERecon

import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import cupy
import time
import math
import sys
import fnmatch
import hashlib
from skimage.restoration import unwrap_phase

from mri_raw import *
from multi_scale_low_rank_recon import *
from llr_recon import *
from svt import *
import numba as nb
import os
import scipy.ndimage as ndimage
from registration_tools import *


def gating_processing(gating_file,gate_delay):
    ## Load gating data
    with open(gating_file, 'rb') as fid:
        raw = np.fromfile(fid, dtype='>i4')
        raw = raw.reshape((-1, 5), order='F') # track.full file is 5 cols, track is 4 cols

    data = {
        'ecg': raw[:, 0],
        'resp': raw[:, 1], #4095 - raw[:, 1],
        'time': raw[:, 2] / 1e6, #s
        'prep': raw[:, 3],
        'acq': raw[:, 4]
    }

    pts = data["ecg"].size
    print(f'Gating files has {pts} measurements')

    ## Estimate the number of encodes based on repeated samples
    pos = pts - 1
    number_of_encodes = 1

    while (data['acq'][pos] - data['acq'][pos - 1]) == 0:
        number_of_encodes += 1
        pos -= 1

    print(f"Estimated the number of encodes = {number_of_encodes}")

    # Now go backwards to set the encode
    encode_number = np.zeros(pts, dtype=int)
    encode_number[-1] = number_of_encodes

    current_encode = number_of_encodes - 1

    #I could not identify where the C++ code assings the value of the last encode entry
    #So I do it here
    encode_number[-1] = current_encode

    for pos in range(pts - 2, -1, -1):
        if data['acq'][pos] == data['acq'][pos + 1]:
            current_encode -= 1
        else:
            current_encode = number_of_encodes - 1
        encode_number[pos] = current_encode

    print(encode_number)

    ## Check for bad ECGs
    bad_ecg_max = 2000; #4s is a long time for a heart not to beat
    bad_ecg_min = 0;    #Can't have negative time

    #find bad ECG
    bad_gates = 0

    for pos in range(pts):
        if (data["ecg"][pos] > bad_ecg_max) or (data["ecg"][pos] < bad_ecg_min):
            print("--Found bad ECG--")
            print(f"  Pos = {pos - 1} ecg = {data['ecg'][pos - 1]} acquisition time = {data['time'][pos - 1]}")
            print(f"  Pos = {pos} ecg = {data['ecg'][pos]} acquisition time = {data['time'][pos]}")
            print(f"  Pos = {pos + 1} ecg = {data['ecg'][pos + 1]} acquisition time = {data['time'][pos + 1]}")
            
            bad_gates += 1

            time_ahead = data['time'][pos + 1] - data['time'][pos]
            time_behind = data['time'][pos] - data['time'][pos - 1]
            print(f"Time ahead = {time_ahead}")
            print(f"Time behind = {time_behind}")

            # Find closest value
            if time_ahead < time_behind:
                data['ecg'][pos] = data['ecg'][pos + 1] - time_ahead * 1000
            else:
                data['ecg'][pos] = data['ecg'][pos - 1] + time_behind * 1000

            print(f"  FIXED::Pos = {pos - 1} ecg = {data['ecg'][pos - 1]} acquisition time = {data['time'][pos - 1]}")
            print(f"  FIXED::Pos = {pos} ecg = {data['ecg'][pos]} acquisition time = {data['time'][pos]}")
            print(f"  FIXED::Pos = {pos + 1} ecg = {data['ecg'][pos + 1]} acquisition time = {data['time'][pos + 1]}")

    print(f'Found {bad_gates} bad gates')

    ## Gate delay: Shift All ECG Time Stamps to account for gating system
    print(f'Delaying Projections by {gate_delay} ms')
    print("Recalculating cardiac trigger locations and shifting")
    print(f'Shifting ECG/PG by {gate_delay} ms')
    
    number_of_triggers = 0
    for pos in range(pts - 1):
        if (data['ecg'][pos + 1] < data['ecg'][pos]) or (pos == (pts - 2)):
            number_of_triggers += 1

    print(f'Counted {number_of_triggers} ecg/pg triggers')
    # Initialize the array for ECG trigger locations
    ecg_trigger_locations = np.zeros(number_of_triggers, dtype=float)
    trigger_index = 0

    for pos in range(pts - 1):
        if (data['ecg'][pos + 1] < data['ecg'][pos]) or (pos == (pts - 2)):
            #print(f'pos {pos} time : {data["time"][pos]} - {data["ecg"][pos]} * 1e-3 - {gate_delay}')
            ecg_trigger_locations[trigger_index] = data['time'][pos] - data['ecg'][pos] * 1e-3 - gate_delay * 1e-3 # check units, ok
            #print(ecg_trigger_locations)
            trigger_index += 1

    # Now update the entire ECG waveform using known trigger locations
    trigger_index_behind = 0
    for pos in range(pts):
        # Make sure we are still in the same window
        if trigger_index_behind < (number_of_triggers - 1):
            if data['time'][pos] > ecg_trigger_locations[trigger_index_behind + 1]:
                trigger_index_behind += 1
        
        data['ecg'][pos] = 1e3 * (data['time'][pos] - ecg_trigger_locations[trigger_index_behind])

    print("---------Gating Stats--------------")
    print(f"ECG Range (ms) = {data['ecg'].min()} - {data['ecg'].max()}")
    print(f"Resp Range = {data['resp'].min()} - {data['resp'].max()}")
    print(f"Acquisition Range (s) = {data['time'].min():.6f} - {data['time'].max():.6f}")
    print(f"Prep Range (ms) = {data['prep'].min() / 1e3:.3f} - {data['prep'].max() / 1e3:.3f}")
    print(f"Acquisition Order Range = {data['acq'].min()} - {data['acq'].max()}")

    ## Print ECG histogram:

    #Put into histgram (0-2s in 50ms spacing)
    total = data['ecg'].size

    print("\n\nECG Histogram - Please Check to Ensure Realistic Values")
    for i in range(40):
        start = 0 + 50 * i
        stop = 0 + 50 * (i + 1)
        
        count = np.sum((data['ecg'] >= start) & (data['ecg'] < stop))
        print(f"ECG Range ({start} to {stop}) = {100 * count / total:.2f}%")

    # Initialize bad_count
    bad_count = 0

    # Count the number of ECG values greater than or equal to 2000
    count = np.sum(data['ecg'] >= 2000)
    print(f"ECG Range ( > 2000ms) = {100 * count / total:.2f}%")
    bad_count = count

    print("\nAnalyzing ECG\n")

    # Reference to ECG values
    ecg_ref = data['ecg']

    # Calculate the median
    med_ecg = np.median(ecg_ref)
    med_ecg *= 2

    print(f"Median RR is {med_ecg} ms")
    print(f"Expected HR is {60000.0 / med_ecg:.2f} bpm")

    # Check for potential bad values
    if med_ecg < 500:
        print("WARNING!!! Calculated Heartrate is greater than 120 bpm.")

    if bad_count / total > 0.1:
        print("WARNING!!! More than 10% of the ECG values are potentially bad ( >2000ms)")

    # Count values outside of median RR
    vals_in_range = np.sum(data['ecg'] < med_ecg)
    percent_good_ecg_data = 100 * vals_in_range / total

    print(f"Values within expected RR = {percent_good_ecg_data:.2f} %")

    if percent_good_ecg_data < 90:
        print(f"WARNING!! More than 10% ({100 - percent_good_ecg_data:.2f}) of gating data lies outside expected ECG range")

    # Scale ECG vals to seconds
    data['ecg'] = data['ecg'] * 1e-3
    print('ecgvals scaled to units of s')

    return data, encode_number


def get_image( m0, v, vencs):
    image = []
    for encode, venc in enumerate(vencs):
        image.append(m0[encode]*np.exp(1j*math.pi*v/venc))
    image =np.stack(image, 0)
    return image


def error_func(im1, im2, vtest, vencs):

    error = np.zeros(im1[0].shape)
    for i in range(len(vencs)):
        error += np.abs(im1[i] - im2[i])

    return error


class recon_file:
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
        return(os.path.join(self.folder, self.filename))


def find(patterns, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    if os.path.islink(os.path.join(root, name)) == False:
                        f = recon_file()
                        f.filename = name
                        f.folder = root
                        #f.short_hash = short_md5(os.path.join(root, name))
                        f.extension = os.path.splitext(name)[1]
                        result.append(f)
    return result

def get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates=False, prep_disdaqs=0):
    logger = logging.getLogger('Get Gate bins')

    #print(gate_signal)
    #print(gate_signal[0].dtype)

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
    elif gate_type == 'prep':
        # Skip a number of projections
        t_min = np.min([np.min(gate) for gate in gate_signal]) + prep_disdaqs


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

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Gate real time images')
    print('Finding Gating information', flush=True)
    files_to_compare = find(['Gating_Track_*.pcvipr_track.full'], os.getcwd())

    for idxx, f in enumerate(files_to_compare):

        full_filename = f.fullname()
        filename = os.path.basename(full_filename)
        folder = f.folder

        print(folder)

        os.chdir(folder)
        os.system(f'ls {filename}')
        
        # load real time dataset
        try:
            with h5py.File('Vel_Time0025.h5', 'r') as hf: # For case 3 and 6 Vel_Time0.0008.h5
                image = np.array(hf['vz'])  # shape (3, 500, 320, 320)
                logger.info(f'Successfully loaded Vel_Time*.h5 from {folder}')
                # process image here...

        except (OSError, FileNotFoundError) as e:
            logger.warning(f'Could not open Vel_Time0025.h5 in {folder}: {e}')
            continue  # move to next folder

        gate_delay=200 # use same gate delay as in cardiac recon
        gating_data, encs = gating_processing(full_filename,gate_delay)

        arms=2000
        num_encodes=3

        # gating is time sorted
        raw_ecg = np.copy(gating_data['ecg'])
        raw_ecg = raw_ecg.reshape(arms,num_encodes)

        # only need gating values of first encode, and find center bin values for the 500 tf realtime recon
        rt_frames = 500
        rt_ecg = []
        for i in range(rt_frames):
            ecg1 = raw_ecg[i*4+1,0]
            ecg2 = raw_ecg[i*4+2,0]
            
            if ecg2 > ecg1:
                mean_ecg = (ecg1 + ecg2)/2.0

            elif ecg2 < ecg1: # ecg counter has reset (need to choose end of trigger or begining of next )
                mean_ecg = ecg1

            else:
                mean_ecg = ecg2

            rt_ecg.append(mean_ecg)
        
        ecg = np.stack(rt_ecg)
        #print(ecg)
        print(ecg.shape)
        
        # now bin 
        num_frames = 30 # cardiac recon results 
        t_min, t_max, delta_time = get_gate_bins(ecg, 'ecg', num_frames)

        #t_max = 1.012
        #delta_time = (t_max - t_min) / num_frames

        print(t_min)
        print(t_max)
        print(delta_time)

        points_per_bin = []
        gate_blood = []
        gate_csf   = []
        gate_ref   = []

        for t in range(num_frames):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # Find index where value is held
            idx = np.logical_and.reduce([
                np.abs(ecg[:]) >= t_start,
                np.abs(ecg[:]) < t_stop])
            
            current_points = np.sum(idx)

            # Gate the data
            points_per_bin.append(current_points)

            #print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
            logger.info(f'Frame {t} [{t_start} to {t_stop} ], Points = {current_points}')

            # if empty bin then copy previous frame
            if current_points == 0:
                ref_img = gate_ref[-1]
                blood_img = gate_blood[-1]
                csf_img = gate_csf[-1]
                print('WARNING EMPTY BINs, copying data from previous bin')
            
            else:
                ref_img = image[0, idx, :, :]
                blood_img = image[1, idx, :, :]
                csf_img = image[2, idx, :, :]

                #median range is a bit larger than mean
                mean_ref = np.median(ref_img,axis=0)
                mean_blood = np.median(blood_img,axis=0)
                mean_csf = np.median(csf_img,axis=0)

            gate_ref.append(mean_ref)
            gate_blood.append(mean_blood)
            gate_csf.append(mean_csf)
            
        new_ref   = np.stack(gate_ref)
        new_blood = np.stack(gate_blood)
        new_csf   = np.stack(gate_csf)

        stacked = np.stack((new_ref, new_blood, new_csf), axis=0)

        print(stacked.shape)

        max_points_per_bin = np.max(np.array(points_per_bin))
        logger.info(f'Max points = {max_points_per_bin}')
        logger.info(f'Points per bin = {points_per_bin}')
        logger.info(
            f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
        logger.info(f'Standard deviation = {np.std(points_per_bin)}')

        with h5py.File(f'tf_{num_frames}_Vel_Time0025.h5','w') as hf:
            hf.create_dataset('vz', data=stacked)






