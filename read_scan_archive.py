'''
The goal of this code is to load Scan Archive data for 2D PC MRI using spiral readout:
    
    It requires k coordiates (from SKOPE will update the coordinates with skope coordinates if available)
    will load the ScanArchive data and header information
    will perform off-iso center correction (oc_shift) and a user defined global demodulation
    will perform a B0 term from SKOPE correction
    will perform kspace whitening 
        there are some issues with Python SDK reading ConfigUD and kacq_uid from header as float which leads to 7 digit precisio only
    will read gating data (ecg, resp, time, prep, acq), and apply gate delay to ecgvals. Only supports *track.full gating file format
    will create an MRI_Raw.h5 like structure to use by the flow recon code

    It will not perform gradient calibration

'''

# %% Setup 
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

# Local libs
from orc_support import * 

# Update GE path
import sys
orc_folder = os.getenv('ORC_PYTHON_SDKTOP', '/home/larivera/Software/Orchestra/orchestra-sdk-2.1-1.python')
sys.path.append(orc_folder)
import GERecon

# %% classes and functions

class MRI_Raw:
    Num_Encodings = 0
    Num_Coils = 0
    trajectory_type = None
    dft_needed = None
    Num_Frames = None
    coords = None
    time = None
    ecg = None
    prep = None
    resp = None
    dcf = None
    kdata = None
    frame = None
    target_image_size = [256, 256, 64]

def pca_coil_compression(kdata=None, axis=0, target_channels=None):
    logger = logging.getLogger('PCA_CoilCompression')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f'Compressing to {target_channels} channels, along axis {axis}')
    logger.info(f'Initial  size = {kdata_cc.shape} ')

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f'Old channels =  {old_channels} ')

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1 - 0.05])

    # Create a subsampled array
    kcc = np.zeros((old_channels, np.sum(mask)), dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[..., c]
        kcc[c, :] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e], -1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e] = np.matmul(u, kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e], -1, axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata, axis, -1)
        kdata = np.expand_dims(kdata, -1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata, axis=-1)
        kdata = kdata[..., :target_channels]

        # Put back
        kdata = np.moveaxis(kdata, -1, axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata

    
def convert_float_to_uint(x):
    # Import the struct module for packing and unpacking binary data
    import struct
    
    # Pack the float into binary data and unpack it as an unsigned int
    val = struct.unpack('I', struct.pack('f', x))[0]
    
    return val

# load skope data and run the recon again
def load_skope_data(path):

    # Open the HDF5 file in read mode
    with h5py.File(path, 'r') as hf:
        # Load the datasets
        coord = hf['coord'][:]
        kw = hf['kw'][:]
        b0 = hf['b0'][:]

    return coord, kw, b0 

@njit(parallel=True)
def apply_offisocenter_data_demod(kx, ky, kdata, oc_xshift, oc_yshift, demod_freq, time_mri):
    PI = np.pi
    for enc in prange(kx.shape[1]):
        for shot in range(kx.shape[0]):
            for i in range(kx.shape[2]):

                # Shift due to off iso-center
                #freq_shift = 2.0 * PI * (-oc_xshift * kx[ shot, enc, i] - oc_yshift * ky[shot, enc, i])
                freq_shift = 2.0 * PI * (-oc_xshift * kx[ shot, enc, i] + oc_yshift * ky[shot, enc, i])

                shift = np.exp(1j * freq_shift)
                demod = np.exp(1j * demod_freq * 2.0 * PI * time_mri[i])
                kdata[shot, enc, i] *= shift * demod 

    return kdata

@njit(parallel=True)
def apply_b0correction(kdata, b0):
    for enc in prange(kdata.shape[1]):
        for coil in range(kdata.shape[-1]):
            kdata[:,enc,:,coil] *= np.exp(1j * b0[:,enc,:])

    return kdata

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

    return data, encode_number

def define_MRI_Raw_dictionary(ksp, coord, kw, ktime, gating_data, arms, Num_Encodings, Num_Coils, Num_Frames, trajectory, spiral_z_encoding ):

    trajectory_type = np.zeros(3)
    dft_needed = np.zeros(3)

    if trajectory == 'NONCARTESIAN':
        for i in range(2):
            trajectory_type[i] = 1
            dft_needed[i] = 1

    if spiral_z_encoding == "CARTESIAN_PHASE_ENCODED":
        dft_needed[2] = 1
        trajectory_type[2] = 0

    elif spiral_z_encoding == "NONCARTESIAN_PHASE_ENCODED":
        dft_needed[2] = 1
        trajectory_type[2] = 1
        
    elif spiral_z_encoding == "SLICE_ENCODED":
        dft_needed[2] = 0
        trajectory_type[2] = 0

    # Create data structure with groups 
    MRI_Raw = {
        'Kdata': {
            'Num_Encodings': Num_Encodings,
            'Num_Coils': Num_Coils,
            'Num_Frames': Num_Frames,
            'trajectory_typeX': int(trajectory_type[0]),
            'trajectory_typeY': int(trajectory_type[1]),
            'trajectory_typeZ': int(trajectory_type[2]),
            'dft_neededX': int(dft_needed[0]),
            'dft_neededY': int(dft_needed[1]),
            'dft_neededZ': int(dft_needed[2])
        },

        'Gating': {

        }
        
    }

    # desire convention shape for export is (1868, 2000) (pts,arms) 
    # time_mri are derived from the data using the TE, BW and num points, is the same across encodes
    # print(time_mri.shape)
    # Reshape the array to have repeats for spiral arms, 
    KTime = np.tile(ktime[:, np.newaxis], (1, arms))
    print(KTime.shape)

    #trajectories will be imported from SKOPE measures
    #print(coord.shape)
    #print(ksp.shape)
    #print(kw.shape)

    export_coord = np.moveaxis(coord, 0, -2)
    print(export_coord.shape) #(3, 1868, 2000, 2)

    export_kw = np.moveaxis(kw, 1, -1)
    print(export_kw.shape) #(3, 1868, 2000)

    export_ksp = np.moveaxis(ksp, 1, -1)
    print(export_ksp.shape) # (48, 3, 1868, 2000)

    # # desire convention shape for gating export is (arms, encodes) 

    #print(gating_data['ecg'].shape)
    #print(gating_data['resp'].shape)
    #print(gating_data['time'].shape)
    #print(gating_data['prep'].shape)
    #print(gating_data['acq'].shape)

    # raw kspace (6000, 1868, 48), ecg data (6000,)
    ecg = np.copy(gating_data['ecg'])
    ecg = ecg.reshape(arms,Num_Encodings)

    resp = np.copy(gating_data['resp'])
    resp = resp.reshape(arms,Num_Encodings)

    time = np.copy(gating_data['time'])
    time = time.reshape(arms,Num_Encodings)

    prep = np.copy(gating_data['prep'])
    prep = prep.reshape(arms,Num_Encodings)

    #print(ecg.shape)
    #print(resp.shape)
    #print(time.shape)
    #print(prep.shape)

    for encode in range(Num_Encodings):
        print(f"Exporting {encode}")
        
        try:
            s = f"KT_E{encode}"
            MRI_Raw['Kdata'][s] = KTime
        except Exception as e:
            print(f"Can't export KT for encode {encode}: {e}")
        
        try:
            s = f"KX_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,0]
        except Exception as e:
            print(f"Can't export KX for encode {encode}: {e}")

        try:
            s = f"KY_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,1]
        except Exception as e:
            print(f"Can't export KY for encode {encode}: {e}")
        
        try:
            s = f"KZ_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,2]
        except Exception as e:
            print(f"Can't export KZ for encode {encode}: {e}, filling KZ with 0s (KX shape)")
            MRI_Raw['Kdata'][s] = np.zeros_like(export_coord[encode, :, :, 0])
        
        try:
            s = f"KW_E{encode}"
            MRI_Raw['Kdata'][s] = export_kw[encode,:,:]
        except Exception as e:
            print(f"Can't export KW for encode {encode}: {e}")
        
        print("Exporting data")
        for coil in range(Num_Coils):
            try:
                s = f"KData_E{encode}_C{coil}"
                MRI_Raw['Kdata'][s] = export_ksp[coil, encode,:,:]
            except Exception as e:
                print(f"Can't export Kdata for encode {encode}, coil {coil}: {e}")

        if ecg.size != 0:
            print("Exporting Gating")

            try:
                s = f"ECG_E{encode}"
                MRI_Raw['Gating'][s] = ecg[:,encode]
            except Exception as e:
                print(f"Can't export ECG data: {e}")

            try:
                s = f"RESP_E{encode}"
                MRI_Raw['Gating'][s] = resp[:,encode]
            except Exception as e:
                print(f"Can't export Resp data: {e}")

            try:
                s = f"PREP_E{encode}"
                MRI_Raw['Gating'][s] = prep[:,encode]
            except Exception as e:
                print(f"Can't export PREP data: {e}")

            try:
                s = f"TIME_E{encode}"
                MRI_Raw['Gating'][s] = time[:,encode]
            except Exception as e:
                print(f"Can't export TIME data: {e}")

    return MRI_Raw


def define_MRI_Raw_structure(ksp, coord, kw, ktime, gating_data, arms, Num_Encodings, Num_Coils, Num_Frames, trajectory, spiral_z_encoding ):

    trajectory_type = np.zeros(3)
    dft_needed = np.zeros(3)

    if trajectory == 'NONCARTESIAN':
        for i in range(2):
            trajectory_type[i] = 1
            dft_needed[i] = 1

    if spiral_z_encoding == "CARTESIAN_PHASE_ENCODED":
        dft_needed[2] = 1
        trajectory_type[2] = 0

    elif spiral_z_encoding == "NONCARTESIAN_PHASE_ENCODED":
        dft_needed[2] = 1
        trajectory_type[2] = 1
        
    elif spiral_z_encoding == "SLICE_ENCODED":
        dft_needed[2] = 0
        trajectory_type[2] = 0

    # Create data structure with groups 
    MRI_Raw = {
        'Kdata': {
            'Num_Encodings': Num_Encodings,
            'Num_Coils': Num_Coils,
            'Num_Frames': Num_Frames,
            'trajectory_typeX': int(trajectory_type[0]),
            'trajectory_typeY': int(trajectory_type[1]),
            'trajectory_typeZ': int(trajectory_type[2]),
            'dft_neededX': int(dft_needed[0]),
            'dft_neededY': int(dft_needed[1]),
            'dft_neededZ': int(dft_needed[2])
        },

        'Gating': {

        }
        
    }

    # desire convention shape for export is (1868, 2000) (pts,arms) 
    # time_mri are derived from the data using the TE, BW and num points, is the same across encodes
    # print(time_mri.shape)
    # Reshape the array to have repeats for spiral arms, 
    KTime = np.tile(ktime[:, np.newaxis], (1, arms))
    print(KTime.shape)

    #trajectories will be imported from SKOPE measures
    #print(coord.shape)
    #print(ksp.shape)
    #print(kw.shape)

    export_coord = np.moveaxis(coord, 0, -2)
    print(export_coord.shape) #(3, 1868, 2000, 2)

    export_kw = np.moveaxis(kw, 1, -1)
    print(export_kw.shape) #(3, 1868, 2000)

    export_ksp = np.moveaxis(ksp, 1, -1)
    print(export_ksp.shape) # (48, 3, 1868, 2000)

    # # desire convention shape for gating export is (arms, encodes) 

    #print(gating_data['ecg'].shape)
    #print(gating_data['resp'].shape)
    #print(gating_data['time'].shape)
    #print(gating_data['prep'].shape)
    #print(gating_data['acq'].shape)

    # raw kspace (6000, 1868, 48), ecg data (6000,)
    ecg = np.copy(gating_data['ecg'])
    ecg = ecg.reshape(arms,Num_Encodings)

    resp = np.copy(gating_data['resp'])
    resp = resp.reshape(arms,Num_Encodings)

    time = np.copy(gating_data['time'])
    time = time.reshape(arms,Num_Encodings)

    prep = np.copy(gating_data['prep'])
    prep = prep.reshape(arms,Num_Encodings)

    #print(ecg.shape)
    #print(resp.shape)
    #print(time.shape)
    #print(prep.shape)

    for encode in range(Num_Encodings):
        print(f"Exporting {encode}")
        
        try:
            s = f"KT_E{encode}"
            MRI_Raw['Kdata'][s] = KTime
        except Exception as e:
            print(f"Can't export KT for encode {encode}: {e}")
        
        try:
            s = f"KX_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,0]
        except Exception as e:
            print(f"Can't export KX for encode {encode}: {e}")

        try:
            s = f"KY_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,1]
        except Exception as e:
            print(f"Can't export KY for encode {encode}: {e}")
        
        try:
            s = f"KZ_E{encode}"
            MRI_Raw['Kdata'][s] = export_coord[encode,:,:,2]
        except Exception as e:
            print(f"Can't export KZ for encode {encode}: {e}, filling KZ with 0s (KX shape)")
            MRI_Raw['Kdata'][s] = np.zeros_like(export_coord[encode, :, :, 0])
        
        try:
            s = f"KW_E{encode}"
            MRI_Raw['Kdata'][s] = export_kw[encode,:,:]
        except Exception as e:
            print(f"Can't export KW for encode {encode}: {e}")
        
        print("Exporting data")
        for coil in range(Num_Coils):
            try:
                s = f"KData_E{encode}_C{coil}"
                MRI_Raw['Kdata'][s] = export_ksp[coil, encode,:,:]
            except Exception as e:
                print(f"Can't export Kdata for encode {encode}, coil {coil}: {e}")

        if ecg.size != 0:
            print("Exporting Gating")

            try:
                s = f"ECG_E{encode}"
                MRI_Raw['Gating'][s] = ecg[:,encode]
            except Exception as e:
                print(f"Can't export ECG data: {e}")

            try:
                s = f"RESP_E{encode}"
                MRI_Raw['Gating'][s] = resp[:,encode]
            except Exception as e:
                print(f"Can't export Resp data: {e}")

            try:
                s = f"PREP_E{encode}"
                MRI_Raw['Gating'][s] = prep[:,encode]
            except Exception as e:
                print(f"Can't export PREP data: {e}")

            try:
                s = f"TIME_E{encode}"
                MRI_Raw['Gating'][s] = time[:,encode]
            except Exception as e:
                print(f"Can't export TIME data: {e}")

    return MRI_Raw


# %% 
def load_ScanArchive(archive_filename_scan, gate_delay, demod, skope_path):
    '''
        Data loaded from ScanArchive 
    '''
    #archive_filename_scan = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/ScanArchive_608WIMRMR2_20240403_152702561.h5'
    archive = GERecon.Archive(archive_filename_scan)
    archive_data_all_scan = read_scan_achive_data(archive) #(6000,1868,44)

    rawdata = np.stack(archive_data_all_scan, axis=0)
    print(f'rawdata shape = {rawdata.shape} type = {rawdata.dtype}')
    
    '''
        Load Some Header information 
    '''
    metadata = archive.Metadata()
    x_res = metadata["acquiredXRes"]
    y_res = metadata["acquiredYRes"]
    num_control = metadata["controlCount"]
    num_channels = metadata["numChannels"]
    num_passes = metadata["passes"]
    slices_per_pass = archive.SlicesPerPass()
    header = archive.Header()

    xres = int(header['rdb_hdr_rec']['rdb_hdr_da_xres'])
    # A more accurate timming may be:
    # rba_extra_data = int(header['rdb_hdr_rec']['rdb_hdr_user11']) (in the 2d pc spiral protocol this about 3 points)
    # xres = int(header['rdb_hdr_rec']['rdb_hdr_da_xres']) - rba_extra_data
    bw = float(header['rdb_hdr_image']['vbw'])
    dt_mri = 1./(2*bw*1000)
    #time_mri = np.arange(xres)*dt_mri 

    Num_Coils = rawdata.shape[-1]
    mri_points = rawdata.shape[-2]
    TE = float(header['rdb_hdr_image']['te']) #us
    time_mri = np.arange(mri_points)*dt_mri + TE*1e-6

    fov = float(header['rdb_hdr_image']['dfov']) # mm
    arms = int(header['rdb_hdr_rec']['rdb_hdr_user10']) # number of spiral arms 

    val = convert_float_to_uint(header['rdb_hdr_rec']['rdb_hdr_user16'])
    resx = int(val % 1024)
    resy = int(((val - resx) % (1024 * 1024)) / 1024)
    #resz = int(((val - resx - 1024 * resy)) / (1024 * 1024))
    rhuser30 = convert_float_to_uint(header['rdb_hdr_rec']['rdb_hdr_user30'])
    total_num_encodes = int((float(rhuser30 % 100) / 1.0))

    print(mri_points)
    print(time_mri)
    print(total_num_encodes)
    print(x_res)
    print(y_res)
    print(Num_Coils)
    print(num_control)
    print(num_passes)
    print(slices_per_pass)
    #print(header)
    print(resx)
    print(resy)
    #print(resz)
    print(fov)
    print(arms)
    print(TE)
    print(metadata)
    
    '''
        Prepare kspace for processing (NOT DOING CALIBRATION) 
    '''
    total_samples = total_num_encodes*arms
    print(total_samples)
    kdata = rawdata[:total_samples,...]
    print(f'raw kspace shape {kdata.shape}')

    kdata = kdata.reshape(arms,total_num_encodes,xres,Num_Coils)
    print(f'kspace shape for recon {kdata.shape}')
    
    '''
        Load SKOPE data 
    '''
    #path = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data'

    coord, kw, b0 = load_skope_data(skope_path)
    print(f' skope coord shape = {coord.shape}') 
    print(f' skope weights (pipe) shape = {kw.shape}') 
    print(f' skope B0 shape = {b0.shape}') 
    
    try:
        kdata= apply_b0correction(kdata, b0)
        print("B0 data found: correction performed")

    except NameError:
        print("No B0 correction")

    #read demod factor
    #demod = -250
    
    '''
        Estimate and apply off-center correction and global demodulation
    '''
    # Off isocenter shifts

    oc_xshift = (-header['rdb_hdr_rec']['rdb_hdr_user24']) / fov
    oc_yshift = (-header['rdb_hdr_rec']['rdb_hdr_user25']) / fov
    #oc_zshift = (-header['rdb_hdr_rec']['rdb_hdr_user26']/100) / fov

    print(f'off-center shift in x = {oc_xshift*fov} mm and y = {oc_yshift*fov} mm')

    #skope coord shape = (2000, 3, 1868, 2)
    #kspace shape for recon (2000, 3, 1868, 48)

    # Do we really need to do it on a coil-by-coil basis ?
    kdata_off_corrected = np.zeros_like(kdata, dtype=kdata.dtype)
    for coil in range(kdata.shape[-1]):
        kdata_off_corrected[:,:,:,coil] = apply_offisocenter_data_demod( coord[:,:,:,0], coord[:,:,:,1], kdata[:,:,:,coil], oc_xshift, oc_yshift, demod, time_mri)

    '''
        Kspace whitening

        Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 7, which is not enough to represent the header value. 
        GE might need to fix this. 
        In the meantime read files using precision of 7 until a fix is available. 
    '''

    # keep the first 7 digits only because of GE error
    Coil_UID = header['rdb_hdr_rec']['rdb_hdr_coilConfigUID']
    strCoil_UID = str(Coil_UID)
    Coil_UID7   = strCoil_UID[:7]
    print(Coil_UID)
    print(Coil_UID7)

    # Define the path to the directory containing the files
    noise_directory = os.path.dirname(archive_filename_scan)
    noise_file_pattern = os.path.join(noise_directory, f'NoiseStatistics-Coil{Coil_UID7}*-Repetition0000.h5')

    # Find all files that match the pattern
    matching_files = glob.glob(noise_file_pattern)
    noise_file = matching_files[0] if matching_files else None

    print(noise_file)

    # Move coils to first dim for noise whitenning 
    ksp = np.moveaxis(kdata_off_corrected, -1, 0)
    print(ksp.shape)

    with h5py.File(noise_file, 'r') as hf:
        noise = hf['Data']['NoiseData']['real'] + 1j * hf['Data']['NoiseData']['imag']
        print(noise.shape)
        cov = sp.mri.get_cov(noise)
        ksp = sp.mri.whiten(ksp, cov)

    '''
    try:
        noise = hf['Kdata']['Noise']['real'] + 1j * hf['Kdata']['Noise']['imag']
        logging.info('Whitening ksp.')
        cov = mr.util.get_cov(noise)
        ksp = mr.util.whiten(ksp, cov)
    except Exception:
        ksp /= np.abs(ksp).max()
        logging.info('No noise data.')
        pass
    '''
    
    '''
        Load Gating Data

        Assuming ScanArchive data was stored sequentially 

        Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 7, which is not enough to represent the header value. 
        GE might need to fix this. 
        In the meantime read files using precision of 7 until a fix is available. 

    '''
    # keep the first 7 digits only because of GE error
    kacq_uid = header['rdb_hdr_rec']['rdb_hdr_kacq_uid']
    str_kacq_uid = str(kacq_uid)
    kacq_uid7   = str_kacq_uid[:7]
    print(kacq_uid)
    print(kacq_uid7)

    # Define the path to the directory containing the files
    gating_directory = os.path.dirname(archive_filename_scan)
    gating_file_pattern = os.path.join(gating_directory, f'Gating_Track_{kacq_uid7}*.pcvipr_track.full')

    # Find all files that match the pattern
    matching_files = glob.glob(gating_file_pattern)
    gating_file = matching_files[0] if matching_files else None

    print(f'Using {gating_file}')

    #gate_delay = 200

    gating_data, gating_num_enc = gating_processing(gating_file,gate_delay)

    # Scale ecg vals to seconds
    gating_data['ecg'] = gating_data['ecg'] * 1e-3
    print(gating_data['ecg'])

    '''
        Arrange data into MRI_Raw structure
        
    '''
    # Initialize some of the variables
    Num_Frames = 1 # Not sure where to get this variable from. Probably comes from kdata thats binned to rcframes ?
    trajectory = 'NONCARTESIAN'
    spiral_z_encoding = 'CARTESIAN_PHASE_ENCODED'

    MRI_Raw = define_MRI_Raw(ksp, coord, kw, time_mri, gating_data, arms, total_num_encodes, Num_Coils, Num_Frames, trajectory, spiral_z_encoding )

    return MRI_Raw



