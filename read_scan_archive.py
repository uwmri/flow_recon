'''
The goal of this code is to load Scan Archive data for 2D PC MRI using center out spiral readout:
    will load the ScanArchive data and header information
    will load coordiates from SKOPE measures or will try to estimate coordinates from gradient data
    will perform off-iso center correction (oc_shift) and a user defined global demodulation
    will perform a B0 term from SKOPE correction
    will perform kspace whitening 
        there are some issues with Python SDK reading ConfigUD and kacq_uid from header as float which leads to 7 digit precisio only
    will read gating data (ecg, resp, time, prep, acq), and apply gate delay to ecgvals. Only supports *track.full gating file format
    will create an MRI_Raw.h5 like structure to use by the flow recon code

    It will not perform gradient calibration
    (consider pip install pyvoro)

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
import struct
from scipy.integrate import cumtrapz

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


def load_skope_data(path):

    # Open the HDF5 file in read mode
    with h5py.File(path, 'r') as hf:
        # Load the datasets
        coord = hf['coord'][:]
        kw = hf['kw'][:]
        b0 = hf['b0'][:]

    return coord, kw, b0 


#could be faster using broadcasting
@njit(parallel=True)
def apply_offisocenter_data_demod(kx, ky, kdata, oc_xshift, oc_yshift, time_mri, demod_freq=0):
    PI = np.pi
    for enc in prange(kx.shape[1]):
        for shot in range(kx.shape[0]):
            for i in range(kx.shape[2]):

                # Shift due to off iso-center ( in the Cpp code the oc_yshift is negative)
                #freq_shift = 2.0 * PI * (-oc_xshift * kx[ shot, enc, i] - oc_yshift * ky[shot, enc, i])
                freq_shift = 2.0 * PI * (-oc_xshift * kx[ shot, enc, i] + oc_yshift * ky[shot, enc, i])

                shift = np.exp(1j * freq_shift)
                demod = np.exp(1j * demod_freq * 2.0 * PI * time_mri[i])
                kdata[shot, enc, i] *= shift * demod 

    return kdata


#could be faster using broadcasting
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
    print('ecgvals scaled to units of s')

    return data, encode_number


def define_MRI_Raw_structure(kdata, coord, kw, ktime, gating_data, arms, Num_Encodings,
                            Num_Coils, Num_Frames, trajectory, spiral_z_encoding,
                            max_coils=None, max_encodes=None, 
                            compress_coils=-1, scale_kdata=True ):

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

    try:
        logging.info(f'Frames {Num_Frames}')
        logging.info(f'Coils {Num_Coils}')
        logging.info(f'Encodings {Num_Encodings}')
        logging.info(f'Trajectory Type {trajectory_type}')
        logging.info(f'DFT Needed {dft_needed}')

    except Exception:
        logging.info('Missing header data')
        pass

    if max_coils is not None:
            Num_Coils = min(max_coils, Num_Coils)

    if max_encodes is not None:
        Num_Encodings = min(max_encodes, Num_Encodings)

    # Get the MRI Raw structure setup
    mri_raw = MRI_Raw()
    mri_raw.Num_Coils = int(Num_Coils)
    mri_raw.Num_Encodings = int(Num_Encodings)
    mri_raw.Num_Frames = int(Num_Frames)
    mri_raw.dft_needed = tuple(dft_needed)
    mri_raw.trajectory_type = tuple(trajectory_type)

    # List array
    mri_raw.coords = []
    mri_raw.dcf = []
    mri_raw.kdata = []
    mri_raw.time = []
    mri_raw.prep = []
    mri_raw.ecg = []
    mri_raw.resp = []

    # desire convention shape for export is (1,2000, 1864) (arms, points) 
    # time_mri are derived from the data using the TE, BW and num points, is the same across encodes
    # print(time_mri.shape)
    # Reshape the array to have repeats for spiral arms, 
    KTime = np.tile(ktime[:, np.newaxis], (1, arms))
    KTime = np.moveaxis(KTime, -1, 0)
    KTime = KTime[np.newaxis,...]

    print(f'ktime shape {KTime.shape}')

    #trajectories will be imported from SKOPE measures
    #print(coord.shape)
    #print(ksp.shape)
    #print(kw.shape)

    export_coord = np.moveaxis(coord, 0, -3)
    export_coord = export_coord[np.newaxis,...]
    print(export_coord.shape) #(1, 3, 2000, 1868, 2)

    export_kw = np.copy(kw)
    export_kw = export_kw[np.newaxis,...]
    print(export_kw.shape) #(1, 3, 2000, 1868)

    export_ksp = np.moveaxis(kdata, 1, -2)
    export_ksp = export_ksp[:,np.newaxis,...]
    print(export_ksp.shape) # (48, 1, 3, 2000, 1868)

    # # desire convention shape for gating export is (1, arms, encodes) 

    #print(gating_data['ecg'].shape)
    #print(gating_data['resp'].shape)
    #print(gating_data['time'].shape)
    #print(gating_data['prep'].shape)
    #print(gating_data['acq'].shape)

    # raw kspace (6000, 1868, 48), ecg data (6000,)
    gecg = np.copy(gating_data['ecg'])
    gecg = gecg.reshape(arms,Num_Encodings)
    gecg = gecg[np.newaxis,...]

    gresp = np.copy(gating_data['resp'])
    gresp = gresp.reshape(arms,Num_Encodings)
    gresp = gresp[np.newaxis,...]

    gtime = np.copy(gating_data['time'])
    gtime = gtime.reshape(arms,Num_Encodings)
    gtime = gtime[np.newaxis,...]

    gprep = np.copy(gating_data['prep'])
    gprep = gprep.reshape(arms,Num_Encodings)
    gprep = gprep[np.newaxis,...]

    print(gecg.shape) # (1,2000,3)
    #print(resp.shape)
    #print(time.shape)
    #print(prep.shape)

    for encode in range(Num_Encodings*Num_Frames):

        logging.info(f'Loading encode {encode}')
        # expecting coords in z,y,x ordering
        #coord = np.stack([ky,kx], axis=-1).astype(np.float32)
        #coord = np.stack([export_coord[encode, :, :, 1], export_coord[encode, :, :, 0]], axis=-1)
        coord = export_coord[:, encode, ...]

        dcf = export_kw[:, encode,...]

        #instead of skope pipe menon lets just use weights of ones for sense recon
        #dcf = np.ones_like(dcf)

        time_readout = gtime[..., encode]
        ecg_readout = gecg[..., encode]
        prep_readout = gprep[..., encode]
        resp_readout = gresp[..., encode]

        if resp_readout.size != dcf.size:

            # This assigns the same time to each point in the readout
            time_readout = np.expand_dims(time_readout, -1)
            ecg_readout = np.expand_dims(ecg_readout, -1)
            resp_readout = np.expand_dims(resp_readout, -1)
            prep_readout = np.expand_dims(prep_readout, -1)

            time = np.tile(time_readout, (1, 1, dcf.shape[2]))
            resp = np.tile(resp_readout, (1, 1, dcf.shape[2]))
            ecg = np.tile(ecg_readout, (1, 1, dcf.shape[2]))
            prep = np.tile(prep_readout, (1, 1, dcf.shape[2]))

            print(f'Min/max = {np.min(time)} {np.max(time)}')

        ksp = export_ksp[:,:,encode,:,:]

        # Append to list
        mri_raw.coords.append(coord)
        mri_raw.dcf.append(dcf)
        mri_raw.kdata.append(ksp)
        mri_raw.time.append(time)
        mri_raw.prep.append(prep)
        mri_raw.ecg.append(ecg)
        mri_raw.resp.append(resp)

        # Log the data
        logging.info(f'MRI coords {mri_raw.coords[encode].shape}')
        logging.info(f'MRI dcf {mri_raw.dcf[encode].shape}')
        logging.info(f'MRI kdata {mri_raw.kdata[encode].shape}')
        logging.info(f'MRI time {mri_raw.time[encode].shape}')
        logging.info(f'MRI ecg {mri_raw.ecg[encode].shape}')
        logging.info(f'MRI resp {mri_raw.resp[encode].shape}')
        logging.info(f'MRI prep {mri_raw.prep[encode].shape}')

    if scale_kdata:
        # Scale k-space to max 1
        logging.info('Scaling k-space to max 1')
        kdata_max = [np.abs(ksp).max() for ksp in mri_raw.kdata]
        kdata_max = np.max(np.array(kdata_max))
        for ksp in mri_raw.kdata:
            ksp /= kdata_max

    if compress_coils > 0:
        # Compress Coils
        logging.info('Doing PCA coil compression')
        mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=compress_coils)
        mri_raw.Num_Coils = compress_coils

    return mri_raw


def load_Rspiral(fname):
    out = {}

    # Open file in binary read mode
    with open(fname, 'rb') as fid:

        # Read the first data fields using struct to unpack binary data
        out['Rspiral_pts'] = struct.unpack('>i', fid.read(4))[0]  # int32 (big-endian)
        out['a_gxSpiral'] = struct.unpack('>f', fid.read(4))[0]  # float32 (big-endian) # gradient amplitude scaling factors ?
        out['a_gySpiral'] = struct.unpack('>f', fid.read(4))[0]  # float32 (big-endian)

        # Read SpiralX and SpiralY
        spiral_x_int = np.fromfile(fid, dtype='>i', count=out['Rspiral_pts'])
        spiral_y_int = np.fromfile(fid, dtype='>i', count=out['Rspiral_pts'])
        
        out['SpiralX'] = out['a_gxSpiral'] * spiral_x_int.astype(np.float64) / 32768
        out['SpiralY'] = out['a_gySpiral'] * spiral_y_int.astype(np.float64) / 32768

        # Read Theta and Phase Encodes (in PSD these may be populated ~ shots first then encodes)
        out['Rspiral_total_shots'] = struct.unpack('>i', fid.read(4))[0]  # int32 (big-endian)
        out['SpiralTheta'] = np.fromfile(fid, dtype='>f4', count=out['Rspiral_total_shots'])
        out['SpiralZpos'] = np.fromfile(fid, dtype='>f4', count=out['Rspiral_total_shots'])

        # Read Timing Information
        out['Rspiral_time'] = struct.unpack('>i', fid.read(4))[0]  # int (big-endian)
        out['Rspiral_sampling_time'] = struct.unpack('>i', fid.read(4))[0]  # int (big-endian)
        out['Rspiral_start'] = struct.unpack('>i', fid.read(4))[0]  # int (big-endian)
        out['Rspiral_stop'] = struct.unpack('>i', fid.read(4))[0]  # int (big-endian)

    return out


def calc_kspiral_trajectory(gdata,DT,FOV, cal_offset=8.5e-6):
 
    # only supports 2d single echo center out spiral 

    gamma = 4258
    factor = 1000
    dt = 4.0e-6 / factor
    dt_grad = 4.0e-6
    time_offset = 5e-3 # zero padding interpolation

    # DT dwell time from bw (sampling rate)

    samples_per_spiral = int(gdata['Rspiral_sampling_time'] / (1e6*DT))

    total_time = 2 * time_offset + samples_per_spiral * DT
    #print(total_time)

    num_points = int(total_time / dt)
    t_highres = np.linspace(0, total_time, num_points)

    # gradient time vector (low-res)
    grad_time = np.arange(gdata['Rspiral_pts']) * dt_grad + time_offset
    print(grad_time)
    print(t_highres)

    # Interpolate gradients to high-res time vector
    gx_highres = np.interp(t_highres, grad_time, gdata['SpiralX'],left=0, right=0)
    gy_highres = np.interp(t_highres, grad_time, gdata['SpiralY'],left=0, right=0)

    # Integrate gradients to get k-space trajectories (try other integrations)
    #kx_highres = np.cumsum(gx_highres * dt )
    #ky_highres = np.cumsum(gy_highres * dt )

    kx_highres = cumtrapz(gx_highres, dx=dt, initial=0)
    ky_highres = cumtrapz(gy_highres, dx=dt, initial=0)

    # Subsample at readout times 
    #scale_factor = DT / dt
    #readout_indices = (time_offset / dt + np.arange(samples_per_spiral) * scale_factor).astype(int)
    #kx = kx_highres[readout_indices]
    #ky = ky_highres[readout_indices]

    #delay=-2*dt_grad works ok, heuristically selected 8.5e-6
    sampling_t = np.arange(samples_per_spiral) * DT + time_offset - cal_offset

    #print(sampling_t)

    kx = np.interp(sampling_t, t_highres, kx_highres, left=0, right=0)
    ky = np.interp(sampling_t, t_highres, ky_highres, left=0, right=0)

    kx *= gamma * (FOV/10)
    ky *= gamma * (FOV/10)

    # Rotate Kx and Ky using table
    nShots = len(gdata['SpiralTheta'])
    nPoints = len(kx)

    Kx_rotated = np.zeros((nShots, nPoints))
    Ky_rotated = np.zeros((nShots, nPoints))

    for i in range(nShots):
        theta = gdata['SpiralTheta'][i]  # angle in radians

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        Kx_rotated[i] = -kx*cos_t - ky*sin_t
        Ky_rotated[i] = kx*sin_t - ky*cos_t

    # not sure why we have to negate Ky, but then it matches the first shot with SKOPE coordinates. There is likely an unaccounted transpose.
    Kx_rotated *= 1
    Ky_rotated *= -1

    return Kx_rotated, Ky_rotated


def load_ScanArchive(archive_filename_scan, gate_delay, demod=0, skope_path=None, max_coils=None, compress_coils=-1, max_encodes=None, cal_offset=8.5e-6):
    '''
        Data loaded from ScanArchive 
    '''
    logger = logging.getLogger('Loading Scan Archive')

    #archive_filename_scan = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/ScanArchive_608WIMRMR2_20240403_152702561.h5'
    archive = GERecon.Archive(archive_filename_scan)
    archive_data_all_scan, slice_order, view_order = read_scan_achive_data(archive) #(6000,1868,44)

    rawdata = np.stack(archive_data_all_scan, axis=0)
    print(f'rawdata shape = {rawdata.shape} type = {rawdata.dtype}')

    slice_index = np.stack(slice_order, axis=0)
    view_index = np.stack(view_order, axis=0)
    
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

    npe_z = int(header['rdb_hdr_rec']['rdb_hdr_user19'])
    flag2d = int(float(int(header['rdb_hdr_rec']['rdb_hdr_user29']+ 11111100) % 100) / 10.0)
    print(f'2D PC flag {flag2d}')
    
    flag2d = 1
    if flag2d:  
        npe_z = 1
    
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
    # get acquisition ordering
    sicnt = (slice_index * (header['rdb_hdr_rec']['rdb_hdr_da_yres'] - 1) + view_index)

    # find spiral table order 
    sort_act_order = np.argsort(sicnt)  

    # sort kspace to match spiral table order 
    kdata_sorted = rawdata[sort_act_order, ...]  

    # remove calibration data which is stored at the end of the array
    total_samples = total_num_encodes*arms
    kdata = kdata_sorted[:total_samples,...]
    print(f'raw kspace shape {kdata.shape}')

    kdata_points = kdata.shape[-2]
    # col ordering
    kdata = kdata.reshape((arms,total_num_encodes,kdata_points,Num_Coils), order='F')
    print(f'kspace shape for recon {kdata.shape}')
    
    '''
        If available load SKOPE data. Else, load Rspiral152656538.kacq file contains z and theta spiral table information and
        estimate k-space trajectory

        Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 7, which is not enough to represent the header value. 
        GE might need to fix this. In the meantime read files using precision of 6-7 until a fix is available. 

    '''
    if skope_path is not None:
        coord, pkw, b0 = load_skope_data(skope_path)
        print('SKOPE data available to load') 
        print(f' skope coord shape = {coord.shape}') 
        print(f' skope weights (pipe) shape = {pkw.shape}') 
        print(f' skope B0 shape = {b0.shape}') 
        dont_use_pipe = True
        if dont_use_pipe:
            kw = np.ones_like(pkw)
        else:
            kw = pkw

    else:
        # keep the first 7 digits only because of GE error
        kacq_uid = header['rdb_hdr_rec']['rdb_hdr_kacq_uid']
        str_kacq_uid = str(kacq_uid)
        kacq_uid7   = str_kacq_uid[:6]
        #Rspiral152656538.kacq

        # Define the path to the directory containing the files
        rspiral_directory = os.path.dirname(archive_filename_scan)
        rspiral_file_pattern = os.path.join(rspiral_directory, f'Rspiral{kacq_uid7}*.kacq')

        # Find all files that match the pattern
        matching_files = glob.glob(rspiral_file_pattern)
        rspiral_file = matching_files[0] if matching_files else None

        print(f'Loading {rspiral_file} to calculate kspace trajectory')

        output = load_Rspiral(rspiral_file)

        logger.info(f'Calc kspace trajectory with cal offset = {cal_offset*1e6} us')


        Spiral_Kx, Spiral_Ky = calc_kspiral_trajectory(output,dt_mri,fov,cal_offset)
        print(f'Spiral shape {Spiral_Kx.shape}')
        
        min_points = int(min(Spiral_Kx.shape[-1], kdata.shape[-2]))
        print(min_points)

        coord = np.stack([Spiral_Kx, Spiral_Ky],axis=-1)

        print(f'coord shape {coord.shape}')

        kdata = kdata[:,:,:min_points,:]
        coord = coord[:,:min_points,:]
        coord_dim = coord.shape[-1]
        
        #col ordering
        coord = coord.reshape((arms,total_num_encodes,min_points,coord_dim), order='F')

        #DCF of ones for sense recon (will not work for pils). We can use pipe menon method too (e.g. for pils), but is not great for sense
        # probably voronoi method is best compromise
        kw = np.ones((total_num_encodes, arms, min_points))

        print(f'kspace shape for recon {kdata.shape}')
        print(f'coord shape for recon {coord.shape}')

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
    logger.info(f'Off-center correction off-center shift in x = {oc_xshift*fov} mm and y = {oc_yshift*fov} mm and global demodulation {demod} Hz')
    kdata_off_corrected = np.zeros_like(kdata, dtype=kdata.dtype)
    for coil in range(kdata.shape[-1]):
        kdata_off_corrected[:,:,:,coil] = apply_offisocenter_data_demod( coord[:,:,:,0], coord[:,:,:,1], kdata[:,:,:,coil], oc_xshift, oc_yshift, time_mri, demod)

    '''
        Kspace pre-whitening

        Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 7, which is not enough to represent the header value. 
        GE might need to fix this. 
        In the meantime read files using precision of 6-7 until a fix is available. 
    '''

    # keep the first 7 digits only because of GE error
    Coil_UID = header['rdb_hdr_rec']['rdb_hdr_coilConfigUID']
    strCoil_UID = str(Coil_UID)
    Coil_UID7   = strCoil_UID[:6]
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
    logger.info(f'Noise whitening with:{noise_file}')

    try:
        with h5py.File(noise_file, 'r') as hf:
            noise = hf['Data']['NoiseData']['real'] + 1j * hf['Data']['NoiseData']['imag']
            print(noise.shape)
            cov = sp.mri.get_cov(noise)
            ksp = sp.mri.whiten(ksp, cov)

    except Exception:
        ksp /= np.abs(ksp).max()
        logging.info('No noise data.')
        pass
    
    '''
        Load Gating Data 
        Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 6-7, which is not enough to represent the header value. 
        GE might need to fix this. 
        In the meantime read files using precision of 6-7 until a fix is available. 

    '''
    # keep the first 6 digits only because of GE error
    kacq_uid = header['rdb_hdr_rec']['rdb_hdr_kacq_uid']
    str_kacq_uid = str(kacq_uid)
    kacq_uid7   = str_kacq_uid[:6]

    # Define the path to the directory containing the files
    gating_directory = os.path.dirname(archive_filename_scan)
    gating_file_pattern = os.path.join(gating_directory, f'Gating_Track_{kacq_uid7}*.pcvipr_track.full')

    # Find all files that match the pattern
    matching_files = glob.glob(gating_file_pattern)
    gating_file = matching_files[0] if matching_files else None

    print(f'Using {gating_file}')

    logger.info(f'Processing gating file:{gating_file} and applying delay {gate_delay}')

    gating_data, gating_num_enc = gating_processing(gating_file,gate_delay)

    '''
        Calculate DCF for PILS recon for mv paper without skope
    '''
    do_pipe_menon = False

    if skope_path is None and do_pipe_menon:
        try:
            device = sp.Device(0)
        except:
            device = sp.cpu_device

        new_coord = sp.to_device(coord, device)
        logger.info(f'Coord shape for dcg {new_coord.shape}')
        print(f'numer of encodes = {new_coord.shape[1]}')
        new_kw = []
        kw = []
        for enc in range(new_coord.shape[1]):
            kw_enc = sp.mri.pipe_menon_dcf(new_coord[:,enc,:,:],img_shape=(319,319),device=device,max_iter=30,
                                        n=320,
                                        beta=8,
                                        width=4,
                                        show_pbar=True,)
            new_kw.append(kw_enc)
        kw = np.stack(new_kw)
        logger.info('Did pipe-menon for case without skope data to recon with pils (wont work for sense)')

    '''
        Arrange data into MRI_Raw like structure

    '''
    # Initialize some of the variables
    Num_Frames = 1 # Not sure where to get this variable from. Probably comes from kdata thats binned to rcframes ?
    trajectory = 'NONCARTESIAN'
    spiral_z_encoding = 'CARTESIAN_PHASE_ENCODED'

    mri_raw = define_MRI_Raw_structure(ksp, coord, kw, time_mri, gating_data, arms, total_num_encodes,
                            Num_Coils, Num_Frames, trajectory, spiral_z_encoding,
                            max_coils=None, max_encodes=max_encodes, 
                            compress_coils=compress_coils, scale_kdata=True )
    
    return mri_raw


# %%
if __name__ == '__main__':

    # %%
    # Try to recon data with MRI structure data (pts, arms)

    '''
        Data loaded from ScanArchive 
    '''
    archive_filename_scan = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/VOL01_DV/01711_00006_Spiral_Dual_Venc_8-75/raw_data/ScanArchive_608WIMRMR2_20240403_152702561.h5'
    archive = GERecon.Archive(archive_filename_scan)
    archive_data_all_scan, slice_order, view_order = read_scan_achive_data(archive) #(6000,1868,44)

    # %%
    rawdata = np.stack(archive_data_all_scan, axis=0)
    print(f'rawdata shape = {rawdata.shape} type = {rawdata.dtype}')

    slice_index = np.stack(slice_order, axis=0)
    print(f'rawdata shape = {slice_index.shape} type = {slice_index.dtype}')
    print(slice_index)

    view_index = np.stack(view_order, axis=0)
    print(f'rawdata shape = {view_index.shape} type = {view_index.dtype}')
    print(view_index)

    np.savetxt("slice_index.csv", slice_index, delimiter=",", fmt='%d')
    np.savetxt("view_index.csv", view_index, delimiter=",", fmt='%d')

    # %%
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
    bw = float(header['rdb_hdr_image']['vbw'])
    dt_mri = 1./(2*bw*1000)

    arms = int(header['rdb_hdr_rec']['rdb_hdr_user10']) # number of spiral arms 
    #rhuser30 = convert_float_to_uint(header['rdb_hdr_rec']['rdb_hdr_user30'])
    #total_num_encodes = int((float(rhuser30 % 100) / 1.0))
    Num_Coils = rawdata.shape[-1]
    print(Num_Coils)
    rba_extra_data = int(header['rdb_hdr_rec']['rdb_hdr_user11'])
    yres = int(header['rdb_hdr_rec']['rdb_hdr_da_yres'])
    print(yres)

    sicnt = (slice_index * (header['rdb_hdr_rec']['rdb_hdr_da_yres'] - 1) + view_index)
    np.savetxt("new_view_index.csv", sicnt, delimiter=",", fmt='%d')


    #%% Now sort kdata to match spiral table
    '''
        Prepare kspace for processing (NOT DOING CALIBRATION) 
    '''
    total_num_encodes = 3 
    total_samples = total_num_encodes*arms
    print(total_samples)

    # find spiral table order 
    sort_act_order = np.argsort(sicnt)
    print(sort_act_order)

    # Sort kspace to match spiral table order 
    kdata_sorted = rawdata[sort_act_order, ...]

    print(f'raw kspace shape {kdata_sorted.shape}')
    #remove calibration data which is stored at the end of the array
    kdata = kdata_sorted[:total_samples,...]
    print(f'raw kspace shape {kdata.shape}')


    # %%

    '''
        Load Rspiral152656538.kacq file contains z and theta spiral table information

        Assuming ScanArchive data was stored sequentially 

         Current limitation:Orchestra Python SDK seems to be read int header values as float, 
        leading to a precision of 7, which is not enough to represent the header value. 
        GE might need to fix this. 
        In the meantime read files using precision of 6-7 until a fix is available. 

    '''
    # keep the first 7 digits only because of GE error
    kacq_uid = header['rdb_hdr_rec']['rdb_hdr_kacq_uid']

    str_kacq_uid = str(kacq_uid)
    kacq_uid7   = str_kacq_uid[:6]
    print(kacq_uid)
    print(kacq_uid7)
    #Rspiral152656538.kacq

    print(f'kacq_uid :{kacq_uid} str_kacq_uid {str_kacq_uid} and kacq_uid7 {kacq_uid7}')

    # Define the path to the directory containing the files
    rspiral_directory = os.path.dirname(archive_filename_scan)
    rspiral_file_pattern = os.path.join(rspiral_directory, f'Rspiral{kacq_uid7}*.kacq')

    # Find all files that match the pattern
    matching_files = glob.glob(rspiral_file_pattern)
    rspiral_file = matching_files[0] if matching_files else None

    print(f'Using {rspiral_file}')

    #scaling factor for gradients (only needed when doing calibration)
    #gamma = 4258
    #area_to_kspace = gamma / cal_ampmod * (10*fov) #units of 1/cm

    output = load_Rspiral(rspiral_file)

    xres = int(header['rdb_hdr_rec']['rdb_hdr_da_xres'])
    fov = float(header['rdb_hdr_image']['dfov']) # mm

    bw = float(header['rdb_hdr_image']['vbw'])
    dt_mri = 1./(2*bw*1000)
    print(f'dt sampling = {dt_mri}')

    delay = 8.5e-6
    Spiral_Kx, Spiral_Ky = calc_kspiral_trajectory(output,dt_mri,fov,delay)

    print(Spiral_Kx.shape)
    print(output['Rspiral_sampling_time'])
    print(Spiral_Kx[0,:])


    # %%
    # Number of shots and points per shot
    nShots, nPoints = Spiral_Kx.shape

    plt.figure(figsize=(6, 6))

    nShots = 1
    # Plot each shot
    for i in range(nShots):
        plt.plot(Spiral_Kx[i], Spiral_Ky[i], lw=0.8)
        plt.plot(Spiral_Kx[i+2000], Spiral_Ky[i+2000], lw=0.8)
        plt.plot(Spiral_Kx[i+4000], Spiral_Ky[i+4000], lw=0.8)


    plt.xlabel('Kx')
    plt.ylabel('Ky')
    plt.title('Spiral K-space Trajectories (Rotated)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # %% Now prepare for recon
    print(Spiral_Kx.shape)
    print(kdata.shape)
    min_points = min(Spiral_Kx.shape[-1], kdata.shape[-2])
    print(min_points)

    TE = float(header['rdb_hdr_image']['te']) #us
    time_mri = np.arange(min_points)*dt_mri + TE*1e-6
    new_coord = np.stack([Spiral_Kx, Spiral_Ky],axis=-1)

    print(new_coord.shape)

    new_ksp = np.copy(kdata[:,:min_points,:])
    new_coord = new_coord[:,:min_points,:]

    print(new_ksp.shape)

    # this is reshape from spiral table to have encode coils
    new_ksp = new_ksp.reshape((arms,total_num_encodes,min_points,Num_Coils), order='F')
    new_coord = new_coord.reshape((arms,total_num_encodes,min_points,2), order='F')

    print(f'kspace shape for recon {new_ksp.shape}')
    print(f'kspace shape for recon {new_coord.shape}')
    
    #%%
    plt.figure(figsize=(6, 6))

    nShots = 3
    # Plot each shot
    for i in range(nShots):
        # Original coordinate trajectory (2D spiral) (2000, 3, 1868, 2)
        plt.plot(new_coord[i, 0, :, 0], new_coord[i, 0, :, 1], lw=0.8, label=f'Shot {i+1} Raw' if i==0 else "")

        # Interpolated / Rotated Spiral trajectory
        plt.plot(Spiral_Kx[i+0000], Spiral_Ky[i+0000], lw=0.8, linestyle='--', label=f'Shot {i+1} Rotated' if i==0 else "")


    plt.xlabel('Kx')
    plt.ylabel('Ky')
    plt.title('Spiral K-space Trajectories (Rotated)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


    # %% run recon
    res = [3,320,320]
    sos_combined = np.zeros(res, dtype=np.float32) 

    try:
        device = sp.Device(0)
    except:
        device = sp.cpu_device

    new_coord = sp.to_device(new_coord, device)
    print(f'numer of encodes = {new_coord.shape[1]}')
    kw = []
    for enc in range(new_coord.shape[1]):
        kw_enc = sp.mri.pipe_menon_dcf(new_coord[:,enc,:,:],img_shape=(320,320),device=device,max_iter=30,
                                    n=320,
                                    beta=8,
                                    width=4,
                                    show_pbar=True,)
        kw.append(kw_enc)
    new_kw = np.stack(kw)

    print(f' kw shape is {new_kw.shape}') #(2000,3,1868)
    print(new_kw.dtype)
    new_kw = new_kw.astype(np.float32)
    new_kw = sp.to_device(new_kw, device)

    new_ksp = sp.to_device(new_ksp, device)
    time_mri = sp.to_device(time_mri, device)

    #demod = -250
    for enc in range(new_ksp.shape[1]):
        images = []
        for coil in range(new_ksp.shape[-1]):
            #print(coil)
            kdata_temp = new_ksp[:,enc,:,coil]
            xp = sp.get_device(kdata_temp).xp
            #for k in range(min_points):
            #        kdata_temp[:,k] *= xp.exp(1j*demod*2*np.pi*time_mri[k])

            image = sp.nufft_adjoint(kdata_temp[:,:]*new_kw[enc,:,:], new_coord[:,enc,:,:], oshape=[320, 320])
            images.append( sp.to_device(image))

        images = np.stack(images,0)
        sos = np.sqrt(np.sum(np.abs(images)**2, axis=0))     


        sos_combined[enc,...] = sos 

    directory = '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/'
    recon_name = f'{directory}/vol_spiral_trajectory_from_header.h5'

    with h5py.File(recon_name, 'w') as hf:
        hf.create_dataset("sos", data=np.abs(sos_combined))

    # image quality still not as expected. Maybe issues with samplin times interpolation of trajectories, 
    # or other kspace processing we should do before recon (e.g. nomalization?)


    # %% 
    skope_path= '/mounts/data/analyses/larivera/projects/multivenc/VOLDATA/SCAN_ARCH/skope_data'
    coord, kw, b0 = load_skope_data(skope_path)
    print(coord.shape) #(2000, 3, 1868, 2)
    print(coord[0,0,:,0])

    plt.plot(coord[0,0,:1865,0], label='SKOPE highres')
    plt.legend()
    plt.show()

    print(coord[0,0,:1864,0] - Spiral_Kx[0])


    #%%
    plt.figure(figsize=(6, 6))

    nShots = 1
    # Plot each shot
    for i in range(nShots):
        plt.plot(coord[i,0,:,0], coord[i,0,:,1], lw=0.8)
        #plt.plot(coord[i,1,:,0], coord[i,1,:,1], lw=0.8)
        #plt.plot(coord[i,2,:,0], coord[i,2,:,1], lw=0.8)

        plt.plot(Spiral_Kx[i], Spiral_Ky[i], lw=0.8)
        #plt.plot(Spiral_Kx[i+2000], Spiral_Ky[i+2000], lw=0.8)
        #plt.plot(Spiral_Kx[i+4000], Spiral_Ky[i+4000], lw=0.8)

    plt.xlabel('Kx')
    plt.ylabel('Ky')
    plt.title('Spiral K-space Trajectories (Rotated)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    print(coord[0,0,:15,0])
    print(Spiral_Kx[0,:15])
    print(Spiral_Ky[0,:15])

    print(Spiral_Kx.shape)


    #%%
    plt.figure(figsize=(6, 6))

    nShots = 3
    # Plot each shot
    for i in range(nShots):
        # Original coordinate trajectory (2D spiral) (2000, 3, 1868, 2)
        plt.plot(coord[i, 2, :, 0], coord[i, 2, :, 1], lw=0.8, label=f'Shot {i+1} Raw' if i==0 else "")

        # Interpolated / Rotated Spiral trajectory
        plt.plot(Spiral_Kx[i+4000], Spiral_Ky[i+4000], lw=0.8, linestyle='--', label=f'Shot {i+1} Rotated' if i==0 else "")


    plt.xlabel('Kx')
    plt.ylabel('Ky')
    plt.title('Spiral K-space Trajectories (Rotated)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

