#! /usr/bin/env python ALMA + BEN CHANGES
import numpy as np
from numpy import sum, sqrt, mean, abs
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import os
import argparse
import matplotlib.pyplot as plt
import cupy
import math
import time
import re


# Laplacian based phase unwrapping
def unwrap_4d(phase_w):

    logger = logging.getLogger('Laplacian Unwrap')

    ts = 2.0  # scales temporal data to spatial dimentions
    # real_flag = 1 # restrict laplacians to real (lowers memory load)
    phase_w = np.moveaxis(phase_w, 0, -1)  # x y z t

    ndim = phase_w.shape

    # create grid
    X, Y, Z, T = np.mgrid[-ndim[0] // 2:ndim[0] // 2:,
                 -ndim[1] // 2:ndim[1] // 2:,
                 -ndim[2] // 2:ndim[2] // 2:,
                 -ndim[3] // 2:ndim[3] // 2:]

    # get mod
    mod = 2.0 * np.cos(np.pi * X / ndim[0]) + 2.0 * np.cos(np.pi * Y / ndim[1]) + 2.0 * np.cos(
        np.pi * Z / ndim[2]) + ts * np.cos(np.pi * T / ndim[3]) - 6.0 - ts

    X = None
    Y = None
    Z = None
    T = None

    logger.info('Laplacian')
    print('Forward')
    lap_phase_w = lap4(phase_w, 1, mod)
    lap_phase = np.cos(phase_w) * lap4(np.sin(phase_w), 1, mod) - np.sin(phase_w) * lap4(np.cos(phase_w), 1, mod)

    logger.info('Inverse Laplacian')
    print('Backwards')
    ilap_phasediff = lap4(lap_phase - lap_phase_w, -1, mod)
    n_u4 = np.int8(np.real(np.ndarray.round(ilap_phasediff / 2 / np.pi)))

    phase_w = np.moveaxis(phase_w, -1, 0)  # t x y z
    n_u4 = np.moveaxis(n_u4, -1, 0)

    return n_u4


def lap4(phase_w, direction, mod):
    ndim = phase_w.shape
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phase_w)))

    if direction == 1:
        K *= mod

    elif direction == -1:
        mod[ndim[0] // 2, ndim[1] // 2, ndim[2] // 2, ndim[3] // 2] = 1
        K /= mod

    else:
        print("ERROR")

    out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out


class MRI_4DFlow:

    def __init__(self, encode_type,venc, unwrap_lap=False):

        'Initialization'
        self.set_encoding_matrix(encode_type)
        self.Venc = venc  #m/s
        self.NoiseLevel = 0.0 #relative to max signal of 1
        self.spatial_resolution = 0.5 # percent of kmax
        self.time_resolution = 0.5 # percent of nominal
        self.background_magnitude = 0.5 #value of background
        self.unwrap_lap = unwrap_lap
        
        # Matrices
        self.signal = None
        self.velocity_estimate = None
        self.angiogram = None
        self.magnitude = None
        

    def set_encoding_matrix(self, encode_type='4pt-referenced'):
        encode_dictionary = {
            '4pt-referenced' : np.pi/2.0*np.array([[-1.0, -1.0, -1.0],
                               [ 1.0, -1.0, -1.0],
                               [-1.0,  1.0, -1.0],
                               [-1.0, -1.0,  1.0]],dtype=np.float32),
            '3pt': np.pi / 2.0 * np.array([[-1.0, -1.0, -1.0],
                                                      [1.0, -1.0, -1.0],
                                                      [-1.0, 1.0, -1.0]], dtype=np.float32),
            '4pt-balanced': np.pi / 2.0/ sqrt(2.0) * np.array([[-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32),
            '5pt': np.pi / sqrt(3.0) * np.array([ [0.0, 0.0, 0.0],
                                                      [-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32)
        }
        self.EncodingMatrix = encode_dictionary[encode_type]
        self.DecodingMatrix = np.linalg.pinv(self.EncodingMatrix)

    """
    :param velocity: a Nt x Nz x Ny x Nx x 3 description of the velocity field
    :param pd: a Nt x Nz x Ny x Nx mask of the vessel locations
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def generate_complex_signal(self,velocity,pd):

        # Get last dimension to (3 x 1)
        velocity = np.expand_dims( velocity,-1)

        # Multiple to get phase
        print(self.EncodingMatrix.shape)
        print(velocity.shape)

        # Get the Phase
        phase = np.matmul( self.EncodingMatrix/self.Venc, velocity)

        # Create Magnitude image (M*exp(i*phase))
        mag = np.copy(pd)
        mag += self.background_magnitude
        mag = np.expand_dims(mag, -1)
        mag = np.expand_dims(mag, -1)
        self.signal = mag*np.exp(1j * phase )

    def solve_for_velocity(self):
        t0 = time.perf_counter()
        print(f"[solve_for_velocity] start: signal shape={self.signal.shape}, dtype={self.signal.dtype}")

        # Multiply by reference
        ref = self.signal[...,0]
        ref = np.expand_dims(ref, -1)
        signal2 = self.signal * np.conj(ref)

        # Convert to .. x Nencodes x 1
        signal2 = np.expand_dims( signal2,-1)

        # Get subtracted decoding matrix
        diffMatrix = self.EncodingMatrix
        diffMatrix -= diffMatrix[0,:]
        self.DecodingMatrix = np.linalg.pinv(diffMatrix)

        # Take angle
        phase = np.angle(signal2)

        # Unwrap phase for all encodes
        num_enc = phase.shape[4]

        if self.unwrap_lap:
            if phase.shape[0] > 1:
                print(f'number of encodes to unwrap {num_enc}')
                # Start loop in second encode (first was use to reference)
                phase_wrap = []
                phase = np.squeeze(phase)
                print('Starting Laplacian based phase unwrapping')
                for i in range(num_enc - 1):
                    print(f'Copy encode {i}')
                    phase_wrap = np.copy(phase[:, :, :, :, i + 1])

                    # Find phase wraps
                    print(f'Unwrap the encode {i}')
                    n_jumps = unwrap_4d(phase_wrap)

                    # Unwrap phase
                    print(f'Apply unwrap {i}')
                    phase[:, :, :, :, i + 1] = phase[:, :, :, :, i + 1] + 2 * np.pi * n_jumps

                phase = np.expand_dims(phase, -1)
                print('Laplacian based phase unwrapping finished')

        #Solve for velocity
        self.velocity_estimate = np.matmul(self.DecodingMatrix*self.Venc,phase)

        # Data comes back as Nt x Nz X Ny x Nz x 3 x 1, reduce to
        #   Nt x Nz x Ny x Nx x 3
        self.velocity_estimate = np.squeeze( self.velocity_estimate, axis=-1)
        print(f"[solve_for_velocity] done in {time.perf_counter() - t0:.2f}s, velocity shape={self.velocity_estimate.shape}, dtype={self.velocity_estimate.dtype}")

    def background_phase_correct(self, mag_thresh=0.08, angiogram_thresh=0.3, fit_order=3):

        # Average time frames
        magnitude_avg = mean(self.magnitude, axis=0)
        angiogram_avg = mean(self.angiogram, axis=0)

        # Threshold
        max_mag = np.max( magnitude_avg)
        max_angiogram = np.max( angiogram_avg)

        # Get the number of coeficients
        pz,py,px = np.meshgrid(range(fit_order+1),range(fit_order+1),range(fit_order+1))
        idx = np.where( (px+py+pz) <= fit_order )
        px = px[idx]
        py = py[idx]
        pz = pz[idx]
        N = len(px)

        #print('Polynomial fitting with %d variables' % (N,))
        AhA = np.zeros((N, N), dtype=np.float32)
        AhBx= np.zeros((N, 1), dtype=np.float32)
        AhBy= np.zeros((N, 1), dtype=np.float32)
        AhBz= np.zeros((N, 1), dtype=np.float32)

        # Now gather terms (Nt x Nz x Ny x Nx x 3 )
        z, y, x = np.meshgrid(np.linspace(-1, 1, self.velocity_estimate.shape[1]),
                              np.linspace(-1, 1, self.velocity_estimate.shape[2]),
                              np.linspace(-1, 1, self.velocity_estimate.shape[3]),
                              indexing='ij')

        # Grab array
        vavg = np.squeeze( mean( self.velocity_estimate, axis=0))
        vx = vavg[:, :, :, 0]
        vy = vavg[:, :, :, 1]
        vz = vavg[:, :, :, 2]

        temp = ( (magnitude_avg > (mag_thresh * max_mag)) &
                 (angiogram_avg < (angiogram_thresh * max_angiogram)) )
        mask = np.zeros(temp.shape, temp.dtype)
        ss = 2 #subsample
        mask[::ss,::ss,::ss] = temp[::ss,::ss,::ss]

        # Subselect values
        idx =np.argwhere(mask)
        x_slice = x[idx[:,0],idx[:,1],idx[:,2]]
        y_slice = y[idx[:,0],idx[:,1],idx[:,2]]
        z_slice = z[idx[:,0],idx[:,1],idx[:,2]]
        vx_slice = vx[idx[:,0],idx[:,1],idx[:,2]]
        vy_slice = vy[idx[:,0],idx[:,1],idx[:,2]]
        vz_slice = vz[idx[:,0],idx[:,1],idx[:,2]]

        for ii in range(N):
            for jj in range(N):
                AhA[ii, jj] = sum( (x_slice ** px[ii] * y_slice ** py[ii] * z_slice ** pz[ii]) *
                                     (x_slice ** px[jj] * y_slice ** py[jj] * z_slice ** pz[jj]) )

        for ii in range(N):
            phi = np.power(x_slice, px[ii]) * np.power(y_slice, py[ii]) * np.power( z_slice, pz[ii])
            AhBx[ii] = sum(vx_slice * phi)
            AhBy[ii] = sum(vy_slice * phi)
            AhBz[ii] = sum(vz_slice * phi)

        polyfit_x = np.linalg.solve(AhA, AhBx)
        polyfit_y = np.linalg.solve(AhA, AhBy)
        polyfit_z = np.linalg.solve(AhA, AhBz)

        # Now Subtract
        background_phase = np.zeros(vx.shape + (3,), vx.dtype)

        #print("Subtract")
        for ii in range(N):
            phi = (x**px[ii])
            phi*= (y**py[ii])
            phi*= (z**pz[ii])
            background_phase[:,:,:,0] += polyfit_x[ii]*phi
            background_phase[:,:,:,1] += polyfit_y[ii]*phi
            background_phase[:,:,:,2] += polyfit_z[ii]*phi

        #Expand and subtract
        background_phase = np.expand_dims( background_phase,0)
        self.velocity_estimate -= background_phase

    """
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def update_magnitude(self):
        self.magnitude = sqrt( sum( abs(self.signal)**2 , -1))

    """
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def update_angiogram(self):

        # Recalc Magnitude
        self.update_magnitude()

        # Ensure velocity is available
        if self.velocity_estimate is None:
            self.solve_for_velocity()

        # New velocity calculation
        vmag = sqrt( sum( self.velocity_estimate ** 2, axis=-1))
        vmag_scaled = np.minimum((vmag / float(self.Venc)) * 2.0, 1.0)
        
        self.angiogram = (self.magnitude * np.sin((math.pi / 2.0) * vmag_scaled)).astype(np.float32)

        
    def thresh_angiogram(self, mag_thresh=0.08, cd_thresh=0.3):
        if self.magnitude is None or self.angiogram is None:
            raise ValueError("Call solve_for_velocity() and update_angiogram() first")
        
         # Time Averaged Magnitude and Angiogram: (Nz, Ny, Nx)
        mag_avg = mean(self.magnitude, axis=0)
        angio_avg = mean(self.angiogram, axis=0)

        mag_max = np.max(mag_avg)
        angio_max = np.max(angio_avg)

        abs_mag_thresh = mag_thresh * mag_max
        abs_cd_thresh = cd_thresh * angio_max
        
        mask = (angio_avg < abs_cd_thresh) & (mag_avg > abs_mag_thresh)

        self.angiogram_mask = mask.astype(np.float32)
        
    def _fit_background_poly_from_mask(self, fit_order=3):
        
        if self.velocity_estimate is None:
            raise ValueError("velocity_estimate is None - run solve_for_velocity() first")
        if getattr(self, "angiogram_mask", None) is None:
            raise ValueError("angiogram_mask is None - run thresh_angiogram() first")
        
        # Time Averaged Velocity: (Nz, Ny, Nx, 3)
        vavg = mean(self.velocity_estimate, axis=0)
        vx = vavg[..., 0]
        vy = vavg[..., 1]
        vz = vavg[..., 2]
        
        Nz, Ny, Nx = vx.shape
        
        # Normalized coordinates in [-1, 1]
        z, y, x = np.meshgrid(
            np.linspace(-1, 1, Nz),
            np.linspace(-1, 1, Ny),
            np.linspace(-1, 1, Nx),
            indexing="ij"
        )
        
        mask = self.angiogram_mask.astype(bool)
        idx = np.argwhere(mask)
        if idx.size == 0:
            raise RuntimeError("Background mask is empty - check thresholds")
        
        x_slice = x[mask]
        y_slice = y[mask]
        z_slice = z[mask]
        vx_slice = vx[mask]
        vy_slice = vy[mask]
        vz_slice = vz[mask]
        
        # Polynomial basis indices
        pz, py, px = np.meshgrid(
            range(fit_order + 1),
            range(fit_order + 1),
            range(fit_order + 1),
        )
        idx_poly = np.where((px+py+pz) <= fit_order)
        px = px[idx_poly]
        py = py[idx_poly]
        pz = pz[idx_poly]
        N = len(px)
        
        AhA = np.zeros((N, N), dtype=np.float64)
        AhBx = np.zeros((N, 1), dtype=np.float64)
        AhBy = np.zeros((N, 1), dtype=np.float64)
        AhBz = np.zeros((N, 1), dtype=np.float64)
        
        # Build normal eqns
        for ii in range(N):
            phi_i = (x_slice ** px[ii]) * (y_slice ** py[ii]) * (z_slice ** pz[ii])
            for jj in range(N):
                phi_j = (x_slice ** px[jj]) * (y_slice ** py[jj]) * (z_slice ** pz[jj])
                AhA[ii, jj] = sum(phi_i * phi_j)
                
            AhBx[ii] = sum(vx_slice * phi_i)
            AhBy[ii] = sum(vy_slice * phi_i)
            AhBz[ii] = sum(vz_slice * phi_i)
            
        # Solve for polynomial coefficients
        coef_x = np.linalg.solve(AhA, AhBx)
        coef_y = np.linalg.solve(AhA, AhBy)
        coef_z = np.linalg.solve(AhA, AhBz)
        
        # Evaluate polynomial on full grid to get background velocity (Nz, Ny, Nx, 3)
        background_vel = np.zeros(vavg.shape, dtype=np.float32)
        
        for ii in range(N):
            phi = (x ** px[ii]) * (y ** py[ii]) * (z ** pz[ii])
            background_vel[..., 0] += (coef_x[ii] * phi).astype(np.float32)
            background_vel[..., 1] += (coef_x[ii] * phi).astype(np.float32)
            background_vel[..., 2] += (coef_x[ii] * phi).astype(np.float32)
            
        return background_vel
    
    def background_phase_fit_iterative(self,
                                fit_number=2,
                                fit_order=3,
                                mag_thresh=0.08,
                                cd_thresh_first=1.0,
                                cd_thresh_other=0.3):
        
        if self.signal is None:
            raise ValueError("self.signal is None - set self.signal before background_phase_fit_iteratvive()")
        
        background_vel = None
        
        for iter in range(fit_number):
            print(f"[background_phase_fit_iterative] Iteration {iter+1}/{fit_number}")
            
            # i) Velocity from current signal
            self.solve_for_velocity()
            
            # ii) Angiogram from current velocity + magnitude
            self.update_angiogram()
            
            # iii) Threshold angiogram to get background mask
            if iter == 0:
                self.thresh_angiogram(mag_thresh=mag_thresh, cd_thresh=cd_thresh_first)
            else:
                self.thresh_angiogram(mag_thresh=mag_thresh, cd_thresh=cd_thresh_other)
                
            # iv) Fit background polynomial velocity field
            background_vel = self._fit_background_poly_from_mask(fit_order=fit_order)
            
        # --- After iterations: apply background correction in phase space ---
        
        if background_vel is None:
            raise RuntimeError("background_vel is None - fitting loop did not run")
        
        # Background velocity (Nz, Ny, Nx, 3)
        Nz, Ny, Nx, _ = background_vel.shape
        
        # Encoding matrix scald by 1/Venc, similar to CPP pcvipr_recon
        ES = self.EncodingMatrix.astype(np.float32) / np.float32(self.Venc)
        # (Nz, Ny, Nx, Nenc)
        poff = np.tensordot(background_vel.astype(np.float32, copy=False), ES.T, axes=([3], [0]))
        poff = poff.astype(np.float32, copy=False)
        
        # Broadcast over time dimension (Nt)
        Nt = self.signal.shape[0]
        phase_full = poff[np.newaxis, ...] # (1, Nz, Ny, Nx, Nenc)
        if Nt > 1:
            phase_full = np.broadcast_to(phase_full, (Nt, Nz, Ny, Nx, ES.shape[0]))
        
        # Apply complex background phase correction
        back_phase = np.exp((-1j * phase_full).astype(np.complex64))
        self.signal = self.signal.astype(np.complex64, copy=False)
        self.signal *= back_phase
        
        print("[background_phase_fit_iterative] Applied background phase correction in signal domain.")
            
            
        # Recompute final velocity and angiogram from corrected signal
        print("[background_phase_fit_iterative] Recomputing final velocity...")
        self.solve_for_velocity()
        print("[background_phase_fit_iterative] Recomputing final angiogram...")
        self.update_angiogram()
        
def export_flow_data(mri_flow, out_name, header_info=None, c_format=False):
    
    # Export to file
    try:
        os.remove(out_name)
    except OSError:
        pass
    
    # Convert to int and scale
    mg = mri_flow.magnitude * 32767/np.max(mri_flow.magnitude)
    mg = mg.astype(np.int16)
    cd = mri_flow.angiogram * 32767/np.max(mri_flow.angiogram)
    cd = cd.astype(np.int16)
    vx = (mri_flow.velocity_estimate[..., 0] * 10).astype(np.int16)
    vy = (mri_flow.velocity_estimate[..., 1] * 10).astype(np.int16)
    vz = (mri_flow.velocity_estimate[..., 2] * 10).astype(np.int16)
    
    # if TA add time dimension
    if len(mri_flow.magnitude.shape) < 4:
        mg = np.expand_dims(mg, axis=0)
        cd = np.expand_dims(cd, axis=0)
        vx = np.expand_dims(vx, axis=0)
        vy = np.expand_dims(vy, axis=0)
        vz = np.expand_dims(vz, axis=0)
        
    print(f"Exporting flow data to {out_name}")
    with h5py.File(out_name, 'w') as hf:
        header_group = hf.create_group("Header")
        data_group = hf.create_group("Data")
        if header_info is not None:
            for attr in header_info.keys():
                header_group.attrs[attr] = header_info[attr]
        else:
            header_group.attrs["venc"] = mri_flow.Venc
            header_group.attrs["frames"] = mg.shape[0]
            header_group.attrs["matrixx"] = mg.shape[1]
            header_group.attrs["matrixy"] = mg.shape[2]
            header_group.attrs["matrixz"] = mg.shape[3]
            
        if c_format:
            frames = mg.shape[0]
            data_group.create_dataset("MAG", data=np.rint(np.squeeze(np.mean(mg, axis=0))).astype(np.int16))
            data_group.create_dataset("CD", data=np.rint(np.squeeze(np.mean(cd, axis=0))).astype(np.int16))
            data_group.create_dataset("comp_vd_1", data=np.rint(np.squeeze(np.mean(vx, axis=0))).astype(np.int16))
            data_group.create_dataset("comp_vd_2", data=np.rint(np.squeeze(np.mean(vy, axis=0))).astype(np.int16))
            data_group.create_dataset("comp_vd_3", data=np.rint(np.squeeze(np.mean(vz, axis=0))).astype(np.int16))

            if frames > 1:
                for i in range(frames):
                    data_group.create_dataset(f"ph_{i:03}_mag", data=np.squeeze(mg[i, ...]))
                    data_group.create_dataset(f"ph_{i:03}_cd", data=np.squeeze(cd[i, ...]))
                    data_group.create_dataset(f"ph_{i:03}_vd_1", data=np.squeeze(vx[i, ...]))
                    data_group.create_dataset(f"ph_{i:03}_vd_2", data=np.squeeze(vy[i, ...]))
                    data_group.create_dataset(f"ph_{i:03}_vd_3", data=np.squeeze(vz[i, ...]))
        else:
            hf.create_dataset("MAG", data=mg)
            hf.create_dataset("CD", data=cd)
            hf.create_dataset("VX", data=vx)
            hf.create_dataset("VY", data=vy)
            hf.create_dataset("VZ", data=vz)


if __name__ == "__main__":


    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--venc', type=float, default=80.0)
    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. FullRecon.h5)', default=None)
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='Flow.h5')
    parser.add_argument('--c_format', dest='c_format', action='store_true', default=True, help='export Flow.h5 file in CPP recon format')
    

    args = parser.parse_args()

    # Put up a file selector if the file is not specified
    if args.filename is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()
        args.filename = askopenfilename()

    if args.out_folder is None:
        out_folder = os.path.dirname(args.filename)
    else:
        out_folder = args.out_folder

    print(f'Loading {args.filename}')
    t_load = time.perf_counter()

    with h5py.File(args.filename, 'r') as hf:
        #temp = np.array(hf['Images'])
        #temp = temp['real'] + 1j*temp['imag']
        #temp = np.moveaxis(temp, -1, 0)
        #frames = int(temp.shape[0]/4)
        #temp = np.reshape(temp, newshape=(frames,4, temp.shape[1], temp.shape[2], temp.shape[3]))
        #temp = np.reshape(temp, newshape=(10,4, temp.shape[-3], temp.shape[-2], temp.shape[-1]))
        #temp = np.squeeze(temp)
        
        images = hf['Images']
        
        items = []
        
        for key in images.keys():
            match = re.search(r"Encode_(\d+)_Frame_(\d+)", key)
            
            if match is None:
                continue
            
            encode = int(match.group(1))
            frame = int(match.group(2))
            
            items.append((frame, encode, key))
            
        frames = len(set(frame for frame, encode, key in items))
        num_encodes = len(set(encode for frame, encode, key in items))
        
        print(f"num of frames = {frames}")
        print(f"num of encodes = {num_encodes}")
        
        items = sorted(items)
        
        data = []
        
        for frame, encode, key in items:
            arr = np.array(images[key])
            arr = arr["real"] + 1j * arr["imag"]
            data.append(arr)
        
        temp = np.stack(data)
        temp = temp.reshape(frames, num_encodes, *temp.shape[1:])
        print(f"The shape is : {temp.shape}")

        if len(temp.shape) == 4:
            temp = np.expand_dims(temp,axis=0)

        print(f' num frames =  {frames}')
        print(f' num encodes = {num_encodes}')
        #temp = np.reshape(temp,newshape=(5, frames,temp.shape[1],temp.shape[2],temp.shape[3]))
        #temp = np.reshape(temp,newshape=(temp.shape[1], frames,temp.shape[2],temp.shape[3],temp.shape[4]))

        temp = np.moveaxis(temp,1,-1)
        print(f"Rearranged shape is {temp.shape}")

    if num_encodes == 5:
        encoding = "5pt"
    elif num_encodes == 4:
        encoding = "4pt-referenced"
    elif num_encodes == 3:
        encoding = "3pt"

    print(f' encoding type is {encoding}')

    # Solve for Velocity
    mri_flow = MRI_4DFlow(encode_type= encoding, venc=args.venc)
    mri_flow.signal = temp
    # mri_flow.solve_for_velocity()             # Now in background_phase_fit_iterative
    # mri_flow.update_angiogram()               # Now in background_phase_fit_iterative
    #mri_flow.background_phase_correct()        # Now in background_phase_fit_iterative
    #mri_flow.update_angiogram()                # Now in background_phase_fit_iterative
    
    mri_flow.background_phase_fit_iterative(
        fit_number=2,
        fit_order=3,
        mag_thresh=0.08,
        cd_thresh_first=1.0,
        cd_thresh_other=0.3
    )

    # Export to file
    print(f"Exporting flow data to {args.out_filename}")
    out_name = os.path.join(out_folder, args.out_filename)
    
    print(f"Exporting flow data to out name = {out_name}")
    
    export_flow_data(mri_flow, out_name, c_format=args.c_format)
    
    # Old exporting, useful, but export_flow_data is better as its CPP format
    # if os.path.exists(out_name):
    #     os.remove(out_name)
    
    # # Outputs for TA datasets    
    # outputs = {
    #     "CD": mri_flow.angiogram,
    #     "MAG": mri_flow.magnitude,
    #     "comp_vd_1": mri_flow.velocity_estimate[..., 0],
    #     "comp_vd_2": mri_flow.velocity_estimate[..., 1],
    #     "comp_vd_3": mri_flow.velocity_estimate[..., 2]
    # }
    
    # with h5py.File(out_name, "w") as hf:
    #     data_group = hf.create_group("Data")
        
    #     # Time averaged dataset creation
    #     for name, arr in outputs.items():
    #         data_group.create_dataset(name, data=mean(arr, axis=0))
            
    #     # Time resolved dataset creation
    #     for frame in range(frames):
            
    #         data_group.create_dataset(
    #             f"ph_{frame:03d}_cd",
    #             data=mri_flow.angiogram[frame]
    #         )
            
    #         data_group.create_dataset(
    #             f"ph_{frame:03d}_mag",
    #             data=mri_flow.magnitude[frame]
    #         )
            
    #         data_group.create_dataset(
    #             f"ph_{frame:03d}_comp_vd_1",
    #             data=mri_flow.velocity_estimate[frame, ..., 0]
    #         )
            
    #         data_group.create_dataset(
    #             f"ph_{frame:03d}_comp_vd_2",
    #             data=mri_flow.velocity_estimate[frame, ..., 1]
    #         )
            
    #         data_group.create_dataset(
    #             f"ph_{frame:03d}_comp_vd_3",
    #             data=mri_flow.velocity_estimate[frame, ..., 2]
    #         )
    # print("Created Flow.h5 file from Images.h5")
        
    # try:
    #     os.remove(out_name)
    # except OSError:
    #     pass
    # with h5py.File(out_name, 'w') as hf:
    #     hf.create_dataset("VX", data=mri_flow.velocity_estimate[..., 0])
    #     hf.create_dataset("VY", data=mri_flow.velocity_estimate[..., 1])
    #     hf.create_dataset("VZ", data=mri_flow.velocity_estimate[..., 2])
    #     hf.create_dataset("CD", data=mri_flow.angiogram)
    #     hf.create_dataset("MAG", data=mri_flow.magnitude)