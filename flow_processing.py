#! /usr/bin/env python
import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import os
import argparse
import matplotlib.pyplot as plt
import cupy
import math


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


# Laplacian based phase unwrapping
def unwrap_3d(phase_w):

    logger = logging.getLogger('Laplacian Unwrap')

    ts = 8.0  # scales temporal data to spatial dimentions
    # real_flag = 1 # restrict laplacians to real (lowers memory load)
    phase_w = np.moveaxis(phase_w, 0, -1)  # x y t

    ndim = phase_w.shape

    # create grid
    X, Y, T = np.mgrid[-ndim[0] // 2:ndim[0] // 2:,
                 -ndim[1] // 2:ndim[1] // 2:,
                 -ndim[2] // 2:ndim[2] // 2:]

    # get mod
    mod = 2*np.cos(np.pi*X / ndim[0]) + 2*np.cos(np.pi*Y / ndim[1]) + ts*np.cos(np.pi*T / ndim[2]) - 6.0 - ts
    X = None
    Y = None
    T = None

    logger.info('Laplacian')
    print('Forward')
    lap_phase_w = lap3(phase_w, 1, mod)
    lap_phase = np.cos(phase_w) * lap3(np.sin(phase_w), 1, mod) - np.sin(phase_w) * lap3(np.cos(phase_w), 1, mod)

    logger.info('Inverse Laplacian')
    print('Backwards')
    ilap_phasediff = lap3(lap_phase - lap_phase_w, -1, mod)
    n_u4 = np.int8(np.real(np.ndarray.round(ilap_phasediff / 2 / np.pi)))
    n_u4 = np.moveaxis(n_u4, -1, 0) # t x y

    return n_u4


def lap3(phase_w, direction, mod):
    ndim = phase_w.shape
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(phase_w)))

    if direction == 1:
        K *= mod

    elif direction == -1:
        mod[ndim[0] // 2, ndim[1] // 2, ndim[2] // 2] = 1
        K /= mod

    else:
        print("ERROR")

    out = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))

    return out


class MRI_4DFlow:

    def __init__(self, encode_type, signal=None, venc=None, unwrap_lap=False):

        # Initialization
        self.encode_type = encode_type
        self.set_encoding_matrix(encode_type)
        self.venc = venc
        self.NoiseLevel = 0.0 #relative to max signal of 1
        self.spatial_resolution = 0.5 # percent of kmax
        self.time_resolution = 0.5 # percent of nominal
        self.background_magnitude = 0.5 #value of background
        self.unwrap_lap = unwrap_lap
        
        # Matrices
        self.signal = signal
        self.velocity_estimate = None
        self.angiogram = None
        self.magnitude = None
        

    def set_encoding_matrix(self, encode_type='4pt-referenced'):
        encode_dictionary = {
            '4pt-referenced': np.pi/2.0*np.array([[-1.0, -1.0, -1.0],
                               [1.0, -1.0, -1.0],
                               [-1.0,  1.0, -1.0],
                               [-1.0, -1.0,  1.0]], dtype=np.float32),
            '3pt': np.pi / 2.0 * np.array([[-1.0, -1.0, -1.0],
                                                      [1.0, -1.0, -1.0],
                                                      [-1.0, 1.0, -1.0]], dtype=np.float32),
            '4pt-balanced': np.pi / 2.0/ np.sqrt(2.0) * np.array([[-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32),
            '5pt': np.pi / np.sqrt(3.0) * np.array([ [0.0, 0.0, 0.0],
                                                      [-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32),
            '2pt': np.pi * np.array([[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 1.0]], dtype=np.float32)
        }
        self.EncodingMatrix = encode_dictionary[encode_type]
        self.DecodingMatrix = np.linalg.pinv(self.EncodingMatrix)

    """
    :param velocity: a Nt x Nz x Ny x Nx x 3 description of the velocity field
    :param pd: a Nt x Nz x Ny x Nx mask of the vessel locations
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def generate_complex_signal(self, velocity, pd):

        # Get last dimension to (3 x 1)
        velocity = np.expand_dims(velocity, -1)

        # Multiple to get phase
        print(self.EncodingMatrix.shape)
        print(velocity.shape)

        # Get the Phase
        phase = np.matmul( self.EncodingMatrix/self.venc, velocity)

        # Create Magnitude image (M*exp(i*phase))
        mag = np.copy(pd)
        mag += self.background_magnitude
        mag = np.expand_dims(mag, -1)
        mag = np.expand_dims(mag, -1)
        self.signal = mag*np.exp(1j * phase )

    def solve_for_velocity(self):

        # Multiply by reference
        ref = self.signal[...,0]
        ref = np.expand_dims(ref, -1)
        signal2 = self.signal * np.conj(ref)

        # Convert to .. x Nencodes x 1
        signal2 = np.expand_dims( signal2,-1)

        # Get subtracted decoding matrix
        diffMatrix = self.EncodingMatrix.copy()
        diffMatrix -= diffMatrix[0,:]
        self.DecodingMatrix = np.linalg.pinv(diffMatrix)

        # Take angle
        phase = np.angle(signal2)

        # Unwrap phase for all encodes
        num_enc = phase.shape[3]

        if self.unwrap_lap:
            if phase.shape[0] > 1:
                print(f'number of encodes to unwrap {num_enc}')
                # Start loop in second encode (first was use to reference)
                phase_wrap = []
                phase = np.squeeze(phase)
                print('Starting Laplacian based phase unwrapping')
                for i in range(num_enc - 1):
                    print(f'Copy encode {i}')
                    phase_wrap = np.copy(phase[:, :, :, i + 1])

                    # Find phase wraps
                    print(f'Unwrap the encode {i}')
                    #n_jumps = unwrap_4d(phase_wrap)
                    n_jumps = unwrap_3d(phase_wrap)

                    # Unwrap phase
                    print(f'Apply unwrap {i}')
                    phase[:, :, :, i + 1] = phase[:, :, :, i + 1] + 2 * np.pi * n_jumps

                phase = np.expand_dims(phase, -1)
                print('Laplacian based phase unwrapping finished')

        #Solve for velocity
        self.velocity_estimate = np.matmul(self.DecodingMatrix*self.venc,phase)

        # Data comes back as Nt x Nz X Ny x Nz x 3 x 1, reduce to
        #   Nt x Nz x Ny x Nx x 3
        self.velocity_estimate = np.squeeze( self.velocity_estimate, axis=-1)

    def background_phase_correct(self,mag_thresh=0.08, angiogram_thresh=0.3,fit_order=3):

        # Average time frames
        magnitude_avg = np.mean(self.magnitude, 0)
        angiogram_avg = np.mean(self.angiogram, 0)

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
        vavg = np.squeeze( np.mean( self.velocity_estimate, axis=0))
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
                AhA[ii, jj] = np.sum( (x_slice ** px[ii] * y_slice ** py[ii] * z_slice ** pz[ii]) *
                                     (x_slice ** px[jj] * y_slice ** py[jj] * z_slice ** pz[jj]) )

        for ii in range(N):
            phi = np.power(x_slice, px[ii]) * np.power(y_slice, py[ii]) * np.power( z_slice, pz[ii])
            AhBx[ii] = np.sum(vx_slice * phi)
            AhBy[ii] = np.sum(vy_slice * phi)
            AhBz[ii] = np.sum(vz_slice * phi)

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

        #Expand and subtract)
        background_phase = np.expand_dims( background_phase,0)
        self.velocity_estimate -= background_phase

    """
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def update_magnitude(self):
        self.magnitude = np.sqrt(np.sum(np.abs(self.signal)**2 , axis=-1))

    """
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def update_angiogram(self):

        # Recalc Magnitude
        self.update_magnitude()

        if self.velocity_estimate is None:
            self.solve_for_velocity()

        # make consistent with C++ recon
        vmag = np.sqrt(np.sum(self.velocity_estimate**2, axis=-1))
        vmag = 2.0 * vmag / self.venc
        vmag = np.minimum(vmag, 1.0)
        self.angiogram = self.magnitude * np.sin(np.pi/2 * vmag)

        # idx = np.where(vmag > self.venc )
        # self.angiogram[idx] = self.magnitude[idx]
        

def export_flow_data(mri_flow, out_name, header_info=None, c_format=False):

    # Export to file
    try:
        os.remove(out_name)
    except OSError:
        pass
    
    # convert to int and scale
    mg = mri_flow.magnitude * 32767/np.max(mri_flow.magnitude)
    mg = mg.astype(np.int16)
    cd = mri_flow.angiogram * 32767/np.max(mri_flow.angiogram)
    cd = cd.astype(np.int16)
    vx = mri_flow.velocity_estimate[..., 0].astype(np.int16)
    vy = mri_flow.velocity_estimate[..., 1].astype(np.int16)
    vz = mri_flow.velocity_estimate[..., 2].astype(np.int16)
    
    # if time averaged, add time dimension
    if len(mri_flow.magnitude.shape) < 4:
        mg = np.expand_dims(mg, axis=0)
        cd = np.expand_dims(cd, axis=0)
        vx = np.expand_dims(vx, axis=0)
        vy = np.expand_dims(vy, axis=0)
        vz = np.expand_dims(vz, axis=0)
    
    print(f'Exporting flow data to {out_name}')
    with h5py.File(out_name, 'w') as hf:
        header_group = hf.create_group("Header")
        data_group = hf.create_group("Data")
        if header_info is not None:
            for attr in header_info.keys():
                header_group.attrs[attr] = header_info[attr]
        else:
            header_group.attrs["venc"] = mri_flow.venc
            header_group.attrs["frames"] = mg.shape[0]
            header_group.attrs["matrixx"] = mg.shape[1]
            header_group.attrs["matrixy"] = mg.shape[2]
            header_group.attrs["matrixz"] = mg.shape[3]
        
        if c_format:
            frames = mg.shape[0]
            data_group.create_dataset("MAG", data=np.squeeze(np.mean(mg, axis=0)))
            data_group.create_dataset("CD", data=np.squeeze(np.mean(cd, axis=0)))
            data_group.create_dataset("comp_vd_1", data=np.squeeze(np.mean(vx, axis=0)))
            data_group.create_dataset("comp_vd_2", data=np.squeeze(np.mean(vy, axis=0)))
            data_group.create_dataset("comp_vd_3", data=np.squeeze(np.mean(vz, axis=0)))
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
    parser.add_argument('--c_format', dest='c_format', action='store_true', default=False, help='export flow HDF5 file in C++ recon format')

    args = parser.parse_args()
    
    # Put up a file selector if the file is not specified
    if args.filename is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()
        args.filename = askopenfilename()
    
    if args.out_folder is None:
        args.out_folder = os.path.dirname(args.filename)
            
    print(f'Loading {args.filename}')
    with h5py.File(args.filename, 'r') as hf:
        signal = np.array(hf['IMAGE'])
    
    print(signal.shape)
    #signal = signal['real'] + 1j*signal['imag']
    #signal = np.moveaxis(signal, -1, 0)
    #frames = int(signal.shape[0]/4)
    #signal = np.reshape(signal, newshape=(frames,4, signal.shape[1], signal.shape[2], signal.shape[3]))
    # signal = np.reshape(signal, newshape=(10,4, signal.shape[-3], signal.shape[-2], signal.shape[-1]))

    signal = np.squeeze(signal)

    if len(signal.shape) == 4:
        signal = np.expand_dims(signal,axis=0)

    frames = int(signal.shape[0])
    num_encodes = int(signal.shape[1])

    print(f' num of frames =  {frames}')
    print(f' num of encodes = {num_encodes}')
    #signal = np.reshape(signal,newshape=(5, frames,signal.shape[1],signal.shape[2],signal.shape[3]))
    #signal = np.reshape(signal,newshape=(signal.shape[1], frames,signal.shape[2],signal.shape[3],signal.shape[4]))

    signal = np.moveaxis(signal,1,-1)
    print(signal.shape)

    if num_encodes == 5:
        encoding = "5pt"
    elif num_encodes == 4:
        encoding = "4pt-referenced"
    elif num_encodes == 3:
        encoding = "3pt"
    elif num_encodes == 2:
        encoding = "2pt"

    print(f' encoding type is {encoding}')

    # Solve for Velocity
    mri_flow = MRI_4DFlow(encoding, signal=signal, venc=args.venc)
    print(f'venc is {mri_flow.venc}')
    mri_flow.solve_for_velocity()
    mri_flow.update_angiogram()
    #mri_flow.background_phase_correct()
    #mri_flow.update_angiogram()
    
    print(f'Exporting flow data to {args.out_filename}')
    export_flow_data(mri_flow, os.path.join(args.out_folder, args.out_filename), c_format=args.c_format)
