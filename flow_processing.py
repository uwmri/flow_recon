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


class MRI_4DFlow:

    def __init__(self, encode_type,venc):

        'Initialization'
        self.set_encoding_matrix(encode_type)
        self.Venc = venc  #m/s
        self.NoiseLevel = 0.0 #relative to max signal of 1
        self.spatial_resolution = 0.5 # percent of kmax
        self.time_resolution = 0.5 # percent of nominal
        self.background_magnitude = 0.5 #value of background

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
            '4pt-balanced': np.pi / 2.0/ np.sqrt(2.0) * np.array([[-1.0, -1.0, -1.0],
                                                      [ 1.0,  1.0, -1.0],
                                                      [ 1.0, -1.0,  1.0],
                                                      [-1.0,  1.0, 1.0]], dtype=np.float32),
            '5pt': np.pi / np.sqrt(3.0) * np.array([ [0.0, 0.0, 0.0],
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

        unwrap_lap = True
        if unwrap_lap:
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

    def background_phase_correct(self,mag_thresh=0.08, angiogram_thresh=0.3,fit_order=2):

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
        self.magnitude = np.sqrt( np.mean( np.abs(self.signal)**2 , -1))

    """
    :return: Nt x Nz x Ny x Nx x Nencode x 1
    """
    def update_angiogram(self):

        # Recalc Magnitude
        self.update_magnitude()

        if self.velocity_estimate is None:
            self.solve_for_velocity()

        vmag = np.sqrt( np.mean( np.abs(self.velocity_estimate)**2 , -1))
        self.angiogram = self.magnitude*np.sin(math.pi/2.0*vmag/self.Venc)

        idx = np.where(vmag > self.Venc )
        self.angiogram[idx] = self.magnitude[idx]


if __name__ == "__main__":


    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--venc', type=float, default=80.0)
    # Input Output
    parser.add_argument('--filename', type=str, help='filename for data (e.g. FullRecon.h5)', default=None)
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='Flow.h5')

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
    with h5py.File(args.filename, 'r') as hf:
        temp = np.array(hf['IMAGE'])
        print(temp.shape)
        #temp = temp['real'] + 1j*temp['imag']
        #temp = np.moveaxis(temp, -1, 0)
        #frames = int(temp.shape[0]/4)
        #temp = np.reshape(temp, newshape=(frames,4, temp.shape[1], temp.shape[2], temp.shape[3]))
        # temp = np.reshape(temp, newshape=(10,4, temp.shape[-3], temp.shape[-2], temp.shape[-1]))

        temp = np.squeeze(temp)

        if len(temp.shape) == 4:
            temp = np.expand_dims(temp,axis=0)

        frames = int(temp.shape[0])
        num_encodes = int(temp.shape[1])

        print(f' num of frames =  {frames}')
        print(f' num of encodes = {num_encodes}')
        #temp = np.reshape(temp,newshape=(5, frames,temp.shape[1],temp.shape[2],temp.shape[3]))
        #temp = np.reshape(temp,newshape=(temp.shape[1], frames,temp.shape[2],temp.shape[3],temp.shape[4]))

        temp = np.moveaxis(temp,1,-1)
        print(temp.shape)

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
    mri_flow.solve_for_velocity()
    mri_flow.update_angiogram()
    #mri_flow.background_phase_correct()
    #mri_flow.update_angiogram()

    # Export to file
    out_name = os.path.join(out_folder, args.out_filename)
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
