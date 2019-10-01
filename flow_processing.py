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
                              )

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

        if self.velocity_estimate is not None:
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
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)')
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory')

    args = parser.parse_args()

    # Put up a file selector if the file is not specified
    if args.filename is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()
        args.filename = askopenfilename()

    with h5py.File('FullRecon.h5', 'r') as hf:
        temp = hf['IMAGE']
        print(temp.shape)
        #temp = temp['real'] + 1j*temp['imag']
        #temp = np.moveaxis(temp, -1, 0)
        #frames = int(temp.shape[0]/5)
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
    #mri_flow.update_angiogram()
    #mri_flow.background_phase_correct()
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
