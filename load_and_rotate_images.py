# ! /usr/bin/env python
# Utilities
import sys
sys.path.append('/data/data_mrcv2/99_LARR/mc_flow/simulation/flow_recon')  # UPDATE!
sys.path.append('/export/home/larivera/CODE/RECON/NoiseOptimalSampling')  # UPDATE!

from flow_processing import MRI_4DFlow
import llr_recon_flow
import logging
import cupy
import argparse
import numpy as np
import h5py
import os
import shutil
import fnmatch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from registration_tools import *
import logging
import os
import numpy as np # array library
import math  # math used to define pi
import h5py  #hdf5 interface, used to load data
from functools import partial #this lets us fix some of the variables in a fucntion call
import pywt #python based wavelets
from scipy.signal import convolve # convolution
import matplotlib.pyplot as plt
from IPython import display
import tkinter as tk
from tkinter import filedialog
import os
import sigpy as sp
import sigpy.mri as mri
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mri
import cupy as cp
import os
from pathlib import Path
import matplotlib.pyplot as plt
import  logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter



def fake_rotations(images, theta_in, psi_in, phi_in, x_in, y_in, z_in):
    
    #push images to torch
    images_torch = torch.from_numpy(images)

    # Rotate
    # for paper revisions think about Bore reference coordinates B0 Zdir,  Up/Dn Y, Lt/Rt X)
    theta = torch.nn.Parameter(torch.tensor([np.radians(theta_in)]).view(1,1)) # rot about X
    phi = torch.nn.Parameter(torch.tensor([np.radians(phi_in)]).view(1,1)) # rot about Y 
    psi = torch.nn.Parameter(torch.tensor([np.radians(psi_in)]).view(1,1)) # rot about Z

    tx = torch.nn.Parameter(torch.tensor([x_in]).view(1,1)) # Y translation 
    ty = torch.nn.Parameter(torch.tensor([y_in]).view(1,1)) # X translation 
    tz = torch.nn.Parameter(torch.tensor([z_in]).view(1,1)) # Z translation 

    rot = torch.zeros(3, 4, dtype=images_torch.dtype)

    rot[0, 0] = torch.cos(theta) * torch.cos(psi)
    rot[0, 1] = -torch.cos(phi) * torch.sin(psi) + torch.sin(phi) * torch.sin(theta) * torch.cos(psi)
    rot[0, 2] =  torch.sin(phi) * torch.sin(psi) + torch.cos(phi) * torch.sin( theta) * torch.cos(psi)

    rot[1, 0] = torch.cos(theta)*torch.sin(psi)
    rot[1, 1] = torch.cos(phi) * torch.cos(psi) + torch.sin(phi) * torch.sin( theta) * torch.sin(psi)
    rot[1, 2] = -torch.sin(phi) * torch.cos(psi) + torch.cos(phi) * torch.sin(theta) * torch.sin( psi)

    rot[2, 0] = -torch.sin(theta)
    rot[2, 1] = torch.sin( phi) * torch.cos( theta)
    rot[2, 2] = torch.cos( phi ) * torch.cos( theta)

    #print(rot)
    rot[0, 3] = tx
    rot[1, 3] = ty
    rot[2, 3] = tz

    # reshape into (Nbatch*Nframes)x2x3 affine matrix
    theta = rot.view(-1, 3, 4)

    images_torch = images_torch.view(-1,1,images_torch.shape[-3], images_torch.shape[-2], images_torch.shape[-1])

    # Create affine grid from affine transform
    # affine grid uses matrices from -1 to 1 along each dimension
    grid = F.affine_grid(theta, images_torch.size(), align_corners=False)

    # Sample the data on the grid
    registered = F.grid_sample(images_torch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return(registered.detach().cpu().numpy())


#def load_images( file_nav=None):
#    # Load an image
#    with h5py.File(file_nav,'r') as hf:
#        images = np.array(hf['IMAGE_MAG'])
#
#    return images


 # Load 4D images
#images = load_images('Dynamic.h5')
#print(images.shape)

#plt.figure(figsize=(10,10))
#plt.imshow(images[0,0,0,64,:,:])
#plt.show()
#(256, 1, 1, 128, 128, 176) input shape of registration pipeline 


# load maginitude and velocity data
mag  = []
vel1 = []
vel2 = []
vel3 = [] 
img_shape = [64,64,64]

filename = ('MAG.dat')
mag.append(np.reshape(np.fromfile(filename,dtype = 'int16'), img_shape))

filename = ('comp_vd_1.dat')
vel1.append(np.reshape(np.fromfile(filename,dtype = 'int16'),img_shape))

filename = ('comp_vd_2.dat')
vel2.append(np.reshape(np.fromfile(filename,dtype = 'int16'),img_shape))

filename = ('comp_vd_3.dat')
vel3.append(np.reshape(np.fromfile(filename,dtype = 'int16'),img_shape))

mag = np.array(mag) # 1,320,320,320,  ; t,z,y,x
vel1 = np.array(vel1)
vel2 = np.array(vel2)
vel3 = np.array(vel3)

mag = mag.astype('float32')
vel1 = vel1.astype('float32')
vel2 = vel2.astype('float32')
vel3 = vel3.astype('float32')

#normalize mag 
mag /= np.amax(mag)

print(mag.shape)
print(mag.dtype)

# lets focus on rotations about 1 axis (the z axis with respect to the bore )
# sine rotations, but can model differently too. 
z_rot_range = np.arange(0,np.pi,np.pi*1e-2)   # start,stop,step 100 rots 
max_rot_deg = 90
z_rot_angles = max_rot_deg*np.sin(z_rot_range) 

print(z_rot_angles)
print(z_rot_angles.shape)

# call function that will create array of rotated images 
all_rot_mag = []
all_rot_v1  = []
all_rot_v2  = []
all_rot_v3  = []

for i in range(1): #len(z_rot_angles)
    
    # KMJ - Grab rotation matrix
    #rot_mat = ge_rotation( 0.0,z_rot_angles[i],0.0,0.0,0.0,0.0)

    rot_mag = fake_rotations(mag,0.0,z_rot_angles[i],0.0,0.0,0.0,0.0)  # fake_rotations(images, theta_in, psi_in, phi_in, x_in, y_in, z_in):
    all_rot_mag.append(rot_mag)

    rot_v1 = fake_rotations(vel1, 0.0,z_rot_angles[i],0.0,0.0,0.0,0.0) 
    all_rot_v1.append(rot_v1)

    rot_v2 = fake_rotations(vel2, 0.0,z_rot_angles[i],0.0,0.0,0.0,0.0)  
    all_rot_v2.append(rot_v2)

    rot_v3 = fake_rotations(vel3, 0.0,z_rot_angles[i],0.0,0.0,0.0,0.0)  
    all_rot_v3.append(rot_v3)

    # KMJ - Need rotation of the velocity direction
    #rot_v1_new = rot_mat[0,0]*rot_v1 + rot_mat[1,0]*rot_v2 + rot_mat[1,0]*rot_v3
    #rot_v2_new = rot_mat[0,1]*rot_v1 + rot_mat[1,1]*rot_v2 + rot_mat[1,1]*rot_v3
    #rot_v3_new = rot_mat[0,2]*rot_v1 + rot_mat[1,2]*rot_v2 + rot_mat[1,2]*rot_v3
    


all_rot_mag = np.array(all_rot_mag)
all_rot_v1 = np.array(all_rot_v1)
all_rot_v2 = np.array(all_rot_v2)
all_rot_v3 = np.array(all_rot_v3)


print(all_rot_mag.shape)

#plt.figure(figsize=(10,10))
#plt.subplot(1,2,1)
#plt.imshow(mag[0,32,:,:])
#plt.subplot(1,2,2)
#plt.imshow(all_rot_mag[0,0,0,32,:,:])
#plt.show()

# Just check images make sense 
#plt.figure(figsize=(10,10))
#for m in range(10):
#    for n in range(10):
#        plt.subplot(10,10,m*10+n+1)
#        plt.imshow(all_rot_mag[m*10+n,0,0,34,:,:])
#plt.show()

#plt.figure(figsize=(10,10))
#for m in range(10):
#    for n in range(10):
#        plt.subplot(10,10,m*10+n+1)
#        plt.imshow(all_rot_v1[m*10+n,0,0,34,:,:])
#plt.show()

#plt.figure(figsize=(10,10))
#for m in range(10):
#    for n in range(10):
#        plt.subplot(10,10,m*10+n+1)
#        plt.imshow(all_rot_v2[m*10+n,0,0,34,:,:])
#plt.show()

#plt.figure(figsize=(10,10))
#for m in range(10):
#    for n in range(10):
#        plt.subplot(10,10,m*10+n+1)
#        plt.imshow(all_rot_v3[m*10+n,0,0,34,:,:])
#plt.show()

# now make complex data
# encoding matrix for dataset (4-pt ref), scalar = pi/(2*Venc*gamma), assume gamma = 1

# Encoding Matrix =
#  -0.0020  -0.0020  -0.0020 1      vx 
#   0.0020  -0.0020  -0.0020 1      vy
#  -0.0020   0.0020  -0.0020 1      vy
#  -0.0020  -0.0020   0.0020 1    theta

# X,Y,Z
E = np.array([ [-0.0020,  -0.0020,  -0.0020],  [0.0020,  -0.0020,  -0.0020], [-0.0020,   0.0020, -0.0020], [-0.0020,  -0.0020,   0.0020]])

# subtract the first measurement as it was done in the recon that generated the dataset
E_S = E - E[0,:]
E_S = E # Don't subtract actually !
print(E_S)

#find pInv 
#Eps = np.linalg.pinv(Encoding_Matrix_Substracted)
#print(Eps)

phi1 = E_S[0,0]*all_rot_v1+E_S[0,1]*all_rot_v2+E_S[0,2]*all_rot_v3 #this is zero
phi2 = E_S[1,0]*all_rot_v1+E_S[1,1]*all_rot_v2+E_S[1,2]*all_rot_v3
phi3 = E_S[2,0]*all_rot_v1+E_S[2,1]*all_rot_v2+E_S[2,2]*all_rot_v3
phi4 = E_S[3,0]*all_rot_v1+E_S[3,1]*all_rot_v2+E_S[3,2]*all_rot_v3

# now combine 
complex_data = np.zeros_like(all_rot_mag).astype('complex64')
complex_data = np.stack((complex_data,complex_data,complex_data,complex_data),axis=0)
print(complex_data.shape)
print(complex_data.dtype)  

# 4 encodes
complex_data[0,...] = np.multiply(all_rot_mag, np.exp(1.0j*phi1))
complex_data[1,...] = np.multiply(all_rot_mag, np.exp(1.0j*phi2))
complex_data[2,...] = np.multiply(all_rot_mag, np.exp(1.0j*phi3))
complex_data[3,...] = np.multiply(all_rot_mag, np.exp(1.0j*phi4))

# should we add some noise now or later on kspace ? Preferentially in k-space, but don't worry about adding noise. 
#n1 = 5e-4*np.max(np.abs(complex_data[0,...])) # should be just 5e-4
#complex_data[0,...] += n1 * np.random.standard_normal(all_rot_mag.shape) + n1 * 1j * np.random.standard_normal(all_rot_mag.shape)
#complex_data[1,...] += n1 * np.random.standard_normal(all_rot_mag.shape) + n1 * 1j * np.random.standard_normal(all_rot_mag.shape)
#complex_data[2,...] += n1 * np.random.standard_normal(all_rot_mag.shape) + n1 * 1j * np.random.standard_normal(all_rot_mag.shape)
#complex_data[3,...] += n1 * np.random.standard_normal(all_rot_mag.shape) + n1 * 1j * np.random.standard_normal(all_rot_mag.shape)
# shape ~ (4,1,1,1,64,64,64)

# now define a sampling trajectory 

# for now lets just use the same num of proj and coords for each tf and encode 

# load smaps for 32ch coil
filename = 'SenseMaps.h5'
with h5py.File(filename, 'r') as hf:
    smaps = []  # smaps
    num_coils = 32
    for z in range(num_coils):
        print(f'Loading kspace, coil {z + 1} / {num_coils}.')
        s = hf['Maps'][f'SenseMaps_{z}']
        smaps.append( s['real'] + 1j * s['imag'])

 
smaps = np.stack(smaps,axis=0)
print(smaps.shape) # 32, 64, 64, 64

# Generate radial coordinates
nproj = 11000
npts = 64
ndims = 3

coord_shape = [ nproj, npts, ndims ]
imag_shape =  [64, 64, 64]

# KMJ - Just load from the data
coords = sp.mri.radial(coord_shape, img_shape, golden=True)
print(coords.shape)
#print(coords)

# Create B which is a lineary operator
B = sp.mri.linop.Sense(smaps, coord=coords, coil_batch_size=None)
print(B)
print(B.linops)

# Now get simulated k-space
num_enc = 4
ksp_enc = []
for i in range(num_enc):
    k_space = B.apply(np.squeeze(complex_data[i,...])) # use adjoint
    ksp_enc.append(k_space)

ksp_enc = np.array(ksp_enc)

print(ksp_enc.shape) # 


# noise here instead ? 
#n1 = 5e-4*np.max(np.abs(k_space))
#k_space += n1 * np.random.standard_normal(k_space.shape)
#k_space += n1 * 1j * np.random.standard_normal(k_space.shape)


# Simple plot using Matplotlib
#plt.figure()
#plt.plot(np.abs(k_space[0,:,:]).transpose())
#plt.show()

device = sp.Device(0)

# KMJ - Just load from the data
dcf = np.ones((coord_shape[0],coord_shape[1])) # is 1 fine ?

coords = sp.to_device(coords, device=device)
dcf_gpu = sp.to_device(dcf, device=device)
smaps_gpu = sp.to_device(smaps, device=device)
k_space_gpu = sp.to_device(ksp_enc, device=device)

sim_complex_data = []
for i in range(num_enc):
    with device:
        # Create a SENSE operator
        sense = sp.mri.app.SenseRecon(k_space_gpu[i,...], mps=smaps_gpu, weights=dcf_gpu, coord=coords, device=device, max_iter=60,
                                        coil_batch_size=None)
        # Run SENSE operator
        image_sense = sense.run()

        # Create SENSE + L1 Wavelet penalty
        lam = 0.1
        #l1wavelet = sp.mri.app.L1WaveletRecon(k_space_gpu, mps=smaps_gpu, lamda=lam, weights=dcf_gpu, coord=coords,
                                                #device=device, accelerate=True, coil_batch_size=None, max_iter=200)
        # Run L1 wavelet penalty
        #image_l1wavelet = l1wavelet.run()

        # Put back onto CPU for visualization
    image_sense = sp.to_device(image_sense, sp.cpu_device)
    sim_complex_data.append(image_sense)


# Put back onto CPU for visualization
#image_sense = sp.to_device(image_sense, sp.cpu_device)
#image_l1wavelet = sp.to_device(image_l1wavelet, sp.cpu_device)

#print(image_sense.shape)
sim_complex_data = np.array(sim_complex_data)
print(sim_complex_data.shape)
# Show the image
#plt.figure(figsize=(20,20))

#plt.subplot(221)
#plt.imshow((np.abs(sim_complex_data[0,32,:,:]).transpose()),cmap='gray')
#plt.axis('off')
#plt.title('SENSE')

#plt.subplot(222)
#plt.imshow((np.abs(image_l1wavelet[0,32,:,:]).transpose()),cmap='gray')
#plt.axis('off')
#plt.title('L1 Wavelet')

#plt.subplot(222)
#plt.imshow((np.abs(complex_data[0,0,0,0,32,:,:]).transpose()),cmap='gray')
#plt.axis('off')
#plt.title('Truth')


#plt.subplot(223)
#plt.imshow((np.angle(sim_complex_data[0,32,:,:]).transpose()),cmap='gray')
#plt.axis('off')
#plt.title('SENSE')

#plt.subplot(224)
#plt.imshow((np.angle(complex_data[0,0,0,0,32,:,:]).transpose()),cmap='gray')
#plt.axis('off')
#plt.title('Truth')

#plt.show()

# generate navigators 
# give correct set of rotations and recover images 

# Now get velocity and magnitude data back 
sim_mag = np.squeeze(np.abs(sim_complex_data[0,...]))

# Get pseudoinverse of encoding matrix (remember we used the subtracted E), might need to account for that in actual recon

Eps = np.linalg.pinv(E_S)
#get vel back
phi1= np.squeeze(np.angle(sim_complex_data[0,...]))
phi2= np.squeeze(np.angle(sim_complex_data[1,...]))
phi3= np.squeeze(np.angle(sim_complex_data[2,...]))
phi4= np.squeeze(np.angle(sim_complex_data[3,...]))

sim_v1 = Eps[0,0]*phi1+Eps[0,1]*phi2+Eps[0,2]*phi3+Eps[0,3]*phi4
sim_v2 = Eps[1,0]*phi1+Eps[1,1]*phi2+Eps[1,2]*phi3+Eps[1,3]*phi4
sim_v3 = Eps[2,0]*phi1+Eps[2,1]*phi2+Eps[2,2]*phi3+Eps[2,3]*phi4

print(sim_v1.shape)

with h5py.File('sim_images_Eps_sub.h5', 'w') as hf:
    hf.create_dataset('sim_v1', data=sim_v1)
    hf.create_dataset('sim_v2', data=sim_v2)
    hf.create_dataset('sim_v3', data=sim_v3)
    hf.create_dataset('sim_mag', data=sim_mag)


#next try rotated images and get ready to mc recon.