#! /usr/bin/env python
import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import cupy
import time
import math
import sys
from mri_raw import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from registration_tools import *
import logging
import os

# Use weighted loss for masks
def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)

def load_images( file_nav=None):
    # Load an image
    with h5py.File(args.file_nav,'r') as hf:
        images = np.array(hf['IMAGE_MAG'])

    return images

def estimate_mask( images):
    r"""Estimates a mask with the zprofile of exciation

    Args:
        images (array): a 4D array [Nt x Nz x Ny x Nz ] to
    Returns:
        mask (array): a mask broad castable to the image size for weighted mean square error
    """


    # Average images
    avg = np.mean(images, axis=tuple(range(len(images.shape)-3)))
    avg /= np.max(avg)

    # Create a mask for the zprofile
    zprofile = np.max( avg, axis=(-2,-1))
    zprofile_idx = np.nonzero(np.squeeze(zprofile > 0.1))
    start_idx = zprofile_idx[0][0] + 10
    stop_idx = zprofile_idx[0][-1] - 10
    print(f'Start idx = {start_idx} , stop idx {stop_idx}')
    mask = np.zeros_like(avg)
    mask[start_idx:stop_idx,:,:] = 1.0
    return mask

def average_rotation( r=None):
    r"""Finds a rotation to a common space

    Args:
        r (array): a 3D array [Nt x 3 x 3] containing rotation matrices over time
    Returns:
        rp (array): a 3D array [Nt x 3 x 3] containing corrected rotation matrices
    """

    # This just defines a rotation matrix using euler angles
    RotModel = Rotation3D()

    # Optimize using Adam
    optimizer = torch.optim.Adam(RotModel.parameters(), lr=1e-4)

    # Need identify to get loss
    id = torch.eye(3)

    # Get the rotation matrix
    R = torch.tensor(r)

    for epoch in range(100000):

        optimizer.zero_grad()

        # B is the correction to R
        B = RotModel()

        # Calculate the corrected matrix
        y = torch.matmul(B, R)

        # Loss is average difference from identity
        loss = torch.sum(torch.abs(y - id) ** 2)

        if epoch % 1000 == 0:
            print(f'Epoch {epoch} Loss = {loss.item()}')

        loss.backward()
        optimizer.step()

    # Apply rotation to R
    B = RotModel()

    print(f'Inverse Common Rotation')
    print(f'  Psi={RotModel.psi[0,0]} ')
    print(f'  Theta={RotModel.theta[0, 0]} ')
    print(f'  Phi={RotModel.phi[0, 0]} ')
    print(f'  B = {B}')

    return(B.detach().cpu().numpy())


def register_images( images, mask, logdir=None, out_reg_images=False):
    r"""Registers a series of 3D images collected over time using pytorch affine and masked mean square error

    Args:
        images (array): a 4D array [Nt x Nz x Ny x Nz ] to
        mask (array): a mask broad castable to the image size for weighted mean square error
        logdir (path): a folder location to save the data
    Returns:
        tuple containing  tx, ty, tz, phi, psi, theta rigid transforms at each timepoint
    """

    # Ensure mask is a tensor on gpu
    mask = torch.tensor(mask).to('cuda')

    # The model is declared once so that we only need to estimate differences from frame to frame
    model = RigidRegistration()
    model.cuda()

    fixed_image = torch.tensor(images[0]).to('cuda')
    fixed_image = fixed_image.view(-1, 1, fixed_image.shape[-3], fixed_image.shape[-2], fixed_image.shape[-1])
    print(f'Fixed image max {torch.max(fixed_image)}')
    fixed_image /= torch.max(fixed_image)
    print(f'Fixed image shape {fixed_image.shape}')

    # Pad to be square
    max_size = torch.max(torch.tensor(fixed_image.shape))

    pad_amount1 = (max_size - fixed_image.shape[-1]) // 2
    pad_amount2 = (max_size - fixed_image.shape[-2]) // 2
    pad_amount3 = (max_size - fixed_image.shape[-3]) // 2

    pad_f = (pad_amount1, pad_amount1, pad_amount2, pad_amount2, pad_amount3, pad_amount3)
    fixed_image = nn.functional.pad(fixed_image, pad_f)

    mask = nn.functional.pad(mask, pad_f)

    # Register the images
    all_tx = []
    all_ty = []
    all_tz = []
    all_phi = []
    all_psi = []
    all_theta = []
    all_images = [fixed_image.detach().cpu().numpy(),]
    all_moving = [fixed_image.detach().cpu().numpy(),]
    all_loss = []
    all_tx.append(model.tx.detach().cpu().numpy())
    all_ty.append(model.ty.detach().cpu().numpy())
    all_tz.append(model.tz.detach().cpu().numpy())
    all_phi.append(model.phi.detach().cpu().numpy())
    all_psi.append(model.psi.detach().cpu().numpy())
    all_theta.append(model.theta.detach().cpu().numpy())

    for idx in range(1, images.shape[0]):

        print(f'Image {idx} of {images.shape[0]}')

        moving_image = torch.tensor( images[idx] ).to('cuda')
        #print(f'Max moving = {torch.max( moving_image)}')
        moving_image /= torch.max( moving_image)
        moving_image = moving_image.view(-1, 1, moving_image.shape[-3], moving_image.shape[-2], moving_image.shape[-1])

        moving_image = nn.functional.pad(moving_image, pad_f)

        # Get optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        #loss_func = torch.nn.MSELoss()
        loss_func = weighted_mse_loss

        loss_monitor = []
        loss_thresh = 1e-12
        #loss_thresh = 1e-1 # Temp

        loss_window = 20
        #loss_window = 2 # temp

        # Grab the images
        model.train()
        for epoch in range(0, 3000):
            optimizer.zero_grad()

            registered = model(moving_image)
            loss = loss_func( registered, fixed_image, mask)

            # compute gradients and update parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            if epoch ==0:
                loss0 = loss.item()
            loss_monitor.append(loss.item()/(loss0+1e-9))
            if epoch > loss_window:
                dloss = loss_monitor[-loss_window] - loss_monitor[-1]
                if dloss < loss_thresh:
                    break

        print(f'Stop reg : epoch = {epoch} loss = {loss_monitor[-1]}')

        all_images.append( registered.detach().cpu().numpy())
        all_moving.append( moving_image.detach().cpu().numpy())
        all_tx.append(model.tx.detach().cpu().numpy())
        all_ty.append(model.ty.detach().cpu().numpy())
        all_tz.append(model.tz.detach().cpu().numpy())
        all_phi.append(model.phi.detach().cpu().numpy())
        all_psi.append(model.psi.detach().cpu().numpy())
        all_theta.append(model.theta.detach().cpu().numpy())
        all_loss.append( loss.item())

    # Export to file
    out_name = os.path.join(logdir, 'RegisteredImages.h5')
    print('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        if out_reg_images:
            hf.create_dataset("REGISTERED", data=np.squeeze(np.stack(all_images)))
            hf.create_dataset("MOVING", data=np.squeeze(np.stack(all_moving)))
        hf.create_dataset("phi", data=180.0/math.pi*np.squeeze(np.array(all_phi)))
        hf.create_dataset("psi", data=180.0 / math.pi * np.squeeze(np.array(all_psi)))
        hf.create_dataset("theta", data=180.0 / math.pi * np.squeeze(np.array(all_theta)))
        hf.create_dataset("tx", data=np.squeeze(np.array(all_tx)) * images.shape[-1]/2.0)
        hf.create_dataset("ty", data=np.squeeze(np.array(all_ty)) * images.shape[-2]/2.0)
        hf.create_dataset("tz", data=np.squeeze(np.array(all_tz)) * images.shape[-3]/2.0)

    # Cast to np array instead of list
    all_tx = np.squeeze(np.array(all_tx))
    all_ty = np.squeeze(np.array(all_ty))
    all_tz = np.squeeze(np.array(all_tz))
    all_phi = np.squeeze(np.array(all_phi))
    all_psi = np.squeeze(np.array(all_psi))
    all_theta = np.squeeze(np.array(all_theta))

    return all_tx, all_ty, all_tz, all_phi, all_psi, all_theta

def correct_MRI_Raw( file_data, tx, ty, tz, phi, psi, theta, out_folder):
    r"""This perfroms rigid motion correction of k-space data using the

    Args:
        file_data (str): name of the MRI file to load
        tx(array): translation in x, in fraction of FOV [-1,1]
        ty(array): translation in y, in fraction of FOV [-1,1]
        tz(array): translation in z, in fraction of FOV [-1,1]
        phi(array): rotation angle [rad]
        psi(array): rotation angle [rad]
        theta(array): rotation angle [rad]

    Returns:
        mri_raw(class): contains to corrected MRI data
    """

    num_frames = len(tx)

    # Before doing the correction we will calculate the transform that would average things.
    R = []
    for t in range(num_frames):
        rot = build_rotation(theta=theta[t], phi=phi[t], psi=psi[t])
        R.append(rot)
    R = np.stack(R,axis=0)

    # This rotation will move the rotation to be as close to identity as possible
    inverse_common_rotation = average_rotation(R)

    # Get the frame center
    mri_raw = load_MRI_raw(h5_filename=file_data)

    gate_signal = mri_raw.time
    gate_type = 'time'
    discrete_gates = False

    # Get the gate positions
    t_min, t_max, delta_time = get_gate_bins(gate_signal, gate_type, num_frames, discrete_gates)

    device = sp.get_device(mri_raw.kdata[0])
    xp = sp.get_device(mri_raw.kdata[0]).xp

    # Now do the correction
    for e in range(mri_raw.Num_Encodings):
        for t in range(num_frames):
            t_start = t_min + delta_time * t
            t_stop = t_start + delta_time

            # Find index where value is held
            idx = np.argwhere(np.logical_and.reduce([
                np.abs(gate_signal[e]) >= t_start,
                np.abs(gate_signal[e]) < t_stop]))
            current_points = len(idx)
            print(f'Current points = {current_points}')

            # Get the signal
            coord = mri_raw.coords[e][idx[:, 0], :]
            kdata = mri_raw.kdata[e][:, idx[:, 0]]

            # Correct the tranlation
            kdata *= xp.exp(xp.array(1j * math.pi * tz[t]) * coord[..., 0])
            kdata *= xp.exp(xp.array(1j * math.pi * ty[t]) * coord[..., 1])
            kdata *= xp.exp(xp.array(1j * math.pi * tx[t]) * coord[..., 2])

            # Build Rotation matrix
            rot = build_rotation(theta=theta[t], phi=phi[t], psi=psi[t])
            rot = inverse_common_rotation @ rot
            rot = sp.to_device(rot, device)
            rot = np.linalg.inv(rot)
            coord_rot = device.xp.flip(coord, axis=-1)  # Swap z/x
            # coord_rot = coord
            coord_rot = device.xp.expand_dims(coord_rot, -1)
            coord_rot = device.xp.matmul(rot, coord_rot)
            coord_rot = device.xp.squeeze(coord_rot)
            coord_rot = device.xp.flip(coord_rot, axis=-1)  # Swap z/x

            # Store in MRI Raw again in case moved to GPU
            mri_raw.coords[e][idx[:, 0], :] = coord_rot
            mri_raw.kdata[e][:, idx[:, 0]] = kdata

    return mri_raw


if __name__ == '__main__':

    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_nav', type=str, help='filename for data (e.g. FullRecon.h5)', default='MRI_Raw.h5')
    parser.add_argument('--file_data', type=str, help='filename for data (e.g. FullRecon.h5)', default='MRI_Raw_Corrected.h5')
    parser.add_argument('--logdir', type=str, help='folder to log files to, default is current directory', default=None)
    parser.add_argument('--out_folder', type=str, default=None)
    parser.add_argument('--out_filename', type=str, default='MRI_Raw_Corrected.h5')
    parser.add_argument('--out_reg_images', dest='out_reg_images', action='store_true')
    parser.set_defaults(out_reg_images=True)

    args = parser.parse_args()

    if args.out_folder is None:
        args.out_folder = os.path.dirname(args.file_nav)

    if args.logdir is None:
        args.logdir = os.path.dirname(args.file_nav)

    # Load 4D images
    images = load_images(args.file_nav)

    # Estimate a mask
    mask = estimate_mask(images)

    # Register the images
    tx, ty, tz, phi, psi, theta = register_images(images, mask, logdir=args.logdir, out_reg_images=args.out_reg_images)

    # Now load and correct the data
    mri_raw = correct_MRI_Raw(args.file_data, tx, ty, tz, phi, psi, theta, args.out_folder)

    # Save to corrected version
    out_name = os.path.join(args.out_folder, args.out_filename)
    print(f'Saving corrected data to {out_name}')
    save_MRI_raw(mri_raw, h5_filename=out_name)


