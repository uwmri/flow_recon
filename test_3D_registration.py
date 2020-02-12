import os
import ctypes
import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import argparse
import matplotlib.pyplot as plt
import cupy
import time
import math
import scipy.ndimage as ndimage
import scipy
from scipy.spatial.transform import Rotation as R


def get_qrotation(params):


    r = R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])


def get_rotation(shifts=[0, 0, 0], rotation=[0, 0, 0], device=sp.cpu_device):
    xp = device.xp
    #print(shifts)
    #print(rotation)

    # Stack is used due to a bug in cupy
    shifts = xp.stack(shifts).astype(np.float64)
    rotation = xp.stack(rotation).astype(np.float64)
    #print(rotation)
    #print(shifts)

    translation = device.xp.array([[1.0, 0.0, 0.0, shifts[0]],
                                   [0.0, 1.0, 0.0, shifts[1]],
                                   [0.0, 0.0, 1.0, shifts[2]],
                                   [0.0, 0.0, 0.0, 1.0]], np.float64)

    c = xp.cos(rotation[0])
    s = xp.sin(rotation[0])
    rot1 = device.xp.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, c, s, 0.0],
                            [0.0, -s, c, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], np.float64)

    c = xp.cos(rotation[1])
    s = xp.sin(rotation[1])
    rot2 = device.xp.array([[c, 0.0, s, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [-s, 0.0, c, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], np.float64)

    c = xp.cos(rotation[2])
    s = xp.sin(rotation[2])
    rot3 = device.xp.array([[c, s, 0.0, 0.0],
                            [-s, c, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], np.float64)
    translation = translation @ rot1 @ rot2 @ rot3

    #print(f'R1 = {rot1}')
    #print(f'R2 = {rot2}')
    #print(f'R3 = {rot3}')

    return translation.astype(np.float32)


def image_gradients( image=None ):
    xp = sp.get_device(image).xp

    gradx = xp.roll(image,shift=1, axis=2) - xp.roll(image,shift=-1, axis=2)
    grady = xp.roll(image,shift=1, axis=1) - xp.roll(image,shift=-1, axis=1)
    gradz = xp.roll(image,shift=1, axis=0) - xp.roll(image,shift=-1, axis=0)

    gradx = xp.expand_dims(gradx, -1)
    grady = xp.expand_dims(grady, -1)
    gradz = xp.expand_dims(gradz, -1)

    grad = xp.stack( [gradz, grady, gradx], axis=-1)

    return grad


def pad_cube(image, extra_padding=0):
    device = sp.get_device(image)
    xp = device.xp

    padz = max(image.shape) - image.shape[0]
    pady = max(image.shape) - image.shape[1]
    padx = max(image.shape) - image.shape[2]
    padx = (padx // 2) + extra_padding
    pady = (pady // 2) + extra_padding
    padz = (padz // 2) + extra_padding

    if len(image.shape) == 3:
        padded_image = xp.pad(image, pad_width=[(padz, padz), (pady, pady), (padx, padx)], mode='constant',
                              constant_values=0)
    else:
        padded_image = xp.pad(image, pad_width=[(padz, padz), (pady, pady), (padx, padx), (0, 0)], mode='constant',
                              constant_values=0)

    return padded_image

def rigid_registration( moving=None, fixed=None):

    xp = sp.get_device(moving).xp

    # Now the Jacobian
    jacobian = xp.zeros(moving.shape + (3, 6), moving.dtype)

    coord, center = get_coord(moving)

    x = coord[..., 2, 0]
    y = coord[..., 1, 0]
    z = coord[..., 0, 0]

    a = xp.array(0.0, np.float64)
    b = xp.array(0.0, np.float64)
    c = xp.array(0.0, np.float64)
    sx = xp.array(0.0, np.float64)
    sy = xp.array(0.0, np.float64)
    sz = xp.array(0.0, np.float64)

    moved = xp.copy(moving)

    print(sp.get_device(coord))
    print(sp.get_device(moving))
    print(sp.get_device(jacobian))

    for iter in range(1000):

        # dxp.cost / dI
        diff = moved - fixed
        diff = xp.expand_dims(diff, -1)
        diff = xp.expand_dims(diff, -1)

        print(f'Cost = {xp.sum(diff**2)} \n\tAngles={a},{b},{c} \n\tShift={sx},{sy},{sz}')

        # gradient dI / dx
        grad = image_gradients(moved)

        # Just to make it a bit more readable
        jacobian[:, :, :, 0, 0] = 0.0
        jacobian[:, :, :, 0, 1] = z*xp.cos(b) - x*(xp.cos(c)*xp.sin(b)) - y*(xp.sin(b)*xp.sin(c))
        jacobian[:, :, :, 0, 2] = y*(xp.cos(b)*xp.cos(c)) - x*(xp.cos(b)*xp.sin(c))
        jacobian[:, :, :, 0, 3] = 1.0
        #jacobian[:, :, :, 0, 4] = 0.0
        #jacobian[:, :, :, 0, 5] = 0.0

        jacobian[:, :, :, 1, 0] = z*(xp.cos(a)*xp.cos(b)) \
                                  + y*( -xp.cos(c)*xp.sin(a) - xp.cos(a)*xp.sin(b)*xp.sin(c)) \
                                  + x*( -xp.cos(a)*xp.cos(c)*xp.sin(b) +  xp.sin(a)*xp.sin(c))
        jacobian[:, :, :, 1, 1] = -x*(xp.cos(b)*xp.cos(c)*xp.sin(a)) - z*(xp.sin(a)*xp.sin(b)) \
                                  - y*(xp.cos(b)*xp.sin(a)*xp.sin(c))
        jacobian[:, :, :, 1, 2] = x*( -xp.cos(a)*xp.cos(c) + xp.sin(a)*xp.sin(b)*xp.sin(c)) \
                                  -y*( xp.cos(c)*xp.sin(a)*xp.sin(b) + xp.cos(a)*xp.sin(c))
        #jacobian[:, :, :, 1, 3] = 0.0
        jacobian[:, :, :, 1, 4] = 1.0
        #jacobian[:, :, :, 1, 5] = 0.0

        jacobian[:, :, :, 2, 0] = - z*xp.cos(b)*xp.sin(a) \
                                  + y*(-xp.cos(a)*xp.cos(c) + xp.sin(a)*xp.sin(b)*xp.sin(c))  \
                                  + x*(xp.cos(c)*xp.sin(a)*xp.sin(b) + xp.cos(a)*xp.sin(c))
        jacobian[:, :, :, 2, 1] = -x*(xp.cos(a)*xp.cos(b)*xp.cos(c)) - z*(xp.cos(a)*xp.sin(b)) \
                                  - y*(xp.cos(a)*xp.cos(b)*xp.sin(c))
        jacobian[:, :, :, 2, 2] = x*(xp.cos(c)*xp.sin(a)  + xp.cos(a)*xp.sin(b)*xp.sin(c)) \
                                  + y*(-xp.cos(a)*xp.cos(c)*xp.sin(b) + xp.sin(a)*xp.sin(c))
        #jacobian[:, :, :, 2, 3] = 0.0
        #jacobian[:, :, :, 2, 4] = 0.0
        jacobian[:, :, :, 2, 5] = 1.0

        #print(f' Diff Shape = {diff.shape}')
        #print(f' Grad Shape = {grad.shape}')
        #print(f' Jacobian shape = {jacobian.shape}')

        gradient = diff @ grad @ jacobian

        grad_p = xp.squeeze(xp.mean( gradient, axis=(0,1,2) ))
        if iter == 0:
            scale = 40*xp.max(xp.abs(grad_p))
        grad_p /= scale

        a += grad_p[0]
        #b += grad_p[1]
        c += grad_p[2]
        #sx += grad_p[3]
        #sy += grad_p[4]
        #sz += grad_p[5]

        tmat = get_rotation(shifts=[sx, sy, sz], rotation=[a, b, c], device=sp.get_device(moving))
        tmat = sp.to_device(tmat, device=sp.get_device(moving))

        moved = rigid_transform(moving, coord=coord, transform_matrix=tmat, center=center)

        do_plot = True
        if do_plot:
            moving_cpu = sp.to_device(moved, sp.cpu_device)
            fixed_cpu = sp.to_device(fixed, sp.cpu_device)
            plt.figure()
            plt.imshow(moving_cpu[64,:,:]-fixed_cpu[64,:,:],cmap='gray')
            plt.show()

        #print(grad_p)


    print(gradient.shape)

    return gradient




def rigid_transform(image=None, coord=None, transform_matrix=None, center=None):
    device = sp.get_device(image)
    xp = device.xp

    coord = sp.to_device(coord, device=device)
    transform_matrix = sp.to_device(transform_matrix, device=device)

    new_coord = xp.matmul(transform_matrix, coord)
    new_coord = xp.squeeze(new_coord)
    new_coord[:, :, :, 0] += center[0]
    new_coord[:, :, :, 1] += center[1]
    new_coord[:, :, :, 2] += center[2]
    new_coord = new_coord[:,:,:,:-1]

    ## Cubic interpolation kernel
    width = 2
    kernel = xp.linspace(1.0, 0.0, 400, dtype=xp.float32)

    #xpos = xp.linspace(0.0, 2.0, 1000, dtype=xp.float32)
    #width = 4
    #kernel = 1.5 * (xpos ** 3) - 2.5 * (xpos ** 2) + 1.0
    #mask = xpos >= 1.0
    #kernel[mask] = -0.5 * (xpos[mask] ** 3) + 2.5 * (xpos[mask] ** 2) - 4 * xpos[mask] + 2
    #kernel[-1] = 0.0

    #print(sp.get_device(image))
    #print(sp.get_device(kernel))
    #print(sp.get_device(new_coord))
    #print(sp.get_device(kernel))

    # Actual rotation
    moved = sp.interpolate(image, kernel=kernel, width=width, coord=new_coord)

    return moved


def get_coord(image):
    xp = sp.get_device(image).xp

    # Rotate the image coordinates
    center = xp.array(image.shape, dtype=xp.float32) / 2
    z, y, x = xp.meshgrid(xp.arange(0, image.shape[0], dtype=xp.float32),
                          xp.arange(0, image.shape[1], dtype=xp.float32),
                          xp.arange(0, image.shape[2], dtype=xp.float32), indexing='ij')
    x -= center[0]
    y -= center[1]
    z -= center[2]

    # Rotate the coordinates
    shift_col = xp.ones_like(z)
    coord = xp.stack((z, y, x, shift_col), axis=-1)
    coord = xp.expand_dims(coord, -1)

    return coord, center

def main():

    # Make phanto
    N = 128
    moving = sp.shepp_logan([N, N, N], dtype=np.float32)
    fixed = sp.shepp_logan([N, N, N], dtype=np.float32)

    #moving = sp.convolve(moving,np.ones((12,12,12)))
    #fixed = sp.convolve(fixed, np.ones((12, 12, 12)))


    moving = pad_cube(moving, extra_padding=16)
    fixed = pad_cube(fixed, extra_padding=16)

    device = sp.Device(0)
    moving = sp.to_device(moving, device)
    fixed = sp.to_device(fixed, device)

    coord, center = get_coord(moving)

    #
    tmat = get_rotation(shifts=[0, 0, 0], rotation=[0.1, 0.0, 0.1])
    moved = rigid_transform(image=moving, coord=coord, transform_matrix=tmat, center=center)
    tmatF = get_rotation(shifts=[0, 0, 0], rotation=[-0.1, -0.0, -0.1])
    moved_back = rigid_transform(image=moved, coord=coord, transform_matrix=tmatF, center=center)
    tmatI = np.linalg.inv(tmat)
    moved_backI = rigid_transform(image=moved, coord=coord, transform_matrix=tmatI, center=center)


    print(tmat)
    print(tmatF)
    print(tmatI)

    xp = sp.get_device(moved).xp
    print(f'Initial = {xp.sum((moved-fixed)**2)}')
    print(f'Back = {xp.sum((moved_back - fixed) ** 2)}')
    print(f'BackI = {xp.sum((moved_backI - fixed) ** 2)}')

    '''
    moved = sp.to_device(moved,sp.cpu_device)
    moved_back = sp.to_device(moved_back, sp.cpu_device)
    fixed = sp.to_device(fixed,sp.cpu_device)
    plt.figure()
    plt.subplot(121)
    plt.imshow(np.abs(moved[64, :, :]-fixed[64, :, :]))
    plt.subplot(122)
    plt.imshow(np.abs(moved_back[64, :, :]-fixed[64, :, :]))
    plt.show()
    '''

    rigid_registration( moved, fixed)


if __name__ == "__main__":
    main()
