#! /usr/bin/env python
import numpy as np
import h5py
import sigpy.mri as mr
import logging
import sigpy as sp
import os
import argparse
#import matplotlib.pyplot as plt
import cupy
import numba as nb
import time
import math
import scipy.ndimage as ndimage


@nb.vectorize  # pragma: no cover
def _thresh_to_binary(lamda, input):
    abs_input = abs(input)
    if abs_input > lamda:
        return 1.0
    else:
        return 0.0


def thresh_to_binary(lamda, input):
    r"""Thresh to binary

    Args:
        lamda (float, or array): Threshold parameter.
        input (array)

    Returns:
        array: soft-thresholded result.

    """
    device = sp.backend.get_device(input)
    xp = device.xp

    lamda = xp.real(lamda)
    with device:
        if device == sp.backend.cpu_device:
            output = _thresh_to_binary(lamda, input)
        else:  # pragma: no cover
            output = _thresh_to_binary_cuda(lamda, input)

        if np.issubdtype(input.dtype, np.floating):
            output = xp.real(output)

    return output


class MRI_Raw:
    Num_Encodings = 0
    Num_Coils = 0
    trajectory_type = None
    dft_needed = None
    Num_Frames = None
    coords = None
    time = None
    dcf = None
    kdata = None
    target_image_size = [256, 256, 64]

class SingularValueThresholding(sp.prox.Prox):

    def __init__(self, ishape, frames, num_encodes, lamda=None, block_size=8, block_stride=8, axis=0):

        self.frames = frames
        self.num_encodes = num_encodes
        self.total_images = frames*num_encodes
        self.axis = axis
        self.old_shape = np.array(ishape)
        self.new_shape = (self.total_images, int(self.old_shape[0] / self.total_images))  + tuple(self.old_shape[1:])

        self.lamda = lamda
        self.block_size = [block_size, block_size, block_size]
        self.block_stride = [block_stride, block_stride, block_stride]
        self.block_shape = (num_encodes, block_size, block_size, block_size)

        print(f'Old shape = {self.old_shape}')
        print(f'New shape = {self.new_shape}')
        print(f'Block size = {self.block_size}')
        print(f'Block stride = {self.block_stride}')
        print(f'Block shape = {self.block_shape}')

        # input is output
        oshape = ishape

        super().__init__(oshape, ishape)

    def _prox(self, alpha, input):

        if math.isclose( self.lamda, 0.0):
            return input

        # Save input device
        initial_device = sp.get_device(input)

        # Input is 3D (Nt*Nz x Ny x Nx)
        print(input.shape)
        input = initial_device.xp.reshape(input, self.new_shape)

        # Block shifts are always negative
        block_shift = [-np.random.randint(0, self.block_stride[e]) for e in range(3)]
        block_ishift = [ -x for x in block_shift]
        input = initial_device.xp.roll(input,block_shift,axis=(-3,-2,-1))

        # Put on CPU
        input = sp.to_device(input, sp.cpu_device)

        # Shifts including hanging blocks
        print(input.shape)
        print(f'Block shift = {block_shift}')
        print(f'Stride = {self.block_stride}')
        print(f'Block size = {self.block_size}')
        print(f'Block shape = {self.block_shape}')

        nshifts = np.ceil((np.asarray(input.shape[1:]) - np.array(block_shift)) / self.block_stride).astype(np.int)

        print(input.shape)
        print(f'Nshifts = {nshifts}')

        t = time.time()
        input = self._svt_thresh_batched(x=input,
                            block_size=self.block_size,
                            block_shape=self.block_shape,
                            block_stride=self.block_stride,
                            block_shift=block_shift,
                            lamda=float(alpha*self.lamda))
        print(f'SVT took {time.time() - t}')

        # Return on same device
        input = sp.to_device(input, initial_device)
        input = initial_device.xp.roll(input, block_ishift, axis=(-3, -2, -1))
        input = initial_device.xp.reshape(input, self.old_shape)

        return(input)

    def _svt_thresh_batched(self, x, block_size, block_shape, block_stride, block_shift, lamda):
        print(f'Block shift = {block_shift}')
        print(f'Stride = {block_stride}')
        print(f'Block size = {block_size}')
        print(f'Block shape = {block_shape}')

        full_block_shift = list([0]) + block_shift
        full_block_stride = list([block_shape[0]] + block_stride)
        print(f'Full stride = {full_block_stride}')

        # 4D Blocking
        B = sp.linop.ArrayToBlocks(list(x.shape), list(block_shape), list(full_block_stride))
        print(B)

        # x = np.roll( x, shift=self.blk_shift, axis=(0,1,2))

        # Parse into blocks
        image = B * x
        print(image.shape)

        # reshape to (Nblocks, encode, prod(block_size) )
        old_shape = image.shape
        image = np.moveaxis(image,0,-1) # First axis is time
        new_shape = (-1, np.prod(block_shape),image.shape[-1])
        print(f'Resize from {old_shape} to {new_shape}')

        image = np.reshape(image, new_shape)

        nuclear_norm = 0.0

        lr_batch_size = 32
        lr_batchs = (image.shape[0] + lr_batch_size - 1) // lr_batch_size
        for batch in range(lr_batchs):
            start = batch * lr_batch_size
            stop = min((batch + 1) * lr_batch_size, image.shape[0])

            image_t = image[start:stop, :, :]

            u, s, vh = np.linalg.svd(image_t, full_matrices=False)

            nuclear_norm += np.sum(np.abs(s))

            # Threshold
            s = sp.soft_thresh(lamda, s)

            image[start:stop, :, :] = np.matmul(u * s[..., None, :], vh)

        # Back to GPU
        image = np.moveaxis(image,-1,0)
        image = np.reshape(image, newshape=old_shape)

        x = B.H * image

        print(f'Nuclear norm = {nuclear_norm}')

        return x

class DiagOnDevice(sp.linop.Diag):
    """Diagonally stack linear operators.

    Create a Linop that splits input, applies linops independently,
    and concatenates outputs. This is a special case in which the devices to run are specified by the operator
    rather than the input.
    In matrix form, given matrices {A1, ..., An}, returns diag([A1, ..., An]).

    Args:
        linops (list of Linops): list of linops with the same input and
            output shape.
        axis (int or None): If None, inputs/outputs are vectorized
            and concatenated.
        run_device: an sp.Device to run the operation on
        in_device: an sp.Device specifiying the device location of the input
        out_device: an sp.Devie specifying the device the output should be stored

    """

    def __init__(self, linops, axis=None, run_device=sp.cpu_device,in_device=sp.cpu_device,out_device=sp.cpu_device):
        self.run_device = run_device
        self.in_device = in_device
        self.out_device = out_device

        super().__init__(linops, axis=axis)

    def _apply(self, input):

        #Allocate space for output
        output = self.out_device.xp.empty(self.oshape, dtype=input.dtype)

        for n, linop in enumerate(self.linops):
            if n == 0:
                istart = 0
                ostart = 0
            else:
                istart = self.iindices[n - 1]
                ostart = self.oindices[n - 1]

            if n == self.nops - 1:
                iend = None
                oend = None
            else:
                iend = self.iindices[n]
                oend = self.oindices[n]

            if self.axis is None:
                op_input =input[istart:iend].reshape(linop.ishape)
                op_input = sp.to_device(op_input, self.run_device)
                output[ostart:oend] = sp.to_device(linop(op_input).ravel(), self.out_device)
            else:
                ndim = len(linop.oshape)
                axis = self.axis % ndim
                oslc = tuple([slice(None)] * axis + [slice(ostart, oend)] +
                             [slice(None)] * (ndim - axis - 1))

                ndim = len(linop.ishape)
                axis = self.axis % ndim
                islc = tuple([slice(None)] * axis + [slice(istart, iend)] +
                             [slice(None)] * (ndim - axis - 1))

                op_input = input[islc]
                op_input = sp.to_device(op_input, self.run_device)
                output[oslc] = sp.to_device(linop(op_input), self.out_device)

        return output

    def _adjoint_linop(self):
        return DiagOnDevice([op.H for op in self.linops], axis=self.axis, run_device=self.run_device, in_device=self.out_device, out_device=self.in_device)

class SubtractArray(sp.linop.Linop):
    """Subtract array operator, subtracts a given array allowing composed operator

    Args:
        shape (tuple of ints): Input shape
        x: array to subtract from the input

    """

    def __init__(self, x):
        self.x = x
        super().__init__(x.shape, x.shape)

    def _apply(self, input):

        return(input-self.x)

    def _adjoint_linop(self):
        return self

class BatchedSenseRecon(sp.app.LinearLeastSquares):
    r"""SENSE Reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 +
        \frac{\lambda}{2} \| x \|_2^2

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        coord (None or array): coordinates.
        device (Device): device to perform reconstruction.
        coil_batch_size (int): batch size to process coils.
            Only affects memory usage.
        comm (Communicator): communicator for distributed computing.
        **kwargs: Other optional arguments.

    References:
        Pruessmann, K. P., Weiger, M., Scheidegger, M. B., & Boesiger, P.
        (1999).
        SENSE: sensitivity encoding for fast MRI.
        Magnetic resonance in medicine, 42(5), 952-962.

        Pruessmann, K. P., Weiger, M., Bornert, P., & Boesiger, P. (2001).
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magnetic resonance in medicine, 46(4), 638-651.

    """

    def __init__(self, y, mps, lamda=0, weights=None,
                 coord=None, device=sp.cpu_device, coil_batch_size=None,
                 comm=None, show_pbar=True, max_power_iter=50, **kwargs):

        # Temp
        self.frames = y.shape[0]//5
        self.num_encodes = 5
        self.num_images = self.frames*self.num_encodes
        self.cpu_device = sp.cpu_device
        self.gpu_device= sp.Device(0)
        self.max_power_iter = max_power_iter
        self.show_pbar = True
        self.log_images = True
        self.log_out_name = 'ReconLog.h5'

        if self.log_images:
            # Export to file
            logger.info('Logging images to ' + self.log_out_name)
            try:
                os.remove(self.log_out_name)
            except OSError:
                pass

        # put coord and mps to gpu
        mps = sp.to_device(mps, sp.Device(0))
        coord = sp.to_device(coord, sp.Device(0))
        weights = sp.to_device(weights,sp.Device(0))

        # Get A operator to return on GPU
        ops_list = [sp.mri.linop.Sense(mps, coord[e, ...], weights[e, ...], ishape=None,
                                             coil_batch_size=coil_batch_size, comm=comm) for e in
                          range(self.num_images)]
        # Run on GPU
        AHA = DiagOnDevice([ op.H*op for op in ops_list], axis=0, run_device=sp.Device(0), in_device=self.gpu_device,
                         out_device=self.gpu_device)
        max_eig = sp.app.MaxEig(AHA, dtype=y.dtype, device=self.gpu_device,
                             max_iter=self.max_power_iter,
                             show_pbar=self.show_pbar).run()

        # Scale the weights, for the max eigen value is one
        with sp.get_device(weights):
            weights *= 1.0/max_eig

        y = sp.to_device(y, self.gpu_device)
        if weights is not None:
            for e in range(self.num_images):
                y[e,...] *= weights[e,...]**0.5

        # Update ops list with weights
        ops_list = [sp.mri.linop.Sense(mps, coord[e, ...], weights[e, ...], ishape=None,
                                             coil_batch_size=coil_batch_size, comm=comm) for e in
                          range(self.num_images)]

        print(y.shape)
        print(ops_list[0].oshape)

        sub_list = [ SubtractArray(y[e,...]) for e in range(self.num_images)]
        grad_ops = [ ops_list[e].H * sub_list[e] *ops_list[e] for e in range(len(ops_list))]
        print(grad_ops)

        # Get AHA opts list
        A = DiagOnDevice(grad_ops, axis=0, run_device=sp.Device(0), in_device=sp.cpu_device,
                         out_device=sp.cpu_device)
        print(A)

        proxg = SingularValueThresholding(A.ishape, frames=self.frames, num_encodes=self.num_encodes, lamda=lamda, block_size=16, block_stride=16)

        # Put x on CPU
        x = self.cpu_device.xp.zeros(A.ishape, dtype=y.dtype)
        self.x = x

        # Reshape Y to flatten axis 1
        #new_shape = (y.shape[0]*y.shape[1],1) + y.shape[2:]
        #y = sp.get_device(y).xp.reshape(y,new_shape)

        print(y.shape)
        print(A)
        print(A.H)
        print(x.shape)

        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, x=x, proxg=proxg, show_pbar=show_pbar, alpha=1.0, accelerate=True, **kwargs)

    def _summarize(self):

        if self.log_images:
            self._write_log()

        super()._summarize()


    def _write_log(self):

        print(f'Logging to file {self.x.shape}')
        xp = sp.get_device(self.x).xp
        out_slice = (self.x.shape[0] / self.frames ) // 2
        out_slice = int(40)
        Xiter = xp.copy( self.x[out_slice,...])
        Xiter = sp.to_device(Xiter, sp.cpu_device)
        Xiter = np.squeeze(Xiter)
        Xiter = np.expand_dims(Xiter, axis=0)

        with h5py.File(self.log_out_name, 'a') as hf:

            # Resize to include additional image
            if "X" in hf.keys():
                hf["X"].resize((hf["X"].shape[0] + Xiter.shape[0]), axis=0)
                hf["X"][-Xiter.shape[0]:]  = np.abs(Xiter)
            else:
                maxshape=np.array(Xiter.shape)
                maxshape[0] *= self.max_iter
                maxshape = tuple(maxshape)
                print(maxshape)
                hf.create_dataset("X", data=np.abs(Xiter), maxshape=maxshape)


    def _get_GradientMethod(self):
        print(f'Using defined alg {self.alpha}')
        self.alg = sp.app.GradientMethod(
            self.A,
            self.x,
            self.alpha,
            proxg=self.proxg,
            max_iter=self.max_iter,
            accelerate=self.accelerate)


    def _output(self):
        self.x = sp.to_device(self.x, sp.cpu_device)
        self.x = np.reshape(self.x, (self.frames, self.num_encodes,-1) + self.x.shape[1:] )
        return self.x

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
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1-0.05])

    # Create a subsampled array
    kcc = np.zeros( (old_channels, np.sum(mask)),dtype=kdata_cc.dtype)
    logger.info(f'Kcc Shape = {kcc.shape} ')
    for c in range(old_channels):
        ktemp = kdata_cc[...,c]
        kcc[c,:] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.info(f'Working on SVD of {kcc.shape}')
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.info(f'S = {s}')

    if isinstance(kdata, list):
        logger.info('Passed k-space is a list, using encode 0 for compression')

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e],-1)
            logger.info(f'Shape = {kdata[e].shape}')
            kdata[e]= np.matmul(u,kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e],-1,axis)

        for ksp in kdata:
            logger.info(f'Final Shape {ksp.shape}')
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata,axis,-1)
        kdata = np.expand_dims(kdata,-1)
        kdata = np.matmul(u, kdata)
        logger.info(f'Shape = {kdata.shape}')

        # Crop to target channels
        kdata = np.squeeze(kdata,axis=-1)
        kdata = kdata[...,:target_channels]

        # Put back
        kdata = np.moveaxis(kdata,-1,axis)
        logger.info(f'Final shape = {kdata.shape}')

    return kdata

def get_smaps(mri_rawdata=None, args=None, smap_type='jsense'):
    logger = logging.getLogger('Get sensitivity maps')

    # Set to GPU
    device = sp.Device(0)
    xp = device.xp

    with device:

        # Reference for shortcut
        coord = mri_rawdata.coords[0]
        dcf = mri_rawdata.dcf[0]
        kdata = mri_rawdata.kdata[0]

        if smap_type == 'espirit':

            # Low resolution images
            res = 8
            lpf = np.sum(coord ** 2, axis=-1)
            lpf = np.exp(-lpf / (2.0 * res * res))

            img_shape = sp.estimate_shape(coord)
            ksp = xp.ones([mri_rawdata.Num_Coils] + img_shape, dtype=xp.complex64)

            for c in range(mri_rawdata.Num_Coils):
                logger.info(f'Reconstructing  coil {c}')
                ksp_t = np.copy(kdata[c, :, :, :])
                ksp_t *= np.squeeze(dcf)
                ksp_t *= np.squeeze(lpf)
                ksp_t = sp.to_device(ksp_t, device=device)
                ksp[c, :, :, :] = sp.nufft_adjoint(ksp_t, coord, img_shape)

            # Put onto CPU due to memory issues in ESPiRIT
            ksp = sp.to_device(ksp, sp.cpu_device)

            # Espirit Cal
            smaps = sp.mri.app.EspiritCalib(ksp, calib_width=24, thresh=0.001, kernel_width=5, crop=0.8, max_iter=3,
                                            device=sp.cpu_device, show_pbar=True).run()
        else:
            smaps = mr.app.JsenseRecon(kdata,
                                       coord=coord,
                                       weights=dcf,
                                       mps_ker_width=args.mps_ker_width,
                                       ksp_calib_width=args.ksp_calib_width,
                                       lamda=args.lamda,
                                       device=device,
                                       max_iter=args.max_iter,
                                       max_inner_iter=args.max_inner_iter).run()

            # Get a composite image
            img_shape = sp.estimate_shape(coord)
            image = 0
            for e in range(mri_rawdata.Num_Encodings):
                kr = sp.get_device(mri_rawdata.coords[e]).xp.sum(mri_rawdata.coords[e] ** 2, axis=-1)
                kr = sp.to_device(kr, device)
                lpf = xp.exp( -kr / (2*(16.**2)))

                for c in range(mri_rawdata.Num_Coils):
                    logger.info(f'Reconstructing encode, coil {e} , {c} ')
                    ksp = sp.to_device(mri_rawdata.kdata[e][c, ...], device)
                    ksp *= sp.to_device(mri_rawdata.dcf[e], device)
                    ksp *= lpf
                    coords_temp = sp.to_device(mri_rawdata.coords[e], device)
                    image += xp.abs(sp.nufft_adjoint(ksp, coords_temp, img_shape)) ** 2

            image = xp.sqrt(image)
            image = sp.to_device(image, sp.get_device(smaps))

            xp = sp.get_device(smaps).xp

            print(sp.get_device(smaps))
            print(sp.get_device(image))

            # Threshold
            image = xp.abs(image)
            image /= xp.max(image)
            thresh = 0.015
            print(thresh)
            mask = image > thresh

            mask = sp.to_device(mask, sp.cpu_device)
            zz, xx, yy = np.meshgrid( np.linspace(-1,1,11), np.linspace(-1,1,11), np.linspace(-1,1,11))
            rad = zz**2 + xx**2 + yy**2
            smap_mask = rad < 1.0
            print(smap_mask)
            mask = ndimage.morphology.binary_dilation(mask, smap_mask)
            mask = np.array( mask, dtype=np.float32)
            mask = sp.to_device(mask, sp.get_device(smaps))

            print(image)
            print(image.shape)
            print(smaps.shape)
            smaps = mask * smaps


    smaps_cpu= sp.to_device(smaps, sp.cpu_device)
    image_cpu = sp.to_device(image, sp.cpu_device)
    mask_cpu = sp.to_device(mask, sp.cpu_device)

    # Export to file
    out_name = 'SenseMaps.h5'
    logger.info('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=np.abs(image_cpu))
        hf.create_dataset("SMAPS", data=np.abs(smaps_cpu))
        hf.create_dataset("MASK", data=np.abs(mask_cpu))

    return smaps


def sos_recon(mri_rawdata=None, device=None):
    # Set to GPU
    if device is None:
        device = sp.Device(0)

    xp = device.xp
    logger = logging.getLogger('SOS Recon')
    img_shape = sp.estimate_shape(mri_rawdata.coords)

    img = 0
    for c in range(mri_rawdata.Num_Coils):
        logger.info(f'Reconstructing coil {c}')
        ksp = sp.to_device(mri_rawdata.kdata[c, ...], device)
        ksp *= sp.to_device(mri_rawdata.dcf, device)
        coords_temp = sp.to_device(mri_rawdata.coords, device)

        img += xp.abs(sp.nufft_adjoint(ksp, coords_temp, img_shape)) ** 2

    img = xp.sqrt(img)

    img = sp.to_device(img, sp.cpu_device)

    return img


def pils_recon(mri_rawdata=None, smaps=None):

    # Set to GPU
    device = sp.Device(0)

    # Reference for shortcut
    coord = mri_rawdata.coords
    dcf = mri_rawdata.dcf
    kdata = mri_rawdata.kdata

    with device:
        pils = sp.mri.app.SenseRecon(kdata, mps=smaps, weights=dcf, coord=coord, device=device, max_iter=1,
                                     coil_batch_size=1)
        img = pils.run()

    img = sp.to_device(img, sp.cpu_device)

    return (img)


def llr_recon(mri_rawdata=None, smaps=None):

    logger = logging.getLogger('Recon images')
    mempool = cupy.get_default_memory_pool()

    # Set to GPU
    device = sp.Device(0)

    logger.info(f'Memory used = {mempool.used_bytes()} of {mempool.total_bytes()}')

    # Reference for shortcut
    coord = mri_rawdata.coords
    dcf = mri_rawdata.dcf
    kdata = mri_rawdata.kdata
    lam = 0.0005
    #lam = 0.01
    #sense = BatchedSenseRecon(kdata, mps=smaps, weights=dcf, coord=coord, device=device, lamda=lam, coil_batch_size=None, max_iter=20)
    sense = BatchedSenseRecon(kdata, mps=smaps, weights=dcf, coord=coord, device=device, lamda=lam,
                              coil_batch_size=1, max_iter=200)

    img = sense.run()
    img = sp.to_device(img, sp.cpu_device)

    return (img)

def crop_kspace(mri_rawdata=None, crop_factor=2):

    logger = logging.getLogger('Recon images')

    # Get initial shape
    if isinstance(mri_rawdata.coords,list):
        img_shape = sp.estimate_shape(mri_rawdata.coords[0])
    else:
        img_shape = sp.estimate_shape(mri_rawdata.coords)

    # Crop kspace
    img_shape_new = np.floor( np.array(img_shape)/ crop_factor )
    logger.info(f'New shape = {img_shape_new}, Old shape = {img_shape}')

    for e in range(len(mri_rawdata.coords)):

        # Find values where kspace is within bounds (square crop)
        idx = np.argwhere( np.logical_and.reduce([
            np.abs(mri_rawdata.coords[e][...,0]) < img_shape_new[0]/2,
            np.abs(mri_rawdata.coords[e][...,1]) < img_shape_new[1]/2,
            np.abs(mri_rawdata.coords[e][...,2]) < img_shape_new[2]/2]))

        # Now crop
        mri_rawdata.coords[e] = mri_rawdata.coords[e][idx[:,0], :]
        mri_rawdata.dcf[e] = mri_rawdata.dcf[e][idx[:,0]]
        mri_rawdata.kdata[e] = mri_rawdata.kdata[e][:,idx[:, 0]]
        mri_rawdata.time[e] = mri_rawdata.time[e][idx[:,0]]

        print(mri_rawdata.coords[e].shape)
        print(mri_rawdata.dcf[e].shape)
        print(mri_rawdata.kdata[e].shape)
        print(f'New shape = {sp.estimate_shape(mri_rawdata.coords[e])}')


def gate_kspace(mri_rawdata=None, num_frames=10):

    logger = logging.getLogger('Gate k-space')

    # Assume the input is a list

    # Get the MRI Raw structure setup
    mri_rawG = MRI_Raw()
    mri_rawG.Num_Coils =  mri_raw.Num_Coils
    mri_rawG.Num_Encodings =  mri_raw.Num_Encodings*num_frames
    mri_rawG.dft_needed = mri_raw.dft_needed
    mri_rawG.trajectory_type = mri_raw.trajectory_type

    # List array
    mri_rawG.coords = []
    mri_rawG.dcf = []
    mri_rawG.kdata = []
    mri_rawG.time = []

    # Loop over all encodes
    t_min = np.min([np.min(mri_raw.time[e]) for e in range(mri_raw.Num_Encodings)])
    t_max = np.max([np.max(mri_raw.time[e]) for e in range(mri_raw.Num_Encodings)])

    logger.info(f'Max time = {t_max}')
    logger.info(f'Min time = {t_min}')

    delta_time = (t_max -t_min)/num_frames
    logger.info(f'Delta = {delta_time}')


    # Get the number of points per bin
    points_per_bin = []
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):

            t_start = t_min + delta_time*t
            t_stop = t_start + delta_time

            # Find index where value is held
            idx = np.argwhere(np.logical_and.reduce([
                np.abs(mri_rawdata.time[e]) >= t_start,
                np.abs(mri_rawdata.time[e]) < t_stop]))

            # Gate the data
            points_per_bin.append(len(idx))

    max_points_per_bin = np.max(np.array(points_per_bin))
    logger.info(f'Max points = {max_points_per_bin}')
    logger.info(f'Points per bin = {points_per_bin}')

    total_encodes = mri_raw.Num_Encodings*num_frames
    core_shape = (total_encodes,1,max_points_per_bin)

    # Append to list
    mri_rawG.coords = np.zeros( core_shape + (3,), dtype=np.float32)
    mri_rawG.dcf = np.zeros( core_shape, dtype=np.float32)
    mri_rawG.kdata = np.zeros( (total_encodes,mri_raw.Num_Coils,1,max_points_per_bin), dtype=np.complex64)
    mri_rawG.time = np.zeros( core_shape, dtype=np.float32)

    logger.info(f'New coords shape {mri_rawG.coords.shape}')
    logger.info(f'New dcf shape {mri_rawG.dcf.shape}')
    logger.info(f'New kdata shape {mri_rawG.kdata.shape}')
    logger.info(f'New time shape {mri_rawG.time.shape}')

    count = 0
    for t in range(num_frames):
        for e in range(mri_raw.Num_Encodings):

            t_start = t_min + delta_time*t
            t_stop = t_start + delta_time

            # Find index where value is held
            idx = np.argwhere(np.logical_and.reduce([
                np.abs(mri_rawdata.time[e]) >= t_start,
                np.abs(mri_rawdata.time[e]) < t_stop]))
            current_points = len(idx)

            mri_rawG.coords[count,0,:current_points,:] = mri_raw.coords[e][idx[:,0],:]
            mri_rawG.dcf[count,0,:current_points] = mri_raw.dcf[e][idx[:,0]]
            mri_rawG.kdata[count,:,:,:current_points] = mri_raw.kdata[e][:,np.newaxis,idx[:,0]]
            mri_rawG.time[count,0,:current_points] = mri_raw.time[e][idx[:,0]]
            count += 1
    return(mri_rawG)

def load_MRI_raw(h5_filename=None):
    with h5py.File(h5_filename, 'r') as hf:

        try:
            Num_Encodings = np.squeeze(hf['Kdata'].attrs['Num_Encodings'])
            Num_Coils = np.squeeze(hf['Kdata'].attrs['Num_Coils'])
            Num_Frames = np.squeeze(hf['Kdata'].attrs['Num_Frames'])

            trajectory_type = [np.squeeze(hf['Kdata'].attrs['trajectory_typeX']),
                               np.squeeze(hf['Kdata'].attrs['trajectory_typeY']),
                               np.squeeze(hf['Kdata'].attrs['trajectory_typeZ'])]

            dft_needed = [np.squeeze(hf['Kdata'].attrs['dft_neededX']), np.squeeze(hf['Kdata'].attrs['dft_neededY']),
                          np.squeeze(hf['Kdata'].attrs['dft_neededZ'])]

            logging.info(f'Frames {Num_Frames}')
            logging.info(f'Coils {Num_Coils}')
            logging.info(f'Encodings {Num_Encodings}')
            logging.info(f'Trajectory Type {trajectory_type}')
            logging.info(f'DFT Needed {dft_needed}')

        except Exception:
            logging.info('Missing header data')
            pass

        #Num_Coils = 2

        # Get the MRI Raw structure setup
        mri_raw = MRI_Raw()
        mri_raw.Num_Coils = int(Num_Coils)
        mri_raw.Num_Encodings = int(Num_Encodings)
        mri_raw.dft_needed = tuple(dft_needed)
        mri_raw.trajectory_type = tuple(trajectory_type)

        # List array
        mri_raw.coords = []
        mri_raw.dcf = []
        mri_raw.kdata = []
        mri_raw.time = []

        for encode in range(Num_Encodings):

            logging.info(f'Loading encode {encode}')

            # Get the coordinates
            coord = []
            for i in ['Z', 'Y', 'X']:
                logging.info(f'Loading {i} coord.')
                coord.append(np.array(hf['Kdata'][f'K{i}_E{encode}']).flatten())
            coord = np.stack(coord, axis=-1)

            dcf = np.array(hf['Kdata'][f'KW_E{encode}'])
            try:
                time_readout = np.array(hf['Gating']['time'])
            except Exception:
                time_readout = np.array(hf['Gating'][f'TIME_E{encode}'])
            time_readout = np.expand_dims(time_readout,-1)
            time = np.tile(time_readout,(1,1,dcf.shape[2]))

            dcf = dcf.flatten()
            time = time.flatten()

            # Get k-space
            ksp = []
            for c in range(Num_Coils):
                logging.info(f'Loading kspace, coil {c + 1} / {Num_Coils}.')

                k = hf['Kdata'][f'KData_E{encode}_C{c}']
                ksp.append(np.array(k['real'] + 1j * k['imag']).flatten())
            ksp = np.stack(ksp, axis=0)

            # Append to list
            mri_raw.coords.append(coord)
            mri_raw.dcf.append(dcf)
            mri_raw.kdata.append(ksp)
            mri_raw.time.append(time)

            # Log the data
            logging.info(f'MRI coords {mri_raw.coords[encode].shape}')
            logging.info(f'MRI dcf {mri_raw.dcf[encode].shape}')
            logging.info(f'MRI kdata {mri_raw.kdata[encode].shape}')
            logging.info(f'MRI time {mri_raw.time[encode].shape}')
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

        # Scale k-space to max 1
        kdata_max = [ np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(kdata_max)
        kdata_max = np.max(np.array(kdata_max))
        for ksp in mri_raw.kdata:
            ksp /= kdata_max

        kdata_max = [np.abs(ksp).max() for ksp in mri_raw.kdata]
        print(kdata_max)

        # Compress Coils
        if Num_Coils > 14:
            mri_raw.kdata = pca_coil_compression(kdata=mri_raw.kdata, axis=0, target_channels=14)
            mri_raw.Num_Coils = 14

        return mri_raw

def autofov(mri_raw=None, device=None,
            thresh=0.05, scale=1):
    logger = logging.getLogger('autofov')

    # Set to GPU
    if device is None:
        device = sp.Device(0)
    print(device)
    xp = device.xp

    with device:
        # Put on GPU
        coord = 2.0 * mri_raw.coords[0]
        dcf = mri_raw.dcf[0]

        # Low resolution filter
        res = 64
        lpf = np.sum(coord ** 2, axis=-1)
        lpf = np.exp(-lpf / (2.0 * res * res))

        # Get reconstructed size
        img_shape = sp.estimate_shape(coord)
        img_shape = [int(min(i, 64)) for i in img_shape]
        images = xp.ones([mri_raw.Num_Coils] + img_shape, dtype=xp.complex64)
        kdata = mri_raw.kdata

        sos = xp.zeros(img_shape, dtype=xp.float32)

        print(kdata[0].shape)
        print(images.shape)

        for c in range(mri_raw.Num_Coils):
            logger.info(f'Reconstructing  coil {c}')
            ksp_t = np.copy(kdata[0][c, ...])
            ksp_t *= np.squeeze(dcf)
            ksp_t *= np.squeeze(lpf)
            ksp_t = sp.to_device(ksp_t, device=device)

            sos += xp.square(xp.abs(sp.nufft_adjoint(ksp_t, coord, img_shape)))

        # Multiply by SOS of low resolution maps
        sos = xp.sqrt(sos)

    sos = sp.to_device(sos)

    boxc = sos > thresh * sos.max()
    boxc_idx = np.nonzero(boxc)
    boxc_center = np.array(img_shape) // 2
    boxc_shape = np.array([int(2 * max(c - min(boxc_idx[i]), max(boxc_idx[i]) - c) * scale)
                           for i, c in zip(range(3), boxc_center)])

    #  Due to double FOV scale by 2
    target_recon_scale = boxc_shape / img_shape
    logger.info(f'Target recon scale: {target_recon_scale}')

    # Scale to new FOV
    target_recon_size = sp.estimate_shape(coord) * target_recon_scale

    # Round to 8 for blocks and FFT
    target_recon_size = 16*np.ceil( target_recon_size / 16 )

    # Get the actual scale without rounding
    ndim = coord.shape[-1]
    with sp.get_device(coord):
        img_scale = [(2.0*target_recon_size[i]/(coord[..., i].max() - coord[..., i].min())) for i in range(ndim)]

    logger.info(f'Target recon size: {target_recon_size}')
    logger.info(f'Kspace Scale: {img_scale}')
    for e in range(len(mri_raw.coords)):
        mri_raw.coords[e] *= img_scale

    new_img_shape = sp.estimate_shape(mri_raw.coords[0])
    logger.info('Image shape: {}'.format(new_img_shape))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')

    # Parse Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=1.1)
    parser.add_argument('--filename', type=str, help='filename for data (e.g. MRI_Raw.h5)',
                        default='D:/TR_FLOW_RECON/MRI_Raw.h5')

    parser.add_argument('--mps_ker_width', type=int, default=16)
    parser.add_argument('--ksp_calib_width', type=int, default=32)
    parser.add_argument('--lamda', type=float, default=0)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--max_inner_iter', type=int, default=4)

    args = parser.parse_args()

    # For tracking memory
    mempool = cupy.get_default_memory_pool()

    # Load Data
    logger.info(f'Load MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    mri_raw = load_MRI_raw(h5_filename=args.filename)

    crop_kspace(mri_rawdata=mri_raw, crop_factor=2)

    # Reconstruct an low res image and get the field of view
    logger.info(f'Estimating FOV MRI ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    autofov(mri_raw=mri_raw, thresh=args.thresh, scale=args.scale)

    # Get sensitivity maps
    logger.info(f'Reconstruct sensitivity maps ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    smaps = get_smaps(mri_rawdata=mri_raw, args=args)

    # Gate k-space
    mri_raw = gate_kspace(mri_rawdata=mri_raw, num_frames=50)

    # Reconstruct the image
    logger.info(f'Reconstruct Images ( Memory used = {mempool.used_bytes()} of {mempool.total_bytes()} )')
    pils = llr_recon(mri_raw, smaps=smaps)

    # Copy back to make easy
    smaps = sp.to_device(smaps, sp.cpu_device)
    smaps_mag = np.abs(smaps)

    pils = sp.to_device(pils)
    pils_mag = np.abs(pils)
    pils_phase = np.angle(pils)

    # Export to file
    out_name = 'FullRecon.h5'
    logger.info('Saving images to ' + out_name)
    try:
        os.remove(out_name)
    except OSError:
        pass
    with h5py.File(out_name, 'w') as hf:
        hf.create_dataset("IMAGE", data=pils)
        hf.create_dataset("IMAGE_MAG", data=pils_mag)
        hf.create_dataset("IMAGE_PHASE", data=pils_phase)
        hf.create_dataset("SMAPS", data=smaps_mag)
