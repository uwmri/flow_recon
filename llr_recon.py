import numpy as np
import h5py
import cupy
import cupy.cudnn
import sigpy as sp
import sigpy.mri as mr
import logging
import cupy
import time

from multi_scale_low_rank_recon import *
from svt import *

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

    def __init__(self, y, mps, lamda=0, weights=None, num_enc=0, gate_type='time',
                 coord=None, device=sp.cpu_device, coil_batch_size=None,
                 comm=None, show_pbar=True, max_power_iter=40, batched_iter=50, fast_maxeig=False,
                 composite_init=True, block_width=16, log_folder=None, **kwargs):

        # Temp
        self.num_encodes = num_enc
        self.frames = len( y )  // self.num_encodes
        self.num_images = self.frames*self.num_encodes
        self.cpu_device = sp.cpu_device
        if device is None:
            self.gpu_device = sp.Device(0)
        else:
            self.gpu_device = device

        self.max_power_iter = max_power_iter
        self.show_pbar = True
        self.log_images = True
        self.log_out_name = os.path.join( log_folder, 'ReconLog.h5')

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('BatchedSenseRecon')

        print(f'Whats the num of frames?  = {self.frames}')
        print(f'Whats the num of encodes?  = {self.num_encodes}')
        print(f'Whats the num of images?  = {self.num_images}')

        if self.log_images:
            # Export to file
            self.logger.info('Logging images to ' + self.log_out_name)
            try:
                os.remove(self.log_out_name)
            except OSError:
                pass

        # put coord and mps to gpu
        mps = sp.to_device(mps, self.gpu_device)

        for i in range(len(coord)):
            coord[i] = sp.to_device(coord[i], sp.Device(self.gpu_device))
            weights[i] = sp.to_device(weights[i], sp.Device(self.gpu_device))

        if fast_maxeig:
            print('Fast Maxeig')
            A = sp.mri.linop.Sense(mps, coord[0], weights[0], ishape=None,
                                             coil_batch_size=coil_batch_size, comm=comm)
            AHA = A.H * A
            max_eig = sp.app.MaxEig(AHA, dtype=y[0].dtype, device=self.gpu_device,
                             max_iter=self.max_power_iter,
                             show_pbar=self.show_pbar).run()
        else:
            #global max eigen
            ops_list = [sp.mri.linop.Sense(mps, coord[e], weights[e], ishape=None,
                        coil_batch_size=coil_batch_size, comm=comm) for e in range(self.num_images)]

            grad_ops_nodev = [ops_list[e].H * ops_list[e] for e in range(len(ops_list))]
            # A.h*A
            # wrap to run GPU
            grad_ops = [sp.linop.ToDevice(op.oshape, self.cpu_device, self.gpu_device) * op * sp.linop.ToDevice(op.ishape, self.gpu_device, self.cpu_device) for op in grad_ops_nodev]
            # Get AHA opts list
            AHA = sp.linop.Diag(grad_ops, oaxis=0, iaxis=0)
            max_eig = sp.app.MaxEig(AHA, dtype=y[0].dtype, device=self.cpu_device, max_iter=self.max_power_iter, show_pbar=self.show_pbar).run()

        # Scale the weights
        for i in range(len(weights)):
            with sp.get_device(weights[i]):
                weights[i] *= 1.0 / max_eig

        # Put on GPU
        for i in range(len(y)):
            y[i] = sp.to_device(y[i], self.gpu_device)

        # Initialize with an average reconstruction
        if composite_init:

            xp = sp.get_device(y[0]).xp

            # Create a composite image
            for e in range(self.num_images):

                # Sense operator
                A = sp.mri.linop.Sense(mps, coord[e], weights[e], ishape=None,
                                       coil_batch_size=coil_batch_size, comm=comm)
                data = xp.copy(y[e])
                data *= weights[e] ** 0.5

                if e == 0:
                    composite = A.H * data
                else:
                    composite = composite + A.H * data

            composite /= self.num_images

            # Scale composite to be max to 1
            composite /= np.max(np.abs(composite))
            print(f'Init with {composite.shape}, Mean = {np.max(np.abs(composite))}')

             # Multiply by sqrt(weights)
            if weights is not None:
                for e in range(self.num_images):
                    y[e] *= weights[e] ** 0.5

            # Now scale the images
            sum_yAx = 0.0
            sum_yy = xp.sum( xp.square( xp.abs(y)))

            for e in range(self.num_images):
                A = sp.mri.linop.Sense(mps, coord[e], weights[e], ishape=None,
                                       coil_batch_size=coil_batch_size, comm=comm)
                data = A * composite
                sum_yAx += xp.sum( data * xp.conj(y[e]))

            y_scale = xp.abs( sum_yAx / sum_yy )
            print(f'Sum yAx = {sum_yAx}')
            print(f'Sum yy = {sum_yy}')

            y *= y_scale

            composite = sp.to_device(composite, sp.cpu_device)
            x = np.vstack([composite for i in range(self.num_images)])
        else:
             # Multiply by sqrt(weights)
            if weights is not None:
                for e in range(self.num_images):
                    y[e] *= weights[e] ** 0.5

        # Update ops list with weights
        ops_list = [sp.mri.linop.Sense(mps, coord[e], weights[e], ishape=None,
                                             coil_batch_size=coil_batch_size, comm=comm) for e in range(self.num_images)]

        sub_list = [ SubtractArray(y[e]) for e in range(self.num_images)]
        #grad_ops_nodev = [ ops_list[e].H * sub_list[e] *ops_list[e] for e in range(len(ops_list))]
        grad_ops_nodev = [ ops_list[e].H * sub_list[e] *ops_list[e] for e in range(len(ops_list))]
        # A.h*(Ax-y)

        # wrap to run GPU
        grad_ops = [sp.linop.ToDevice(op.oshape, self.cpu_device, self.gpu_device)*op*sp.linop.ToDevice(op.ishape,self.gpu_device,self.cpu_device) for op in grad_ops_nodev]
        # * is a function

        # Get AHA opts list
        A = sp.linop.Diag(grad_ops, oaxis=0, iaxis=0)

        if composite_init == False:
            x = self.cpu_device.xp.zeros(A.ishape, dtype=y[0].dtype)

        # block size and stride should be equal, now testing different stride for block shifting problem
        # cardiac recon expected to be lower rank than temporal recon, thus smaller block size (as in cpp wrapper)
        print('batched iter = ', batched_iter)
        proxg = SingularValueThresholdingNumba(A.ishape, frames=self.frames, num_encodes=self.num_encodes,
                                              lamda=lamda, block_size=block_width, block_stride=block_width, batched_iter=batched_iter)


        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, x=x, proxg=proxg, show_pbar=show_pbar, alpha=1.0, accelerate=True, **kwargs) #default alpha = 1

        # log the initial guess
        if self.log_images:
            self._write_log()
            # self._write_x()

    def _write_x(self):
        with h5py.File('X.h5', 'w') as hf:
            hf.create_dataset("X", data=np.abs(self.x))

    def _summarize(self):

        if self.log_images:
            self._write_log()

        super()._summarize()


    def _write_log(self):

        self.logger.info(f'Logging to file {self.log_out_name}')
        xp = sp.get_device(self.x).xp

        # Reshape X if 2D
        if len(self.x.shape) == 2:
            temp = xp.reshape(self.x, (self.frames, self.num_encodes, -1) + self.x.shape[1:])
            out_frame = self.frames // 2
            out_encode = self.num_encodes // 2
            Xiter = xp.copy( temp[out_frame, out_encode])
        else:
            out_slice = int((self.x.shape[0] / self.frames ) // 2)
            Xiter = xp.copy( self.x[out_slice])

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
                maxshape[0] *= (self.max_iter  + 1)
                maxshape = tuple(maxshape)
                print(f'Init {self.log_out_name} with maxshape = {maxshape}')
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