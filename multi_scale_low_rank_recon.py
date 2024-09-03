import argparse
import logging
import numpy as np
import sigpy as sp
from math import ceil
from tqdm.auto import tqdm
from multi_scale_low_rank_image import MultiScaleLowRankImage
import os
import h5py

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


class MultiScaleLowRankRecon(object):
    r"""Multi-scale low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| ksp_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        ksp (array): k-space measurements of shape (C, num_tr, num_ro, D).
            where C is the number of channels,
            num_tr is the number of TRs, num_ro is the readout points,
            and D is the number of spatial dimensions.
        coord (array): k-space coordinates of shape (num_tr, num_ro, D).
        dcf (array): density compensation factor of shape (num_tr, num_ro).
        mps (array): sensitivity maps of shape (C, N_D, ..., N_1).
            where (N_D, ..., N_1) represents the image shape.
        T (int): number of frames.
        lamda (float): regularization parameter.
        blk_widths (tuple of ints): block widths for multi-scale low rank.
        alpha (float): initial step-size.
        beta (float): step-size decay factor.
        sgw (None or array): soft-gating weights.
            Shape should be compatible with dcf.
        device (sp.Device): computing device.
        comm (None or sp.Communicator): distributed communicator.
        seed (int): random seed.
        max_epoch (int): maximum number of epochs.
        decay_epoch (int): number of epochs to decay step-size.
        max_power_iter (int): maximum number of power iteration.
        show_pbar (bool): show progress bar.

    """
    def __init__(self, ksp, coord, dcf, mps, lamda,
                 blk_widths=[32, 64, 128], alpha=1, beta=0.5, sgw=None,
                 device=sp.cpu_device, comm=None, seed=0,
                 max_epoch=120, decay_epoch=30, max_power_iter=5,
                 show_pbar=True, num_encodings=1, log_dir=None, out_iter_mon=False):
        self.ksp = ksp
        self.coord = coord
        self.dcf = dcf
        self.mps = mps
        self.sgw = sgw
        self.blk_widths = blk_widths
        self.lamda = lamda
        self.alpha = alpha
        self.beta = beta
        self.device = sp.Device(device)
        self.comm = comm
        self.seed = seed
        self.max_epoch = max_epoch
        self.decay_epoch = decay_epoch
        self.max_power_iter = max_power_iter
        self.show_pbar = show_pbar and (comm is None or comm.rank == 0)
        self.log_dir = log_dir
        self.out_iter_mon = out_iter_mon

        # Initialize random seed so code is reproducible
        np.random.seed(self.seed)
        self.xp = self.device.xp
        with self.device:
            self.xp.random.seed(self.seed)

        self.dtype = self.ksp[0].dtype
        self.num_coils = self.mps.shape[0]
        self.img_shape = self.mps.shape[1:]
        self.num_encodings = num_encodings

        # Handle the correlations across encodings differently
        self.total_images = len(ksp)
        self.T = self.total_images // self.num_encodings
        self.t_map = [t // self.num_encodings for t in range(self.total_images)]
        self.e_map = [t % self.num_encodings for t in range(self.total_images)]
        print(f'Time mapping = {self.t_map}')
        print(f'Encode mapping = {self.e_map}')

        self.num_dim = len(self.img_shape)
        self.num_scales = len(self.blk_widths)
        if self.sgw is not None:
            self.dcf *= np.expand_dims(self.sgw, -1)

        # This returns block operators
        self.B = [self._get_B(j) for j in range(self.num_scales)]

        # Scale factors for each scale
        self.G = [self._get_G(j) for j in range(self.num_scales)]

        self._normalize()

    def _get_B(self, j):
        # Block widths with width set smalled that the image
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]

        # Block stride
        s_j = [(b + 1) // 2 for b in b_j]

        # Shifts for blocks for tiling/overlap
        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        # Reshape operation, will crop the LR images when its large
        C_j = sp.linop.Resize(self.img_shape, i_j,
                              ishift=[0] * self.num_dim, oshift=[0] * self.num_dim)

        # Block to array operator with overlapping blocks
        B_j = sp.linop.BlocksToArray(i_j, b_j, s_j)

        # Hanning window to reduce block artifacts
        with self.device:
            w_j = sp.hanning(b_j, dtype=self.dtype, device=self.device)**0.5
        W_j = sp.linop.Multiply(B_j.ishape, w_j)
        return C_j * B_j * W_j

    def _get_G(self, j):
        # Block widths with width set smalled that the image
        b_j = [min(i, self.blk_widths[j]) for i in self.img_shape]

        # Block stride
        s_j = [(b + 1) // 2 for b in b_j]

        # Shifts for blocks for tiling/overlap
        i_j = [ceil((i - b + s) / s) * s + b - s
               for i, b, s in zip(self.img_shape, b_j, s_j)]

        # number of blocks
        n_j = [(i - b + s) // s for i, b, s in zip(i_j, b_j, s_j)]

        M_j = sp.prod(b_j) # number of elements of blocks
        P_j = sp.prod(n_j) # number of blocks

        return M_j**0.5 + self.T**0.5 + (2 * np.log(P_j))**0.5

    def _normalize(self):
        with self.device:
            # Estimate maximum eigenvalue using the first encode
            coord_t = sp.to_device(self.coord[0], self.device)
            dcf_t = sp.to_device(self.dcf[0], self.device)
            F = sp.linop.NUFFT(self.img_shape, coord_t)
            W = sp.linop.Multiply(F.oshape, dcf_t)

            max_eig = sp.app.MaxEig(F.H * W * F, max_iter=self.max_power_iter,
                                    dtype=self.dtype, device=self.device,
                                    show_pbar=self.show_pbar).run()
            print(f'Max_eig {max_eig}')
            for d in self.dcf:
                d /= max_eig

            # Estimate scaling by gridding all the frames
            img_adj = 0
            for c in range(self.num_coils):
                mps_c = sp.to_device(self.mps[c], self.device)
                for frame in range(self.total_images):
                    dcf_t = sp.to_device(self.dcf[frame], self.device)
                    ksp_c = sp.to_device(self.ksp[frame][c], self.device)
                    coord_t = sp.to_device(self.coord[frame], self.device)
                    img_adj_c = sp.nufft_adjoint(ksp_c * dcf_t, coord_t, self.img_shape)
                    img_adj_c *= self.xp.conj(mps_c)
                    img_adj += img_adj_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            img_adj_norm = self.xp.linalg.norm(img_adj).item()
            print(f'img_adj_norm = {img_adj_norm}')
            for k in self.ksp:
                k /= img_adj_norm

    def _init_vars(self):

        # Initialize L.
        self.L = [] # Left vectors
        self.R = [] # Right vectors

        # Initialize L and R for each scale
        for j in range(self.num_scales):

            # L shape is the in input size to the block dimension
            L_j_shape = (self.num_encodings,) + tuple(self.B[j].ishape)

            # Initialize with Gaussian random numbers, normalized to be unit vectors
            L_j = sp.randn(L_j_shape, dtype=self.dtype, device=self.device)
            L_j_norm = self.xp.sum(self.xp.abs(L_j) ** 2,
                                   axis=range(-self.num_dim, 0), keepdims=True) ** 0.5
            L_j /= L_j_norm

            # R shape is the number of time frames x number of scales
            R_j_shape = (self.T, ) + L_j_norm.shape[1:]
            R_j = self.xp.zeros(R_j_shape, dtype=self.dtype)
            self.L.append(L_j)
            self.R.append(R_j)

            print(f'Init scale with L {L_j_shape} , R {R_j_shape}')

    def _power_method(self):
        for it in range(self.max_power_iter):
            # R = A^H(y)^H L
            with tqdm(desc='PowerIter R {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for t in range(self.total_images):
                    self._AHyH_L(t)
                    pbar.update()

            # Normalize R
            for j in range(self.num_scales):
                R_j_norm = self.xp.sum(self.xp.abs(self.R[j])**2,
                                       axis=0, keepdims=True)**0.5
                self.R[j] /= R_j_norm

            # L = A^H(y) R
            with tqdm(desc='PowerIter L {}/{}'.format(
                    it + 1, self.max_power_iter),
                      total=self.T, disable=not self.show_pbar, leave=True) as pbar:
                for j in range(self.num_scales):
                    self.L[j].fill(0)

                for t in range(self.total_images):
                    self._AHy_R(t)
                    pbar.update()

            # Normalize L.
            sigma = []
            for j in range(self.num_scales):
                L_j_norm = self.xp.sum(self.xp.abs(self.L[j]) ** 2,
                                       axis=range(-self.num_dim, 0), keepdims=True) ** 0.5
                self.L[j] /= L_j_norm
                sigma.append(L_j_norm)

        sigma_max = 0
        for j in range(self.num_scales):
            self.L[j] *= sigma[j]**0.5
            self.R[j] *= self.xp.sum( sigma[j]**0.5, 0, keepdims=True) # Sum over encodes
            sigma_max = max(sigma[j].max().item(), sigma_max)

        self.alpha /= sigma_max

    def _AHyH_L(self, idx):

        # Get the time index
        t = self.t_map[idx]
        e = self.e_map[idx]

        # Download k-space arrays.
        coord_t = sp.to_device(self.coord[idx], self.device)
        dcf_t = sp.to_device(self.dcf[idx], self.device)
        ksp_t = sp.to_device(self.ksp[idx], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.num_coils):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.num_scales):
            AHy_tj= self.B[j].H(AHy_t)
            self.R[j][t] = self.xp.sum(AHy_tj * self.xp.conj(self.L[j][e]),
                                       axis=range(-self.num_dim, 0), keepdims=True)

    def _AHy_R(self, idx):

        # Get the time index
        t = self.t_map[idx]
        e = self.e_map[idx]

        # Download k-space arrays.
        coord_t = sp.to_device(self.coord[idx], self.device)
        dcf_t = sp.to_device(self.dcf[idx], self.device)
        ksp_t = sp.to_device(self.ksp[idx], self.device)

        # A^H(y_t)
        AHy_t = 0
        for c in range(self.num_coils):
            mps_c = sp.to_device(self.mps[c], self.device)
            AHy_tc = sp.nufft_adjoint(dcf_t * ksp_t[c], coord_t,
                                      oshape=self.img_shape)
            AHy_tc *= self.xp.conj(mps_c)
            AHy_t += AHy_tc

        if self.comm is not None:
            self.comm.allreduce(AHy_t)

        for j in range(self.num_scales):
            AHy_tj = self.B[j].H(AHy_t)
            self.L[j][e] += AHy_tj * self.xp.conj(self.R[j][t])

    def run(self):
        with self.device:
            self._init_vars()
            self._power_method()
            self.L_init = []
            self.R_init = []
            for j in range(self.num_scales):
                self.L_init.append(sp.to_device(self.L[j]))
                self.R_init.append(sp.to_device(self.R[j]))

            done = False
            while not done:
                try:
                    self.L = []
                    self.R = []
                    for j in range(self.num_scales):
                        self.L.append(sp.to_device(self.L_init[j], self.device))
                        self.R.append(sp.to_device(self.R_init[j], self.device))

                    self._sgd()
                    done = True
                except OverflowError:
                    self.alpha *= self.beta
                    if self.show_pbar:
                        tqdm.write('\nReconstruction diverged. '
                                   'Scaling step-size by {}.'.format(self.beta))

            if self.comm is None or self.comm.rank == 0:
                return MultiScaleLowRankImage(
                    (self.T, self.num_encodings) + self.img_shape,
                    [sp.to_device(L_j, sp.cpu_device) for L_j in self.L],
                    [sp.to_device(R_j, sp.cpu_device) for R_j in self.R])

    def _sgd(self):
        for self.epoch in range(self.max_epoch):
            desc = 'Epoch {}/{}'.format(self.epoch + 1, self.max_epoch)
            disable = not self.show_pbar
            total = self.total_images
            with tqdm(desc=desc, total=total,
                      disable=disable, leave=True) as pbar:
                loss = 0
                for i, t in enumerate(np.random.permutation(self.total_images)):
                    loss += self._update(t)
                    pbar.set_postfix(loss=loss * self.total_images / (i + 1))
                    pbar.update()

                # Form image for export.
                t = int( self.T // 2)
                e = 0
                img_t = 0
                for j in range(self.num_scales):
                    img_t += self.B[j](self.L[j][e] * self.R[j][t])
                im = sp.to_device(img_t, sp.cpu_device)

                # Form Dynamic images
                im_y = []
                for t in range(self.T):
                    img_t = 0
                    for j in range(self.num_scales):
                        img_t += self.B[j](self.L[j][e] * self.R[j][t])

                    if len(img_t.shape) == 3:
                        im_slice = img_t[:, img_t.shape[1]//2, :]
                    elif len(img_t.shape) == 2:
                        im_slice = img_t
                    im_y.append(sp.to_device(im_slice, sp.cpu_device))
                im_y = np.stack(im_y)

                if self.out_iter_mon:
                    out_name = os.path.join(self.log_dir, 'IterMon.h5')
                    if self.epoch == 0:
                        try:
                            os.remove(out_name)
                        except OSError:
                            pass
                        with h5py.File(out_name, 'w') as hf:
                            hf.create_dataset(f'Im{self.epoch:05}', data=np.abs(im))
                            hf.create_dataset(f'Phase{self.epoch:05}', data=np.angle(im))
                            hf.create_dataset(f'Y{self.epoch:05}', data=np.abs(im_y))
                    else:
                        with h5py.File(out_name, 'a') as hf:
                            hf.create_dataset(f'Im{self.epoch:05}', data=np.abs(im))
                            hf.create_dataset(f'Phase{self.epoch:05}', data=np.angle(im))
                            hf.create_dataset(f'Y{self.epoch:05}', data=np.abs(im_y))

    def _update(self, idx):

        t = self.t_map[idx]
        e = self.e_map[idx]

        # Form image.
        img_t = 0
        for j in range(self.num_scales):
            img_t += self.B[j](self.L[j][e] * self.R[j][t])

        # Transfer k-space arrays to device
        coord_t = sp.to_device(self.coord[idx], self.device)
        dcf_t = sp.to_device(self.dcf[idx], self.device)
        ksp_t = sp.to_device(self.ksp[idx], self.device)

        # Data consistency.
        e_t = 0
        loss_t = 0
        for c in range(self.num_coils):
            mps_c = sp.to_device(self.mps[c], self.device)
            e_tc = sp.nufft(img_t * mps_c, coord_t)
            e_tc -= ksp_t[c]
            e_tc *= dcf_t**0.5
            loss_t += self.xp.linalg.norm(e_tc)**2
            e_tc *= dcf_t**0.5
            e_tc = sp.nufft_adjoint(e_tc, coord_t, oshape=self.img_shape)
            e_tc *= self.xp.conj(mps_c)
            e_t += e_tc

        if self.comm is not None:
            self.comm.allreduce(e_t)
            self.comm.allreduce(loss_t)

        loss_t = loss_t.item()

        # Compute gradient for each scale
        for j in range(self.num_scales):
            lamda_j = self.lamda * self.G[j]

            # L gradient.
            g_L_j = self.B[j].H(e_t) #Image to block
            g_L_j *= self.xp.conj(self.R[j][t])
            g_L_j += lamda_j / self.T * self.L[j][e]
            g_L_j *= self.T

            # R gradient.
            g_R_jt = self.B[j].H(e_t) # Image to blocks
            g_R_jt *= self.xp.conj(self.L[j][e])
            g_R_jt = self.xp.sum(g_R_jt, axis=range(-self.num_dim, 0), keepdims=True)
            g_R_jt += lamda_j * self.R[j][t]

            # Loss.
            loss_t += lamda_j / self.T * self.xp.linalg.norm(self.L[j][e]).item()**2
            loss_t += lamda_j * self.xp.linalg.norm(self.R[j][t]).item()**2
            if np.isinf(loss_t) or np.isnan(loss_t):
                raise OverflowError

            # Add.
            self.L[j][e] -= self.alpha * self.beta**(self.epoch // self.decay_epoch) * g_L_j
            self.R[j][t] -= self.alpha * g_R_jt

        loss_t /= 2
        return loss_t


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Multi-Scale Low rank reconstruction.')
    parser.add_argument('--blk_widths', type=int, nargs='+', default=[32, 64, 128],
                        help='Block widths for low rank.')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Step-size')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Step-size decay.')
    parser.add_argument('--max_epoch', type=int, default=120,
                        help='Maximum epochs.')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='Decay epochs.')
    parser.add_argument('--max_power_iter', type=int, default=5,
                        help='Maximum power iterations.')
    parser.add_argument('--device', type=int, default=-1,
                        help='Computing device.')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Toggle multi-gpu. Require MPI. '
                        'Ignore device when toggled.')
    parser.add_argument('--sgw_file', type=str,
                        help='Soft gating weights.')

    parser.add_argument('ksp_file', type=str,
                        help='k-space file.')
    parser.add_argument('coord_file', type=str,
                        help='Coordinate file.')
    parser.add_argument('dcf_file', type=str,
                        help='Density compensation file.')
    parser.add_argument('mps_file', type=str,
                        help='Sensitivity maps file.')
    parser.add_argument('T', type=int,
                        help='Number of frames.')
    parser.add_argument('lamda', type=float,
                        help='Regularization. Recommend 1e-8 to start.')
    parser.add_argument('img_file', type=str,
                        help='Output image file.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    ksp = np.load(args.ksp_file, 'r')
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    mps = np.load(args.mps_file, 'r')

    comm = sp.Communicator()
    if args.multi_gpu:
        device = sp.Device(comm.rank)
    else:
        device = sp.Device(args.device)

    if args.sgw_file is not None:
        sgw = np.load(args.sgw_file)
    else:
        sgw = None

    # Split between nodes.
    ksp = ksp[comm.rank::comm.size].copy()
    mps = mps[comm.rank::comm.size].copy()

    app = MultiScaleLowRankRecon(ksp, coord, dcf, mps, args.T, args.lamda,
                                 sgw=sgw,
                                 blk_widths=args.blk_widths,
                                 alpha=args.alpha,
                                 beta=args.beta,
                                 max_epoch=args.max_epoch,
                                 decay_epoch=args.decay_epoch,
                                 max_power_iter=args.max_power_iter,
                                 device=device, comm=comm)
    img = app.run()

    if comm.rank == 0:
        img.save(args.img_file)
