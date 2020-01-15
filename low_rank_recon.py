import argparse
import pickle
import logging
import numpy as np
import sigpy as sp
import sigpy.mri as mr
from tqdm import tqdm
import os
import h5py

'''
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass
'''

class ShuffledNumbers(object):
    """Produces shuffled numbers between given range.
    Args:
        Arguments to numpy.arange.
    """

    def __init__(self, *args):
        self.numbers = np.arange(*args)
        np.random.shuffle(self.numbers)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ret = self.numbers[self.i]

        self.i += 1
        if self.i == len(self.numbers):
            np.random.shuffle(self.numbers)
            self.i = 0

        return ret


class LowRankRecon(sp.app.App):
    r"""Low rank reconstruction.

    Considers the objective function,

    .. math::
        f(l, r) = sum_t \| y_t - \mathcal{A}(L, R_t) \|_2^2 +
        \lambda ( \| L \|_F^2 + \| R_t \|_F^2)

    where :math:`\mathcal{A}_t` is the forward operator for time :math:`t`.

    Args:
        y (array): k-space measurements of shape (num_tr, num_ro, img_ndim).
        coord (array): coordinates.
        weights (array): weights.
        mps (array): sensitivity maps of shape (num_coils, ...).
        num_frames (int): number of frames.

    """

    def __init__(self, y, coord, weights, mps,
                 num_levels=3, scale_ratio=2,
                 normalize=True, coil_batch_size=1,
                 alpha=0.1, lamda=1e-10, init_scale=1e-5,
                 device=sp.cpu_device, comm=sp.Communicator(), seed=0,
                 max_inner_iter=10, max_power_iter=30, max_iter=10,
                 show_pbar=True):
        self.y = y
        self.coord = coord
        self.weights = weights
        self.mps = mps
        self.num_levels = num_levels
        self.scale_ratio = scale_ratio
        self.num_frames = y.shape[0] # Data is pregated
        self.normalize = normalize
        self.coil_batch_size = coil_batch_size
        self.alpha = alpha
        self.lamda = lamda
        self.init_scale = init_scale
        self.device = device
        self.comm = comm
        self.seed = seed
        self.max_inner_iter = max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_iter = max_iter
        self.show_pbar = show_pbar and comm.rank == 0
        self.log_images = True
        self.log_out_name = 'ReconLog.h5'

        self.dtype = self.y.dtype
        self.num_coils = self.y.shape[1] # data is Nframes x Ncoils x Nr x Npts

        self.kdata_shape = self.y.shape[2:]
        self.img_shape = self.mps.shape[1:]
        self.img_ndim = len(self.img_shape)
        self.device = sp.Device(self.device)

        self._get_vars()
        self._get_batch_vars()
        self._get_alg()

        with self.device:
            if self.normalize:
                self.A_scale = 1 / self._get_max_eig() ** 0.5
                self.y_scale = 1 / self._get_img_adj_max()
            else:
                self.A_scale = 1
                self.y_scale = 1

        if self.log_images:
            # Export to file
            print('Logging images to ' + self.log_out_name)
            try:
                os.remove(self.log_out_name)
            except OSError:
                pass

    def _get_vars(self):
        """Get variables.
        """
        self.blk_shapes = []
        self.blk_strides = []
        self.num_blks = []
        self.blk_frames = []

        self.L = []
        self.R = []
        self.L_initialize = []
        for j in range(self.num_levels):
            self.blk_shapes.append([max(i // self.scale_ratio ** j, 1) for i in self.img_shape])
            self.blk_strides.append([max(b // 2, 1) for b in self.blk_shapes[j]])

            if j == 0:
                self.num_blks.append([2] * self.img_ndim)
            else:
                self.num_blks.append([(i - b + s) // s for i, b, s in zip(self.img_shape,
                                                                          self.blk_shapes[j],
                                                                          self.blk_strides[j])])

            self.blk_frames.append(max(self.num_frames // self.scale_ratio ** j, 1))
            print('Scale {}: {} x {}'.format(j, self.blk_frames[j], self.blk_shapes[j]))

            num_time_blks = (self.num_frames + self.blk_frames[j] - 1) // self.blk_frames[j]
            self.L.append(np.empty([num_time_blks] + self.num_blks[j] + self.blk_shapes[j],
                                   dtype=self.dtype))
            self.R.append(np.zeros([self.num_frames] + self.num_blks[j] + [1] * self.img_ndim,
                                   dtype=self.dtype))
            self.L_initialize.append([False] * num_time_blks)

        self.img = LowRankImage(self.L, self.R, self.img_shape, self.num_blks,
                                self.blk_shapes, self.blk_strides, self.blk_frames)

    def _get_batch_vars(self):
        np.random.seed(self.seed)
        self.t_idx = ShuffledNumbers(self.num_frames)
        xp = self.device.xp
        with self.device:
            self.y_t = xp.empty((self.num_coils,) + self.kdata_shape, dtype=self.dtype)
            self.coord_t = xp.empty(self.kdata_shape + (self.img_ndim,),
                                    dtype=self.coord.dtype)
            self.weights_t = xp.empty(self.kdata_shape, dtype=self.weights.dtype)

    def _get_A_L_t(self):
        """Get linear operator that with fixed R_t.
        """
        S_t = mr.linop.Sense(self.mps, weights=self.weights_t,
                             coord=self.coord_t,
                             coil_batch_size=self.coil_batch_size,
                             comm=self.comm)
        M_L_t = []
        for j in range(self.num_levels):
            M_L_t.append(self.B[j] * sp.linop.Multiply(self.L[j][self.t // self.blk_frames[j]].shape,
                                                       self.R[j][self.t]))
        M_L_t = sp.linop.Hstack(M_L_t)
        return self.A_scale * S_t * M_L_t

    def _get_A_R_t(self):
        """Get linear operator that with fixed L_t.
        """
        S_t = mr.linop.Sense(self.mps, weights=self.weights_t,
                             coord=self.coord_t,
                             coil_batch_size=self.coil_batch_size,
                             comm=self.comm)
        M_R_t = []
        for j in range(self.num_levels):
            M_R_t.append(self.B[j] * sp.linop.Multiply(self.R[j][self.t].shape,
                                                       self.L[j][self.t // self.blk_frames[j]]))

        M_R_t = sp.linop.Hstack(M_R_t)
        return self.A_scale * S_t * M_R_t

    def _get_L_t(self):
        xp = self.device.xp
        with self.device:
            L_t = xp.empty(sum(self.L[j][self.t // self.blk_frames[j]].size
                               for j in range(self.num_levels)), dtype=self.dtype)
            i = 0
            for j in range(self.num_levels):
                t = self.t // self.blk_frames[j]

                if self.L_initialize[j][t]:
                    sp.copyto(L_t[i:(i + self.L[j][t].size)],
                              self.L[j][t].ravel())
                else:
                    L_tj = sp.randn([self.L[j][t].size],
                                    scale=self.init_scale,
                                    dtype=self.dtype, device=self.device)
                    if self.comm is not None:
                        self.comm.allreduce(L_tj)
                        L_tj /= self.comm.size ** 0.5

                    sp.copyto(L_t[i:(i + self.L[j][t].size)], L_tj)
                    self.L_initialize[j][t] = True

                i += self.L[j][t].size

        return L_t

    def _get_R_t(self):
        xp = self.device.xp
        with self.device:
            R_t = xp.empty(sum(self.R[j][self.t].size
                               for j in range(self.num_levels)), dtype=self.dtype)
            i = 0
            for j in range(self.num_levels):
                sp.copyto(R_t[i:(i + self.R[j][self.t].size)],
                          self.R[j][self.t].ravel())
                i += self.R[j][self.t].size

        return R_t

    def _set_L_t(self, L_t):
        with self.device:
            i = 0
            for j in range(self.num_levels):
                t = self.t // self.blk_frames[j]
                sp.copyto(self.L[j][t],
                          L_t[i:(i + self.L[j][t].size)].reshape(self.L[j][t].shape))
                i += self.L[j][t].size

    def _set_R_t(self, R_t):
        with self.device:
            i = 0
            for j in range(self.num_levels):
                sp.copyto(self.R[j][self.t],
                          R_t[i:(i + self.R[j][self.t].size)].reshape(self.R[j][self.t].shape))
                i += self.R[j][self.t].size

    def _min_R_t(self):
        sp.copyto(self.coord_t, self.coord[self.t,...])
        sp.copyto(self.y_t, self.y[self.t,...])
        sp.copyto(self.weights_t, self.weights[self.t,...])
        with self.device:
            self.y_t *= self.weights_t ** 0.5 * self.y_scale

        R_t = self._get_R_t()
        self.A_R_t = self._get_A_R_t()
        sp.app.LinearLeastSquares(self.A_R_t, self.y_t, x=R_t,
                                  lamda=self.lamda,
                                  max_iter=self.max_inner_iter,
                                  show_pbar=False).run()
        self._set_R_t(R_t)

    def _min_L_t(self):
        L_t = self._get_L_t()
        self.A_L_t = self._get_A_L_t()
        sp.app.LinearLeastSquares(self.A_L_t, self.y_t, x=L_t,
                                  alpha=self.alpha, z=L_t,
                                  lamda=self.lamda / self.num_frames,
                                  max_iter=self.max_inner_iter,
                                  show_pbar=False).run()
        self._set_L_t(L_t)

    def _get_alg(self):
        self.B = _get_B(self.img_shape, self.num_frames, self.num_blks,
                        self.blk_shapes, self.blk_strides, self.blk_frames)
        alg = sp.alg.AltMin(self._min_R_t, self._min_L_t, max_iter=self.max_iter)
        super().__init__(alg, show_pbar=self.show_pbar)

    def _get_max_eig(self):
        sp.copyto(self.coord_t, self.coord[0,...])
        sp.copyto(self.weights_t, self.weights[0,...])

        F = sp.linop.NUFFT(self.img_shape, self.coord_t)
        W = sp.linop.Multiply(F.oshape, self.weights_t)

        return sp.app.MaxEig(F.H * W * F,
                             dtype=self.dtype, device=self.device,
                             max_iter=self.max_power_iter,
                             show_pbar=self.show_pbar).run()

    def _get_img_adj_max(self):
        xp = self.device.xp
        with self.device:
            weights = sp.to_device(self.weights, self.device)
            img_adj = 0
            for t in range(self.num_frames):
                for c in range(self.num_coils):
                    y_c = sp.to_device(self.y[t,c], self.device) * self.A_scale * weights[t]
                    img_c = sp.nufft_adjoint(y_c, self.coord[t], self.img_shape)
                    mps_c = sp.to_device(self.mps[c], self.device)
                    img_c *= xp.conj(mps_c)
                    img_adj += img_c

            if self.comm is not None:
                self.comm.allreduce(img_adj)

            return xp.abs(img_adj).max()

    def _pre_update(self):
        self.t = self.t_idx.next()

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(t=self.t, refresh=False)

        if self.log_images:
            self._write_log()

    def _write_log(self):

        xp = self.device.xp
        with self.device:

            # Grab the first image in series
            img_t = self.img[int(self.t)]

            # Central slice
            out_slices = np.array(img_t.shape) // 2

            with h5py.File(self.log_out_name, 'a') as hf:

                for dim,slice in enumerate(out_slices):
                    out_slice = xp.copy(xp.take( img_t, slice, axis=dim))
                    out_name = f'Slice{slice}_Dim{dim}'
                    out_slice = sp.to_device(out_slice, sp.cpu_device)
                    out_slice = np.expand_dims(out_slice,0)

                    # Resize to include additional image
                    if out_name in hf.keys():
                        hf[out_name].resize((hf[out_name].shape[0] + out_slice.shape[0]), axis=0)
                        hf[out_name][-out_slice.shape[0]:] = np.abs(out_slice)
                    else:
                        maxshape = np.array(out_slice.shape)
                        maxshape[0] *= (self.max_iter + 10)
                        maxshape = tuple(maxshape)
                        print(f'Init {self.log_out_name}  / {out_name} with maxshape = {maxshape}')
                        hf.create_dataset(out_name, data=np.abs(out_slice), maxshape=maxshape)

    def _output(self):
        trange = range(self.num_frames)
        if self.show_pbar:
            trange = tqdm(trange, desc="Final Recon")

        for self.t in trange:
            self._min_R_t()

        return self.img


class LowRankImage(object):
    """Low rank image representation.

    Args:
        L (array): Left singular vectors of length num_levels.
            Each scale is of shape [blk_frames] + num_blks + blk_shape.
        R (array): Right singular vectors of length num_levels.
            Each scale is of shape [num_frames] + num_blks + [1] * num_img_dim.
        img_shape (tuple of ints): Image shape.

    """

    def __init__(self, L, R, img_shape, num_blks, blk_shapes, blk_strides, blk_frames,
                 device=sp.cpu_device):
        self.img_shape = img_shape
        self.num_blks = num_blks
        self.blk_shapes = blk_shapes
        self.blk_strides = blk_strides
        self.blk_frames = blk_frames

        self.img_ndim = len(img_shape)
        self.num_frames = len(R[0])
        self.shape = (self.num_frames,) + tuple(img_shape)
        self.ndim = len(self.shape)
        self.dtype = L[0].dtype
        self.num_levels = len(num_blks)
        self.B = _get_B(self.img_shape, self.num_frames,
                        self.num_blks, self.blk_shapes, self.blk_strides, self.blk_frames)

        self.L = L
        self.R = R
        self.use_device(device)

    def __len__(self):
        return self.num_frames

    def use_device(self, device):
        self.device = sp.Device(device)

    def _get_img(self, t):
        xp = self.device.xp
        with self.device:
            img_t = 0
            for j in range(self.num_levels):
                img_t += self.B[j](sp.to_device(self.L[j][t // self.blk_frames[j]], self.device) *
                                   sp.to_device(self.R[j][t], self.device))

        return img_t

    def __getitem__(self, index):
        xp = self.device.xp
        with self.device:
            if isinstance(index, int):
                output = self._get_img(index)
            elif isinstance(index, slice):
                output = xp.stack([self._get_img(t) for t in index])
            elif isinstance(index, tuple) or isinstance(index, list):
                if isinstance(index[0], int):
                    output = self._get_img(index[0])[index[1:]]
                elif isinstance(index[0], slice):
                    output = xp.stack([self._get_img(t)[index[1:]] for t in index[0]])

        return output

    def decom(self):
        return LowRankDecom(self.L, self.R, self.img_shape, self.num_blks,
                            self.blk_shapes, self.blk_strides, self.blk_frames, device=self.device)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


class LowRankDecom(object):
    """Low rank image representation.

    Args:
        L (array): Left singular vectors of length num_levels.
            Each scale is of shape [blk_frames] + num_blks + blk_shape.
        R (array): Right singular vectors of length num_levels.
            Each scale is of shape [num_frames] + num_blks + [1] * num_img_dim.
        img_shape (tuple of ints): Image shape.

    """

    def __init__(self, L, R, img_shape, num_blks, blk_shapes, blk_strides, blk_frames,
                 device=sp.cpu_device):
        self.img_shape = img_shape
        self.num_blks = num_blks
        self.blk_shapes = blk_shapes
        self.blk_strides = blk_strides
        self.blk_frames = blk_frames

        self.img_ndim = len(img_shape)
        self.num_frames = len(R[0])
        self.num_levels = len(num_blks)
        self.shape = (self.num_frames, self.num_levels) + tuple(img_shape)
        self.ndim = len(self.shape)
        self.dtype = L[0].dtype
        self.B = _get_B(self.img_shape, self.num_frames,
                        self.num_blks, self.blk_shapes, self.blk_strides, self.blk_frames)

        self.L = L
        self.R = R
        self.use_device(device)

    def __len__(self):
        return self.num_frames

    def use_device(self, device):
        self.device = sp.Device(device)

    def _get_decom(self, t):
        xp = self.device.xp
        with self.device:
            return xp.stack([self.B[j](sp.to_device(self.L[j][t // self.blk_frames[j]],
                                                    self.device) *
                                       sp.to_device(self.R[j][t], self.device))
                             for j in range(self.num_levels)])

    def __getitem__(self, index):
        xp = self.device.xp
        with self.device:
            if isinstance(index, int):
                output = self._get_decom(index)
            elif isinstance(index, slice):
                output = xp.stack([self._get_decom(t) for t in index])
            elif isinstance(index, tuple) or isinstance(index, list):
                if isinstance(index[0], int):
                    output = self._get_decom(index[0])[index[1:]]
                elif isinstance(index[0], slice):
                    output = xp.stack([self._get_decom(t)[index[1:]] for t in index[0]])

        return output

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


def _get_B(img_shape, num_frames, num_blks, blk_shapes, blk_strides, blk_frames):
    """Get block to array linear operator.

    """
    num_levels = len(num_blks)
    img_ndim = len(img_shape)
    B = []
    for j in range(num_levels):
        m = sp.prod(blk_shapes[j])
        n = blk_frames[j]
        gwidth_j = m ** 0.5 + n ** 0.5

        if j == 0:
            B_j = sp.linop.Sum(num_blks[0] + blk_shapes[0], range(img_ndim))
            B.append((1 / gwidth_j) * B_j)
        else:
            B_j = sp.linop.BlocksToArray(img_shape, blk_shapes[j], blk_strides[j])
            B.append((1 / gwidth_j) * B_j)

    return B


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Low rank reconstruction.')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of levels for low rank.')
    parser.add_argument('--scale_ratio', type=int, default=2, help='Scaling factor.')
    parser.add_argument('--num_frames', type=int, default=1, help='Number of frames.')
    parser.add_argument('--coil_batch_size', type=int, default=1, help='Coil batch size.')
    parser.add_argument('--init_scale', type=float, default=1e-5, help='Initial scaling.')
    parser.add_argument('--alpha', type=float, default=0.1, help='Step-size.')
    parser.add_argument('--lamda', type=float, default=1e-10, help='Regularization.')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum iterations.')
    parser.add_argument('--max_inner_iter', type=int, default=10, help='Maximum inner iterations.')
    parser.add_argument('--max_power_iter', type=int, default=30, help='Maximum power iterations.')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--multi_gpu', action='store_true')

    parser.add_argument('y_file', type=str)
    parser.add_argument('coord_file', type=str)
    parser.add_argument('dcf_file', type=str)
    parser.add_argument('mps_file', type=str)
    parser.add_argument('img_file', type=str)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    y = np.load(args.y_file, 'r')
    coord = np.load(args.coord_file)
    dcf = np.load(args.dcf_file)
    mps = np.load(args.mps_file, 'r')

    # Choose device
    comm = sp.Communicator()
    if args.multi_gpu:
        device = sp.Device(comm.rank)
    else:
        device = sp.Device(args.device)

    y = np.array_split(y, comm.size)[comm.rank]
    mps = np.array_split(mps, comm.size)[comm.rank]
    img = LowRankRecon(y, coord, dcf, mps,
                       num_levels=args.num_levels,
                       scale_ratio=args.scale_ratio,
                       num_frames=args.num_frames,
                       coil_batch_size=args.coil_batch_size,
                       init_scale=args.init_scale,
                       alpha=args.alpha,
                       lamda=args.lamda,
                       max_iter=args.max_iter,
                       max_inner_iter=args.max_inner_iter,
                       max_power_iter=args.max_power_iter,
                       device=device, comm=comm).run()

    if comm.rank == 0:
        img.save(args.img_file)
