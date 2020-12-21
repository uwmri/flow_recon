import time
import math
import numba as nb
import torch as torch
import sigpy as sp
import numpy as np

__all__ = [ 'SingularValueThresholding', 'SingularValueThresholdingNumba' ]

@nb.jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def svt_numba3(output, input, lamda, blk_shape, blk_strides, block_iter, num_encodes):

    bz = input.shape[1] // blk_strides[0]
    by = input.shape[2] // blk_strides[1]
    bx = input.shape[3] // blk_strides[2]

    Sz = int(input.shape[1])
    Sy = int(input.shape[2])
    Sx = int(input.shape[3])

    scale = float(1.0 / block_iter)

    num_frames = int(input.shape[0]/num_encodes)
    # bmat_shape = (num_frames, num_encodes*blk_shape[0] * blk_shape[1] * blk_shape[2])
    bmat_shape = (num_encodes*blk_shape[0] * blk_shape[1] * blk_shape[2], num_frames)

    shifts = np.zeros((3, block_iter), np.int32)
    for d in range(3):
        for biter in range(block_iter):
            shifts[d,biter] = np.random.randint(blk_shape[d])

    for iter in range(block_iter):
        print('block iter = ',iter)
        shiftz = shifts[0, iter]
        shifty = shifts[1, iter]
        shiftx = shifts[2, iter]

        for nz in nb.prange(bz):
            sz = nz * blk_strides[0] + shiftz
            ez = sz + blk_shape[0]

            for ny in range(by):
                sy = ny * blk_strides[1] + shifty
                ey = sy + blk_shape[1]

                for nx in range(bx):

                    sx = nx * blk_strides[2] + shiftx
                    ex = sx + blk_shape[2]

                    block = np.zeros(bmat_shape, input.dtype)

                    # Grab a block
                    for tframe in range(num_frames):
                        count = 0
                        for encode in range(num_encodes):
                            store_pos = int(tframe * num_encodes + encode)
                            for k in range(sz, ez):
                                for j in range(sy, ey):
                                    for i in range(sx, ex):
                                        # block[tframe, count] = input[store_pos, k % Sz, j % Sy, i % Sx]
                                        block[count, tframe] = input[store_pos, k % Sz, j % Sy, i % Sx]
                                        count += 1

                    # Svd
                    u, s, vh = np.linalg.svd(block, full_matrices=False)

                    for k in range(u.shape[1]):

                        # s[k] = max(s[k] - lamda, 0)
                        abs_input = abs(s[k])
                        if abs_input == 0:
                            sign = 0
                        else:
                            sign = s[k] / abs_input

                        s[k] = abs_input - lamda
                        s[k] = (abs(s[k]) + s[k]) / 2
                        s[k] = s[k] * sign

                        for i in range(u.shape[0]):
                            u[i, k] *= s[k]

                    block = np.dot(u, vh)

                    # Put block back
                    for tframe in range(num_frames):
                        count = 0
                        for encode in range(num_encodes):
                            store_pos = int(tframe * num_encodes + encode)
                            for k in range(sz, ez):
                                for j in range(sy, ey):
                                    for i in range(sx, ex):
                                        # output[store_pos, k % Sz, j % Sy, i % Sx] += scale*block[tframe, count]
                                        output[store_pos, k % Sz, j % Sy, i % Sx] += scale*block[count, tframe]
                                        count += 1

    return output

@nb.jit(nopython=True, cache=True, parallel=True)  # pragma: no cover
def svt_numba2(output, input, lamda, blk_shape, blk_strides, block_iter, num_encodes):

    by = input.shape[1] // blk_strides[0]
    bx = input.shape[2] // blk_strides[1]

    Sy = int(input.shape[1])
    Sx = int(input.shape[2])

    scale = float(1.0 / block_iter)

    num_frames = int(input.shape[0]/num_encodes)
    bmat_shape = (num_encodes*blk_shape[0] * blk_shape[1], num_frames)

    shifts = np.zeros((2, block_iter), np.int32)
    for d in range(2):
        for biter in range(block_iter):
            shifts[d,biter] = np.random.randint(blk_shape[d])

    for iter in range(block_iter):
        print('block iter = ',iter)
        shiftx = shifts[0, iter]
        shifty = shifts[1, iter]

        for ny in nb.prange(by):
            sy = ny * blk_strides[0] + shifty
            ey = sy + blk_shape[0]

            for nx in range(bx):

                sx = nx * blk_strides[1] + shiftx
                ex = sx + blk_shape[1]

                block = np.zeros(bmat_shape, input.dtype)

                # Grab a block
                for tframe in range(num_frames):
                    count = 0
                    for encode in range(num_encodes):
                        store_pos = int(tframe * num_encodes + encode)
                        for j in range(sy, ey):
                            for i in range(sx, ex):
                                block[count, tframe] = input[store_pos, j % Sy, i % Sx]
                                count += 1

                # Svd
                u, s, vh = np.linalg.svd(block, full_matrices=False)

                for k in range(u.shape[1]):

                    # s[k] = max(s[k] - lamda, 0)
                    abs_input = abs(s[k])
                    if abs_input == 0:
                        sign = 0
                    else:
                        sign = s[k] / abs_input

                    s[k] = abs_input - lamda
                    s[k] = (abs(s[k]) + s[k]) / 2
                    s[k] = s[k] * sign

                    for i in range(u.shape[0]):
                        u[i, k] *= s[k]

                block = np.dot(u, vh)

                # Put block back
                for tframe in range(num_frames):
                    count = 0
                    for encode in range(num_encodes):
                        store_pos = int(tframe * num_encodes + encode)
                        for j in range(sy, ey):
                            for i in range(sx, ex):
                                output[store_pos, j % Sy, i % Sx] += scale*block[count, tframe]
                                count += 1

    return output



class SingularValueThresholdingNumba(sp.prox.Prox):

    def __init__(self, ishape, frames, num_encodes, lamda=None, block_size=8, block_stride=8, axis=0, block_iter=4, batched_iter=0):

        self.frames = frames
        self.num_encodes = num_encodes
        self.total_images = frames * num_encodes
        self.axis = axis
        self.block_iter = block_iter
        self.old_shape = np.array(ishape)
        self.new_shape = (self.total_images, int(self.old_shape[0] / self.total_images)) + tuple(self.old_shape[1:])

        self.lamda = lamda
        ndim = len(self.new_shape[1:])
        self.block_size = [block_size for _ in range(ndim)]
        self.block_stride = [block_stride for _ in range(ndim)]
        self.block_shape = tuple(self.block_size)

        print(f'Old shape = {self.old_shape}')
        print(f'New shape = {self.new_shape}')
        print(f'Block size = {self.block_size}')
        print(f'Block stride = {self.block_stride}')
        print(f'Block shape = {self.block_shape}')
        print(f'Block iter = {self.block_iter}')

        # input is output
        oshape = ishape

        super().__init__(oshape, ishape)

    def _prox(self, alpha, input):

        if math.isclose(self.lamda, 0.0):
            return input
        # org_image = np.copy(input)
        t = time.time()

        # Save input device
        initial_device = sp.get_device(input)
        input = initial_device.xp.reshape(input, self.new_shape)

        # Put on CPU
        input = sp.to_device(input, sp.cpu_device)

        #print(f'Pre {np.linalg.norm(input)}')
        # SVT thresholding
        output = np.zeros_like(input)
        # noticed block_shape is define differently in the numba svt.
        bthresh = float(self.lamda*alpha * np.sqrt(self.num_encodes*np.prod(self.block_shape)))
        print(f'Numba {bthresh}')
        if len(self.block_size) == 3:
            output = svt_numba3(output, input, bthresh, tuple(self.block_shape), tuple(self.block_stride), self.block_iter, self.num_encodes)
        elif len(self.block_size) ==2:
            output = svt_numba2(output, input, bthresh, tuple(self.block_shape), tuple(self.block_stride),
                                self.block_iter, self.num_encodes)
        else:
            raise RuntimeError(f'SVT only support 2 or 3 blocks but {len(self.block_size)} were requested')

        # Return on same device
        output = sp.to_device(output, initial_device)
        output = initial_device.xp.reshape(output, self.old_shape)

        print(f'SVT took {time.time() - t}')

        return output

def svt_torch_batch(x, lamda, blk_size, blk_stride, num_enc, num_frames, blk_shape):
    # 4D Blocking
    B = sp.linop.ArrayToBlocks(list(x.shape), list(blk_size), list(blk_stride))

    # Parse into blocks
    image = B * x

    # print('image shape B*x to get the number of blocks')
    # print(image.shape)

    # if blk_size[0] >= 2:
    # this probably brakes the image, but lets try to only use 1 reshape
    # reshape to (Nblocks, encode*blocks_size, frames)

    # image = np.moveaxis(image, 0, -1)
    # print(image.shape)

    # old_shape = image.shape

    # new_shape = (-1,  np.prod(blk_size), image.shape[-1])
    # new_shape = (-1,  np.prod(blk_shape), image.shape[-1]//num_enc)

    # image = torch.reshape(torch.from_numpy(image), new_shape)
    # print('image reshape', image.shape)
    # image = image.numpy()

    image = np.moveaxis(image, 0, 3)
    # print(image.shape)

    old_shape = image.shape

    new_shape = (-1, image.shape[3] // num_enc, num_enc * np.prod(blk_size))
    print(f'Old shape = {old_shape} new shape = {new_shape}')
    image = torch.reshape(torch.from_numpy(image), new_shape)
    # print('image reshape', image.shape)
    image = image.numpy()

    image = np.moveaxis(image, 1, -1)
    # print(image.shape)

    # else:
    # reshape to (Nblocks, encode*frames, prod(block_size))
    # image = np.moveaxis(image, 0, -4)
    # print(image.shape)

    # old_shape = image.shape

    # new_shape = (-1, image.shape[-4], np.prod(blk_size))
    # image = torch.reshape(torch.from_numpy(image), new_shape)
    # print('image reshape', image.shape)
    # image = image.numpy()

    # Scale lamda by block elements
    lamda *= np.sqrt(np.prod(blk_shape))  # is this dependent on the array shape of the svd?
    print(f'Torch inner {lamda}')
    nuclear_norm = 0.0
    # do the batched SVD, want shape of (batch, much larger ,much smaller) typically is fastest
    lr_batch_size = 64
    lr_batchs = (image.shape[0] + lr_batch_size - 1) // lr_batch_size
    for batch in range(lr_batchs):
        start = batch * lr_batch_size
        stop = min((batch + 1) * lr_batch_size, image.shape[0])

        image_t = image[start:stop, :, :]

        u, s, v = torch.svd(torch.from_numpy(image_t))

        # pytorch does not support complex tensor multiplication

        u = u.numpy()
        s = s.numpy()
        v = v.transpose(-2, -1)
        v = v.numpy()

        s = sp.soft_thresh(lamda, s)

        image[start:stop, :, :] = np.matmul(u * s[..., None, :], v)

    # if blk_size[0] >= 2:

    # image = torch.reshape(torch.from_numpy(image), old_shape)
    # image = image.numpy()
    # print(image.shape)

    # image = np.moveaxis(image, -1, 0)
    # print(image.shape)

    image = np.moveaxis(image, -1, 1)
    # print(image.shape)

    image = torch.reshape(torch.from_numpy(image), old_shape)
    image = image.numpy()
    # print(image.shape)

    image = np.moveaxis(image, 3, 0)
    # print(image.shape)

    # else:
    # image = torch.reshape(torch.from_numpy(image), old_shape)
    # image = image.numpy()
    # print('image shape after svt', image.shape)

    # image = np.moveaxis(image, -4, 0)

    nuclear_norm /= np.sqrt(np.prod(blk_shape)) * float(lr_batchs)
    x = B.H * image

    return x


class SingularValueThresholding(sp.prox.Prox):

    def __init__(self, ishape, frames, num_encodes, lamda=None,
                 block_size=8, block_stride=8, axis=0, block_iter=4, batched_iter=50):

        self.batched_iter = batched_iter
        self.frames = frames
        self.num_encodes = num_encodes
        self.total_images = frames * num_encodes
        self.axis = axis
        self.block_iter = block_iter
        self.old_shape = np.array(ishape)
        self.new_shape = (self.total_images, int(self.old_shape[0] / self.total_images)) + tuple(self.old_shape[1:])

        self.lamda = lamda
        self.block_size = [block_size, block_size, block_size]
        self.block_stride = [block_stride, block_stride, block_stride]

        self.block_shape = (num_encodes, block_size, block_size, block_size)
        self.iter_counter = 0

        print(f'Old shape = {self.old_shape}')
        print(f'New shape = {self.new_shape}')
        print(f'Block size = {self.block_size}')
        print(f'Block stride = {self.block_stride}')
        print(f'Block shape = {self.block_shape}')
        print(f'Block iter = {self.block_iter}')

        # input is output
        oshape = ishape

        super().__init__(oshape, ishape)

    def _prox(self, alpha, input):

        # Input is 3D (Nt*Nz x Ny x Nx)
        ########################New code###################################################
        if math.isclose(self.lamda, 0.0):
            return input

        # Save input device
        # initial_device = sp.get_device(input)
        # input = initial_device.xp.reshape(input, self.new_shape)
        # print('SVT input shape', input.shape)  # frames*encodes, z, y, x

        # test svt insite cycle spinning
        #
        #

        # t = time.time()
        # Put on CPU
        # input = sp.to_device(input, sp.cpu_device)

        # SVT thresholding
        # input_avg = np.zeros(input.shape,dtype=np.complex64)
        # bthresh = float(alpha * self.lamda)
        # for biter in range(self.block_iter):
        #    block_shift = [-np.random.randint(0, self.block_stride[e]) for e in range(3)]
        # bthresh = float(alpha*self.lamda/float(self.block_iter))
        # print('Shift {block_shift} with {bthresh}')
        #    input = svt_inplace(input, bthresh, tuple(self.block_size), tuple(self.block_stride), tuple(block_shift))
        #    input_avg += input

        # Return on same device
        # input = input_avg.copy()/self.block_iter
        # input = sp.to_device(input, initial_device)
        # input = initial_device.xp.reshape(input, self.old_shape)

        # print(f'SVT insite with cycle spinning took {time.time() - t}')

        # test svt pytorch with cycle spinning
        #
        #

        input = torch.reshape(torch.from_numpy(input), self.new_shape)
        bthresh = float(alpha * self.lamda)
        print(f'Torch {bthresh}')
        t = time.time()

        if self.iter_counter < self.batched_iter // 2:
            # SVT biter (cycle spinning) = 1
            # Block shifts are always negative
            block_shift = [-np.random.randint(0, self.block_stride[e]) for e in range(3)]
            block_ishift = [-x for x in block_shift]

            input = torch.roll(input, block_shift, dims=(-3, -2, -1))
            input = input.numpy()

            # input = initial_device.xp.roll(input, block_shift, axis=(-3, -2, -1))
            # Put on CPU
            # input = sp.to_device(input, sp.cpu_device)
            input = svt_torch_batch(input, bthresh, tuple(self.block_size), tuple(self.block_stride),
                                    self.num_encodes, self.frames, tuple(self.block_shape))
            # Return on same device
            # input = sp.to_device(input, initial_device)
            # input = initial_device.xp.roll(input, block_ishift, axis=(-3, -2, -1))

            input = torch.roll(torch.from_numpy(input), block_ishift, dims=(-3, -2, -1))

        else:
            input_avg = torch.zeros_like(input)
            # input_avg = initial_device.xp.zeros(input.shape,dtype=np.complex64)
            # SVT biter
            for biter in range(self.block_iter):
                print('iter: ', biter)

                # copy input
                input_copy = input.detach().clone()

                # Block shifts are always negative
                block_shift = [-np.random.randint(0, self.block_stride[e]) for e in range(3)]
                # print(block_shift)

                block_ishift = [-x for x in block_shift]
                # print(block_ishift)
                ### input copy of input avg????

                input_copy = torch.roll(input_copy, block_shift, dims=(-3, -2, -1))
                input_copy = input_copy.numpy()

                # input = torch.roll(input, block_shift, dims=(-3, -2, -1))
                # input = input.numpy()

                # input = initial_device.xp.roll(input, block_shift, axis=(-3, -2, -1))
                # Put on CPU
                # input = sp.to_device(input, sp.cpu_device)

                # input = svt_torch_batch_test(x=input, lamda=bthresh, blk_size=self.block_size,
                #                             blk_stride=tuple(self.block_stride),
                #                             num_enc=self.num_encodes, num_frames=self.frames,
                #                             blk_shape=tuple(self.block_shape))

                input_copy = svt_torch_batch(x=input_copy, lamda=bthresh, blk_size=self.block_size,
                                             blk_stride=tuple(self.block_stride),
                                             num_enc=self.num_encodes, num_frames=self.frames,
                                             blk_shape=tuple(self.block_shape))

                # Return on same device
                # input = sp.to_device(input, initial_device)
                # input = initial_device.xp.roll(input, block_ishift, axis=(-3, -2, -1))
                # input = torch.roll(torch.from_numpy(input), block_ishift, dims=(-3, -2, -1))

                input_copy = torch.roll(torch.from_numpy(input_copy), block_ishift, dims=(-3, -2, -1))

                input_avg += input_copy

            input_avg /= self.block_iter
            input = input_avg.detach().clone()

        input = torch.reshape(input, tuple(self.old_shape))
        input = input.numpy()

        self.iter_counter = self.iter_counter + 1

        # input = initial_device.xp.copy(input_avg)/self.block_iter
        # input = initial_device.xp.reshape(input, self.old_shape)
        print(f'SVT torch took {time.time() - t}')

        return (input)

    def _svt_thresh_batched(self, x, block_size, block_shape, block_stride, block_shift, lamda):
        # print(f'Block shift = {block_shift}')
        # print(f'Stride = {block_stride}')
        # print(f'Block size = {block_size}')
        # print(f'Block shape = {block_shape}')
        full_block_shift = list([0]) + block_shift
        # print('full block shift', full_block_shift)
        full_block_stride = list([block_shape[0]] + block_stride)
        # print('full block stride', full_block_stride)
        # print('block shape', block_shape)
        # print('x' , x.shape)
        print(f'Full stride = {full_block_stride}')

        # 4D Blocking
        B = sp.linop.ArrayToBlocks(list(x.shape), list(block_shape), list(full_block_stride))

        # x = np.roll( x, shift=self.blk_shift, axis=(0,1,2))

        # Parse into blocks
        image = B * x

        # print('image shape B*x')
        # print(image.shape)

        # reshape to (Nblocks, encode, prod(block_size) )
        old_shape = image.shape
        image = np.moveaxis(image, 0, -1)  # First axis is time
        new_shape = (-1, np.prod(block_shape), image.shape[-1])
        # print(f'Resize from {old_shape} to {new_shape}')

        image = np.reshape(image, new_shape)
        # print('image reshape')
        # print(image.shape)

        # Scale lamda by block elements
        lamda *= np.sqrt(np.prod(block_shape))
        nuclear_norm = 0.0
        # print(image.shape)
        lr_batch_size = 256
        lr_batchs = (image.shape[0] + lr_batch_size - 1) // lr_batch_size
        for batch in range(lr_batchs):
            start = batch * lr_batch_size
            stop = min((batch + 1) * lr_batch_size, image.shape[0])

            image_t = image[start:stop, :, :]
            # print(image_t.shape)

            u, s, vh = np.linalg.svd(image_t, full_matrices=False)

            nuclear_norm += np.mean(np.abs(s))

            # Threshold
            s = sp.soft_thresh(lamda, s)

            image[start:stop, :, :] = np.matmul(u * s[..., None, :], vh)

        # Back to GPU
        image = np.moveaxis(image, -1, 0)
        image = np.reshape(image, newshape=old_shape)

        nuclear_norm /= np.sqrt(np.prod(block_shape)) * float(lr_batchs)

        x = B.H * image
        # th_x = B.H * image
        # mask = (th_x == 0)
        # th_x[mask] = x[mask]
        # x = th_x

        return x

    def _svt_thresh_batched_stack_3D_linops(self, x, block_size, block_shape, block_stride, block_shift, lamda, frames,
                                            num_encodes):

        # print('num of frames and encodes', frames, num_encodes)
        # print('shape x',x.shape)

        Block_op = sp.linop.ArrayToBlocks(list(x.shape[-3:]), list(block_size), list(block_stride))  # z,y,x
        # print(Block_op)
        Block_reshape = sp.linop.Reshape(oshape=Block_op.oshape + [1], ishape=Block_op.oshape)
        # print(Block_reshape)
        x_reshape = sp.linop.Reshape(x.shape[-3:], (1,) + x.shape[-3:])
        # print(x_reshape)
        BB = sp.linop.Diag([Block_reshape * Block_op * x_reshape for i in range(x.shape[0])], oaxis=6, iaxis=0)
        # print(BB)
        image = BB * x
        # print('old image shape', image.shape)

        image = np.transpose(np.expand_dims(image, axis=7), (6, 7, 0, 1, 2, 3, 4, 5))
        # print('image shape after transpose', image.shape)
        image_old_block_shape = image.shape

        image_new_block_shape = tuple([frames]) + tuple([num_encodes]) + tuple(image.shape[2:])
        image = np.reshape(image, image_new_block_shape)
        # print('image shape after new reshape', image.shape)

        image = np.transpose(image, (0, 2, 3, 4, 1, 5, 6, 7))
        # print('image shape after transpose', image.shape)

        # print('new image shape', image.shape)

        # reshape to (Nblocks, encode, prod(block_size) )
        old_shape = image.shape
        image = np.moveaxis(image, 0, -1)  # First axis is time
        new_shape = (-1, np.prod(block_shape), image.shape[-1])
        # print(f'Resize from {old_shape} to {new_shape}')

        image = np.reshape(image, new_shape)
        # print('image reshape')
        # print(image.shape)

        # print('ARRAY type ', image.dtype)
        # print('ARRAY size ', image.shape)
        # print('ARRAY max ', np.amax(image))
        # print('ARRAY min ', np.amin(image))
        # print('ARRAY median ', np.median(image))

        # Scale lamda by block elements
        lamda *= np.sqrt(np.prod(block_shape))
        nuclear_norm = 0.0
        # print(image.shape)
        lr_batch_size = 128
        lr_batchs = (image.shape[0] + lr_batch_size - 1) // lr_batch_size
        for batch in range(lr_batchs):
            start = batch * lr_batch_size
            stop = min((batch + 1) * lr_batch_size, image.shape[0])

            image_t = image[start:stop, :, :]
            # print(image_t.shape)

            u, s, vh = np.linalg.svd(image_t, full_matrices=False)

            nuclear_norm += np.mean(np.abs(s))

            # Threshold
            s = sp.soft_thresh(lamda, s)

            image[start:stop, :, :] = np.matmul(u * s[..., None, :], vh)

        # Back to GPU
        image = np.moveaxis(image, -1, 0)
        image = np.reshape(image, newshape=old_shape)
        # print('image shape after svt', image.shape)

        nuclear_norm /= np.sqrt(np.prod(block_shape)) * float(lr_batchs)

        image = np.transpose(image, (0, 4, 1, 2, 3, 5, 6, 7))
        # print('image shape after transpose', image.shape)

        image = np.reshape(image, image_old_block_shape)
        # print('image shape after reshape', image.shape)
        image = np.transpose(image, (1, 2, 3, 4, 5, 6, 7, 0))
        # print('image shape after transpose', image.shape)

        image = np.squeeze(image, axis=0)
        # print('image shape after squeeze', image.shape)

        x = BB.H * image

        return x

    def _svt_thresh_batched_with_matrix_manipulation(self, x, block_size, block_shape, block_stride, block_shift,
                                                     lamda):

        # print('num of frames and encodes', frames, num_encodes)
        print('shape x', x.shape)

        blocks = [x.shape[-3] // block_size[-3], x.shape[-2] // block_size[-2], x.shape[-1] // block_size[-1]]
        # print(blocks)

        old_shape = x.shape  # [tf*#encodes, z, y, x]
        # Scale lamda by block elements
        lamda *= np.sqrt(np.prod(block_shape))
        nuclear_norm = 0.0

        for bz in range(blocks[0]):
            for by in range(blocks[1]):
                for bx in range(blocks[2]):
                    # print('Block %d %d %s' % (bz, by, bx))

                    bx_shift = bx * block_size[2]
                    by_shift = by * block_size[1]
                    bz_shift = bz * block_size[0]

                    # Get start stop
                    istart = bx_shift
                    jstart = by_shift
                    kstart = bz_shift

                    istop = istart + block_size[2]
                    jstop = jstart + block_size[1]
                    kstop = kstart + block_size[0]

                    # Grab the block
                    # Grab the block
                    image_block = x[:, kstart:kstop, jstart:jstop, istart:istop]
                    old_shape_block = image_block.shape  # [tf*#encodes, 16, 16, 16]

                    new_shape_block = (-1, np.prod(block_shape))  # [tf, #encodes*16*16*16]
                    image_block = np.reshape(image_block, new_shape_block)
                    image_block = np.moveaxis(image_block, 0, -1)  # [#encodes*16*16*16, tf]

                    # print(image_block.shape)
                    u, s, vh = np.linalg.svd(image_block, full_matrices=False)
                    nuclear_norm += np.mean(np.abs(s))
                    # Threshold
                    s = sp.soft_thresh(lamda, s)

                    image_block[:, :] = np.matmul(u * s[..., None, :], vh)
                    image_block = np.moveaxis(image_block, -1, 0)  # [tf, encodes*16*16*16]
                    image_block = np.reshape(image_block, old_shape_block)  # [tf*#encodes, 16, 16, 16]
                    # print(image_block.shape)
                    # print('Block %d %d %s' % (bz, by, bx))
                    x[:, kstart:kstop, jstart:jstop, istart:istop] = image_block  # [tf*#encodes, z, x, y]

        return x

    def _svt_thresh_batched_with_matrix_manipulation_2SVT(self, x, block_size, block_shape, block_stride, block_shift,
                                                          lamda):

        blocks = [x.shape[-3] // block_size[-3], x.shape[-2] // block_size[-2], x.shape[-1] // block_size[-1]]
        # print(blocks)
        num_shifts = 2
        old_shape = x.shape  # [tf*#encodes, z, y, x]
        x = np.reshape(x, (self.frames, self.num_encodes, -1) + x.shape[1:])  # [tf, #encodes, 1, z, y, x]

        # Scale lamda by block elements and num shifts
        lamda *= np.sqrt(np.prod(block_shape))
        lamda *= 1 / num_shifts

        for nt in range(num_shifts):
            nuclear_norm = 0.0
            for bz in range(blocks[0]):
                for by in range(blocks[1]):
                    for bx in range(blocks[2]):
                        # print('Block %d %d %s' % (bz, by, bx))

                        bx_shift = bx * block_size[2]
                        by_shift = by * block_size[1]
                        bz_shift = bz * block_size[0]

                        # Get start stop
                        istart = bx_shift
                        jstart = by_shift
                        kstart = bz_shift

                        istop = istart + block_size[2]
                        jstop = jstart + block_size[1]
                        kstop = kstart + block_size[0]

                        # Grab the block
                        if istop < x.shape[-1] and jstop < x.shape[-2] and kstop < x.shape[-3]:  # from new shape
                            # Grab the block
                            image_block = x[:, :, 0, kstart:kstop, jstart:jstop, istart:istop]
                            old_shape_block = image_block.shape  # [tf, #encodes, 1, 16, 16, 16]

                            new_shape_block = (-1, np.prod(block_shape))  # [tf, #encodes*16*16*16]
                            image_block = np.reshape(image_block, new_shape_block)

                            # print(image_block.shape)
                            u, s, vh = np.linalg.svd(image_block, full_matrices=False)
                            nuclear_norm += np.mean(np.abs(s))
                            # Threshold
                            s = sp.soft_thresh(lamda, s)

                            image_block[:, :] = np.matmul(u * s[..., None, :], vh)
                            image_block = np.reshape(image_block, old_shape_block)  # [tf, #encodes, 1, 16, 16, 16]
                            # print(image_block.shape)
                            # print('Block %d %d %s' % (bz, by, bx))
                            x[:, :, 0, kstart:kstop, jstart:jstop,
                            istart:istop] = image_block  # [tf, #encodes, 1, z, x, y]

        x = np.reshape(x, old_shape)  # [tf*#encodes, z, x, y]

        return x

if __name__ == '__main__':

    num_encodes = 4
    ishape = (40*128, 128, 128)
    frames = 10

    # random noise
    noise = np.random.random_sample(ishape).astype(np.complex64) + \
            1j * np.random.random_sample(ishape).astype(np.complex64)
    noise -= 0.5 + 0.5*1j

    x = np.zeros_like(noise)
    for t in range(ishape[0]):
        spos = t // 128
        x[t,...] = (t // 128 )  // num_encodes

    y = x + noise

    svtn = SingularValueThresholdingNumba(ishape, frames, num_encodes, lamda=1, block_size=4, block_stride=4, axis=0, block_iter=4, batched_iter=0)
    svtp = SingularValueThresholding(ishape, frames, num_encodes, lamda=1, block_size=4, block_stride=4, axis=0, block_iter=4, batched_iter=0)

    t = time.time()
    out_svtn = svtn._prox(1.0, y)
    print(f'Numba took {time.time() - t}')

    t = time.time()
    out_svtp = svtp._prox(1.0, y)
    print(f'Pytorch took {time.time() - t}')

    diff = np.linalg.norm(np.abs(out_svtn-out_svtp))
    base = np.linalg.norm(np.abs(y))

    print(f'Diff = {diff/base}')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(out_svtn[:, 64, 64])
    plt.plot(out_svtp[:, 64, 64])
    plt.plot(y[:, 64, 64])
    plt.plot(x[:, 64, 64])

    plt.legend(('Numba','Pytorch','Original','Ideal'))
    plt.show()

    plt.figure()
    plt.subplot(221)
    temp = np.concatenate((np.abs(out_svtn[64+0, :,:]), np.abs(out_svtn[64+128, :,:])))
    plt.imshow(temp, vmin=-0.1, vmax=1.2)
    plt.title('Numba')

    plt.subplot(222)
    temp = np.concatenate((np.abs(out_svtp[64+0, :,:]), np.abs(out_svtp[64+128, :,:])))
    plt.imshow(temp, vmin=-0.1, vmax=1.2)
    plt.title('Pytorch')

    plt.subplot(223)
    temp = np.concatenate((np.abs(y[64+0, :,:]), np.abs(y[64+128, :,:])))
    plt.imshow(temp, vmin=-0.1, vmax=1.2)
    plt.title('Ideal + Noise')

    plt.subplot(224)
    temp = np.concatenate((np.abs(x[64+0, :,:]), np.abs(x[64+128, :,:])))
    plt.imshow(temp, vmin=-0.1, vmax=1.2)
    plt.title('Ideal')
    plt.show()








