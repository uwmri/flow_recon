# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import numpy as np
import numba as nb

from sigpy import backend, config, util
from gpu_ops import *

__all__ = ['readout_gridding']


KERNELS = ['spline', 'kaiser_bessel']

import os
os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"
os.environ["CUPY_CUDA_COMPILE_WITH_DEBUG"] = "1"
os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"



def readout_gridding(input, coord, dcf, npts, kernel="spline", width=2, param=1):
    r"""Gridding of points specified by coordinates to array.

    Let :math:`y` be the input, :math:`x` be the output,
    :math:`c` be the coordinates, :math:`W` be the kernel width,
    and :math:`K` be the interpolation kernel, then the function computes,

    .. math ::
        x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
               K\left(\frac{i - c[j]}{W / 2}\right) y[j]

    There are two types of kernels: 'spline' and 'kaiser_bessel'.

    'spline' uses the cardinal B-spline functions as kernels.
    The order of the spline can be specified using param.
    For example, param=1 performs linear interpolation.
    Concretely, for param=0, :math:`K(x) = 1`,
    for param=1, :math:`K(x) = 1 - |x|`, and
    for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
    for :math:`|x| > \frac{1}{3}`
    and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.

    These function expressions are derived from the reference wikipedia
    page by shifting and scaling the range to -1 to 1.
    When the coordinates specifies a uniformly spaced grid,
    it is recommended to use the original scaling with width=param + 1
    so that the interpolation weights add up to one.

    'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
    Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
    where :math:`I_0` is the modified Bessel function of the first kind.
    The beta parameter can be specified with param.
    The modified Bessel function of the first kind is approximated
    using the power series, following the reference.

    Args:
        input (array): Input array  (same size as input)
        coord (array): 1D array coordinates  (same size as input)
        dcf (array): Density compensation (same size as input)
        width (float or tuple of floats): Interpolation kernel full-width.
        kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
        param (float or tuple of floats): Kernel parameter.

    Returns:
        output (array): Output array.

    References:
        https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
        http://people.math.sfu.ca/~cbm/aands/page_378.htm
    """

    # Copy to GPU
    input = array_to_gpu(input)
    coord = array_to_gpu(coord)
    dcf = array_to_gpu(dcf)

    # Coords is [batch size, xres]
    ndim = 1

    # Get the number of batches (this i)
    readouts_shape = coord.shape[:-1]
    readouts_size = util.prod(readouts_shape)

    output_shape = list(readouts_shape) + list(npts)
    output_shape_flat = [readouts_size, ] + list(npts)

    xp = backend.get_array_module(input)
    isreal = np.issubdtype(input.dtype, np.floating)

    # Reshape to the input 
    input = input.reshape([readouts_size, -1])
    coord = coord.reshape([readouts_size, -1])
    dcf = dcf.reshape([readouts_size, -1])

    output = xp.zeros(output_shape_flat, dtype=input.dtype)

    if np.isscalar(param):
        param = xp.array([param] * ndim, coord.dtype)
    else:
        param = xp.array(param, coord.dtype)

    if np.isscalar(width):
        width = xp.array([width] * ndim, coord.dtype)
    else:
        width = xp.array(width, coord.dtype)

    if xp == np:
        _gridding[kernel](output, input, coord, dcf, width, param)
    else:  # pragma: no cover
        if isreal:
            _gridding_cuda[kernel](
                input, coord, dcf, width, param, output, size=input.shape[0])
        else:
            _gridding_cuda_complex[kernel](
                input, coord, dcf, width, param, output, size=input.shape[0])

    output = backend.to_device(output)
    coord = backend.to_device(coord)
    dcf = backend.to_device(dcf)

    return output.reshape(output_shape)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _spline_kernel(x, order):
    if abs(x) > 1:
        return 0

    if order == 0:
        return 1
    elif order == 1:
        return 1 - abs(x)
    elif order == 2:
        if abs(x) > 1 / 3:
            return 9 / 8 * (1 - abs(x))**2
        else:
            return 3 / 4 * (1 - 3 * x**2)


@nb.jit(nopython=True, cache=True)  # pragma: no cover
def _kaiser_bessel_kernel(x, beta):
    if abs(x) > 1:
        return 0

    x = beta * (1 - x**2)**0.5
    t = x / 3.75
    if x < 3.75:
        return 1 + 3.5156229 * t**2 + 3.0899424 * t**4 +\
            1.2067492 * t**6 + 0.2659732 * t**8 +\
            0.0360768 * t**10 + 0.0045813 * t**12
    else:
        return x**-0.5 * np.exp(x) * (
            0.39894228 + 0.01328592 * t**-1 +
            0.00225319 * t**-2 - 0.00157565 * t**-3 +
            0.00916281 * t**-4 - 0.02057706 * t**-5 +
            0.02635537 * t**-6 - 0.01647633 * t**-7 +
            0.00392377 * t**-8)


def _get_gridding(kernel):
    if kernel == 'spline':
        kernel = _spline_kernel
    elif kernel == 'kaiser_bessel':
        kernel = _kaiser_bessel_kernel

    @nb.jit(nopython=True)  # pragma: no cover
    def _gridding1(output, input, coord, dcf, width, param):
        readouts, nx = input.shape
        readouts, nx_out = output.shape 

        for readout in range(readouts):
            for i in range(nx):
                kx = coord[readout, i]
                dcf_t = dcf[readout, i]
                x0 = max(int(np.ceil(kx - width[-1] / 2)),0)
                x1 = min(int(np.floor(kx + width[-1] / 2)),nx_out-1)
                for x in range(x0, x1 + 1):
                    w = dcf_t * kernel((x - kx) / (width[-1] / 2), param[-1])
                    output[readout, x] += w * input[readout, i]

        return output

    return _gridding1


_gridding = {}
for kernel in KERNELS:
    _gridding[kernel] = _get_gridding(kernel)

if config.cupy_enabled:  # pragma: no cover
    import cupy as cp

    _spline_kernel_cuda = """
    __device__ inline S kernel(S x, S order) {
        if (fabsf(x) > 1)
            return 0;

        if (order == 0)
            return 1;
        else if (order == 1)
            return 1 - fabsf(x);
        else if (fabsf(x) > 1 / 3)
            return 9 / 8 * (1 - fabsf(x)) * (1 - fabsf(x));
        else
            return 3 / 4 * (1 - 3 * x * x);
    }
    """

    _kaiser_bessel_kernel_cuda = """
    __device__ inline S kernel(S x, S beta) {
        if (fabsf(x) > 1)
            return 0;

        x = beta * sqrt(1 - x * x);
        S t = x / 3.75;
        S t2 = t * t;
        S t4 = t2 * t2;
        S t6 = t4 * t2;
        S t8 = t6 * t2;
        if (x < 3.75) {
            S t10 = t8 * t2;
            S t12 = t10 * t2;
            return 1 + 3.5156229 * t2 + 3.0899424 * t4 +
                1.2067492 * t6 + 0.2659732 * t8 +
                0.0360768 * t10 + 0.0045813 * t12;
        } else {
            S t3 = t * t2;
            S t5 = t3 * t2;
            S t7 = t5 * t2;

            return exp(x) / sqrt(x) * (
                0.39894228 + 0.01328592 / t +
                0.00225319 / t2 - 0.00157565 / t3 +
                0.00916281 / t4 - 0.02057706 / t5 +
                0.02635537 / t6 - 0.01647633 / t7 +
                0.00392377 / t8);
        }
    }
    """

    mod_cuda = """
    __device__ inline int mod(int x, int n) {
        return (x % n + n) % n;
    }
    """

    def _get_gridding_cuda(kernel):
        if kernel == 'spline':
            kernel = _spline_kernel_cuda
        elif kernel == 'kaiser_bessel':
            kernel = _kaiser_bessel_kernel_cuda

        _gridding1_cuda = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S dcf, raw S width, raw S param',
            'raw T output',
            """
            
            const int xres = input.shape()[1];
            for (int j=0; j < xres; j++){
                const int coord_idx[] = {i, j};
                const S kx = coord[coord_idx];
                const S dcf_t = dcf[coord_idx];
                const int x0 = max( (int)ceil(kx - width[0] / 2.0), (int)0);
                const int x1 = min( (int)floor(kx + width[0] / 2.0), (int)(xres-1));

                for (int x = x0; x < x1 + 1; x++) {
                    const S w = dcf_t * kernel(((S) x - kx) / (width[0] / 2.0), param[0]);
                    const T v = (T) w * input[coord_idx];
                    const int output_idx[] = {i, x};
                    atomicAdd(&output[output_idx], v);   
                }
            }
            

            """,
            name='gridding1',
            preamble=kernel + mod_cuda,
            reduce_dims=False)
        
        return _gridding1_cuda


    def _get_gridding_cuda_complex(kernel):
        if kernel == 'spline':
            kernel = _spline_kernel_cuda
        elif kernel == 'kaiser_bessel':
            kernel = _kaiser_bessel_kernel_cuda

        _gridding1_cuda_complex = cp.ElementwiseKernel(
            'raw T input, raw S coord, raw S dcf, raw S width, raw S param',
            'raw T output',
            """

            const int ndim = 1;
            const int readouts = input.shape()[0];
            const int xres = input.shape()[1];

            for (int j=0; j < xres; j++){
                const int coord_idx[] = {i, j};
                const S kx = coord[coord_idx];
                const S dcf_t = dcf[coord_idx];
                const int x0 = max( (int)ceil(kx - width[ndim - 1] / 2.0), (int)0);
                const int x1 = min( (int)floor(kx + width[ndim - 1] / 2.0), (int)(xres-1));

                for (int x = x0; x < x1 + 1; x++) {
                    const S w = dcf_t * kernel(
                        ((S) x - kx) / (width[ndim - 1] / 2.0), param[ndim - 1]);
                    
                    if( (x > 0) && ( x < xres) ){
                        const int output_idx[] = {i, x};
                        const T v = (T) w * input[coord_idx];
                        atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])), v.real());
                        atomicAdd(reinterpret_cast<T::value_type*>(&(output[output_idx])) + 1, v.imag());
                    }

                }
            }


            """,
            name='gridding1_complex',
            preamble=kernel + mod_cuda,
            reduce_dims=False)
        
        return _gridding1_cuda_complex

    _gridding_cuda = {}
    _gridding_cuda_complex = {}
    for kernel in KERNELS:
        _gridding_cuda[kernel] = _get_gridding_cuda(kernel)
        _gridding_cuda_complex[kernel] = _get_gridding_cuda_complex(kernel)
