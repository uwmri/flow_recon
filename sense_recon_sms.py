import sigpy as sp
import numpy as np
import math

def _estimate_weights(y, weights, coord):
    if weights is None and coord is None:
        with sp.get_device(y):
            weights = (sp.rss(y, axes=(0,)) > 0).astype(y.dtype)

    return weights

class SenseSMSRecon(sp.app.LinearLeastSquares):
    r"""SENSE Reconstruction.

    Considers the problem

    .. math::
        \min_x \frac{1}{2} \| P F S x - y \|_2^2 +
        \frac{\lambda}{2} \| x \|_2^2

    where P is the sampling operator, F is the Fourier transform operator,
    S is the SENSE operator, x is the image, and y is the k-space measurements.
    
    Modified to add operators for simultaneous NUFFT and SMS phase modulation

    Args:
        y (array): k-space measurements.
        mps (array): sensitivity maps.
        sms_factor (int): SMS factor.
        blips (None or array): precalculated phase modulation blips for SMS.
        lamda (float): regularization parameter.
        weights (float or array): weights for data consistency.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
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

    def __init__(
        self,
        y,
        mps,
        sms_factor=1,
        blips=None,
        lamda=0,
        weights=None,
        tseg=None,
        coord=None,
        device=sp.cpu_device,
        coil_batch_size=None,
        comm=None,
        show_pbar=True,
        transp_nufft=False,
        **kwargs
    ):
        weights = _estimate_weights(y, weights, coord)
        if weights is not None:
            y = sp.to_device(y * weights**0.5, device=device)
        else:
            y = sp.to_device(y, device=device)
        
        # extend ishape dimensions to include multiple slices
        if sms_factor > 1:
            ishape = tuple(list(mps.shape[1:]) + [sms_factor])
            mps = np.repeat(mps[..., None], sms_factor, axis=-1)
        else:
            ishape = mps.shape[1:]

        A = SenseSMS(
            mps,
            sms_factor=sms_factor,
            blips=blips,
            coord=coord,
            weights=weights,
            ishape=ishape,
            tseg=tseg,
            coil_batch_size=coil_batch_size,
            comm=comm,
            transp_nufft=transp_nufft,
        )

        if comm is not None:
            show_pbar = show_pbar and comm.rank == 0

        super().__init__(A, y, lamda=lamda, show_pbar=show_pbar, **kwargs)



def SenseSMS(
    mps,
    sms_factor=1,
    blips=None,
    coord=None,
    weights=None,
    tseg=None,
    ishape=None,
    coil_batch_size=None,
    comm=None,
    transp_nufft=False,
):
    """Sense linear operator.
    
    Modified to add operators for simultaneous NUFFT and SMS phase modulation

    Args:
        mps (array): sensitivity maps of length = number of channels.
        sms_factor (int): SMS factor.
        blips (None or array): precalculated phase modulation blips for SMS.
        coord (None or array): coordinates.
        weights (None or array): k-space weights.
            Useful for soft-gating or density compensation.
        tseg (None or Dictionary): parameters for time-segmented off-resonance
            correction. Parameters are 'b0' (array), 'dt' (float),
            'lseg' (int), and 'n_bins' (int). Lseg is the number of
            time segments used, and n_bins is the number of histogram bins.
        ishape (None or tuple): image shape.
        coil_batch_size (None or int): batch size for processing multi-channel.
            When None, process all coils at the same time.
            Useful for saving memory.
        comm (None or `sigpy.Communicator`): communicator
            for distributed computing.

    """
    # Get image shape and dimension.
    if ishape is None:
        ishape = mps.shape[1:]
        img_ndim = mps.ndim - 1
    else:
        img_ndim = len(ishape) - 1 # - 1 for sms_factor (fix later)

    # Serialize linop if coil_batch_size is smaller than num_coils.
    num_coils = len(mps)
    if coil_batch_size is None:
        coil_batch_size = num_coils

    if coil_batch_size < len(mps):
        num_coil_batches = (num_coils + coil_batch_size - 1) // coil_batch_size
        A = sp.linop.Vstack(
            [
                SenseSMS(
                    mps[c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                    sms_factor=sms_factor,
                    blips=blips[c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                    coord=coord,
                    weights=weights,
                    ishape=ishape,
                )
                for c in range(num_coil_batches)
            ],
            axis=0,
        )

        if comm is not None:
            C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
            A = A * C

        return A

    # Create Sense linear operator
    S = sp.linop.Multiply(ishape, mps)
        
    if tseg is None:
        if coord is None:
            F = sp.linop.FFT(S.oshape, axes=range(-img_ndim, 0))
        else:
            if sms_factor > 1:
                # SMS NUFFT operator to do NUFFT on multiple slices
                F = SMS_NUFFT(S.oshape, coord)
            else:
                if transp_nufft is False:
                    F = sp.linop.NUFFT(S.oshape, coord)
                else:
                    F = sp.linop.NUFFT(S.oshape, -coord).H

        A = F * S

    # If B0 provided, perform time-segmented off-resonance compensation
    else:
        if transp_nufft is False:
            F = sp.linop.NUFFT(S.oshape, coord)
        else:
            F = sp.linop.NUFFT(S.oshape, -coord).H
        time = len(coord) * tseg["dt"]
        b, ct = sp.mri.util.tseg_off_res_b_ct(
            tseg["b0"], tseg["n_bins"], tseg["lseg"], tseg["dt"], time
        )
        for ii in range(tseg["lseg"]):
            Bi = sp.linop.Multiply(F.oshape, b[:, ii])
            Cti = sp.linop.Multiply(S.ishape, ct[:, ii].reshape(S.ishape))

            # operation below is effectively A = A + Bi * F(Cti * S)
            if ii == 0:
                A = Bi * F * S * Cti
            else:
                A = A + Bi * F * S * Cti
    
    if sms_factor > 1:
        # apply phase modulation and sum
        Z = SMSMultiply(F.oshape, F.oshape[:-1], blips, sms_factor)
        A = Z * A
    
    if weights is not None:
        with sp.get_device(weights):
            P = sp.linop.Multiply(F.oshape[:-1], weights**0.5)
            A = P * A

    if comm is not None:
        C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
        A = A * C
    
    A.repr_str = "SenseSMS"
    return A


# Preforms NUFFT on multiple slices
class SMS_NUFFT(sp.linop.Linop):
    def __init__(self, ishape, coord, oversamp=1.25, width=4):
        self.coord = coord
        self.oversamp = oversamp
        self.width = width
        self.nslices = ishape[-1] 

        # Input shape: [ncoils, nx, ny, nslices]
        self.ishape = ishape
        # Output shape: [ncoils, nprojs, nslices]
        oshape = list(ishape[:-coord.shape[-1]-1]) + list(coord.shape[:-1]) + [self.nslices]

        super().__init__(oshape, ishape)
    
    def _apply(self, x):
        # x = [ncoils, nx, ny, nslices]
        device = sp.get_device(x)
        with device:
            coord = sp.to_device(self.coord, device)
            output_list = []
            for s in range(self.nslices):
                # [ncoils, nx, ny] -> [ncoils, nprojs]
                slice_output = sp.fourier.nufft(
                    x[..., s], coord, 
                    oversamp=self.oversamp, 
                    width=self.width
                )
                output_list.append(slice_output)
                
            # output = [ncoils, nprojs, nslices]
            output = np.stack(output_list, axis=-1)
            return output
    
    def _adjoint_linop(self):
        return SMS_NUFFTAdjoint(
            self.ishape, self.coord,
            oversamp=self.oversamp, 
            width=self.width
        )

class SMS_NUFFTAdjoint(sp.linop.Linop):
    def __init__(self, oshape, coord, oversamp=1.25, width=4):
        self.coord = coord
        self.oversamp = oversamp
        self.width = width
        self.nslices = oshape[-1]
        
        # Output shape: [ncoils, nx, ny, nslices]
        self.oshape = oshape
        # Input shape: [ncoils, nproj, nslices]
        ishape = list(oshape[:-coord.shape[-1]-1]) + list(coord.shape[:-1]) + [self.nslices]
        
        super().__init__(oshape, ishape)
    
    def _apply(self, x):
        # x = [ncoils, nprojs, nslices]
        device = sp.get_device(x)
        with device:
            coord = sp.to_device(self.coord, device)
            output_list = []
            for s in range(self.nslices):
                # [ncoils, nprojs] -> [ncoils, nx, ny]
                slice_output = sp.fourier.nufft_adjoint(
                    x[..., s], coord, 
                    oshape=self.oshape[:-1],
                    oversamp=self.oversamp,
                    width=self.width
                )
                output_list.append(slice_output)
                
            # output = [ncoils, nx, ny, nslices]
            output = np.stack(output_list, axis=-1)
            return output
    
    def _adjoint_linop(self):
        return SMS_NUFFT(
            self.oshape, self.coord,
            oversamp=self.oversamp,
            width=self.width
        )
        
# Applies SMS phase modulation to multiple slices using precomputed blips
class SMSMultiply(sp.linop.Linop):
    def __init__(self, ishape, oshape, blips, sms_factor):
        self.sms_factor = sms_factor
        self.blips = blips
        
        # [ncoils, nprojs, nslices]
        self.ishape = ishape
        # [ncoils, nprojs]                   
        self.oshape = oshape              
               
        super().__init__(self.oshape, self.ishape)

    def _apply(self, x):
        # x = [ncoils, nprojs, nslices]
        device = sp.get_device(x)
        with device:
            output = sp.to_device(np.zeros(self.oshape, dtype=np.complex64), device)
            for s in range(self.sms_factor):
                # Apply SMS phase modulation
                modulated = x[..., s] * self.blips[..., s]
                output += modulated 
                
        # output = [ncoils, nprojs]
        return output
    
    def _adjoint_linop(self):
        return SMSMultiplyAdjoint(self.oshape, self.ishape, self.blips, self.sms_factor)

class SMSMultiplyAdjoint(sp.linop.Linop):
    def __init__(self, ishape, oshape, blips, sms_factor):
        self.sms_factor = sms_factor
        self.blips = blips
        
        # [ncoils, nprojs]
        self.ishape = ishape
        # [ncoils, nprojs, nslices]         
        self.oshape = oshape         
        
        super().__init__(self.oshape, self.ishape)
    
    def _apply(self, x):
        # x = [ncoils, nprojs]
        device = sp.get_device(x)
        with device:
            slice_list = []
            for s in range(self.sms_factor):
                # Apply SMS phase demodulation (conjugate)
                demodulated = x * np.conj(self.blips[..., s])
                slice_list.append(demodulated)
                
            # output = [ncoils, nx, ny, nslices]
            output = sp.to_device(np.stack(slice_list, axis=-1), device)
            return output
    
    def _adjoint_linop(self):
        return SMSMultiply(self.oshape, self.ishape, self.blips, self.sms_factor)
    
    