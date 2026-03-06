import ants
import numpy as np 
import cupy as cp
import sigpy as sp
import h5py
import nibabel
from scipy import signal, ndimage, interpolate
from cupyx.scipy import signal as csignal
from cupyx.scipy import ndimage as cndimage
from cupyx.scipy import interpolate as cinterpolate

from mri_raw import *

__all__ = ['iMoCoRecon']

class iMoCoRecon:
    '''
    Zhu X, Chan M, Lustig M, Johnson KM, Larson PEZ. Iterative motion-compensation reconstruction 
    ultra-short TE (iMoCo UTE) for high-resolution free-breathing pulmonary MRI. Magn Reson Med. 
    2020; 83: 1208–1221. https://doi.org/10.1002/mrm.27998
    
    https://github.com/PulmonaryMRI/imoco_recon
    
    '''
    def __init__ (self, mri_data, mps, gate_type='ecg', card_frames=1, resp_frames=1, time_ranges=None, lamda=0,
                 device=None, res_scale=2, coil_batch_size=None, max_iter=50, show_pbar=True,
                 comm=None, resp_filter_window=5, out_folder=None, debug=False, **kwargs):
        
        
        self.mri_data = mri_data
        self.mps = mps
        self.gate_type = gate_type
        self.card_frames = card_frames
        self.resp_frames = resp_frames
        self.time_ranges = time_ranges
        self.lamda = lamda
        self.device = device
        self.res_scale = res_scale
        self.coil_batch_size = coil_batch_size
        self.max_iter = max_iter
        self.show_pbar = show_pbar
        self.comm = comm
        self.resp_filter_window = resp_filter_window
        self.out_folder = out_folder
        self.logger = logging.getLogger('iMoCoRecon')
        self.debug = debug
                
        
    def register(self, fixed, moving):
        fixed = ants.from_numpy(fixed)
        moving = ants.from_numpy(moving)
        
        reg = ants.registration(fixed, moving, type_of_transform='SyNOnly', initial_transform="identity",\
                                syn_metric='demons', syn_sampling=4, \
                                grad_step=0.1, flow_sigma=5, total_sigma=3,\
                                reg_iterations=(100,100,40,20,10), \
                                verbose=False, outprefix=self.out_folder, \
                                w='[0.1,1]', write_composite_transform=False)
        
        M = nibabel.load(reg['fwdtransforms'][0])
        iM = nibabel.load(reg['invtransforms'][-1]) 
        Mt = np.squeeze(M.get_fdata())
        iMt = np.squeeze(iM.get_fdata())
        
        return Mt, iMt
    
    def M_scale(self, M, oshape):
        Mscale = [oshape[i]/M.shape[i] for i in range(M.shape[-1])]
        Mo = np.zeros(oshape+(M.shape[-1],))
        for i in range(M.shape[-1]):
            M[...,i] = M[...,i]*(Mscale[i])
            Mo[...,i] = ndimage.zoom(M[...,i],zoom=tuple(Mscale),order=2)

        return Mo
    
    # use GPU acceleration for resp filtering
    def resp_gate_all(self):
        resp_kdata = []
        for i in range(self.resp_frames):
            temp = self.mri_data.copy_data(full=False)
            resp_kdata.append(temp)
        
        for e in range(self.mri_data.Num_Encodings):
            self.logger.info(f"Encode {e}")
            # sort resp by time for filtering
            time = cp.asarray(self.mri_data.time[e].flatten())
            resp = cp.asarray(self.mri_data.resp[e].flatten())
            
            kdata = self.mri_data.kdata[e]
            coords = self.mri_data.coords[e]
            dcf = self.mri_data.dcf[e]
            prep = self.mri_data.prep[e]
            ecg = self.mri_data.ecg[e]
            sms_blips = self.mri_data.sms_blips[e]
            
            # shorten time range
            if self.time_ranges is not None:
                self.logger.info(f'Using data from: ')
                time_mask = np.zeros_like(resp, dtype=bool)
                for time_range in self.time_ranges:
                    self.logger.info(f'{time_range[0]} to {time_range[1]} s')
                    time_mask |= cp.logical_and(time > time_range[0], time < time_range[1])
                
                time = time[time_mask]
                resp = resp[time_mask]
                temp = cp.asnumpy(time_mask)
                kdata = kdata[:, temp]
                coords = coords[temp,:]
                dcf = dcf[temp]
                prep = prep[temp]
                ecg = ecg[temp]
                sms_blips = sms_blips[temp]
            
            # Estimate the TR
            dt = cp.max(time).item() / len(time)
            resp_filter_width = int(round(self.resp_filter_window / cp.asnumpy(dt)))

            self.logger.info(f'Estimated TR = {dt} based on {cp.max(time)} s acquisition with {len(time)} points')
            self.logger.info(f'Using a filter window of {self.resp_filter_window} s')
            
            resp_max = cndimage.maximum_filter(resp, size=resp_filter_width)
            resp_min = cndimage.minimum_filter(resp, size=resp_filter_width)
            
            signal_m = cp.mean((resp_max + resp_min)/2)
            signal_s = cp.mean(resp_max - resp_min)
            
            index = cp.arange(len(resp))
            upper_bound = signal_m + 1.2 * signal_s
            lower_bound = signal_m - 0.8 * signal_s
            eff_index = index[(resp < upper_bound) & (resp > lower_bound)]
            
            exhale_th = cp.asnumpy(signal_m + 0.2 * signal_s)
            exhale_pos, ex_dict = csignal.find_peaks(resp[eff_index], distance=resp_filter_width, height = exhale_th)
            ex_signal = ex_dict['peak_heights']
            drift = cinterpolate.interp1d(eff_index[exhale_pos], ex_signal, kind='cubic', fill_value = "extrapolate")(index)
            
            resp_d = resp - drift
            exhale_pos, ex_dict = csignal.find_peaks(resp_d[eff_index], distance=resp_filter_width, height = -1000)
            ex_signal = ex_dict['peak_heights']
            
            ex_std = np.std(ex_signal)    
            resp_d = resp_d[eff_index] + cp.random.rand(len(eff_index)) * .01 * ex_std
            
            tmp_ind = cp.arange(self.resp_frames * (len(resp_d) // self.resp_frames))
            index = eff_index[cp.argsort(resp_d[tmp_ind])]
            index = cp.asnumpy(index)
            
            time = cp.asnumpy(time)
            resp = cp.asnumpy(resp)
            
            npe = len(index) // self.resp_frames
            
            for r in range(self.resp_frames):
                idx = index[npe * int(r) : npe * int(r + 1)]

                resp_kdata[r].kdata.append(kdata[:, idx])
                resp_kdata[r].coords.append(coords[idx,:])
                resp_kdata[r].dcf.append(dcf[idx])
                resp_kdata[r].time.append(time[idx])
                resp_kdata[r].ecg.append(ecg[idx])
                resp_kdata[r].prep.append(prep[idx])
                resp_kdata[r].resp.append(resp[idx])
                resp_kdata[r].sms_blips.append(sms_blips[idx])
                
                self.logger.info(f"Points in resp bin {r}: {len(resp_kdata[r].dcf[e])}")
        
        return resp_kdata
    
    def cardiac_gate_all(self, resp_data, discrete_gates=False, ecg_delay=0):
        logger = logging.getLogger('Gate k-space')
        
        gated_data = []

        # gate off only resp phase 0 so all resp frames have same number of points
        mri_raw = resp_data[0]

        gate_signals = {
            'ecg': mri_raw.ecg,
            'time': mri_raw.time,
            'prep': mri_raw.prep,
            'resp': mri_raw.resp
        }
        gate_signal = gate_signals.get(self.gate_type, f'Cannot interpret gate signal {self.gate_type}')

        # For ECG, delay the waveform
        if self.gate_type == 'ecg':
            time = mri_raw.time

            for e in range(mri_raw.Num_Encodings):
                time_encode = time[e].flatten()
                ecg_encode = gate_signal[e].flatten()

                    #Sort the data by time
                idx = np.argsort(time_encode)
                idx_inverse = idx.argsort()

                # Estimate the delay
                if e == 0:
                    logger.info(f'Time max {time_encode.max()}')
                    logger.info(f'Time size {time_encode.size}')
                    logger.info(f'Time ecg delay {ecg_delay}')
                    
                    ecg_shift = int(ecg_delay / time_encode.max() * time_encode.size)
                    logger.info(f'Shifting by {ecg_shift}')

                #Using circular shift for now. This should be fixed
                ecg_sorted = ecg_encode[idx]
                ecg_shifted = np.roll( ecg_sorted, -ecg_shift)
                gate_signal[e] = np.reshape(ecg_shifted[idx_inverse], time[e].shape)

        logger.info(f'Gating off of {self.gate_type}')

        t_min, t_max, delta_time = get_gate_bins(gate_signal, self.gate_type, self.card_frames, discrete_gates)

        points_per_bin = []
        count = 0

        for r in range(self.resp_frames):
            mri_rawG = resp_data[r].copy_data(full=False)
            for t in range(self.card_frames):
                for e in range(mri_raw.Num_Encodings):
                    t_start = t_min + delta_time * t
                    t_stop = t_start + delta_time

                    # Find index where value is held
                    idx = np.logical_and.reduce([
                        np.abs(gate_signal[e]) >= t_start,
                        np.abs(gate_signal[e]) < t_stop])
                    current_points = np.sum(idx)

                    # post_gate = gate_signal[e][idx]
                    #print(f'Post gate min = {np.min(post_gate)}')
                    #print(f'Post gate max = {np.max(post_gate)}')
                    #print(f'Size of gate = {gate_signal[e].shape}')

                    # ecg = mri_raw.ecg[e][idx]
                    #print(f'Post ecg min = {np.min(ecg)}')
                    #print(f'Post ecg max = {np.max(ecg)}')
                    #print(f'Size of ecg = {mri_raw.ecg[e].shape}')


                    # Gate the data
                    points_per_bin.append(current_points)

                    #print('(t_start,t_stop) = (', t_start, ',', t_stop, ')')
                    logger.info(f'Frame {t} [{t_start} to {t_stop} ] | {e}, Points = {current_points}')

                    # Coords and K-space have extra dimensions (coils, directions)
                    mri_rawG.dcf.append(mri_raw.dcf[e][idx])
                    mri_rawG.time.append(mri_raw.time[e][idx])
                    mri_rawG.resp.append(mri_raw.resp[e][idx])
                    mri_rawG.prep.append(mri_raw.prep[e][idx])
                    mri_rawG.ecg.append(mri_raw.ecg[e][idx])
                    
                    new_kdata = mri_raw.kdata[e][:, idx]
                    mri_rawG.kdata.append(new_kdata)

                    new_coords = mri_raw.coords[e][idx, :]
                    mri_rawG.coords.append(new_coords)
                    
                    new_sms_blips = mri_raw.sms_blips[e][idx, :]
                    mri_rawG.sms_blips.append(new_sms_blips)
                    
                    count += 1

            max_points_per_bin = np.max(np.array(points_per_bin))
            logger.info(f'Max points = {max_points_per_bin}')
            logger.info(f'Points per bin = {points_per_bin}')
            logger.info(
                f'Average points per bin = {np.mean(points_per_bin)} [ {np.min(points_per_bin)}  {np.max(points_per_bin)} ]')
            logger.info(f'Standard deviation = {np.std(points_per_bin)}')

            mri_rawG.Num_Frames = self.card_frames
            if self.gate_type == "ecg":
                mri_rawG.median_rr = t_max
            else:
                mri_rawG.median_rr = mri_raw.median_rr
            
            gated_data.append(mri_rawG)

        return gated_data
        
    def xd_grasp_recon(self):
        self.logger.info(f'Performing XD-GRASP recon with {self.resp_frames} respiratory phases')

        # scale down
        # if self.res_scale > 1:
        #     self.logger.info(f"Scaling matrix down by {self.res_scale}")
        #     mps = cndimage.zoom(mps, (1, 1/self.res_scale, 1/self.res_scale, 1/self.res_scale), order=2)
        
        self.logger.info(f'Performing respiratory gating')
        resp_kdata = self.resp_gate_all()
        
        mps = array_to_gpu(self.mps, sp.Device(self.device))
        
        resp_imgs = []
        for r in range(self.resp_frames):
            resp_phase = resp_kdata[r]
            
            self.logger.info(f'Reconstructing phase {r}')
            # resp_img = pils_recon(resp_phase, smaps=mps, device=self.device)
            resp_img = []
            for i in range(resp_phase.Num_Encodings):
                self.logger.info(f'Sense Recon : Encode {i}')
                kdata = array_to_gpu(resp_phase.kdata[i], sp.Device(self.device))
                coord = array_to_gpu(resp_phase.coords[i], sp.Device(self.device))
                dcf = array_to_gpu(resp_phase.dcf[i], sp.Device(self.device))
                
                '''
                Note: sigpy has an issue with holding onto items in memory when running the pre-packaged apps.
                Will cause this code to eventually run out of GPU memory if enough respiratory phases are run,
                regardless if you delete the variables. Only fix for right now is to modify the 
                LinearLeastSquares function in sigpy/app.py file (if using conda it's probably located in 
                ~/$USER/.conda/envs/<env name>/lib/python<version>/site-packages)
                and replace this:
                
                def _output(self):
                    return self.x
                
                with this
                
                def _output(self):
                    x = backend.to_device(self.x, backend.cpu_device)
                    del self.x
                    del self.alg
                    return x
                '''
                
                # recon = sp.mri.app.SenseRecon(kdata, mps, lamda=self.lamda, weights=dcf, coord=coord, max_iter=self.max_iter//2, 
                #             coil_batch_size=self.coil_batch_size, device=sp.Device(self.device), solver="ConjugateGradient")
                recon = sp.mri.app.TotalVariationRecon(kdata, mps, lamda=self.lamda, weights=dcf, coord=coord, max_iter=self.max_iter//5,
                            coil_batch_size=self.coil_batch_size, device=sp.Device(self.device), solver="ADMM", save_objective_values=True)
                
                X = sp.to_device(recon.run(), sp.cpu_device)
                resp_img.append(X)
                
                # del kdata, coords, dcf, recon
                # cp.get_default_memory_pool().free_all_blocks()
                
                
            img = np.stack(resp_img, axis=-1)
            mag = np.sqrt(np.sum(np.abs(img)**2, axis=-1))
            resp_imgs.append(mag)
            
        return resp_imgs, resp_kdata
        
    def full_recon(self, gated_data, M_fields, iM_fields):
        def g(x):
            xp = sp.Device(self.device).xp
            with sp.Device(self.device):
                return self.lamda * xp.sum(xp.abs(x)).item()

        # motion operator
        Ms = []
        for i in range(self.resp_frames):
            M = interp_op(tuple(self.mri_data.tshape), M_fields[i], iM_fields[i])
            Ms.append(M)
        
        recon_image = []
        for e in range(self.card_frames*self.mri_data.Num_Encodings):
            self.logger.info(f'Recon Frame {e}')
            
            kdata_list = [gated_data[r].kdata[e] for r in range(self.resp_frames)]
            dcf_list = [gated_data[r].dcf[e] for r in range(self.resp_frames)]
            coords_list = [gated_data[r].coords[e] for r in range(self.resp_frames)]

            kdata = np.stack(kdata_list, axis=0)
            dcf = np.stack(dcf_list, axis=0)
            coords = np.stack(coords_list, axis=0)

            A = iMoCo_operator(kdata, self.mps, coords, dcf, Ms, tuple(self.mri_data.tshape), self.coil_batch_size)
            G = sp.linop.FiniteDifference(A.ishape)
            proxg = sp.prox.L1Reg(G.oshape, self.lamda)
            
            y = array_to_gpu(kdata * dcf[:,np.newaxis,...]**0.5, sp.Device(self.device))
            
            # recon = sp.app.LinearLeastSquares(A, y, max_iter=self.max_iter, lamda=self.lamda, solver="ConjugateGradient")
            recon = sp.app.LinearLeastSquares(A, y, proxg=proxg, G=G, g=g, max_iter=self.max_iter, lamda=self.lamda, solver="ADMM", save_objective_values=True)
            
            X = recon.run()
            recon_image.append(sp.to_device(X, sp.cpu_device))
        
        return recon_image
        
        
    def run(self):
        out_name = os.path.join(self.out_folder, 'debug_imoco.h5')
        
        # XD-GRASP recon
        if not self.debug:
            if self.resp_frames > 1:
                motion_images, resp_kdata = self.xd_grasp_recon()
            else:
                motion_images = [np.zeros(tuple(self.mri_data.tshape))]
                resp_kdata = [resp_gate(self.mri_data, resp_lower=0, resp_upper=0.5,
                                    resp_filter_window=self.resp_filter_window, debug_folder=None)]
            
            try:
                os.remove(out_name)
            except OSError:
                pass
            
            if self.resp_frames > 1:
                # save motion images
                self.logger.info(f'Saving Images to {out_name}')
                with h5py.File(out_name, 'w') as hf:
                    for r in range(self.resp_frames):
                        hf.create_dataset(f"MAG_RESP{r}", data=motion_images[r])
                
                self.logger.info('Performing image registration')
                M_fields = []
                iM_fields = []
                for r in range(self.resp_frames):
                    if r == 0: # frame 0 should be end expiration, register all other frames to it
                        M_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                        iM_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                    else:
                        self.logger.info(f'Registering phase {r} to phase 0')
                        M_field, iM_field = self.register(motion_images[0], motion_images[r])
                    M_fields.append(M_field)
                    iM_fields.append(iM_field)
            
                # save motion transforms
                self.logger.info(f'Saving motion fields to {out_name}')
                with h5py.File(os.path.join(self.out_folder, 'debug_imoco.h5'), 'a') as hf:
                    hf.create_dataset(f"M_fields", data=np.stack(M_fields, axis=-1))
                    hf.create_dataset(f"iM_fields", data=np.stack(iM_fields, axis=-1))
            else:
                M_fields = [np.zeros(tuple(self.mri_data.tshape)+(3,))]
                iM_fields = [np.zeros(tuple(self.mri_data.tshape)+(3,))]
        else:
            motion_images = []
            with h5py.File(out_name, 'r') as hf:
                for r in range(self.resp_frames):
                    motion_images.append(hf[f"MAG_RESP{r}"][:])
                M_fields = hf[f"M_fields"][:]
                iM_fields = hf[f"iM_fields"][:]
            M_fields = [M_fields[...,i] for i in range(M_fields.shape[-1])]
            iM_fields = [iM_fields[...,i] for i in range(iM_fields.shape[-1])]
        
        if self.res_scale > 1:
            self.logger.info('Scaling motion fields to match original dimensions')
            M_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in M_fields]
            iM_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in iM_fields]
        
        # remove to save memory
        del motion_images
        
        # gated_data = []
        if self.card_frames > 1:
            gated_data = self.cardiac_gate_all(resp_kdata)
            # for r in range(self.resp_frames):
            #     card_data = gate_kspace(mri_raw=resp_kdata[r], num_frames=self.card_frames, gate_type=self.gate_type)
            #     gated_data.append(card_data)
        else:
            gated_data = resp_kdata

        image = self.full_recon(gated_data, M_fields, iM_fields)
        
        return image


def iMoCo_operator(
    kdata,
    mps,
    coord,
    weights,
    motion_fields,
    ishape,
    coil_batch_size=None,
):
    
    if ishape is None:
        ishape = mps.shape[1:]
    resp_frames = kdata.shape[0]

    # batch coils
    num_coils = len(mps)
    if coil_batch_size is None:
        coil_batch_size = num_coils

    if coil_batch_size < len(mps):
        num_coil_batches = (num_coils + coil_batch_size - 1) // coil_batch_size
        A = sp.linop.Vstack(
            [
                iMoCo_operator(
                    kdata[c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                    mps[c * coil_batch_size : ((c + 1) * coil_batch_size), ...],
                    coord,
                    weights,
                    motion_fields,
                    ishape
                )
                for c in range(num_coil_batches)
            ],
            axis=0,
        )
        
    # construct iMoCo operator
    # A = W F S M (stacked along repiratory dimension)
    # density compensation x Fourier transform x Sense operator x motion registration
            
    A = []
    I = []
    S = sp.linop.Multiply(ishape, mps)
    for r in range(resp_frames):
        I.append(sp.linop.Identity(ishape)) # identity operator for stacking 
        W = sp.linop.Multiply(kdata[r].shape, weights[r,np.newaxis,...]**0.5) 
        F = NFTs((num_coils,)+ishape, coord[r,...])
        WFSM = W*F*S*motion_fields[r]
        A.append(WFSM)
    A = Diags(A, oshape=kdata.shape, ishape=(resp_frames,)+ishape) * Vstacks(I, oshape=(resp_frames,)+ishape)
            
    return A
    
class interp_op(sp.linop.Linop):
    def __init__(self, ishape, M_field, iM_field=None):
        assert list(ishape) == list(M_field.shape[:-1]),"Dimension mismatch!"
        oshape = ishape
        dkernel = np.array([0,1,0])
        
        # 2d/3d
        if M_field.shape[-1] == 3:
            self.dkernel = dkernel[:,None,None]*dkernel[None,:,None]*dkernel[None,None,:]
            Nx,Ny,Nz = ishape
            my,mx,mz = np.meshgrid(np.arange(Ny),np.arange(Nx),np.arange(Nz))
            self.m = np.stack((mx,my,mz),axis=-1)
        else:
            self.dkernel = dkernel[:,None]*dkernel[None,:]
            Nx,Ny = ishape
            my,mx = np.meshgrid(np.arange(Ny),np.arange(Nx))
            self.m = np.stack((mx,my),axis=-1)
            
        self.M_field = M_field
        self.iM_field = iM_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.backend.get_device(input)
        with device:
            return interp(input, self.M_field, self.m, self.dkernel, device) # major change

    def _adjoint_linop(self):
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field
        return interp_op(self.ishape, iM_field, M_field)
    
def interp(I, M_field, m, dkernel, device=sp.Device(0), deblur=False):  
    M_field = M_field + m
    I = sp.to_device(I, device)
    M= sp.to_device(M_field.astype(np.float64), device) # v0.1.17
    I = sp.interp.interpolate(I, coord=M) # v0.1.17 (input, coord, kernel='spline', width=2, param=1)
    # deconv
    if deblur is True:
        I = sp.conv.convolve(I, dkernel)
    I = sp.to_device(I, device)
    return I

def Vstacks(L_Linop, oshape):
    assert oshape[0]==len(L_Linop), 'Number of Linop mismatch!'
    Linops = sp.linop.Vstack(L_Linop)
    R1 = sp.linop.Reshape(oshape=(math.prod(oshape),),ishape=oshape)
    Linops = R1.H*Linops
    return Linops

def Diags(L_Linop, oshape, ishape):
    assert oshape[0]==ishape[0], 'First dim mismatch!'
    assert oshape[0]==len(L_Linop), 'Number of Linop mismatch!'
    Linops = sp.linop.Diag(L_Linop)
    R1 = sp.linop.Reshape(oshape=(math.prod(oshape),),ishape=oshape)
    R2 = sp.linop.Reshape(oshape=(math.prod(ishape),),ishape=ishape)
    Linops = R1.H*Linops*R2
    return Linops
    
def NFTs(ishape, coord):
    n_Channel = ishape[0]
    oshape = list((n_Channel,)) + list(coord.shape[:-1])
    NFT = sp.linop.NUFFT(ishape[1:], coord=coord)
    NFTs = Diags([NFT for i in range(n_Channel)],oshape,ishape)
    return NFTs     
        