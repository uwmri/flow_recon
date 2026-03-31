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
from flow_processing import *

__all__ = ['iMoCoRecon']

class iMoCoRecon:
    '''
    Zhu X, Chan M, Lustig M, Johnson KM, Larson PEZ. Iterative motion-compensation reconstruction 
    ultra-short TE (iMoCo UTE) for high-resolution free-breathing pulmonary MRI. Magn Reson Med. 
    2020; 83: 1208-1221. https://doi.org/10.1002/mrm.27998
    
    https://github.com/PulmonaryMRI/imoco_recon
    
    '''
    def __init__ (self, mri_data, mps, gate_type='ecg', card_frames=1, resp_frames=1, time_ranges=None, lamda=0, venc=1500,
                 device=None, res_scale=2, coil_batch_size=None, max_iter=50, show_pbar=True,
                 comm=None, resp_filter_window=5, out_folder=None, debug=False, **kwargs):
        
        
        self.mri_data = mri_data
        self.mps = mps
        self.gate_type = gate_type
        self.card_frames = card_frames
        self.resp_frames = resp_frames
        self.time_ranges = time_ranges
        self.venc = venc
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
                
        
    def register(self, fixed, moving, r):
        # ants will add transforms to same output files if run repeatedly messing up future recons so delete each run
        try:
            os.remove(f"{self.out_folder}/resp{r}_1Warp.nii.gz")
            os.remove(f"{self.out_folder}/resp{r}_1InverseWarp.nii.gz")
            os.remove(f"{self.out_folder}/resp{r}_0GenericAffine.mat")
        except OSError:
            pass
        
        # returns a dictionary containing filepaths to the transforms, not the transforms themselves
        reg = ants.registration(ants.from_numpy(fixed), ants.from_numpy(moving), 
                                type_of_transform='SyNAggro', initial_transform="identity", \
                                syn_metric='demons', syn_sampling=6, \
                                grad_step=0.3, flow_sigma=3, total_sigma=1, \
                                reg_iterations=(200,200,150,50,10), \
                                aff_iterations=(2100,1200,1200,10,5), \
                                aff_shrink_factors=(12,8,4,2,1), aff_smoothing_sigmas=(5,4,2,1,0), \
                                verbose=False, outprefix=f"{self.out_folder}/resp{r}_", \
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
            
            self.logger.info(f'Finding mins and maxes...')
            resp_max = cndimage.maximum_filter(resp, size=resp_filter_width)
            resp_min = cndimage.minimum_filter(resp, size=resp_filter_width)
            
            signal_m = cp.mean((resp_max + resp_min)/2)
            signal_s = cp.mean(resp_max - resp_min)
            
            index = cp.arange(len(resp))
            upper_bound = signal_m + 1.2 * signal_s
            lower_bound = signal_m - 0.8 * signal_s
            eff_index = index[(resp < upper_bound) & (resp > lower_bound)]
            
            self.logger.info(f'Correcting respiratory drift...')
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
    
    def cardiac_gate_all(self, resp_data, ecg_delay=0):
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

        # t_min, t_max, delta_time = get_gate_bins(gate_signal, self.gate_type, self.card_frames, discrete_gates)
        t_min = np.min([np.min(gate) for gate in gate_signal])
        t_max = np.max([np.max(gate) for gate in gate_signal])
        median_rr = np.mean([np.median(gate) for gate in gate_signal])
        all_points = np.concatenate([gate.flatten() for gate in gate_signal])
        median_rr = 2.0 * (median_rr - t_min) + t_min
        bin_edges = np.quantile(all_points, np.linspace(0, 1, self.card_frames + 1))
        logger.info(f'Bin edges = {bin_edges}')

        points_per_bin = []
        count = 0

        for r in range(self.resp_frames):
            mri_rawG = resp_data[r].copy_data(full=False)
            for t in range(self.card_frames):
                for e in range(mri_raw.Num_Encodings):
                    # t_start = t_min + delta_time * t
                    # t_stop = t_start + delta_time

                    # # Find index where value is held
                    # idx = np.logical_and.reduce([
                    #     np.abs(gate_signal[e]) >= t_start,
                    #     np.abs(gate_signal[e]) < t_stop])
                    
                    t_start = bin_edges[t]
                    t_stop = bin_edges[t + 1]
                    idx = np.logical_and(gate_signal[e] >= t_start, gate_signal[e] < t_stop)
                    
                    current_points = np.sum(idx)

                    # Gate the data
                    points_per_bin.append(current_points)

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
        
    def motion_recon(self):
        self.logger.info(f'Performing motion-resolved recon with {self.resp_frames} respiratory phases')

        # scale down
        # if self.res_scale > 1:
        #     self.logger.info(f"Scaling matrix down by {self.res_scale}")
        #     mps = cndimage.zoom(mps, (1, 1/self.res_scale, 1/self.res_scale, 1/self.res_scale), order=2)
        
        self.logger.info(f'Performing respiratory gating')
        resp_kdata = self.resp_gate_all()
        
        mps = array_to_gpu(self.mps, sp.Device(self.device))
        
        if self.mri_data.Num_Encodings == 5:
            encoding = "5pt"
        elif self.mri_data.Num_Encodings == 4:
            encoding = "4pt-referenced"
        elif self.mri_data.Num_Encodings == 3:
            encoding = "3pt"
        elif self.mri_data.Num_Encodings == 2:
            encoding = "2pt"
        
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
                It'll cause this code to eventually run out of GPU memory if enough respiratory phases are run,
                regardless of whether you derefernce the variables after each loop. Only fix for right now is 
                to modify the LinearLeastSquares function in sigpy/app.py file (if using conda it's probably 
                located in ~/$USER/.conda/envs/<env name>/lib/python<version>/site-packages/sigpy)
                and replace this:
                
                def _output(self):
                    return self.x
                
                with this
                
                def _output(self):
                    x = backend.to_device(self.x, backend.cpu_device)
                    del self.x
                    del self.alg
                    return x
                
                https://github.com/mikgroup/sigpy/issues/49
                '''
                
                # recon = sp.mri.app.SenseRecon(kdata, mps, lamda=self.lamda, weights=dcf, coord=coord, max_iter=self.max_iter, 
                #             coil_batch_size=self.coil_batch_size, device=sp.Device(self.device), solver="ConjugateGradient")
                recon = sp.mri.app.L1WaveletRecon(kdata, mps, lamda=self.lamda*10, weights=dcf, coord=coord, max_iter=self.max_iter/2, 
                            coil_batch_size=self.coil_batch_size, device=sp.Device(self.device))
                # recon = sp.mri.app.TotalVariationRecon(kdata, mps, lamda=self.lamda*10, weights=dcf, coord=coord, max_iter=1,
                #             coil_batch_size=self.coil_batch_size, device=sp.Device(self.device), solver="ADMM", save_objective_values=True)
                
                X = sp.to_device(recon.run(), sp.cpu_device)
                resp_img.append(X)
                
                # del kdata, coords, dcf, recon
                # cp.get_default_memory_pool().free_all_blocks()
                
            self.logger.info("Solving for velocity")
            resp_img = np.stack(resp_img, axis=-1)
            mri_flow = MRI_4DFlow(encoding, signal=resp_img, venc=self.venc)
            mri_flow.solve_for_velocity()
            mri_flow.update_angiogram()
            resp_imgs.append(mri_flow)
            
        return resp_imgs, resp_kdata
        
    def full_recon(self, gated_data, M_fields, iM_fields):
        # def g(x):
        #     xp = sp.Device(self.device).xp
        #     with sp.Device(self.device):
        #         return self.lamda * xp.sum(xp.abs(x)).item()
        def g(input):
            device = sp.get_device(input)
            xp = device.xp
            with device:
                return self.lamda * xp.sum(xp.abs(W(input))).item()

        tshape = tuple(self.mri_data.tshape)
        
        dkernel = np.array([0,1,0])
        if M_fields[0].shape[-1] == 3:
            dkernel = dkernel[:,None,None]*dkernel[None,:,None]*dkernel[None,None,:]
            Nx,Ny,Nz = tshape
            my,mx,mz = np.meshgrid(np.arange(Ny),np.arange(Nx),np.arange(Nz))
            m = np.stack((mx,my,mz), axis=-1)
        else:
            dkernel = dkernel[:,None]*dkernel[None,:]
            Nx,Ny = tshape
            my,mx = np.meshgrid(np.arange(Ny),np.arange(Nx))
            m = np.stack((mx,my), axis=-1)
        
        # motion operator
        Ms = []
        for i in range(self.resp_frames):
            M = interp_op(tshape, m, dkernel, iM_fields[i], M_fields[i])
            # M = interp_op(tshape, m, dkernel, M_fields[i])
            Ms.append(M)
            
        # mps_list = [self.mps for r in range(self.resp_frames)]
        # mps = np.stack(mps_list, axis=0)
        # mps = np.tile(self.mps, (self.resp_frames, self.card_frames))
        mps = array_to_gpu(self.mps, sp.Device(self.device))
        
        recon_image = []
        for e in range(self.card_frames*self.mri_data.Num_Encodings):
        # for e in range(self.mri_data.Num_Encodings):
            self.logger.info(f'Recon Frame {e+1}')
            
            kdata_list = [gated_data[r].kdata[e] for r in range(self.resp_frames)]
            dcf_list = [gated_data[r].dcf[e] for r in range(self.resp_frames)]
            coords_list = [gated_data[r].coords[e] for r in range(self.resp_frames)]
            
            kdata = np.stack(kdata_list, axis=0)
            dcf = np.stack(dcf_list, axis=0)
            coords = np.stack(coords_list, axis=0)
            
            # stack data so shape is (resp, card, coils, projs)
            # kdata = np.array([gated_data[r].kdata[e*self.card_frames:(e+1)*self.card_frames] for r in range(self.resp_frames)])
            # dcf = np.array([gated_data[r].dcf[e*self.card_frames:(e+1)*self.card_frames] for r in range(self.resp_frames)])
            # coords = np.array([gated_data[r].coords[e*self.card_frames:(e+1)*self.card_frames] for r in range(self.resp_frames)])
            
            A = iMoCo_operator(kdata, mps, coords, dcf, Ms, tshape, self.coil_batch_size)
            # G = sp.linop.FiniteDifference(A.ishape)
            # proxg = sp.prox.L1Reg(G.oshape, self.lamda)
            W = sp.linop.Wavelet(tshape, wave_name="db4")
            proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(W.oshape, self.lamda), W)
            
            # G_s = sp.linop.FiniteDifference(tshape)                     # spatial
            # G_t = sp.linop.FiniteDifference((self.card_frames,)+tshape, append_axes=(0,))  # temporal
            # G = sp.linop.Vstack([G_s, G_t])
            # proxg = sp.prox.Stack([sp.prox.L1Reg(G_s.oshape, self.lamda),
            #                     sp.prox.L1Reg(G_t.oshape, self.lamda*100)])
            
            y = array_to_gpu(kdata * dcf[:,np.newaxis,...]**0.5, sp.Device(self.device))
            
            recon = sp.app.LinearLeastSquares(A, y, proxg=proxg, g=g, max_iter=self.max_iter, lamda=self.lamda)
            
            # recon = sp.app.LinearLeastSquares(A, y, max_iter=self.max_iter, lamda=self.lamda, solver="ConjugateGradient")
            # recon = sp.app.LinearLeastSquares(A, y, proxg=proxg, G=G, g=g, max_iter=self.max_iter, lamda=self.lamda, solver="ADMM", save_objective_values=True)
            
            X = recon.run()
            recon_image.append(sp.to_device(X, sp.cpu_device))
        
        return recon_image
        
        
    def run(self):
        out_name = os.path.join(self.out_folder, 'debug_imoco.h5')
        
        if self.debug:
            resp_kdata = self.resp_gate_all()
            motion_images = []
            # with h5py.File(out_name, 'r') as hf:
                # for r in range(self.resp_frames):
                #     encodes = []
                #     encodes.append(hf[f"MAG_RESP{r}"][:])
                #     encodes.append(hf[f"CD_RESP{r}"][:])
                #     motion_images.append(encodes)
                # M_fields = hf[f"M_fields"][:]
                # iM_fields = hf[f"iM_fields"][:]
            M_fields = []
            iM_fields = []
            for r in range(self.resp_frames):
                if r == self.resp_frames - 1:
                    mfield = np.zeros(tuple(self.mri_data.tshape)+(3,))
                    imfield = np.zeros(tuple(self.mri_data.tshape)+(3,))
                    M_fields.append(mfield)
                    iM_fields.append(imfield)
                else:
                    mfield = nibabel.load(f"{self.out_folder}/resp{r}_1Warp.nii.gz")
                    imfield = nibabel.load(f"{self.out_folder}/resp{r}_1InverseWarp.nii.gz")
                    M_fields.append(np.squeeze(mfield.get_fdata()))
                    iM_fields.append(np.squeeze(imfield.get_fdata()))
            # M_fields = [M_fields[...,i] for i in range(M_fields.shape[-1])]
            # iM_fields = [iM_fields[...,i] for i in range(iM_fields.shape[-1])]
            
            # self.logger.info('Performing image registration')
            # M_fields = []
            # iM_fields = []
            # for r in range(self.resp_frames):
            #     if r == 0: # frame 0 should be end expiration, register all other frames to it
            #         M_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
            #         iM_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
            #     else:
            #         self.logger.info(f'Registering phase {r} to phase 0')
            #         M_field, iM_field = self.register(motion_images[0][0], motion_images[r][0], r)
            #         # M_field, iM_field = self.register(motion_images[0].angiogram, motion_images[r].angiogram, r)
            #     M_fields.append(M_field)
            #     iM_fields.append(iM_field)
        
            # # remove to save memory
            # del motion_images
        else:
            try:
                os.remove(out_name)
            except OSError:
                pass
            if self.resp_frames > 1:
                motion_images, resp_kdata = self.motion_recon()

                # save motion images
                self.logger.info(f'Saving respiratory resolved images to {out_name}')
                with h5py.File(out_name, 'w') as hf:
                    for r in range(self.resp_frames):
                        hf.create_dataset(f"MAG_RESP{r}", data=motion_images[r].magnitude)
                        hf.create_dataset(f"CD_RESP{r}", data=motion_images[r].angiogram)
                
                self.logger.info('Performing image registration')
                M_fields = []
                iM_fields = []
                for r in range(self.resp_frames):
                    if r == self.resp_frames - 1: # last frame should be end expiration, register all other frames to it
                        M_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                        iM_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                    else:
                        self.logger.info(f'Registering phase {r} to phase {self.resp_frames - 1}')
                        M_field, iM_field = self.register(motion_images[-1].angiogram, motion_images[r].angiogram, r)
                        # M_field, iM_field = self.register(motion_images[0].angiogram, motion_images[r].angiogram, r)
                    M_fields.append(M_field)
                    iM_fields.append(iM_field)
            
                # save motion transforms
                self.logger.info(f'Saved motion fields to {self.out_folder}')
            
                # remove to save memory
                del motion_images
            else:
                resp_kdata = [resp_gate(self.mri_data, resp_lower=0, resp_upper=0.5, time_ranges=self.time_ranges, resp_filter_window=self.resp_filter_window, debug_folder=None)]
                
                M_fields = [np.zeros(tuple(self.mri_data.tshape)+(3,))]
                iM_fields = [np.zeros(tuple(self.mri_data.tshape)+(3,))]  
                      
        if self.res_scale > 1:
            self.logger.info('Scaling motion fields to match original dimensions')
            M_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in M_fields]
            iM_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in iM_fields]
        
        gated_data = []
        if self.card_frames > 1:
            gated_data = self.cardiac_gate_all(resp_kdata)
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
    # card_frames = kdata.shape[1]

    # batch coils
    num_coils = mps.shape[0]
    if coil_batch_size is None:
        coil_batch_size = num_coils
        
    
    # construct iMoCo operator
    # A = W F S M (stacked along repiratory dimension)
    # density compensation x Fourier transform x Sense operator x motion registration
    if coil_batch_size < num_coils:
        num_coil_batches = (num_coils + coil_batch_size - 1) // coil_batch_size
        
        A = []
        I = []
        for r in range(resp_frames):
            As = []
            for c in range(num_coil_batches):
                W = sp.linop.Multiply(kdata[r, c*coil_batch_size:((c+1)*coil_batch_size)].shape, weights[r]**0.5)
                # print(f"W.ishape = {W.ishape} W.oshape = {W.oshape}")
                F = sp.linop.NUFFT((coil_batch_size,)+ishape, coord[r])
                # print(f"F.ishape = {F.ishape} F.oshape = {F.oshape}")
                S = sp.linop.Multiply(ishape, mps[c*coil_batch_size:((c+1)*coil_batch_size)])
                # print(f"S.ishape = {S.ishape} S.oshape = {S.oshape}")
                M = motion_fields[r]
                # print(f"M.ishape = {M.ishape} M.oshape = {M.oshape}")
                Ab = W * F * S * M
                # print(f"Ab.ishape = {Ab.ishape} Ab.oshape = {Ab.oshape}")
                As.append(Ab)
            Ar = sp.linop.Vstack(As, axis=0)
            # print(f"Ar.ishape = {Ar.ishape} Ar.oshape = {Ar.oshape}")
            A.append(Ar)
            I.append(sp.linop.Identity(ishape)) # identity operator for stacking 
        # A = sp.linop.Diag(A)
        A = Diags(A, oshape=kdata.shape, ishape=(resp_frames,)+ishape) * Vstacks(I, oshape=(resp_frames,)+ishape)
        # print(f"A.ishape = {A.ishape} A.oshape = {A.oshape}")
    else:
    
        As = []
        I = []
        for r in range(resp_frames):
            I.append(sp.linop.Identity(ishape)) # identity operator for stacking 
            W = sp.linop.Multiply(kdata[r].shape, weights[r]**0.5)
            F = sp.linop.NUFFT((num_coils,)+ishape, coord[r])
            S = sp.linop.Multiply(ishape, mps)
            M = motion_fields[r]
            As.append(W * F * S * M)
        A = Diags(As, oshape=kdata.shape, ishape=(resp_frames,)+ishape) * Vstacks(I, oshape=(resp_frames,)+ishape)
        # A = sp.linop.Diag(As)
    
    # print(f"A.ishape = {A.ishape} A.oshape = {A.oshape}")
            
    return A
    
class interp_op(sp.linop.Linop):
    def __init__(self, ishape, m, dkernel, M_field, iM_field=None):
        assert list(ishape) == list(M_field.shape[:-1]),"Dimension mismatch!"
        oshape = ishape
        self.m=m  
        self.dkernel = dkernel
        self.M_field = M_field
        self.iM_field = iM_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        with sp.backend.get_device(input):
            return interp(input, self.M_field, self.m, self.dkernel) # major change

    def _adjoint_linop(self):
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field
        return interp_op(self.ishape, self.m, self.dkernel, iM_field, M_field)
    
def interp(I, M_field, m, dkernel, deblur=False):  
    M_field = M_field + m
    M = sp.to_device(M_field, sp.get_device(I))
    I = sp.interp.interpolate(I, coord=M) # v0.1.17 (input, coord, kernel='spline', width=2, param=1)
    # deconv
    if deblur is True:
        I = sp.conv.convolve(I, dkernel)
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
        