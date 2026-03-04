import ants
import numpy as np 
import sigpy as sp
import h5py
import nibabel
import scipy.ndimage as ndimage
from tqdm import tqdm

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
            M[...,i] = M[...,i]*(Mscale[i]*self.res_scale)
            Mo[...,i] = ndimage.zoom(M[...,i],zoom=tuple(Mscale),order=2)

        return Mo
    
    def xd_grasp_recon(self):
        self.logger.info(f'Performing XD-GRASP recon with {self.resp_frames} respiratory phases')
        resp_bins = np.linspace(0, 1, self.resp_frames+1)
        self.logger.info(f'Respiratory bins: {resp_bins}')
        
        # crop_kspace(data, crop_factor=self.res_scale, crop_type='radius')
        mps = array_to_gpu(self.mps, sp.Device(self.device))
        resp_imgs = []
        resp_kdata = []

        for r in range(self.resp_frames):
            data = self.mri_data.copy_data(full=True)
            self.logger.info(f'Gating respiratory phase {r}')
            resp_phase = resp_gate(data, resp_lower=resp_bins[r], resp_upper=resp_bins[r+1], time_ranges=self.time_ranges,
                                   resp_filter_window=self.resp_filter_window, debug_folder=self.out_folder, debug_name=r)
            resp_kdata.append(resp_phase)
            
            if not self.debug:
                self.logger.info(f'Reconstructing phase {r}')
                # resp_img = pils_recon(resp_phase, smaps=mps, device=self.device)
                resp_img = []
                for i in range(resp_phase.Num_Encodings):
                    self.logger.info(f'Sense Recon : Encode {i}')
                    kdata = array_to_gpu(resp_phase.kdata[i], sp.Device(self.device))
                    coord = array_to_gpu(resp_phase.coords[i], sp.Device(self.device))
                    dcf = array_to_gpu(resp_phase.dcf[i], sp.Device(self.device))
                
                    sense = sp.mri.app.SenseRecon(kdata, mps, lamda=self.lamda, weights=dcf, coord=coord, max_iter=self.max_iter//2, 
                                coil_batch_size=self.coil_batch_size, device=sp.Device(self.device), solver="ConjugateGradient")
                    resp_img.append(sp.to_device(sense.run(), sp.cpu_device))
                # resp_img = sp.to_device(resp_img, sp.cpu_device)
                img = np.stack(resp_img, axis=-1)
                mag = np.sqrt(np.sum(np.abs(img)**2, axis=-1))
                resp_imgs.append(mag)
            
            del data
        
        return resp_imgs, resp_kdata
        
    def full_recon(self, gated_data, M_fields, iM_fields):
        nCoils = int(self.mps.shape[0])
        # if self.coil_batch_size is None:
        #     self.coil_batch_size = nCoils
        # batches = [
        #     range(i, min(i + self.coil_batch_size, nCoils))
        #     for i in range(0, nCoils, self.coil_batch_size)
        # ]

        tshape = tuple(self.mri_data.tshape)
        ## low rank
        Ms = []
        # M0s = []
        for i in range(self.resp_frames):
            M = interp_op(tshape, M_fields[i], iM_fields[i])
            # M = interp_op(tshape, M_fields[i])
            # M0 = interp_op(tshape,np.zeros(tshape+(3,)))
            # M = DLD(M,device=sp.Device(self.device))
            # M0 = DLD(M0,device=sp.Device(self.device))
            Ms.append(M)
            # M0s.append(M0)
        # Ms = Diags(Ms,oshape=(self.resp_frames,)+tshape,ishape=(self.resp_frames,)+tshape)
        # M0s = Diags(M0s,oshape=(self.resp_frames,)+tshape,ishape=(self.resp_frames,)+tshape)
        
        # print(f'Ms ishape: {Ms[0].ishape}, Ms shape: {Ms[0].oshape}')
        # print(f'M0s ishape: {M0s.ishape}, M0s shape: {M0s.oshape}')
        
        recon_image = []
        for e in range(self.card_frames*self.mri_data.Num_Encodings):
            self.logger.info(f'Recon Frame {e}')
            
            # pad kdata, dcf, and coord to have the same shape across respiratory phases
            # max_size = np.max([gated_data[r].kdata[e].shape[-1] for r in range(self.resp_frames)], axis=0)
            min_size = np.min([gated_data[r].kdata[e].shape[-1] for r in range(self.resp_frames)], axis=0)
            # pad then stack kdata, dcf, and coord
            
            kdata_list = []
            dcf_list = []
            coords_list = []
            for r in range(self.resp_frames):
                kdata = gated_data[r].kdata[e]
                dcf = gated_data[r].dcf[e]
                coords = gated_data[r].coords[e]
                
                # # pad to max_size
                # if kdata.shape[-1] < max_size:
                #     pad_width = max_size - np.array(kdata.shape[-1])
                #     pad_width = np.max(pad_width, 0)
                #     kdata_pad = np.pad(kdata, [(0,0), [0, pad_width]], mode='constant', constant_values=0)
                #     dcf_pad = np.pad(dcf, [[0, pad_width]], mode='constant', constant_values=0)
                #     coords_pad = np.pad(coords, [[0, pad_width], [0, 0]], mode='constant', constant_values=0)
                # else:
                #     kdata_pad = kdata
                #     dcf_pad = dcf
                #     coords_pad = coords
                
                # crop to min_size
                if kdata.shape[-1] > min_size:
                    # crop = np.array(kdata.shape[-1]) - min_size
                    kdata_new = kdata[:, :min_size]
                    dcf_new = dcf[:min_size]
                    coords_new = coords[:min_size, :]
                else:
                    kdata_new = kdata
                    dcf_new = dcf
                    coords_new = coords
                
                kdata_list.append(kdata_new)
                dcf_list.append(dcf_new)
                coords_list.append(coords_new)

            kdata = np.stack(kdata_list, axis=0)
            dcf = np.stack(dcf_list, axis=0)
            coords = np.stack(coords_list, axis=0)
            
            # PFTSMs_all = []
            # Is_all = []
            # for b in batches:
                
            #     S_batch = sp.linop.Multiply(tshape, self.mps[b, ...])
                
            #     PFTSMs_batch = []
            #     Is_batch = []
            #     for r in range(self.resp_frames):
            #         Is_batch.append(sp.linop.Identity(tshape))
            #         # FTs = sp.linop.NUFFT(kdata[r].shape, coord=coords[r])
            #         FTs = NFTs((len(b),)+tshape,coords[r,...],device=sp.Device(self.device))
            #         # M = interp_op(tshape,M_fields[r])
            #         # M = DLD(M,device=sp.Device(self.device))
            #         # print(f'M ishape: {M.ishape}, M oshape: {M.oshape}')
            #         W = sp.linop.Multiply(kdata[r,b].shape,dcf[r]) 
            #         # print(f'W ishape: {W.ishape}, W oshape: {W.oshape}')
            #         FTSM = W*FTs*S_batch*Ms[r]
            #         # print(f'FTSM ishape: {FTSM.ishape}, FTSM oshape: {FTSM.oshape}')
            #         PFTSMs_batch.append(FTSM)
            #     Is_all.append(Is_batch)
            #     PFTSMs_batch = Diags(PFTSMs_batch,oshape=kdata[:,b].shape,
            #                          ishape=(self.resp_frames,)+tshape)*Vstacks(Is_batch,oshape=(self.resp_frames,)+tshape)
            #     PFTSMs_all.append(PFTSMs_batch)
            
            
            # PFTSMs_all = []
            # Is_all = []
            # for b in batches:
                
           

            # print(f'mps {self.mps.shape}')
            # print(sp.get_device(self.mps))
            # print(f'kdata {kdata.shape}')
            # print(sp.get_device(kdata))
            # print(f'dcf {dcf.shape}')
            # print(sp.get_device(dcf))
            # print(f'coords {coords.shape}')   
            # print(sp.get_device(coords))
            
            # F = sp.linop.NUFFT(self.mps.shape, coord=coords[r]) 
            S = sp.linop.Multiply(tshape, self.mps)
            PFTSMs = []
            Is = []
            for r in range(self.resp_frames):
                Is.append(sp.linop.Identity(tshape))
               
                # FTs = sp.linop.Diag([F for i in range(nCoils)])
                FTs = NFTs((nCoils,)+tshape,coords[r,...],device=sp.Device(self.device))
                # print(f'FTs ishape: {FTs.ishape}, FTs oshape: {FTs.oshape}')
                # M = interp_op(tshape,M_fields[r])
                # M = DLD(M,device=sp.Device(self.device))
                # print(f'M ishape: {Ms[r].ishape}, M oshape: {Ms[r].oshape}')
                W = sp.linop.Multiply(kdata[r].shape, dcf[r,np.newaxis,...]**0.5) 
                # print(f'W ishape: {W.ishape}, W oshape: {W.oshape}')
                # FTSM = W*FTs*S*Ms[r]
                FTSM = W*FTs*S*Ms[r]
                # print(f'FTSM ishape: {FTSM.ishape}, FTSM oshape: {FTSM.oshape}')
                PFTSMs.append(FTSM)
            # PFTSMs = sp.linop.Diag(PFTSMs)
            # PFTSMs = sp.linop.Vstack(PFTSMs)
            PFTSMs = Diags(PFTSMs,oshape=kdata.shape,
                                    ishape=(self.resp_frames,)+tshape)*Vstacks(Is,oshape=(self.resp_frames,)+tshape)
                
            # print(f'PFTSMs ishape: {PFTSMs.ishape} PFTSMs oshape: {PFTSMs.oshape}')
            
             # for r in range(self.resp_frames):

            #     S = sp.linop.Multiply(tshape, self.mps)
            #     F = sp.linop.NUFFT(self.mps.shape, coord=coords[r])
            #     W = sp.linop.Multiply(kdata[r].shape, dcf[r, np.newaxis,...]**0.5)

            #     PFTSMs = W * F * S
            y = kdata * dcf[:,np.newaxis,...]**0.5
            # y = array_to_gpu(y.reshape(1, -1), sp.Device(self.device))
            y = array_to_gpu(y, sp.Device(self.device))
                
            alg = sp.app.LinearLeastSquares(PFTSMs, y, max_iter=self.max_iter, lamda=self.lamda, solver="ConjugateGradient")
                
            X = alg.run()
            
            recon_image.append(sp.to_device(X, sp.cpu_device))
                
                
            # TV = sp.linop.FiniteDifference(PFTSMs_batch.ishape,axes = (0,1,2))
            # TV = sp.linop.FiniteDifference(PFTSMs.ishape,axes = (0,1,2))
            # print(f'TV ishape: {TV.ishape}, TV oshape: {TV.oshape}')
                
            ####### debug
            #proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(TV.oshape, lambda_TV), TV)
            
            ## precondition
            # wdata_all = []
            # p_all = []
            # L=0
            # for b, PFTSM in enumerate(PFTSMs_all):
            #     wdata = kdata[:, batches[b], ...]*dcf[:,np.newaxis, ...]*1e4
            #     wdata_all.append(wdata)
            #     p_all.append(np.zeros_like(wdata))
            #     L += PFTSM.H*PFTSM*np.complex64(np.ones(tshape))
            # L = np.mean(np.abs(L))
            # alpha = np.max(np.abs(PFTSMs_all[0].H*wdata_all[0]))
            
            # tmp = PFTSMs.H*PFTSMs*np.complex64(np.ones(tshape))
            # L=np.mean(np.abs(tmp))
            # wdata = kdata*dcf[:,np.newaxis, ...]
            # alpha = np.max(np.abs(PFTSMs.H*wdata))
            
            # ADMM
            
            ###### debug
            # print('alpha:{}'.format(alpha))
            
            # X = np.zeros(tshape,dtype=np.complex64)
            # X0 = np.zeros_like(X)
            # q = np.zeros((3,)+tshape,dtype=np.complex64)
            # pbar = tqdm(range(self.max_iter), desc="ADMM")
            # for j in pbar:
            #     adjoint_sum = 0
            #     for b, PFTSM in enumerate(PFTSMs_all):
            #         # Forward
            #         AX = PFTSM * X

            #         # Update p for this batch
            #         p_all[b] = (
            #             p_all[b] + sigma * (AX - wdata_all[b])
            #         ) / (1 + sigma)

            #         # Accumulate adjoint
            #         adjoint_sum += PFTSM.H * p_all[b]

            #     q = (q + sigma*TV*X)
            #     q = q / np.maximum(1, np.abs(q) / alpha)
            #     X0 = X
            #     X = X - tau * ((1/L) * adjoint_sum + self.lamda * TV.H * q)
                
            #     if j % 5 == 0:
            #         resid = np.linalg.norm(X-X0)/np.linalg.norm(X)
            #         pbar.set_postfix(resid="{0:.2E}".format(resid))
            
            # recon_image.append(sp.to_device(X, sp.cpu_device))
            
            # X = np.zeros(tshape,dtype=np.complex64)
            # X0 = np.zeros_like(X)
            # p = np.zeros_like(wdata)
            # q = np.zeros((3,)+tshape,dtype=np.complex64)
            # pbar = tqdm(range(self.max_iter), desc="ADMM")
            # for j in pbar:
            #     p = (p + sigma*(PFTSMs*X-wdata))/(1+sigma)
            #     q = (q + sigma*TV*X)
            #     q = q/(np.maximum(np.abs(q),alpha)/alpha)
            #     # q = q / np.maximum(1, np.abs(q)/alpha)
            #     X0 = X
            #     X = X-tau*(1/L*PFTSMs.H*p + self.lamda*TV.H*q)
                
            #     if j % 2 == 0:
            #         resid = np.linalg.norm(X-X0)/np.linalg.norm(X)
            #         pbar.set_postfix(resid="{0:.2E}".format(resid))
            
            
            # kdata = array_to_gpu(kdata, sp.Device(self.device))
            # dcf = array_to_gpu(dcf, sp.Device(self.device))
            # coord = array_to_gpu(coord, sp.Device(self.device))
            
           
        
        return recon_image
        
        
    def run(self):
        out_name = os.path.join(self.out_folder, 'debug_imoco.h5')
        
        if self.resp_frames > 1:
            motion_images, resp_kdata = self.xd_grasp_recon()
        else:
            motion_images = [np.zeros(tuple(self.mri_data.tshape))]
            resp_kdata = [resp_gate(self.mri_data, resp_lower=0, resp_upper=0.5,
                                   resp_filter_window=self.resp_filter_window, debug_folder=None)]
            resp_kdata = [self.mri_data]
            
        if not self.debug:
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
                # frame 0 should be end expiration, register all other frames to it
                for r in range(self.resp_frames):
                    if r == 0:
                        M_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                        iM_field = np.zeros(tuple(self.mri_data.tshape)+(3,))
                    else:
                        self.logger.info(f'Registering phase {r} to phase 0')
                        M_field, iM_field = self.register(motion_images[0], motion_images[r])
                        # print(M_field.shape)
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
        
        # self.logger.info('Scaling motion fields to match original dimensions')
        # M_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in M_fields]
        # iM_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in iM_fields]
        
        # remove to save space
        del motion_images
        
        gated_data = []
        if self.card_frames > 1:
            for r in range(self.resp_frames):
                card_data = gate_kspace(mri_raw=resp_kdata[r], num_frames=self.card_frames, gate_type=self.gate_type)
                gated_data.append(card_data)
        else:
            gated_data = resp_kdata

        image = self.full_recon(gated_data, M_fields, iM_fields)
        
        return image
        
    
class interp_op(sp.linop.Linop):
    def __init__(self, ishape, M_field, iM_field=None):
        assert list(ishape) == list(M_field.shape[:-1]),"Dimension mismatch!"
        oshape = ishape
        # print("init")
        # N = 64
        # b spline interpolation
        # if k_id == 0:
        #     self.kernel = [(3*(x/N)**3-6*(x/N)**2+4)/6 for x in range(0,N)]+[(2-x/N)**3/6 for x in range(N,2*N)]
        #     dkernel = np.array([-.2,1.4,-.2])
        #     k_wid = 4
        # else:
        # self.kernel = np.asarray([1-x/(2*N) for x in range(0,2*N)])
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
        # print("apply called")
        device = sp.backend.get_device(input)
        with device:
            return interp(input, self.M_field, self.m, self.dkernel, device) # major change

    def _adjoint_linop(self):
        # print("adjoint called")
        # device = sp.backend.get_device(input)
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
    # print("interp called")
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

# def DLD(Linop, device=sp.Device(0)):
#     B1 = sp.linop.ToDevice(Linop.ishape,idevice=sp.Device(-1),odevice=device)
#     B2 = sp.linop.ToDevice(Linop.oshape,idevice=sp.Device(-1),odevice=device)
#     Linop = B2.H*Linop*B1
#     return Linop
    
def NFTs(ishape, coord, device=sp.Device(0)):
    n_Channel = ishape[0]
    oshape = list((n_Channel,)) + list(coord.shape[:-1])
    NFT = sp.linop.NUFFT(ishape[1:], coord=coord)
    NFTs = Diags([NFT for i in range(n_Channel)],oshape,ishape)
    return NFTs     
        