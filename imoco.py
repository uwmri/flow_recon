import ants
import numpy as np 
import sigpy as sp
import h5py
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
    def __init__ (self, mri_data, mps, gate_type='ecg', card_frames=1, resp_frames=1, lamda=0,
                 device=None, res_scale=2, coil_batch_size=None, max_iter=50, show_pbar=True,
                 comm=None, resp_filter_window=5, out_folder=None, **kwargs):
        
        
        self.mri_data = mri_data
        self.mps = mps
        self.gate_type = gate_type
        self.card_frames = card_frames
        self.resp_frames = resp_frames
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
    
    
    def xd_grasp_recon(self):
        self.logger.info(f'Performing XD-GRASP recon with {self.resp_frames} respiratory phases')
        if self.resp_frames == 1:
            resp_bins = [0, 0.5]
        else:
            resp_bins = np.linspace(0, 1, self.resp_frames+1)
        self.logger.info(f'Respiratory bins: {resp_bins}')
        
        # data = self.mri_data.copy_data(full=True)
        # crop_kspace(data, crop_factor=self.res_scale, crop_type='radius')
        
        resp_imgs = []
        resp_kdata = []
        for r in range(self.resp_frames):
            self.logger.info(f'Gating respiratory phase {r}')
            resp_phase = resp_gate(self.mri_data, resp_lower=resp_bins[r], resp_upper=resp_bins[r+1],
                                   resp_filter_window=self.resp_filter_window, debug_folder=None)
            resp_kdata.append(resp_phase)
            
            self.logger.info(f'Reconstructing phase {r}')
            resp_img = pils_recon(resp_phase, smaps=self.mps, device=self.device)
            img = np.stack(resp_img, axis=-1)
            mag = np.sqrt(np.sum(np.abs(img)**2, axis=-1))
            resp_imgs.append(mag)
        
        return resp_imgs, resp_kdata
    
    def register(self, fixed, moving):
        fixed = ants.from_numpy(fixed)
        moving = ants.from_numpy(moving)
        
        reg = ants.registration(fixed, moving, type_of_transform='SyNOnly', initial_transform="identity",\
                                syn_metric='demons', syn_sampling=4, \
                                grad_step=0.1, flow_sigma=5, total_sigma=3,\
                                reg_iterations=(100,100,40,20,10), \
                                verbose=False, outprefix=self.out_folder, \
                                w='[0.1,1]', write_composite_transform=False)
        
        M_field = ants.image_read(reg['fwdtransforms'][0])
        iM_field = ants.image_read(reg['invtransforms'][-1])   
        
        return M_field.numpy(), iM_field.numpy()
    
    def M_scale(self, M, oshape):
        Mscale = [oshape[i]/M.shape[i] for i in range(M.shape[-1])]
        Mo = np.zeros(oshape+(M.shape[-1],))
        for i in range(M.shape[-1]):
            M[...,i] = M[...,i]*(Mscale[i]*self.res_scale)
            Mo[...,i] = ndimage.zoom(M[...,i],zoom=tuple(Mscale),order=2)

        return Mo
        
    def full_recon(self, resp_kdata, M_fields, iM_fields):
        gated_data = []
        if self.card_frames > 1:
            for r in range(self.resp_frames):
                card_data = gate_kspace(mri_raw=resp_kdata[r], num_frames=self.card_frames, gate_type=self.gate_type)
                gated_data.append(card_data)
        else:
            gated_data = resp_kdata
            
        nCoils = self.mps.shape[0]
        if self.coil_batch_size is None:
            self.coil_batch_size = nCoils
        batches = [
            range(i, min(i + self.coil_batch_size, nCoils))
            for i in range(0, nCoils, self.coil_batch_size)
        ]

        tshape = tuple(self.mri_data.tshape)
        ## low rank
        Ms = []
        M0s = []
        for i in range(self.resp_frames):
            # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
            M = interp_op(tshape,M_fields[i])
            M0 = interp_op(tshape,np.zeros(tshape+(3,)))
            M = DLD(M,device=sp.Device(self.device))
            M0 = DLD(M0,device=sp.Device(self.device))
            Ms.append(M)
            M0s.append(M0)
        Ms = Diags(Ms,oshape=(self.resp_frames,)+tshape,ishape=(self.resp_frames,)+tshape)
        M0s = Diags(M0s,oshape=(self.resp_frames,)+tshape,ishape=(self.resp_frames,)+tshape)
        
        # print(f'Ms ishape: {Ms.ishape}, Ms shape: {Ms.oshape}')
        # print(f'M0s ishape: {M0s.ishape}, M0s shape: {M0s.oshape}')
        
        sigma = 0.4
        tau = 0.4
        recon_image = []
        for e in range(self.card_frames*self.mri_data.Num_Encodings):
            self.logger.info(f'TV Recon: Frame {e}')
            
            # pad kdata, dcf, and coord to have the same shape across respiratory phases
            max_size = np.max([gated_data[r].kdata[e].shape[-1] for r in range(self.resp_frames)], axis=0)
            # pad then stack kdata, dcf, and coord
            
            kdata_list = []
            dcf_list = []
            coord_list = []
            for r in range(self.resp_frames):
                kdata = gated_data[r].kdata[e]
                dcf = gated_data[r].dcf[e]
                coord = gated_data[r].coords[e]
                
                # pad to max_size
                if kdata.shape[-1] < max_size:
                    pad_width = max_size - np.array(kdata.shape[-1])
                    pad_width = np.max(pad_width, 0)
                    kdata_pad = np.pad(kdata, [(0,0), [0, pad_width]], mode='constant', constant_values=0)
                    dcf_pad = np.pad(dcf, [[0, pad_width]], mode='constant', constant_values=0)
                    coord_pad = np.pad(coord, [[0, pad_width], [0, 0]], mode='constant', constant_values=0)
                else:
                    kdata_pad = kdata
                    dcf_pad = dcf
                    coord_pad = coord
                
                kdata_list.append(kdata_pad)
                dcf_list.append(dcf_pad)
                coord_list.append(coord_pad)

            kdata = np.stack(kdata_list, axis=0)
            dcf = np.stack(dcf_list, axis=0)
            coord = np.stack(coord_list, axis=0)
            
            PFTSMs_all = []
            Is_all = []
            for b in batches:
                
                S_batch = sp.linop.Multiply(tshape, self.mps[b, ...])
                
                PFTSMs_batch = []
                Is_batch = []
                for i in range(self.resp_frames):
                    Is_batch.append(sp.linop.Identity(tshape))
                    FTs = NFTs((len(b),)+tshape,coord[i,...],device=sp.Device(self.device))
                    M = interp_op(tshape,M_fields[i])
                    M = DLD(M,device=sp.Device(self.device))
                    # print(f'M ishape: {M.ishape}, M oshape: {M.oshape}')
                    W = sp.linop.Multiply((kdata[i,b].shape),dcf[i]) 
                    # print(f'W ishape: {W.ishape}, W oshape: {W.oshape}')
                    FTSM = W*FTs*S_batch*M
                    # print(f'FTSM ishape: {FTSM.ishape}, FTSM oshape: {FTSM.oshape}')
                    PFTSMs_batch.append(FTSM)
                Is_all.append(Is_batch)
                PFTSMs_batch = Diags(PFTSMs_batch,oshape=kdata[:,b].shape,
                                     ishape=(self.resp_frames,)+tshape)*Vstacks(Is_batch,ishape=tshape,oshape=(self.resp_frames,)+tshape)
                PFTSMs_all.append(PFTSMs_batch)
                
                # print(f'PFTSMs ishape: {PFTSMs_batch.ishape} PFTSMs oshape: {PFTSMs_batch.oshape}')
                
            TV = sp.linop.FiniteDifference(PFTSMs_batch.ishape,axes = (0,1,2))
            # print(f'TV ishape: {TV.ishape}, TV oshape: {TV.oshape}')
                
            ####### debug
            #proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(TV.oshape, lambda_TV), TV)
            
            ## precondition
            wdata_all = []
            p_all = []
            L=0
            for b, PFTSM in enumerate(PFTSMs_all):
                wdata = kdata[:, batches[b], ...]*dcf[:,np.newaxis, ...]*1e4
                wdata_all.append(wdata)
                p_all.append(np.zeros_like(wdata))
                L += PFTSM.H*PFTSM*np.complex64(np.ones(tshape))
            L = np.mean(np.abs(L))
            alpha = np.max(np.abs(PFTSMs_all[0].H*wdata_all[0]))
            
            # ADMM
            
            ###### debug
            # print('alpha:{}'.format(alpha))
            
            X = np.zeros(tshape,dtype=np.complex64)
            X0 = np.zeros_like(X)
            q = np.zeros((3,)+tshape,dtype=np.complex64)
            pbar = tqdm(range(self.max_iter), desc="ADMM")
            for j in pbar:
                adjoint_sum = 0
                for b, PFTSM in enumerate(PFTSMs_all):
                    # Forward
                    AX = PFTSM * X

                    # Update p for this batch
                    p_all[b] = (
                        p_all[b] + sigma * (AX - wdata_all[b])
                    ) / (1 + sigma)

                    # Accumulate adjoint
                    adjoint_sum += PFTSM.H * p_all[b]

                q = (q + sigma*TV*X)
                q = q / np.maximum(1, np.abs(q) / alpha)
                X0 = X
                X = X - tau * ((1/L) * adjoint_sum + self.lamda * TV.H * q)
                resid = np.linalg.norm(X-X0)/np.linalg.norm(X)
                
                pbar.set_postfix(resid="{0:.2E}".format(resid))
            
            recon_image.append(sp.to_device(X, sp.cpu_device))
        
        return recon_image
        
        
    def run(self):
        out_name = os.path.join(self.out_folder, 'debug_imoco.h5')
        
        motion_images, resp_kdata = self.xd_grasp_recon()
        
        # save motion images
        try:
            os.remove(out_name)
        except OSError:
            pass
        self.logger.info(f'Saving Images to {out_name}')
        with h5py.File(out_name, 'w') as hf:
            for r in range(self.resp_frames):
                hf.create_dataset(f"MAG_RESP{r}", data=motion_images[r])
        
        self.logger.info('Performing image registration')
        M_fields = []
        iM_fields = []
        # frame 0 should be end expiration, register all other frames to it
        for r in range(self.resp_frames):
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
        
        ## Debug
        # motion_images = []
        # with h5py.File(out_name, 'r') as hf:
        #     for r in range(self.resp_frames):
        #         motion_images.append(hf[f"MAG_RESP{r}"][:])
        #     M_fields = hf[f"M_fields"][:]
        #     iM_fields = hf[f"iM_fields"][:]
        # M_fields = [M_fields[...,i] for i in range(M_fields.shape[-1])]
        # iM_fields = [iM_fields[...,i] for i in range(iM_fields.shape[-1])]
        
        # self.logger.info('Scaling motion fields to match original dimensions')
        # M_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in M_fields]
        # iM_fields = [self.M_scale(M, tuple(self.mri_data.tshape)) for M in iM_fields]

        image = self.full_recon(resp_kdata, M_fields, iM_fields)
        
        return image
        
    
class interp_op(sp.linop.Linop):
    def __init__(self, ishape, M_field, iM_field=None):
        assert list(ishape) == list(M_field.shape[:-1]),"Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        self.iM_field = iM_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = sp.backend.get_device(input)
        with device:
            return interp(input, self.M_field, device, 1) # major change

    def _adjoint_linop(self):
        # device = sp.backend.get_device(input)
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field

        return interp_op(self.ishape, iM_field, M_field)
    
def interp(I, M_field, device=sp.Device(0), k_id=1, deblur=True):
    # b spline interpolation
    N = 64
    if k_id == 0:
        kernel = [(3*(x/N)**3-6*(x/N)**2+4)/6 for x in range(0,N)]+[(2-x/N)**3/6 for x in range(N,2*N)]
        dkernel = np.array([-.2,1.4,-.2])
        
        k_wid = 4
    else:
        kernel = [1-x/(2*N) for x in range(0,2*N)]
        dkernel = np.array([0,1,0])
        deblur = False
        k_wid = 2
    kernel = np.asarray(kernel)
    
    c_device = sp.get_device(I)
    ndim = M_field.shape[-1]
    
    # 2d/3d
    if ndim == 3:
        dkernel = dkernel[:,None,None]*dkernel[None,:,None]*dkernel[None,None,:]
        Nx,Ny,Nz = I.shape
        my,mx,mz = np.meshgrid(np.arange(Ny),np.arange(Nx),np.arange(Nz))
        m = np.stack((mx,my,mz),axis=-1)
        M_field = M_field + m
    else:
        dkernel = dkernel[:,None]*dkernel[None,:]
        Nx,Ny = I.shape
        my,mx = np.meshgrid(np.arange(Ny),np.arange(Nx))
        m = np.stack((mx,my,mz),axis=-1)
        M_field = M_field + m
    # TODO remove out of range values
    
    # image warp
    
    g_device = device
    I = sp.to_device(input=I,device=g_device)
    M_field_device = sp.to_device(input=M_field.astype(np.float64), device=g_device) # v0.1.17
    I = sp.interp.interpolate(input=I,coord=M_field_device) # v0.1.17 (input, coord, kernel='spline', width=2, param=1)
    # deconv
    if deblur is True:
        sp.conv.convolve(I,dkernel)
    I = sp.to_device(input=I,device=c_device)
    
    return I

def Vstacks(L_Linop, oshape, ishape):
    assert oshape[0]==len(L_Linop), 'Number of Linop mismatch!'
    
    Linops = sp.linop.Vstack(L_Linop)
    i_vec_len = 1
    for tmp in ishape:
        i_vec_len = i_vec_len * tmp
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp
    
    R1 = sp.linop.Reshape(oshape=(o_vec_len,),ishape=oshape)
    Linops = R1.H*Linops
    
    return Linops

def Hstacks(L_Linop, oshape, ishape):
    # assert oshape[0]==len(L_Linop), 'Number of Linop mismatch!'
    
    Linops = sp.linop.Hstack(L_Linop)
    i_vec_len = 1
    for tmp in ishape:
        i_vec_len = i_vec_len * tmp
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp
    
    R2 = sp.linop.Reshape(oshape=(i_vec_len,),ishape=ishape)
    Linops = Linops*R2
    
    return Linops

def Diags(L_Linop, oshape, ishape):
    assert oshape[0]==ishape[0], 'First dim mismatch!'
    assert oshape[0]==len(L_Linop), 'Number of Linop mismatch!'
    Linops = sp.linop.Diag(L_Linop)
    i_vec_len = 1
    for tmp in ishape:
        i_vec_len = i_vec_len * tmp
    o_vec_len = 1
    for tmp in oshape:
        o_vec_len = o_vec_len * tmp
    
    R1 = sp.linop.Reshape(oshape=(o_vec_len,),ishape=oshape)
    R2 = sp.linop.Reshape(oshape=(i_vec_len,),ishape=ishape)
    Linops = R1.H*Linops*R2
    
    return Linops

def DLD(Linop, device=sp.Device(0)):
    B1 = sp.linop.ToDevice(Linop.ishape,idevice=sp.Device(-1),odevice=device)
    B2 = sp.linop.ToDevice(Linop.oshape,idevice=sp.Device(-1),odevice=device)
    Linop = B2.H*Linop*B1
    return Linop
    
def NFTs(ishape, coord, device=sp.Device(0)):
    n_Channel = ishape[0]
    oshape = list((n_Channel,)) + list(coord.shape[:-1])
    
    NFT = sp.linop.NUFFT(ishape[1:], coord=coord)
    NFTs = Diags([DLD(NFT,device=device) for i in range(n_Channel)],oshape,ishape)
    
    return NFTs     
        