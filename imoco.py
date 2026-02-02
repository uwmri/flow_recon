import antspyx as ants
import numpy as np 
import sigpy as sp
import h5py
from scipy.ndimage import minimum_filter1d, maximum_filter1d



class iMoCoRecon:
    def __init__ (self, mri_data, mps, gate_type='ecg', frames1=1, frames2=1, lamda=0,
                 device=None, coil_batch_size=None, batched_iter=50, show_pbar=True,
                 comm=None, log_folder=None, resp_filter_window=5, **kwargs):
        
        self.mri_data = mri_data
        self.mps = mps
        self.gate_type = gate_type
        self.frames1 = frames1
        self.frames2 = frames2
        self.lamda = lamda
        self.device = device
        self.coil_batch_size = coil_batch_size
        self.batched_iter = batched_iter
        self.show_pbar = show_pbar
        self.comm = comm
        self.log_folder = log_folder
        self.resp_filter_window = resp_filter_window

    def resp_gate(self, bins):
        time = self.mri_data.time[0].flatten()
        resp = self.mri_data.resp[0].flatten()
    
        dt = np.max(time) / len(time)
        resp_filter_width = int(self.resp_filter_window / dt)
        
        mins = minimum_filter1d(resp, resp_filter_width) 
        maxs = maximum_filter1d(resp, resp_filter_width)
        norm = (resp - mins) / (maxs - mins + 1e-8)
        
        # assign index to each resp phase
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        resp_bin_idx = np.clip(np.digitize(norm, bin_edges)-1, 0, bins-1)
        return resp_bin_idx
        
    def resp_recon(self, resp_bin_idx):
        
        return
        
        
    def register(self, fixed, moving):
        fixed = ants.from_numpy(fixed)
        moving = ants.from_numpy(moving)
        
        reg = ants.registration(fixed, moving, type_of_transform='SyNOnly', initial_transform="identity",\
                                syn_metric='demons', syn_sampling=4, \
                                grad_step=0.1, flow_sigma=5, total_sigma=3,\
                                reg_iterations=(100,100,40,20,10), \
                                verbose=False, outprefix=self.log_folder, \
                                w='[0.1,1]', write_composite_transform=False)
        
        M_field = reg['fwdtransforms'][0]
        iM_field = reg['invtransforms'][0]   
        
        return M_field, iM_field
        
        
    def full_recon(self):
        
        return
        
        
    def run(self):
        debug = h5py.File('debug_imoco.h5', 'w')
        resp_phases = []
        for i in range(self.frames2):
            resp_phase = self.gate_data(i)
            resp_phases.append(resp_phase)

        debug['resp_phases'] = np.stack(resp_phases, axis=-1)

        M_fields = []
        iM_fields = []
        for i in range(self.frames2 - 1):
            M_field, iM_field = self.register(np.abs(resp_phases[0]), np.abs(resp_phases[i+1]))
            M_fields.append(M_field)
            iM_fields.append(iM_field)
        
        debug['M_fields'] = np.stack(M_fields, axis=-1)
        debug['iM_fields'] = np.stack(iM_fields, axis=-1)

        return
        
    

class interp_op(sp.Linop):
    def __init__(self, ishape, M_field, iM_field = None):
        ndim = M_field.shape[-1]
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
        device = sp.backend.get_device(input)
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field

        return interp_op(self.ishape, iM_field, M_field)
    
def interp(I, M_field, device = sp.Device(-1), k_id = 1, deblur = True):
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
    from importlib_metadata import version
    if version('sigpy') <= '0.1.16':
        I = sp.interp.interpolate(I,k_wid,kernel,M_field.astype(np.float64)) # v0.1.16 (input, width, kernel, coord)
    else:    
        M_field_device = sp.to_device(input=M_field.astype(np.float64), device=g_device) # v0.1.17
        I = sp.interp.interpolate(input=I,coord=M_field_device) # v0.1.17 (input, coord, kernel='spline', width=2, param=1)
    # deconv
    if deblur is True:
        sp.conv.convolve(I,dkernel)
    I = sp.to_device(input=I,device=c_device)
    
    return I
        
        