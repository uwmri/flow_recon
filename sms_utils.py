import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from glob import glob
import h5py
import math
# from tkinter import Tk


# Load raw k-space data (MRI_Raw.h5)
# NOTE: Raw k-space is output from C++ pcvipr_recon_binary with "-export_kdata" flag
def load_mri_raw(directory, file):
    filename = os.path.join(directory, file)
    with h5py.File(filename, 'r') as hf:
        num_encs = 0  #get number of encodes
        while f'KW_E{num_encs}' in hf['Kdata']:
            num_encs += 1
        num_coils = 0 #get number of coils
        while f'KData_E0_C{num_coils}' in hf['Kdata']:  # find number of coils
            num_coils += 1

        kcoord = []  # kcoord: [encode, dimension(x,y,z,w,t), projections, coords]
        ksp = []  # ksp --> [encode, coils, projections, data]
        for e in range(num_encs):
            print(f'Encode: {e}')
            kx = np.array(hf['Kdata'][f'KX_E{e}'])
            ky = np.array(hf['Kdata'][f'KY_E{e}'])
            kz = np.array(hf['Kdata'][f'KZ_E{e}'])
            kw = np.array(hf['Kdata'][f'KW_E{e}'])
            kt = np.array(hf['Kdata'][f'KT_E{e}'])
            kcoord.append(np.stack((kx, ky, kz, kw, kt)))
            del kx, ky, kz, kw, kt

            kdata = []
            for c in range(num_coils):
                print(f'Coil: {c}')
                k = hf['Kdata'][f'KData_E{e}_C{c}']
                kdata.append(k['real'] + 1j * k['imag'])
            kdata = np.stack(kdata, 0)
            ksp.append(kdata)

        kcoord = np.squeeze(kcoord)
        ksp = np.squeeze(ksp)
        return kcoord, ksp

def show_raw(sms_dir):
    kcoord, ksp = load_mri_raw(sms_dir, 'MRI_Raw.h5')

    encode = 0
    Kx = kcoord[encode, 0, :, :]
    Ky = kcoord[encode, 1, :, :]

    sample = 0
    Kxd = Kx[:, sample]
    Kyd = Ky[:, sample]
    angles = (180/math.pi)*np.arctan(Kyd/Kxd) + 180
    plt.plot(angles)
    plt.show()

    numproj = Kx.shape[0]
    for p in range(numproj):
        Kxa = Kx[p, :]
        Kya = Ky[p, :]
        plt.scatter(Kxa, Kya, s=0.1, marker='o')
        plt.show()

class VolumeViewer(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap="gray")
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

# def plot3d(image):
#     fig, ax = plt.subplots(1, 1)
#     tracker = VolumeViewer(ax, image)
#     fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#     plt.show()

# show reconstructed SMS images
def show_images(sms_dir, time_resolved=False):
    sms_file = os.path.join(sms_dir, 'Flow.h5')
    with h5py.File(sms_file, 'r') as hf:
        frames = hf['Header'].attrs['frames']
        sms_factor = hf['Header'].attrs['sms_factor']
        comp_mag = hf['MAG']
        comp_cd = hf['CD']
        comp_vz = hf['comp_vd_3']
        if time_resolved:
            mag = np.zeros((comp_mag.shape)+(frames,))
            cd = np.zeros((comp_mag.shape)+(frames,))
            vz = np.zeros((comp_mag.shape)+(frames,))
            for i in range(frames):
                mag[..., i] = hf[f'ph_{i:03}_mag']
                cd[..., i] = hf[f'ph_{i:03}_cd']
                vz[..., i] = hf[f'ph_{i:03}_vd_3']
                
    # mag = np.rot90(mag, k=1, axes=(0, 1))
    # cd = np.rot90(cd, k=1, axes=(0, 1))

    fig, axs = plt.subplots(sms_factor, 3, figsize=(10, 10))
    for i in range(sms_factor):
        if time_resolved:
            tracker = VolumeViewer(axs[i, 0], mag)
            fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            tracker2 = VolumeViewer(axs[i, 1], cd)
            fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
            tracker3 = VolumeViewer(axs[i, 2], vz)
            fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)
        else:
            axs[i, 0].imshow(comp_mag, cmap='gray')
            axs[i, 0].set_title(f'Magnitude - slice {i}')
            axs[i, 0].axis('off')
            axs[i, 1].imshow(comp_cd, cmap='gray')
            axs[i, 1].set_title(f'Complex Difference - slice {i}')
            axs[i, 1].axis('off')
            axs[i, 2].imshow(comp_vz, cmap='gray')
            axs[i, 2].set_title(f'Velocity - slice {i}')
            axs[i, 2].axis('off')
            

    fig.tight_layout()
    plt.show()
    
# SMS sequence has wonky prescription so need to use dummy prescription to match needed sms fov, enter top and bottom slice locations (S-I, where I is negative S)
def calc_rx(sms_factor, slice_locs):
    sms_factor = int(sms_factor)
    slice_locs = slice_locs.split(',')
    slice_locs = [float(i) for i in slice_locs]
    slice_locs = sorted(slice_locs, reverse=True)
    if len(slice_locs) < 2:
        raise ValueError("At least two slice locations are required.")
    if sms_factor < 2:
        raise ValueError("SMS factor must be at least 2.")
    
    rad_height = 186
    sms_height = 150
   
    scan_height = slice_locs[0] - slice_locs[-1]
    # sms_gap = scan_height / (sms_factor - 1.0)
    sms_fov = scan_height * sms_factor/(sms_factor - 1.0)
    
    aao_rad = [slice_locs[0] - rad_height/2, slice_locs[0] + rad_height/2]
    tho_rad = [slice_locs[1] - rad_height/2, slice_locs[1] + rad_height/2]
    abd_rad = [slice_locs[2] - rad_height/2, slice_locs[2] + rad_height/2]
    
    sms = [slice_locs[1] - sms_height/2, slice_locs[1] + sms_height/2]
    
    print(f"Slice locations (mm S-I): {slice_locs}")
    print(f"AAo radial location: {aao_rad}")
    print(f"Tho radial location: {tho_rad}")
    print(f"Abd radial location: {abd_rad}")
    print(f"SMS FOV: {sms_fov} mm")
    print(f"SMS location: {sms}")
    

    # z_min = 2.0
    # z_max = 20.0
    # z_step = 1
    # max_M = 100
    # best_M = None
    # best_z = None
    # min_error = 99999
    # z_vals = np.arange(z_min, z_max + z_step, z_step)
    
    # for z in z_vals:
    #     for M in range(2, max_M + 1, 2): 
    #         dummy_height = M * z
    #         error = abs(dummy_height - scan_height)
            
    #         if error < min_error:
    #             min_error = error
    #             best_M = M
    #             best_z = z
    
    # all_slice_locs = [round(slice_locs[-1] - i * sms_gap, 1) for i in range(sms_factor)]
    # results = [round(sms_fov, 2), all_slice_locs, round(best_M), round(best_z), round(min_error, 1)]
    # print(f"SMS FOV: {results[0]} mm")
    # print(f"Slice locations (mm S-I): {results[1]}")
    # print(f"Number of dummy slices: {results[2]} mm")
    # print(f"Dummy slice thickness: {results[3]} mm")
    # print(f"Error (each side): {results[4]} mm")

if __name__ == "__main__":

    if sys.argv[1] == 'calc_rx':
        calc_rx(sys.argv[2], sys.argv[3]) # sms_factor, slice_locs
    elif sys.argv[1] == 'show_images':
        show_images(sys.argv[2]) # sms_dir
    elif sys.argv[1] == 'show_raw':
        show_raw(sys.argv[2])
    else:
        print("Invalid command")
