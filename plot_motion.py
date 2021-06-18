import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib
#matplotlib.rcParams['text.usetex'] = True

with h5py.File('Q:/BBF/RegisteredImages.h5','r') as hf:
    all_phi = np.array(hf['phi'])
    all_theta = np.array(hf['theta'])
    all_psi = np.array(hf['psi'])

    all_tx = np.array(hf['tx'])
    all_ty = np.array(hf['ty'])
    all_tz = np.array(hf['tz'])

    moving = np.array(hf['MOVING'])
    registered = np.array(hf['REGISTERED'])

dx = 220/320 * 2
dy = 220/320 * 2
dz = 220/320 * 2
dt = 427.7 / 128

tt = np.arange(len(all_tx)) * dt

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.figure()
plt.plot(tt, all_phi,label=r'$\phi$', linewidth=4)
plt.plot(tt, all_psi,label=r'$\psi$', linewidth=4)
plt.plot(tt, all_theta,label=r'$\theta$', linewidth=4)
plt.xlim((np.min(tt), np.max(tt)))
plt.xlabel(r'Time [s]')
plt.ylabel(r'Rotation [$^{\circ}$]')

plt.legend(frameon=False)
plt.show()

plt.figure()

plt.plot(tt, all_tx * dx, linewidth=4, label=r'$\Delta$ x')
plt.plot(tt, all_ty * dy, linewidth=4, label=r'$\Delta$ y')
plt.plot(tt, all_tz * dz, linewidth=4, label=r'$\Delta$ z')
plt.xlim((np.min(tt), np.max(tt)))
plt.xlabel('Time [s]')
plt.ylabel('Translation [mm]')
plt.legend(frameon=False)
plt.show()

sliced_moving = np.concatenate( (moving[:, moving.shape[1]//2, :,:], \
                                 moving[:, :, moving.shape[2]//2,:], \
                                 moving[:, :, :, moving.shape[3]//2]), axis=2)

sliced_reg = np.concatenate( (registered[:, moving.shape[1]//2, :,:], \
                                 registered[:, :, moving.shape[2]//2,:], \
                                 registered[:, :, :, moving.shape[3]//2]), axis=2)

# Export to file
import os
out_name = os.path.join('Q:\BBF\Slices.h5')
print('Saving images to ' + out_name)
try:
    os.remove(out_name)
except OSError:
    pass
with h5py.File(out_name, 'w') as hf:
    hf.create_dataset("REGISTERED", data=np.squeeze(np.stack(sliced_moving)))
    hf.create_dataset("MOVING", data=np.squeeze(np.stack(sliced_reg)))
