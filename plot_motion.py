import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
from registration_tools import *
import math

folders = [f'Q:/FlowMotion/Flow{i}/' for i in range(10)]

#folder = 'Q:/FlowMotion/Flow5/'
folders = ['D:\\mc_flow\\DATA\\mc_adrc01086',]

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# For storing the images
phi_cases = []
theta_cases = []
psi_cases = []
tx_cases = []
ty_cases = []
tz_cases = []

range_translation = []
for folder in folders:

    dx = 220/320 * 2
    dy = 220/320 * 2
    dz = 220/320 * 2

    with h5py.File(os.path.join(folder,'RegisteredImages.h5'),'r') as hf:
        phi = np.array(hf['phi'])
        theta = np.array(hf['theta'])
        psi = np.array(hf['psi'])

        tx = dx * np.array(hf['tx'])
        ty = dy * np.array(hf['ty'])
        tz = dz * np.array(hf['tz'])

        moving = np.array(hf['MOVING'])
        registered = np.array(hf['REGISTERED'])

    im_avg = np.mean(registered, 0)
    mask = (im_avg > 0.1*np.max(im_avg)).astype(np.float32)

    dt = 427.7 / float(len(tx))
    tt = np.arange(len(tx)) * dt

    fig = plt.figure()
    plt.plot(tt, phi,label=r'$\phi$', linewidth=4)
    plt.plot(tt, psi,label=r'$\psi$', linewidth=4)
    plt.plot(tt, theta,label=r'$\theta$', linewidth=4)
    plt.xlim((np.min(tt), np.max(tt)))
    plt.ylim([-4, 4])
    plt.xlabel(r'Time [s]')
    plt.ylabel(r'Rotation [$^{\circ}$]')

    plt.legend(frameon=False)
    plt.show()
    fig.savefig(os.path.join(folder, 'Rotation.png'), dpi=fig.dpi)

    fig = plt.figure()

    plt.plot(tt, tx * dx, linewidth=4, label=r'$\Delta$ x')
    plt.plot(tt, ty * dy, linewidth=4, label=r'$\Delta$ y')
    plt.plot(tt, tz * dz, linewidth=4, label=r'$\Delta$ z')
    plt.xlim((np.min(tt), np.max(tt)))
    plt.ylim([-2,2])
    plt.xlabel('Time [s]')
    plt.ylabel('Translation [mm]')
    plt.legend(frameon=False)
    plt.show()
    fig.savefig(os.path.join(folder, 'Translation.png'), dpi=fig.dpi)

    # sliced_moving = np.concatenate( (moving[:, moving.shape[1]//2, :,:], \
    #                                  moving[:, :, moving.shape[2]//2,:], \
    #                                  moving[:, :, :, moving.shape[3]//2]), axis=2)
    #
    # sliced_reg = np.concatenate( (registered[:, moving.shape[1]//2, :,:], \
    #                                  registered[:, :, moving.shape[2]//2,:], \
    #                                  registered[:, :, :, moving.shape[3]//2]), axis=2)
    #
    # # Export to file
    # import os
    # out_name = os.path.join(folder,'Motion_Slices.h5')
    # print('Saving images to ' + out_name)
    # try:
    #     os.remove(out_name)
    # except OSError:
    #     pass
    # with h5py.File(out_name, 'w') as hf:
    #     hf.create_dataset("REGISTERED", data=np.squeeze(np.stack(sliced_moving)))
    #     hf.create_dataset("MOVING", data=np.squeeze(np.stack(sliced_reg)))

    #
    FOV = 220
    [x0, y0, z0] = np.meshgrid(np.linspace(-FOV/2, FOV/2, mask.shape[0]),
                               np.linspace(-FOV/2, FOV/2, mask.shape[1]),
                               np.linspace(-FOV/2, FOV/2, mask.shape[2]),
                               sparse=False, indexing='ij')
    d0 = np.stack([x0.flatten(), y0.flatten(), z0.flatten()])

    dmean = np.zeros_like(d0)

    # Get the average position
    for t in range(len(tx)):
        rot = build_rotation(theta=theta[t] * math.pi / 180.0,
                             phi=phi[t] * math.pi / 180.0,
                             psi=psi[t] * math.pi / 180.0)
        d1 = rot @ d0
        d1[0, :] += tx[t]
        d1[1, :] += ty[t]
        d1[2, :] += tz[t]
        dmean += d1
    dmean /= len(tx)

    # Get the rotation
    dist = []
    for t in range(len(tx)):
        rot = build_rotation(theta=theta[t] * math.pi / 180.0,
                             phi=phi[t] * math.pi / 180.0,
                             psi=psi[t] * math.pi / 180.0)
        d1 = rot @ d0
        d1[0, :] += tx[t]
        d1[1, :] += ty[t]
        d1[2, :] += tz[t]
        diff = d1 - dmean

        delta = np.sqrt(np.sum(diff ** 2, 0))

        dist.append(np.sum(delta * mask.flatten()) / np.sum(mask.flatten()))

    range_translation.append(dist)


range_translation = np.array(range_translation)
#
fig = plt.figure()
plt.boxplot(range_translation.transpose())
plt.xlabel('Case')
plt.ylabel('Translation [mm]')
plt.show()
fig.savefig(os.path.join('C:/Users/kmjohnso/Desktop', 'MotionPlots.png'), dpi=fig.dpi)

# #plt.xlim((np.min(tt), np.max(tt)))
# #plt.ylim([-2, 2])
# plt.legend(frameon=False)
# plt.show()
# #fig.savefig(os.path.join(folder, 'Translation.png'), dpi=fig.dpi)