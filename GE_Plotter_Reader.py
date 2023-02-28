import re
import numpy as np
import scipy.integrate as integr
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os


# <?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
# <!-- ICE Pulse Sequence Data File -->
# <PulseSequence name="PSD::EP" date="_0304201908_57_52" author="ICE Seq Plotter" beginTime="0.0"  endTime="0.000000">
# <sequencer id="0"  title="Seq(0) | Core(0) | Hw(0,0,1) | Gradient X" xtitle="Time (ms)" ytitle="Values" >
# <data waveform="sequencerData">

class PulseSequence:
    gx = None
    gy = None
    gz = None
    rho1 = None
    rho2 = None
    theta = None
    omega = None
    m1x = None
    m1y = None
    m1z = None


from tkinter import filedialog
from tkinter import *
import fnmatch

# root = Tk()
# dirname = filedialog.askdirectory(parent=root, title='Please select a directory')
dirname = '/data/data_mrcv2/99_GSR/SMS_Testing/RobertsPhantom_05771_2022-10-04/blip1_pc1'

# Get Pulse sequence structure
ps = PulseSequence()

# Open all the files
time_offset = 0.0
verbose = False


def tryconvertint(value, default, *types):
    print(value)
    try:
        val = int(value)
    except (ValueError, TypeError):
        val = default
    print(val)
    return val


# All files
files = os.listdir(dirname)
files.sort(key=lambda f: tryconvertint(os.path.splitext(f)[-1][1:], -1, 'int'))

for file in files:
    if fnmatch.fnmatch(file, '*.xml*') and not fnmatch.fnmatch(file, '*.ssp'):
        print(os.path.join(dirname, file))

        # Open the file
        e = ET.parse(os.path.join(dirname, file)).getroot()
        time_step = 0.0

        # Get the sequencers
        for child in e.findall('sequencer'):

            # Grab the data
            sequence_data = child.find('data')

            if verbose:
                print(child.attrib['title'])
                print(child.attrib['id'])
                print(child.attrib['xtitle'])
                print(child.attrib['ytitle'])
                print(sequence_data.attrib['waveform'])

            temp_text = sequence_data.text
            wave = np.matrix(temp_text).astype(np.float32)
            wave = np.reshape(wave, (-1, 2))

            # Offset
            wave[:, 0] += time_offset

            temp = re.search(string=str(child.attrib['title']), pattern='Gradient')
            if re.search(string=str(child.attrib['title']), pattern='Gradient X|SeqMGD::X') is not None:
                if ps.gx is None:
                    ps.gx = wave
                else:
                    ps.gx = np.concatenate((ps.gx, wave))
                time_step = np.max(ps.gx[:, 0])
            elif re.search(string=str(child.attrib['title']), pattern='Gradient Y|SeqMGD::Y') is not None:
                if ps.gy is None:
                    ps.gy = wave
                else:
                    ps.gy = np.concatenate((ps.gy, wave))
            elif re.search(string=str(child.attrib['title']), pattern='Gradient Z|SeqMGD::Z') is not None:
                if ps.gz is None:
                    ps.gz = wave
                else:
                    ps.gz = np.concatenate((ps.gz, wave))
            elif re.search(string=str(child.attrib['title']), pattern='RHO|SeqMGD::RHO1') is not None:
                if ps.rho1 is None:
                    ps.rho1 = wave
                else:
                    ps.rho1 = np.concatenate((ps.rho1, wave))
            else:
                if verbose:
                    print('Not matched')

        # Increment time
        print(time_offset)
        time_offset = time_step

fig, ax = plt.subplots(5, 1, sharex=True, sharey=False)

ax[0].plot(ps.gx[:, 0], ps.gx[:, 1])
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim((0, np.max(wave[:, 0])))

ax[1].plot(ps.gy[:, 0], ps.gy[:, 1])
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim((0, np.max(wave[:, 0])))

ax[2].plot(ps.gz[:, 0], ps.gz[:, 1])
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim((0, np.max(wave[:, 0])))

ax[3].plot(ps.rho1[:, 0], ps.rho1[:, 1])
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim((0, np.max(wave[:, 0])))

# Plot 1st moment of gradients (rad/cm)
# NOTE: MUST CONVERT TO PHYSICAL UNITS ON PLOTTER
fig2, ax2 = plt.subplots(3, 1, sharex=True, sharey=False)
FOV = 24  # fov in cm
C = 4257 * 2*np.pi * 1e-9 * FOV  # Hz/G to rad/G*s to rad/G*us to cm*rad/G*us
ps.m1x = integr.cumtrapz(np.ravel(ps.gx[:, 0]), np.ravel(ps.gx[:, 1])) * C
ax2[0].plot(ps.gx[1:, 0], ps.m1x)
plt.xlabel('Time [ms]')
plt.ylabel('M1 [rad/m]')
plt.xlim(0, np.max(wave[:, 0]))

ps.m1y = integr.cumtrapz(np.ravel(ps.gy[:, 0]), np.ravel(ps.gy[:, 1])) * C
ax2[1].plot(ps.gy[1:, 0], ps.m1y)
plt.xlabel('Time [ms]')
plt.ylabel('M1 [rad/m]')
plt.xlim(0, np.max(wave[:, 0]))

ps.m1z = integr.cumtrapz(np.ravel(ps.gz[:, 0]), np.ravel(ps.gz[:, 1])) * C
ax2[2].plot(ps.gz[1:, 0], ps.m1z)
plt.xlabel('Time [ms]')
plt.ylabel('M1 [rad/m]')
plt.xlim(0, np.max(wave[:, 0]))
plt.show()