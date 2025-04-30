
# Import necessary libraries
import numpy as np
import h5py
import ctypes
import os

# Update GE path
import sys
orc_folder = os.getenv('ORC_PYTHON_SDKTOP', '/home/kxj135/Data/CVMRIGroup/Software/Orchestra/orchestra-sdk-2.1-1.python')
sys.path.append(orc_folder)
import GERecon


def float_to_int(data):
    # Convert float to int as a workaround to rhuser storage
    # Input: data - the float value from rhuser variable
    # Output: the integer value of the float data  
    return (ctypes.c_uint32.from_buffer(ctypes.c_float(data))).value

def read_scan_achive_data( archive):
    # Read the data from the scan archive
    # Input: archive - the scan archive
    # Output: data_all - the data from the scan archive

    data_all = []
    slice_index_all = []
    view_index_all = []
    #ontrol_all

    # Get the control count
    metadata = archive.Metadata()
    num_control = metadata["controlCount"]

    # Loop over all the control packets
    progress_update_interval = num_control // 10
    for control_packet_index in range(num_control):

        if control_packet_index % progress_update_interval == 0:
            print(f'Control {control_packet_index} of {num_control} ({ 100*control_packet_index // num_control} % )')

        # Retrieve the next control packet
        control = archive.NextControl()

        # This is a raw control packet, get the next frame so that the control and the frames are in sync.
        # But do not use this frame to fill kspace.
        if control["opcode"] == 16:
            next_frame = np.squeeze(archive.NextFrame())

        # This is a programmable control packet, so use next frame to fill a line of kspace.
        elif control["opcode"] == 1:
            # Frame data is organized as: ReadoutSize x NumChannels x NumFrames
            # where NumFrames is the number of frames corresponding to this control
            # packet. Each programmable packet corresponds to a single frame. Thus,
            # for this example, the frames dimension will always have a size of 1
            next_frame = np.squeeze(archive.NextFrame()).astype(np.complex64)

            if len(next_frame.shape) == 1:
                next_frame = np.expand_dims(next_frame, -1)

            # Place the frame into kSpace
            data_all.append(next_frame)
            #control_all.append(control)

            #echoNum = control['echoNum']
            slice_index = control['sliceNum']
            view_index = control['viewNum'] - 1

            slice_index_all.append(slice_index)
            view_index_all.append(view_index)


    print('Data loaded')

    return data_all, slice_index_all, view_index_all