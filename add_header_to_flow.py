#%% This script is used to add in a /Header group and automatically fill it in to copy that of the cpp output so it works with QVT and other scripts
import h5py
import numpy as np
from pathlib import Path
import re

def add_header_to_flow(new_flow_h5):
    new_flow_h5 = Path(new_flow_h5)
    header_txt = new_flow_h5.parent / "pcvipr_header.txt"
    
    if not new_flow_h5.exists():
        raise FileNotFoundError(f"Flow file not found: {new_flow_h5}")
    
    if not header_txt.exists():
        raise FileNotFoundError(f"pcvipr header file not found: {header_txt}")
    
    values = {}
    
    with open(header_txt, "r") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
        
            key, value = line.split(maxsplit=1)
            
            if key == "pfile":
                values[key] = value
            else:
                values[key] = float(value)
    
    recon_log = new_flow_h5.parent / "recon.log"
    median_rr_ms = None

    # Parsing through recon.log to fill out /Header
    if recon_log.exists():

        with open(recon_log, "r") as f:

            for line in f:

                # vals_within_expected_rr_pct
                match = re.search(r"Values within expected RR\s*=\s*([\d.]+)\s*%", line)
                if match and "vals_within_expected_rr_pct" not in values:
                    values['vals_within_expected_rr_pct'] = float(match.group(1))
                    
                    print(f"Found vals_within_expected_rr_pct with a value of: {values['vals_within_expected_rr_pct']}")


                # expected_hr_bpm
                match = re.search(r"Expected HR is\s*([\d.]+)\s*bpm", line)
                if match and "expected_hr_bpm" not in values:
                    values['expected_hr_bpm'] = int(float(match.group(1)))

                    print(f"Found expected_hr_bpm with a value of: {values['expected_hr_bpm']}")


                # xres
                match = re.search(r"Xres:\s*(\d+)", line)
                if match and "xres" not in values:
                    values['xres'] = int(match.group(1))

                    print(f"Found xres with a value of: {values['xres']}")


                # numrecv
                match = re.search(r"Recv Number\s+(\d+)", line)
                if match and "numrecv" not in values:
                    values['numrecv'] = int(match.group(1))

                    print(f"Found numrecv with a value of: {values['numrecv']}")


                # acq_bw
                match = re.search(r"Acq BW\s*=\s*([\d.]+)", line)
                if match and "acq_bw" not in values:
                    values['acq_bw'] = int(float(match.group(1)))

                    print(f"Found acq_bw with a value of: {values['acq_bw']}")

                # nproj
                match = re.search(r"Nproj:\s*(\d+)", line)
                if match and "nproj" not in values:
                    values['nproj'] = int(match.group(1))

                    print(f"Found nproj with a value of: {values['nproj']}")
                    
                # frames
                match = re.search(r"num of frames =\s*(\d+)", line)
                if match:
                    values['frames'] = int(match.group(1))
                    
                    print(f"Found frames with a value of: {values['frames']}")

                # matrixz, matrixy, matrixx from INFO:autofov:Image shape: [{z}, {y}, {x}]
                # used because pcvipr_header.txt will write in as 320, 320, 320, but python will crop as needed
                match = re.search(r"INFO:autofov:Image shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", line)
                if match:
                    values['matrixz'] = int(match.group(1))
                    values['matrixy'] = int(match.group(2))
                    values['matrixx'] = int(match.group(3))
                    
                

                # Median RR format 1 (preferred):
                # Median RR is ___ ms {ms}
                match_ms = re.search(r"Median RR is\s+([\d.]+)\s*ms", line)

                if match_ms and median_rr_ms is None:
                    median_rr_ms = float(match_ms.group(1))


                # Median RR format 2 (backup):
                # INFO:Get Gate bins:Median RR = _.___ {s}
                match_sec = re.search(r"Median RR\s*=\s*([\d.]+)", line)

                if match_sec and median_rr_ms is None:
                    median_rr_ms = float(match_sec.group(1)) * 1000
            
            print(f"Changed values for matrixz, matrixy, matrixx to : {values['matrixz'], values['matrixy'], values['matrixx']}")
            
    # Write in timeres and median_rr_interval_ms    
    if median_rr_ms is not None:
        nframes = 20
        values['timeres'] = median_rr_ms / nframes
        values['median_rr_interval_ms'] = median_rr_ms

        print(f"timeres was 0, by using recon.log, Median RR {median_rr_ms} ms / {nframes} frames = {values['timeres']} ms")
    else:
        values['timeres'] = 52.9
        values['median_rr_interval_ms'] = 1058
        
        print("timeres was 0 and NO Median RR was found, using fallback timeres=52.9 ms and median_rr_interval_ms=1058, may NOT be correct, verify!")
        

    """
        # Build a 4-point referenced encoding matrix, 
        # as pcvipr_header.txt provides encodings 0, 1, and 2
        -----------------
        # Already done, in flow_processing.py, earlier, just needs to be converted to the header
    """            
    
    a = abs(values['vx0'])
    
    encoding_matrix = np.array([
        [-a, a, -a, -a],
        [-a, -a, a, -a],
        [-a, -a, -a, a],
    ], dtype=np.float64)
    
    # Input values into the encoding_matrix from the pcvipr_header.txt
    encoding_keys = {
        "vx0", "vy0", "vz0",
        "vx1", "vy1", "vz1",
        "vx2", "vy2", "vz2",
    }
    
    # Add /Header to recently created Flow.h5
    with h5py.File(new_flow_h5, "a") as f:
        
        # Prevent accidetnally destroying existing Header
        if "Header" in f:
            del f['Header']
        
        header = f.create_group("Header")
        
        for key, value in values.items():
            
            if key in encoding_keys:
                continue
            
            if key == "pfile":
                header.attrs[key] = value
            else:
                header.attrs[key] = value
                
        # Num encodes not listed in pcvipr_header.txt, so add here
        header.attrs['num_encodes'] = 4
        
        # Add /Header/encoding_matrix
        header.create_dataset(
            "encoding_matrix",
            data=encoding_matrix,
            dtype=np.float64
        )
        
    print(f"Added /Header to : {new_flow_h5}\n\n")
            
new_flow = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/gate_delay_300/Flow3D.h5"
add_header_to_flow(new_flow)

# Comparisons
ref_flow = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/cppRecon/standard/Flow.h5"
with h5py.File(ref_flow, "r") as f:
    print("Reference encoding matrix is:\n")
    print(f['/Header/encoding_matrix'][()])
    print("\n Shape of matrix is:")
    print(f['/Header/encoding_matrix'].shape)

print("\n\n")

with h5py.File(new_flow, "r") as f:
    print("New encoding matrix is:\n")
    print(f['/Header/encoding_matrix'][()])
    print("\n Shape of matrix is:")
    print(f['/Header/encoding_matrix'].shape)
#%%
