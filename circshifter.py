import numpy as np
import h5py
import shutil
from pathlib import Path
import argparse

def circshift(input_3d, input_4d, path, shift_ms=None, shift_frames=None):

    path = Path(path)

    path_3d = path / input_3d
    path_4d = path / input_4d

    shifted_4d = path / f"{path_4d.stem}_shifted.h5"
    shifted_3d = path / f"{path_3d.stem}_shifted.h5"

    dataset_mapping = {
        "ANGIO": ("CD", "cd", 32000),
        "MAG":   ("MAG", "mag", 32000),
        "VX":    ("comp_vd_1", "vd_1", 10),
        "VY":    ("comp_vd_2", "vd_2", 10),
        "VZ":    ("comp_vd_3", "vd_3", 10),
    }

    for file in (path_3d, path_4d):
        if not file.is_file():
            raise FileNotFoundError(file)

    if len({
        path_3d.resolve(), path_4d.resolve(),
        shifted_3d.resolve(), shifted_4d.resolve()
    }) != 4:
        raise ValueError("Input and output filenames must be distinct")

    # -------------------------------------
    # STEP 1: Read metadata from original 3D
    # and frame count from original 4D
    # -------------------------------------

    with h5py.File(path_3d, "r") as f3d:
        if "Header" not in f3d:
            raise KeyError("Original 3D file has no /Header group")
        
        RR = float(f3d["/Header"].attrs["median_rr_interval_ms"])
        print(f"Median RR interval: {RR} ms")

    with h5py.File(path_4d, "r") as f4d:
        for key in dataset_mapping:
            if key not in f4d:
                raise KeyError(f"Missing 4D dataset: /{key}")
            if f4d[key].ndim != 4:
                raise ValueError(
                    f"/{key}: expected (t,z,y,x), "
                    f"got {f4d[key].shape}"
                )

        nframes = f4d["/MAG"].shape[0]

        for key in dataset_mapping:
            if f4d[key].shape[0] != nframes:
                raise ValueError(
                    f"/{key} has a different frame count"
                )

        
    if shift_frames is not None and shift_ms is not None:
        raise ValueError("You listed both how many frames and ms you want to shift, specify one.")

    if shift_frames is not None:
        print(f"The frame shift is: {shift_frames}")
    elif shift_ms is not None:
        dt = RR / nframes
        shift_frames = round(shift_ms / dt)
        print(f"The frame shift is: {shift_frames}")
    else:
        raise ValueError("Must specify either frames to shift or ms to shift, you did none.")

    

    print(f"RR interval: {RR} ms")
    print(f"Time frames: {nframes}")
    print(f"Frame duration: {dt:.2f} ms")
    print(f"Shift: {-shift_frames} frames")
    print(f"Actual shift: {-shift_frames * dt:.2f} ms")

    # -------------------------------------
    # STEP 2: Copy and shift the 4D flow file
    # -------------------------------------

    shutil.copy2(path_4d, shifted_4d)

    with h5py.File(shifted_4d, "r+") as f:
        for key in dataset_mapping:
            dataset = f[key]

            print(f"Shifting /{key}: {dataset.shape}")

            # Process one spatial slice at a time
            for z in range(dataset.shape[1]):
                data = dataset[:, z, :, :]
                dataset[:, z, :, :] = np.roll(
                    data, shift=-shift_frames, axis=0
                )

    print(f"Saved shifted 4D: {shifted_4d}")

    # -------------------------------------
    # STEP 3: Convert shifted 4D -> 3D format
    # -------------------------------------

    with h5py.File(shifted_4d, "r") as fin, \
         h5py.File(path_3d, "r") as f3d, \
         h5py.File(shifted_3d, "w") as fout:

        # Copy metadata directly from original 3D
        f3d.copy("Header", fout)

        # Copy root-level attributes
        for key, value in f3d.attrs.items():
            fout.attrs[key] = value

        data_group = fout.create_group("Data")

        for input_name, (avg_name, frame_name, scaling) in dataset_mapping.items():

            dataset = fin[input_name]
            nt = dataset.shape[0]

            print(f"Converting /{input_name}: {dataset.shape}")

            avg = np.zeros(dataset.shape[1:], dtype=np.float64)

            if input_name in ("MAG", "ANGIO"):
                max_val = max(
                    np.max(dataset[t]) for t in range(nt)
                )
                scale = scaling / max_val if max_val > 0 else 0
            else:
                scale = scaling

            for t in range(nt):
                frame = dataset[t]
                avg += frame / nt

                data_group.create_dataset(
                    f"ph_{t:03d}_{frame_name}",
                    data=frame * scale,
                    dtype=np.int16
                )

            data_group.create_dataset(
                avg_name,
                data=avg * scale,
                dtype=np.int16
            )

            print(f"  Created average + {nt} frames")

    print(f"Saved shifted 3D: {shifted_3d}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_3D', required=True, help='List the Flow_3D.h5 file for copying the header')
    parser.add_argument('--input_4D', required=True, help='List the Flow_4D.h5 file for copying and circshifting')
    parser.add_argument('--path', required=True, type=Path, help='List the path of where these files are')
    parser.add_argument('--shift_ms', default=None, type=int, help='How much do you want to shift your curve by in ms (easy to find in QVT)')
    parser.add_argument('--shift_frames', default=None, type=int, help='How much do you want to shift your curve by in frames (easy to find in FIJI)')

    args = parser.parse_args()
    
    circshift(input_3d=args.input_3D, input_4d=args.input_4D, path=args.path, shift_ms=args.shift_ms, shift_frames=args.shift_frames)
