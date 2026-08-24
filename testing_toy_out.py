# %%
import h5py
import numpy as np
from pathlib import Path


# def convert_h5(input_file, output_file):
input_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/llr/python_recon/100iter/Flow.h5"
output_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/llr/python_recon/100iter/Flow_3D.h5"
"""
Convert 4D (t, z, y, x) datasets into:
    /Data/CD
    /Data/ph_{t1}_cd
    /Data/pd_{t2}_cd
    ...
    /Data/MAG
    /Data/ph_{t1}_mag
    ...
    
"""

# input_file = Path(input_file)
# output_file = Path(output_file)

dataset_mapping = {
    "ANGIO":      ("CD", "cd"),
    "MAG":        ("MAG", "mag"),
    "comp_vd_1":  ("comp_vd_1", "v1"),
    "comp_vd_2":  ("comp_vd_2", "v2"),
    "comp_vd_3":  ("comp_vd_3", "v3"),
}

with h5py.File(input_file, "r") as fin, \
        h5py.File(output_file, "w") as fout:

    data_group = fout.create_group("Data")

    for input_name, (avg_name, frame_name) in dataset_mapping.items():

        dataset_path = f"/{input_name}"

        if dataset_path not in fin:
            print(f"Skipping {dataset_path}: not found")
            continue

        dataset = fin[dataset_path]

        if dataset.ndim != 4:
            print(
                f"Skipping {dataset_path}: expected (t,z,y,x), "
                f"got {dataset.shape}"
            )
            continue

        nt = dataset.shape[0]

        print(f"Processing {dataset_path}: {dataset.shape}")

        # Average over time
        avg = np.mean(dataset, axis=0)
        data_group.create_dataset(
            f"{avg_name}",
            data=avg,
            compression="gzip"
        )

        # Individual time frames
        for t in range(nt):
            data_group.create_dataset(
                f"ph_{t:03d}_{frame_name}",
                data=dataset[t],
                compression="gzip"
            )

        print(f"  Created average + {nt} time frames")

print(f"Saved to: {output_file}")
# %%
