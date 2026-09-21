#%% Imports

import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


#%% Files

original_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/all_img/Images.h5"     # 8-encoding reconstruction
split1_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im1_retest_skip_1463/Images.h5"      # encodings [1, 6, 3, 8]
split2_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im2_retest_skip_0257/Images.h5"      # [5, 2, 7, 4]

split_files = {
    "Split 1": {
        "file": split1_file,
        "encodes": [0, 5, 2, 7]
    },
    "Split 2": {
        "file": split2_file,
        "encodes": [4, 1, 6, 3]
    }
}

#%% Check dataset types

with h5py.File(original_file, "r") as original, \
     h5py.File(split1_file, "r") as split:

    original_ds = original["/Images/Encode_000_Frame_000"]
    split_ds = split["/Images/Encode_000_Frame_000"]

    print("Original:")
    print("  shape:", original_ds.shape)
    print("  dtype:", original_ds.dtype)

    print("\nSplit:")
    print("  shape:", split_ds.shape)
    print("  dtype:", split_ds.dtype)


#%% Determine image dimensions

with h5py.File(original_file, "r") as hf:

    test = hf["/Images/Encode_000_Frame_000"]

    print("Image shape:", test.shape)

    nz = test.shape[0]

    # Find number of frames
    frame = 0

    while f"/Images/Encode_000_Frame_{frame:03}" in hf:
        frame += 1

    num_frames = frame

print("Z slices:", nz)
print("Frames:", num_frames)


#%% Interactive comparison

split_widget = widgets.Dropdown(
    options=["Split 1", "Split 2"],
    value="Split 1",
    description="File:"
)

encoding_widget = widgets.IntSlider(
    value=0,
    min=0,
    max=3,
    step=1,
    description="Encoding:",
    continuous_update=False
)

frame_widget = widgets.IntSlider(
    value=0,
    min=0,
    max=num_frames - 1,
    step=1,
    description="Frame:",
    continuous_update=False
)

z_widget = widgets.IntSlider(
    value=nz // 2,
    min=0,
    max=nz - 1,
    step=1,
    description="Z:",
    continuous_update=False
)


def compare_images(split_name, encoding, frame, z):

    split_info = split_files[split_name]

    split_file = split_info["file"]

    # Corresponding encoding in original 8-encoding file
    original_encoding = split_info["encodes"][encoding]

    original_name = (
        f"/Images/Encode_{original_encoding:03}_Frame_{frame:03}"
    )

    split_name_h5 = (
        f"/Images/Encode_{encoding:03}_Frame_{frame:03}"
    )

    # Load only selected Z slice
    with h5py.File(original_file, "r") as original, \
         h5py.File(split_file, "r") as split:

        original_raw = original[original_name][z]
        split_raw = split[split_name_h5][z]

    # Convert structured real/imag data -> complex
    original_complex = (
        original_raw["real"] +
        1j * original_raw["imag"]
    )

    split_complex = (
        split_raw["real"] +
        1j * split_raw["imag"]
    )

    # Magnitude
    original_data = np.abs(original_complex)
    split_data = np.abs(split_complex)

    # Magnitude difference
    diff = original_data - split_data

    # Shared scale for original and split
    vmin = min(
        original_data.min(),
        split_data.min()
    )

    vmax = max(
        original_data.max(),
        split_data.max()
    )

    # Symmetric difference scale
    diff_max = np.max(np.abs(diff))

    print(
        f"Original E{original_encoding} "
        f"vs {split_name} E{encoding}"
    )

    print(f"Frame: {frame}    Z: {z}")

    print(
        f"Difference: "
        f"min={diff.min():.6g}, "
        f"max={diff.max():.6g}, "
        f"max|diff|={diff_max:.6g}, "
        f"mean|diff|={np.mean(np.abs(diff)):.6g}"
    )

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].imshow(
        original_data,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    ax[0].set_title(
        f"Original\nEncoding {original_encoding}"
    )

    ax[1].imshow(
        split_data,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    ax[1].set_title(
        f"{split_name}\nEncoding {encoding}"
    )

    im = ax[2].imshow(
        diff,
        cmap="bwr",
        vmin=-diff_max,
        vmax=diff_max
    )

    ax[2].set_title(
        f"Magnitude Difference\n"
        f"max |diff| = {diff_max:.4g}"
    )

    for a in ax:
        a.axis("off")

    fig.colorbar(
        im,
        ax=ax[2],
        fraction=0.046,
        pad=0.04
    )

    plt.show()

#%% Display widgets

interactive_plot = widgets.interactive(
    compare_images,
    split_name=split_widget,
    encoding=encoding_widget,
    frame=frame_widget,
    z=z_widget
)

display(interactive_plot)
#%% Imports

import h5py
import numpy as np


#%% Files

# original_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/all_img/Images.h5"     # 8-encoding reconstruction
# split1_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im1_retest_0527/Images.h5"      # [0, 5, 2, 7]
# split2_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im2_5274/Images.h5"      # [4, 1, 6, 3]

split_files = {
    "Split 1": split1_file,
    "Split 2": split2_file
}


#%% Compare every split encoding from Images.h5 against every original encoding

frame = 0

for split_label, split_file in split_files.items():

    print(f"\n{'=' * 65}")
    print(split_label)
    print(f"{'=' * 65}")

    with h5py.File(original_file, "r") as original, \
         h5py.File(split_file, "r") as split:

        for split_enc in range(4):

            split_name = (
                f"/Images/Encode_{split_enc:03}_Frame_{frame:03}"
            )

            split_raw = split[split_name][:]

            # Convert structured real/imag -> complex
            split_data = (
                split_raw["real"] +
                1j * split_raw["imag"]
            )

            errors = []

            for original_enc in range(8):

                original_name = (
                    f"/Images/Encode_{original_enc:03}_Frame_{frame:03}"
                )

                original_raw = original[original_name][:]

                # Convert structured real/imag -> complex
                original_data = (
                    original_raw["real"] +
                    1j * original_raw["imag"]
                )

                # Complex difference
                diff = original_data - split_data

                # Mean absolute complex difference
                error = np.mean(np.abs(diff))

                errors.append(error)

            # Find best match
            best_enc = np.argmin(errors)

            print(f"\nSplit E{split_enc}:")

            for original_enc, error in enumerate(errors):

                marker = "  <-- BEST" if original_enc == best_enc else ""

                print(
                    f"  Original E{original_enc}: "
                    f"{error:.6g}{marker}"
                )
#%%
#%% Compare every split MRI_Raw encoding against every original encoding

import h5py
import numpy as np

original_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/all_img/MRI_Raw.h5"

input_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im1_retest_skip_1463/MRI_Raw.h5"
output_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im1_retest_skip_1463/MRI_Raw_proper_arrange.h5"

split_files = {
    "Split 1": input_file,
    "Split 2": output_file,
}

num_original_enc = 8
num_split_enc = 4
num_coils = 44

for split_label, split_file in split_files.items():

    print(f"\n{'=' * 65}")
    print(split_label)
    print(f"{'=' * 65}")

    with h5py.File(original_file, "r") as original, \
         h5py.File(split_file, "r") as split:

        for split_enc in range(num_split_enc):

            errors = []

            for original_enc in range(num_original_enc):

                coil_errors = []

                for coil in range(num_coils):

                    split_name = f"Kdata/KData_E{split_enc}_C{coil}"
                    original_name = f"Kdata/KData_E{original_enc}_C{coil}"

                    split_raw = split[split_name][:]
                    original_raw = original[original_name][:]

                    # Convert structured real/imag -> complex if necessary
                    if split_raw.dtype.fields is not None:
                        split_data = (
                            split_raw["real"]
                            + 1j * split_raw["imag"]
                        )
                    else:
                        split_data = split_raw

                    if original_raw.dtype.fields is not None:
                        original_data = (
                            original_raw["real"]
                            + 1j * original_raw["imag"]
                        )
                    else:
                        original_data = original_raw

                    # Mean absolute complex difference
                    error = np.mean(
                        np.abs(original_data - split_data)
                    )

                    coil_errors.append(error)

                # Average error across all 44 coils
                errors.append(np.mean(coil_errors))

            # Find original encoding that best matches this split encoding
            best_enc = np.argmin(errors)

            print(f"\nSplit E{split_enc}:")

            for original_enc, error in enumerate(errors):

                marker = "  <-- BEST" if original_enc == best_enc else ""

                print(
                    f"  Original E{original_enc}: "
                    f"{error:.6g}{marker}"
                )

#%%