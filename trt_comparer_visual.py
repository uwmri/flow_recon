# %% Imports

import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
# %% Files

# 8-encoding reconstruction
original_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/all_img/Images.h5"
# encodings [1, 6, 3, 8]
split1_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/espirit/pils/pythonRecon/8enc/cpp_split/im1_retest_skip_1463/Images.h5"

original_encodes = [0, 5, 2, 7]


# %% Check datasets

with h5py.File(original_file, "r") as original, \
     h5py.File(split1_file, "r") as split:

    print("Original:")
    for dataset in ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]:
        print(
            f"  {dataset}: "
            f"shape={original[dataset].shape}, "
            f"dtype={original[dataset].dtype}"
        )

    print("\nSplit 1:")
    for dataset in ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]:
        print(
            f"  {dataset}: "
            f"shape={split[dataset].shape}, "
            f"dtype={split[dataset].dtype}"
        )


# %% Determine dimensions

with h5py.File(original_file, "r") as hf:
    shape = hf["IMAGE"].shape

num_frames, num_encodings, nz, ny, nx = shape

print("\nOriginal IMAGE dimensions:")
print("Frames:   ", num_frames)
print("Encodings:", num_encodings)
print("Z slices: ", nz)
print("Y:        ", ny)
print("X:        ", nx)


# %% Widgets

dataset_widget = widgets.Dropdown(
    options=["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"],
    value="IMAGE",
    description="Dataset:"
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

time_average_widget = widgets.Checkbox(
    value=False,
    description="Time Averaged"
)


# %% Interactive comparison

def compare_images(dataset, encoding, frame, z, time_averaged):

    # Split encoding -> corresponding original encoding
    original_encoding = original_encodes[encoding]

    with h5py.File(original_file, "r") as original, \
         h5py.File(split1_file, "r") as split:

        if time_averaged:

            # Average across all time frames
            # Shape before averaging: (t, y, x)
            # Shape after averaging:  (y, x)

            original_raw = np.mean(
                original[dataset][:, original_encoding, z, :, :],
                axis=0
            )

            split_raw = np.mean(
                split[dataset][:, encoding, z, :, :],
                axis=0
            )

        else:

            # Single time frame
            original_raw = original[dataset][
                frame,
                original_encoding,
                z,
                :,
                :
            ]

            split_raw = split[dataset][
                frame,
                encoding,
                z,
                :,
                :
            ]

    # ---------------------------------------------------------
    # Convert IMAGE to magnitude
    # ---------------------------------------------------------

    if dataset == "IMAGE":

        # Structured real/imaginary dtype
        if original_raw.dtype.fields is not None:

            original_complex = (
                original_raw["real"]
                + 1j * original_raw["imag"]
            )

            split_complex = (
                split_raw["real"]
                + 1j * split_raw["imag"]
            )

            original_data = np.abs(original_complex)
            split_data = np.abs(split_complex)

        # Native complex dtype
        elif np.iscomplexobj(original_raw):

            original_data = np.abs(original_raw)
            split_data = np.abs(split_raw)

        else:

            original_data = original_raw
            split_data = split_raw

    else:

        original_data = original_raw
        split_data = split_raw


    # ---------------------------------------------------------
    # Difference
    # ---------------------------------------------------------

    diff = original_data - split_data

    vmin = min(
        original_data.min(),
        split_data.min()
    )

    vmax = max(
        original_data.max(),
        split_data.max()
    )

    diff_max = np.max(np.abs(diff))


    # ---------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------

    print(
        f"Original E{original_encoding} "
        f"vs Split 1 E{encoding}"
    )

    if time_averaged:
        print(
            f"Dataset: {dataset}    "
            f"Time Averaged    "
            f"Z: {z}"
        )
    else:
        print(
            f"Dataset: {dataset}    "
            f"Frame: {frame}    "
            f"Z: {z}"
        )

    print(
        f"Difference: "
        f"min={diff.min():.6g}, "
        f"max={diff.max():.6g}, "
        f"max|diff|={diff_max:.6g}, "
        f"mean|diff|={np.mean(np.abs(diff)):.6g}"
    )


    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    if dataset == "IMAGE_PHASE":
        image_cmap = "twilight"
    else:
        image_cmap = "gray"

    ax[0].imshow(
        original_data,
        cmap=image_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax[0].set_title(
        f"Original\nEncoding {original_encoding}"
    )

    ax[1].imshow(
        split_data,
        cmap=image_cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax[1].set_title(
        f"Split 1\nEncoding {encoding}"
    )

    if diff_max == 0:
        plot_diff_max = 1e-12
    else:
        plot_diff_max = diff_max

    im = ax[2].imshow(
        diff,
        cmap="bwr",
        vmin=-plot_diff_max,
        vmax=plot_diff_max
    )

    if time_averaged:
        diff_title = "Time-Averaged Difference"
    else:
        diff_title = "Difference"

    ax[2].set_title(
        f"{dataset} {diff_title}\n"
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

# %% Display widgets

interactive_plot = widgets.interactive(
    compare_images,
    dataset=dataset_widget,
    encoding=encoding_widget,
    frame=frame_widget,
    z=z_widget,
    time_averaged=time_average_widget
)

display(interactive_plot)
#%%

split_files = {
    "Split 1": split1_file,
    "Split 2": split2_file
}


# %% Compare every split encoding from Images.h5 against every original encoding

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
# %%
# %% Compare every split encoding against every original encoding


frame = 0

datasets = [
    "IMAGE",
    "IMAGE_MAG",
    "IMAGE_PHASE"
]

for split_label, split_file in split_files.items():

    print(f"\n{'=' * 65}")
    print(split_label)
    print(f"{'=' * 65}")

    with h5py.File(original_file, "r") as original, \
            h5py.File(split_file, "r") as split:

        for dataset_name in datasets:

            print(f"\n{'-' * 65}")
            print(dataset_name)
            print(f"{'-' * 65}")

            for split_enc in range(4):

                # (t, e, z, y, x)
                split_raw = split[dataset_name][frame, split_enc, ...]

                # Convert structured real/imag -> complex if necessary
                if split_raw.dtype.fields is not None:
                    split_data = (
                        split_raw["real"]
                        + 1j * split_raw["imag"]
                    )
                else:
                    split_data = split_raw

                errors = []

                for original_enc in range(8):

                    # (t, e, z, y, x)
                    original_raw = original[dataset_name][
                        frame, original_enc, ...
                    ]

                    # Convert structured real/imag -> complex if necessary
                    if original_raw.dtype.fields is not None:
                        original_data = (
                            original_raw["real"]
                            + 1j * original_raw["imag"]
                        )
                    else:
                        original_data = original_raw

                    # Difference
                    diff = original_data - split_data

                    # Mean absolute difference
                    error = np.mean(np.abs(diff))

                    errors.append(error)

                # Best matching original encoding
                best_enc = np.argmin(errors)

                print(f"\nSplit E{split_enc}:")

                for original_enc, error in enumerate(errors):

                    marker = (
                        "  <-- BEST"
                        if original_enc == best_enc
                        else ""
                    )

                    print(
                        f"  Original E{original_enc}: "
                        f"{error:.6g}{marker}"
                    )

# %%
# %% Imports

import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# %% Files

# Original 8-encoding reconstruction
original_file = "/all_img/Images.h5"

# Split 1 reconstruction
# Contains original encodings [0, 5, 2, 7]
split1_file = "/im1_retest_skip_1463/Images.h5"

# Mapping:
# Split E0 -> Original E0
# Split E1 -> Original E5
# Split E2 -> Original E2
# Split E3 -> Original E7
original_encodes = [0, 5, 2, 7]


# %% Check datasets

with h5py.File(original_file, "r") as original, \
     h5py.File(split1_file, "r") as split:

    print("Original:")
    for dataset in ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]:
        print(
            f"  {dataset}: "
            f"shape={original[dataset].shape}, "
            f"dtype={original[dataset].dtype}"
        )

    print("\nSplit 1:")
    for dataset in ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]:
        print(
            f"  {dataset}: "
            f"shape={split[dataset].shape}, "
            f"dtype={split[dataset].dtype}"
        )


# %% Determine dimensions

with h5py.File(original_file, "r") as hf:

    shape = hf["IMAGE"].shape

    num_frames = shape[0]
    num_encodings = shape[1]
    nz = shape[2]
    ny = shape[3]
    nx = shape[4]


print("\nOriginal IMAGE dimensions:")
print("Frames:   ", num_frames)
print("Encodings:", num_encodings)
print("Z slices: ", nz)
print("Y:        ", ny)
print("X:        ", nx)


# %% Helper function for converting IMAGE to magnitude

def convert_image_data(data):

    # Structured real/imaginary dtype
    if data.dtype.fields is not None:

        complex_data = (
            data["real"]
            + 1j * data["imag"]
        )

        return np.abs(complex_data)

    # Native complex dtype
    elif np.iscomplexobj(data):

        return np.abs(data)

    # Already real
    else:

        return data


# %% Calculate constant color scales
#
# These are calculated ONCE.
#
# Original and Split use the same image scale.
# Difference gets its own symmetric scale.
#
# The scales therefore do NOT change when changing:
#   - frame
#   - encoding
#   - Z slice


fixed_scales = {}


with h5py.File(original_file, "r") as original, \
     h5py.File(split1_file, "r") as split:

    for dataset in ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]:

        print(f"\nCalculating fixed scale for {dataset}...")


        # -----------------------------------------------------
        # Load datasets
        # -----------------------------------------------------

        original_raw = original[dataset][:]
        split_raw = split[dataset][:]


        # -----------------------------------------------------
        # Convert IMAGE to magnitude
        # -----------------------------------------------------

        if dataset == "IMAGE":

            original_data = convert_image_data(
                original_raw
            )

            split_data = convert_image_data(
                split_raw
            )

        else:

            original_data = original_raw
            split_data = split_raw


        # -----------------------------------------------------
        # Fixed image scale
        # -----------------------------------------------------

        vmin = min(
            np.min(original_data),
            np.min(split_data)
        )

        vmax = max(
            np.max(original_data),
            np.max(split_data)
        )


        # -----------------------------------------------------
        # Fixed difference scale
        #
        # Compare:
        #
        # Split E0 -> Original E0
        # Split E1 -> Original E5
        # Split E2 -> Original E2
        # Split E3 -> Original E7
        # -----------------------------------------------------

        diff_max = 0.0


        for split_enc, original_enc in enumerate(
            original_encodes
        ):

            diff = (
                original_data[:, original_enc, :, :, :]
                -
                split_data[:, split_enc, :, :, :]
            )

            current_max = np.max(
                np.abs(diff)
            )

            if current_max > diff_max:
                diff_max = current_max


        # Avoid zero-width color scale
        if diff_max == 0:
            diff_max = 1e-12


        # -----------------------------------------------------
        # Save scales
        # -----------------------------------------------------

        fixed_scales[dataset] = {

            "vmin": vmin,

            "vmax": vmax,

            "diff_max": diff_max

        }


        print(
            f"  Image scale: "
            f"{vmin:.6g} to {vmax:.6g}"
        )

        print(
            f"  Difference scale: "
            f"{-diff_max:.6g} to {diff_max:.6g}"
        )


        # Free large arrays before moving to next dataset
        del original_raw
        del split_raw
        del original_data
        del split_data


# %% Widgets


dataset_widget = widgets.Dropdown(

    options=[
        "IMAGE",
        "IMAGE_MAG",
        "IMAGE_PHASE"
    ],

    value="IMAGE",

    description="Dataset:"

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


time_average_widget = widgets.Checkbox(

    value=False,

    description="Time Averaged"

)


# %% Interactive comparison


def compare_images(
    dataset,
    encoding,
    frame,
    z,
    time_averaged
):

    # ---------------------------------------------------------
    # Determine corresponding original encoding
    # ---------------------------------------------------------

    original_encoding = original_encodes[encoding]


    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    with h5py.File(original_file, "r") as original, \
         h5py.File(split1_file, "r") as split:


        if time_averaged:

            # Load all time points for selected encoding/Z
            #
            # Shape:
            # (t, y, x)

            original_raw = original[dataset][
                :,
                original_encoding,
                z,
                :,
                :
            ]

            split_raw = split[dataset][
                :,
                encoding,
                z,
                :,
                :
            ]


            # -------------------------------------------------
            # IMAGE
            #
            # Convert each time frame to magnitude FIRST,
            # then average the magnitude over time.
            #
            # mean(|S(t)|)
            # -------------------------------------------------

            if dataset == "IMAGE":

                original_all = convert_image_data(
                    original_raw
                )

                split_all = convert_image_data(
                    split_raw
                )

                original_data = np.mean(
                    original_all,
                    axis=0
                )

                split_data = np.mean(
                    split_all,
                    axis=0
                )


            # -------------------------------------------------
            # IMAGE_MAG / IMAGE_PHASE
            # -------------------------------------------------

            else:

                original_data = np.mean(
                    original_raw,
                    axis=0
                )

                split_data = np.mean(
                    split_raw,
                    axis=0
                )


        else:

            # -------------------------------------------------
            # Single frame
            # -------------------------------------------------

            original_raw = original[dataset][
                frame,
                original_encoding,
                z,
                :,
                :
            ]

            split_raw = split[dataset][
                frame,
                encoding,
                z,
                :,
                :
            ]


            # IMAGE -> magnitude
            if dataset == "IMAGE":

                original_data = convert_image_data(
                    original_raw
                )

                split_data = convert_image_data(
                    split_raw
                )

            else:

                original_data = original_raw
                split_data = split_raw


    # ---------------------------------------------------------
    # Difference
    # ---------------------------------------------------------

    diff = (
        original_data
        -
        split_data
    )


    # ---------------------------------------------------------
    # Current difference statistics
    # ---------------------------------------------------------

    current_diff_max = np.max(
        np.abs(diff)
    )

    current_diff_mean = np.mean(
        np.abs(diff)
    )


    # ---------------------------------------------------------
    # Retrieve FIXED color scales
    # ---------------------------------------------------------

    vmin = fixed_scales[dataset]["vmin"]

    vmax = fixed_scales[dataset]["vmax"]

    diff_max = fixed_scales[dataset]["diff_max"]


    # ---------------------------------------------------------
    # Print information
    # ---------------------------------------------------------

    print(
        f"Original E{original_encoding} "
        f"vs Split 1 E{encoding}"
    )


    if time_averaged:

        print(
            f"Dataset: {dataset}    "
            f"Time Averaged    "
            f"Z: {z}"
        )

    else:

        print(
            f"Dataset: {dataset}    "
            f"Frame: {frame}    "
            f"Z: {z}"
        )


    print(
        f"Difference: "
        f"min={diff.min():.6g}, "
        f"max={diff.max():.6g}, "
        f"max|diff|={current_diff_max:.6g}, "
        f"mean|diff|={current_diff_mean:.6g}"
    )


    print(
        f"Fixed image scale: "
        f"{vmin:.6g} to {vmax:.6g}"
    )


    print(
        f"Fixed difference scale: "
        f"{-diff_max:.6g} to {diff_max:.6g}"
    )


    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )


    # ---------------------------------------------------------
    # Colormap
    # ---------------------------------------------------------

    if dataset == "IMAGE_PHASE":

        image_cmap = "twilight"

    else:

        image_cmap = "gray"


    # ---------------------------------------------------------
    # Original
    # ---------------------------------------------------------

    im0 = ax[0].imshow(

        original_data,

        cmap=image_cmap,

        vmin=vmin,

        vmax=vmax

    )


    if time_averaged:

        ax[0].set_title(
            f"Original\n"
            f"Encoding {original_encoding} "
            f"(Time Averaged)"
        )

    else:

        ax[0].set_title(
            f"Original\n"
            f"Encoding {original_encoding}, "
            f"Frame {frame}"
        )


    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    im1 = ax[1].imshow(

        split_data,

        cmap=image_cmap,

        vmin=vmin,

        vmax=vmax

    )


    if time_averaged:

        ax[1].set_title(
            f"Split 1\n"
            f"Encoding {encoding} "
            f"(Time Averaged)"
        )

    else:

        ax[1].set_title(
            f"Split 1\n"
            f"Encoding {encoding}, "
            f"Frame {frame}"
        )


    # ---------------------------------------------------------
    # Difference
    # ---------------------------------------------------------

    im2 = ax[2].imshow(

        diff,

        cmap="bwr",

        vmin=-diff_max,

        vmax=diff_max

    )


    if time_averaged:

        ax[2].set_title(
            f"Time-Averaged Difference\n"
            f"Current max |diff| = "
            f"{current_diff_max:.4g}"
        )

    else:

        ax[2].set_title(
            f"Difference\n"
            f"Current max |diff| = "
            f"{current_diff_max:.4g}"
        )


    # ---------------------------------------------------------
    # Remove axes
    # ---------------------------------------------------------

    for a in ax:

        a.axis("off")


    # ---------------------------------------------------------
    # Colorbars
    # ---------------------------------------------------------

    # Same colorbar scale for Original and Split
    fig.colorbar(

        im1,

        ax=ax[:2],

        fraction=0.025,

        pad=0.02,

        label=dataset

    )


    # Separate fixed colorbar for difference
    fig.colorbar(

        im2,

        ax=ax[2],

        fraction=0.046,

        pad=0.04,

        label="Difference"

    )


    plt.show()


# %% Display widgets


interactive_plot = widgets.interactive(

    compare_images,

    dataset=dataset_widget,

    encoding=encoding_widget,

    frame=frame_widget,

    z=z_widget,

    time_averaged=time_average_widget

)


display(interactive_plot)