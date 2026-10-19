# %% Imports

import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
# Files

# 8-encoding reconstruction
original_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/interleaved_python/20_tf_llr_1e-5/Images.h5"
# encodings [1, 6, 3, 8]
split1_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/interleaved_python/20_tf_llr_1e-5/Images1.h5"
split2_file = "/home/bxa033/Data/CVMRIGroup/Users/bxa033/trtstudyvol2/interleaved_python/20_tf_llr_1e-5/Images2.h5"

#%%

original_encodes = [0, 5, 2, 7]


# Check datasets

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


# Determine dimensions

with h5py.File(original_file, "r") as hf:
    shape = hf["IMAGE"].shape

num_frames, num_encodings, nz, ny, nx = shape

print("\nOriginal IMAGE dimensions:")
print("Frames:   ", num_frames)
print("Encodings:", num_encodings)
print("Z slices: ", nz)
print("Y:        ", ny)
print("X:        ", nx)


# Widgets

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


# Interactive comparison

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

    if dataset == "IMAGE_PHASE":
        diff = np.angle(np.exp(1j * (original_data - split_data)))
    else:
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

# Display widgets

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
# %%

# %%
import h5py
import numpy as np
from itertools import permutations
from pathlib import Path
from time import perf_counter


# ============================================================
# Configuration
# ============================================================

split_files = {
    "Split 1": split1_file,
    "Split 2": split2_file,
}

# Expected mapping: split encoding -> original encoding
expected_encodes = {
    "Split 1": [0, 5, 2, 7],
    "Split 2": [4, 1, 6, 3],
}

dataset_name = "IMAGE"

# Stage 1: fast identification
identification_frames = [0, 5, 10, 15]
spatial_step = 4

# Stage 2: full-resolution verification
# None means all frames.
verification_frames = None

# Number of z slices to process per block
block_size = 8

# Phase is undefined where either complex signal is zero.
# Set a positive value to exclude very-low-signal voxels.
phase_magnitude_threshold = 0.0

# Print progress during verification
print_progress = True


# ============================================================
# Convert HDF5 data to complex64
# ============================================================

def to_complex(data):
    """Convert structured real/imag data to complex64."""

    if data.dtype.fields is not None:
        fields = data.dtype.fields

        if "real" in fields and "imag" in fields:
            result = np.empty(data.shape, dtype=np.complex64)
            result.real = data["real"]
            result.imag = data["imag"]
            return result

        raise ValueError(
            f"Unsupported structured dtype: {data.dtype}"
        )

    if np.iscomplexobj(data):
        return data.astype(np.complex64, copy=False)

    return data.astype(np.float32, copy=False)


# ============================================================
# Dataset validation
# ============================================================

def validate_datasets(original_ds, split_ds):

    if original_ds.ndim != 5 or split_ds.ndim != 5:
        raise ValueError(
            "Expected 5D datasets: "
            "(frames, encodings, z, y, x)"
        )

    nf, ne, nz, ny, nx = original_ds.shape
    sf, se, sz, sy, sx = split_ds.shape

    if nf != sf:
        raise ValueError("Frame counts do not match")

    if (nz, ny, nx) != (sz, sy, sx):
        raise ValueError("Spatial dimensions do not match")

    if ne != 8:
        raise ValueError("Expected 8 original encodings")

    if se != 4:
        raise ValueError("Expected 4 split encodings")

    return nf, ne, se, nz, ny, nx


def select_frames(requested, total):

    if requested is None:
        return list(range(total))

    frames = list(requested)

    if not frames or any(
        f < 0 or f >= total for f in frames
    ):
        raise ValueError("Invalid frame selection")

    return frames


# ============================================================
# Stage 1: identify encoding correspondence
# ============================================================

def identify_encodings(
    original_ds,
    split_ds,
    frames,
    spatial_step,
):
    """
    Compare spatially subsampled volumes using complex error.

    Each original and split volume is read once per frame.
    Only complex error is calculated.

    Returns:
        error_matrix: shape (4, 8)
        mapping: unique original encoding for each split encoding
    """

    num_split = split_ds.shape[1]
    num_original = original_ds.shape[1]

    error_sums = np.zeros(
        (num_split, num_original), dtype=np.float64
    )

    voxel_count = 0

    spatial_slice = (
        slice(None, None, spatial_step),
        slice(None, None, spatial_step),
        slice(None, None, spatial_step),
    )

    print("\nSTAGE 1: IDENTIFY ENCODINGS")
    print(f"Frames: {frames}")
    print(f"Spatial step: {spatial_step}")

    start = perf_counter()

    for frame in frames:

        # Load each subsampled volume only once.
        originals = [
            to_complex(
                original_ds[(frame, enc) + spatial_slice]
            )
            for enc in range(num_original)
        ]

        splits = [
            to_complex(
                split_ds[(frame, enc) + spatial_slice]
            )
            for enc in range(num_split)
        ]

        voxel_count += originals[0].size

        for split_enc, split_data in enumerate(splits):
            for original_enc, original_data in enumerate(originals):

                diff = original_data - split_data

                error_sums[split_enc, original_enc] += (
                    np.abs(diff).sum(dtype=np.float64)
                )

        print(f"  Frame {frame + 1} complete")

    error_matrix = error_sums / voxel_count

    # Find the best one-to-one mapping.
    # Four split encodings must map to four distinct originals.
    best_total = np.inf
    best_mapping = None

    for candidate in permutations(
        range(num_original), num_split
    ):
        total = sum(
            error_matrix[i, candidate[i]]
            for i in range(num_split)
        )

        if total < best_total:
            best_total = total
            best_mapping = list(candidate)

    elapsed = perf_counter() - start

    print(f"\nIdentification time: {elapsed:.2f} s")

    print("\nComplex mean absolute error:")
    header = "Split       " + "".join(
        f"E{i:<12}" for i in range(num_original)
    )
    print(header)

    for split_enc in range(num_split):
        values = "".join(
            f"{v:<13.5g}"
            for v in error_matrix[split_enc]
        )
        print(f"E{split_enc:<10}{values}")

    print("\nIdentified mapping:")

    for split_enc, original_enc in enumerate(best_mapping):

        sorted_errors = np.sort(error_matrix[split_enc])
        best_individual = int(
            np.argmin(error_matrix[split_enc])
        )

        ratio = (
            sorted_errors[1] / sorted_errors[0]
            if sorted_errors[0] > 0
            else (
                np.inf if sorted_errors[1] > 0
                else 1.0
            )
        )

        print(
            f"  Split E{split_enc} -> Original E{original_enc}"
            f" | Individual best: E{best_individual}"
            f" | Best/second ratio: {ratio:.5g}"
        )

    return error_matrix, best_mapping


# ============================================================
# Stage 2: full-resolution verification
# ============================================================

def verify_mapping(
    original_ds,
    split_ds,
    mapping,
    frames,
    block_size,
    phase_threshold=0.0,
):
    """
    Verify only matched encoding pairs.

    Processes one spatial block at a time.

    Calculates:
        Mean complex absolute error
        Mean magnitude absolute error
        Mean wrapped phase absolute error
        Normalized complex error
    """

    nz = original_ds.shape[2]

    num_split = len(mapping)

    complex_sums = np.zeros(num_split, dtype=np.float64)
    magnitude_sums = np.zeros(num_split, dtype=np.float64)
    phase_sums = np.zeros(num_split, dtype=np.float64)
    original_mag_sums = np.zeros(num_split, dtype=np.float64)

    voxel_counts = np.zeros(num_split, dtype=np.int64)
    phase_counts = np.zeros(num_split, dtype=np.int64)

    print("\nSTAGE 2: FULL-RESOLUTION VERIFICATION")
    print(f"Frames: {len(frames)}")
    print(f"Block size: {block_size} slices")

    start = perf_counter()

    for frame in frames:

        for split_enc, original_enc in enumerate(mapping):

            for z_start in range(0, nz, block_size):

                z_end = min(z_start + block_size, nz)

                original_data = to_complex(
                    original_ds[
                        frame,
                        original_enc,
                        z_start:z_end,
                        :,
                        :
                    ]
                )

                split_data = to_complex(
                    split_ds[
                        frame,
                        split_enc,
                        z_start:z_end,
                        :,
                        :
                    ]
                )

                # Complex error
                complex_diff = original_data - split_data

                complex_sums[split_enc] += np.abs(
                    complex_diff
                ).sum(dtype=np.float64)

                # Magnitude error
                original_mag = np.abs(original_data)
                split_mag = np.abs(split_data)

                magnitude_sums[split_enc] += np.abs(
                    original_mag - split_mag
                ).sum(dtype=np.float64)

                original_mag_sums[split_enc] += (
                    original_mag.sum(dtype=np.float64)
                )

                # Wrapped phase difference:
                # angle(a * conj(b)) is equivalent to
                # wrapping angle(a) - angle(b).
                phase_product = (
                    original_data * np.conj(split_data)
                )

                valid_phase = (
                    (original_mag > phase_threshold)
                    & (split_mag > phase_threshold)
                )

                if np.any(valid_phase):
                    phase_diff = np.angle(
                        phase_product[valid_phase]
                    )

                    phase_sums[split_enc] += np.abs(
                        phase_diff
                    ).sum(dtype=np.float64)

                    phase_counts[split_enc] += (
                        phase_diff.size
                    )

                voxel_counts[split_enc] += (
                    original_data.size
                )

        if print_progress:
            print(
                f"  Verified frame {frame + 1} "
                f"({frames.index(frame) + 1}/{len(frames)})"
            )

    results = []

    for split_enc, original_enc in enumerate(mapping):

        count = voxel_counts[split_enc]
        phase_count = phase_counts[split_enc]

        mean_complex = complex_sums[split_enc] / count
        mean_magnitude = magnitude_sums[split_enc] / count

        mean_phase = (
            phase_sums[split_enc] / phase_count
            if phase_count > 0
            else np.nan
        )

        mean_original_mag = (
            original_mag_sums[split_enc] / count
        )

        normalized = (
            mean_complex / mean_original_mag
            if mean_original_mag > 0
            else (
                0.0 if mean_complex == 0
                else np.inf
            )
        )

        results.append({
            "split_encoding": split_enc,
            "original_encoding": original_enc,
            "complex_error": mean_complex,
            "magnitude_error": mean_magnitude,
            "phase_error": mean_phase,
            "normalized_error": normalized,
            "phase_voxels": phase_count,
        })

    elapsed = perf_counter() - start

    print(f"\nVerification time: {elapsed:.2f} s")

    print("\nFull-resolution results:")
    print(
        f"{'Split':<8}"
        f"{'Original':<10}"
        f"{'Complex':<14}"
        f"{'Magnitude':<14}"
        f"{'Phase (rad)':<14}"
        f"{'Normalized':<14}"
    )

    for result in results:
        print(
            f"E{result['split_encoding']:<7}"
            f"E{result['original_encoding']:<9}"
            f"{result['complex_error']:<14.6g}"
            f"{result['magnitude_error']:<14.6g}"
            f"{result['phase_error']:<14.6g}"
            f"{result['normalized_error']:<14.6g}"
        )

    return results


# ============================================================
# Main comparison
# ============================================================

def compare_encodings(
    original_file,
    split_file,
    split_label,
):

    print("\n" + "=" * 80)
    print(split_label)
    print("=" * 80)

    start = perf_counter()

    with (
        h5py.File(original_file, "r") as original,
        h5py.File(split_file, "r") as split
    ):

        original_ds = original[dataset_name]
        split_ds = split[dataset_name]

        (
            num_frames,
            num_original,
            num_split,
            nz,
            ny,
            nx,
        ) = validate_datasets(original_ds, split_ds)

        print(f"Original shape: {original_ds.shape}")
        print(f"Split shape:    {split_ds.shape}")
        print(f"Original dtype: {original_ds.dtype}")
        print(f"Split dtype:    {split_ds.dtype}")

        id_frames = select_frames(
            identification_frames,
            num_frames,
        )

        verify_frames = select_frames(
            verification_frames,
            num_frames,
        )

        # Stage 1
        error_matrix, mapping = identify_encodings(
            original_ds,
            split_ds,
            id_frames,
            spatial_step,
        )

        # Check expected mapping
        expected = expected_encodes[split_label]

        print("\nExpected mapping validation:")

        for split_enc, original_enc in enumerate(mapping):
            expected_enc = expected[split_enc]

            status = (
                "MATCH"
                if original_enc == expected_enc
                else "MISMATCH"
            )

            print(
                f"  Split E{split_enc}: "
                f"Found E{original_enc}, "
                f"Expected E{expected_enc} -> {status}"
            )

        # Stage 2
        results = verify_mapping(
            original_ds,
            split_ds,
            mapping,
            verify_frames,
            block_size,
            phase_magnitude_threshold,
        )

    elapsed = perf_counter() - start

    print(f"\nTotal time for {split_label}: {elapsed:.2f} s")

    return {
        "mapping": mapping,
        "error_matrix": error_matrix,
        "verification": results,
        "elapsed_seconds": elapsed,
    }


# ============================================================
# Run comparisons
# ============================================================

all_results = {}

total_start = perf_counter()

for split_label, split_file in split_files.items():

    all_results[split_label] = compare_encodings(
        original_file,
        split_file,
        split_label,
    )

total_elapsed = perf_counter() - total_start

print("\n" + "=" * 80)
print("ALL COMPARISONS COMPLETE")
print("=" * 80)
print(f"Total elapsed time: {total_elapsed:.2f} s")

# %%
