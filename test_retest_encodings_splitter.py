#%%
from pathlib import Path
import argparse
import h5py
import numpy as np
import re

def encSplitter(inputFile, outputImg1, outputImg2, encodeOrder = "interleaf"):

    # Split from MRI_Raw.h5
    def mri_raw_enc_splitter(inputFile, outputImg1, outputImg2, encodeOrder = "interleaf"):

        # Which encoding type to use?
        if encodeOrder == "interleaf":
            output_sets = {
                outputImg1: [0, 5, 2, 7],
                outputImg2: [4, 1, 6, 3]
            }

        elif encodeOrder == "regular":
            output_sets = {
                outputImg1: [0, 1, 2, 3],
                outputImg2: [4, 5, 6, 7]
            }

        # Fallback issue just in case
        else:
            raise ValueError(
                f"Unknown encodeOrder: {encodeOrder}. "
                "Use 'interleaf' or 'regular'."
            )

        groups_to_split = ["Gating", "Kdata"]
        with h5py.File(inputFile, "r") as hf:
            for output_name, encodes in output_sets.items():

                # Create a map of encodings
                encode_map = {
                    old_encode: new_encode
                    for new_encode, old_encode in enumerate(encodes)
                }

                # List out what was found
                print(f"Encoding map: {encode_map}")
                print(f"\nCreating: {output_name}")
                print(f"Encodings: {encodes}")

                with h5py.File(output_name, "w") as out_hf:
                    for group_name in groups_to_split:
                        if group_name not in hf:
                            print(f"/{group_name} not found")
                            continue

                        input_group = hf[group_name]
                        output_group = out_hf.create_group(group_name)

                        for key in input_group.keys():
                            # Find _E# anywhere in the name to split as encoding
                            match = re.search(r"_E(\d+)(?:_|$)", key)

                            if match is None:
                                print(f"SKIPPING: /{group_name}/{key} -- no encoding found")
                                continue

                            encode_num = int(match.group(1))
                            if encode_num in encode_map:
                                new_encode = encode_map[encode_num]

                                new_key = re.sub(r"_E\d+",
                                                f"_E{new_encode}",
                                                key,
                                                count=1
                                                )

                                print(
                                    f"/{group_name}/{key} -> /{group_name}/{new_key}"
                                )

                                input_group.copy(
                                    key,
                                    output_group,
                                    name=new_key
                                ) 


                    # Writing in headers for the Kdata dataset from the original MRI_Raw file
                    if 'Kdata' in hf and 'Kdata' in out_hf:
                        src_kdata = hf['Kdata']
                        dst_kdata = out_hf['Kdata']

                        for key, value in src_kdata.attrs.items():
                            dst_kdata.attrs[key] = value

                        dst_kdata.attrs['Num_Encodings'] = 4

    # def reorder_encodings(filename, proper_order):

    #     with h5py.File(filename, "r+") as hf:

    #         for group_name in ["Kdata", "Gating"]:

    #             if group_name not in hf:
    #                 continue

    #             group = hf[group_name]

    #             # First move ALL encoding datasets to temporary names
    #             for old_enc in range(4):

    #                 names = [
    #                     name for name in list(group.keys())
    #                     if f"_E{old_enc}" in name
    #                 ]

    #                 for name in names:
    #                     temp_name = name.replace(
    #                         f"_E{old_enc}",
    #                         f"_ETEMP{old_enc}"
    #                     )

    #                     group.move(name, temp_name)

    #             # Now rename temporary datasets into desired encoding positions
    #             for new_enc, old_enc in enumerate(proper_order):

    #                 names = [
    #                     name for name in list(group.keys())
    #                     if f"_ETEMP{old_enc}" in name
    #                 ]

    #                 for name in names:
    #                     new_name = name.replace(
    #                         f"_ETEMP{old_enc}",
    #                         f"_E{new_enc}"
    #                     )

    #                     group.move(name, new_name)

    #                     print(
    #                         f"/{group_name}/{name} -> "
    #                         f"/{group_name}/{new_name}"
    #                     )

    # Split from Images.h5
    def images_encoding_splitter(inputFile, outputImg1, outputImg2, encodeOrder="interleaf"):

        with h5py.File(inputFile, 'r') as hf:
            temp_img = hf['IMAGE'][:]
            temp_mag = hf['IMAGE_MAG'][:]
            temp_phase = hf['IMAGE_PHASE'][:]

        print(f"Loaded image shape: {temp_img.shape}")
        print(f"Loaded mag shape: {temp_mag.shape}")
        print(f"Loaded phase shape: {temp_phase.shape}")

        if encodeOrder == "interleaf":
            order1 = [0, 5, 2, 7]
            order2 = [4, 1, 6, 3]

        elif encodeOrder == "regular":
            order1 = [0, 1, 2, 3]
            order2 = [4, 5, 6, 7]

        else:
            raise ValueError(
                f"Unknown encodeOrder: {encodeOrder}. "
                "Use 'interleaf' or 'regular'."
            )

        # Split/reorder encodings
        img1 = temp_img[:, order1, ...]
        img2 = temp_img[:, order2, ...]

        mag1 = temp_mag[:, order1, ...]
        mag2 = temp_mag[:, order2, ...]

        phase1 = temp_phase[:, order1, ...]
        phase2 = temp_phase[:, order2, ...]

        print(f"Image1 shape: {img1.shape}")
        print(f"Image2 shape: {img2.shape}")
        print(f"Magnitude1 shape: {mag1.shape}")
        print(f"Magnitude2 shape: {mag2.shape}")
        print(f"Phase1 shape: {phase1.shape}")
        print(f"Phase2 shape: {phase2.shape}")

        # Write Image 1
        with h5py.File(outputImg1, 'w') as hf:
            hf.create_dataset('IMAGE', data=img1)
            hf.create_dataset('IMAGE_MAG', data=mag1)
            hf.create_dataset('IMAGE_PHASE', data=phase1)

            print(f"Created: {outputImg1}")
            print("Datasets:", list(hf.keys()))

        # Write Image 2
        with h5py.File(outputImg2, 'w') as hf:
            hf.create_dataset('IMAGE', data=img2)
            hf.create_dataset('IMAGE_MAG', data=mag2)
            hf.create_dataset('IMAGE_PHASE', data=phase2)

            print(f"Created: {outputImg2}")
            print("Datasets:", list(hf.keys()))

    input_lower = str(inputFile).lower()

    if "mri_raw" in input_lower:
        mri_raw_enc_splitter(inputFile, outputImg1, outputImg2, encodeOrder)
        # reorder_encodings(outputImg1, [0, 2, 1, 3])
        # reorder_encodings(outputImg2, [2, 0, 3, 1])
    elif "images" in input_lower:
        images_encoding_splitter(inputFile, outputImg1, outputImg2, encodeOrder)
    else:
        raise ValueError(f"Could not determine input file type from: {inputFile}. Expected filename to contain either 'mri_raw' or 'images' ")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split an 8-encoding HDF5 image into two 4-encoding images.")
    parser.add_argument("-f", "--input", required=True, 
                        type=str, help="Input Images.h5 file")
    parser.add_argument("--image1", required=True, 
                        type=str, help="Output name for Image 1")
    parser.add_argument("--image2", required=True, 
                        type=str, help="Output name for Image 2")
    parser.add_argument("--encode-order", choices=["interleaf", "regular"], 
                        default="interleaf", help="Encoding order (default: interleaf)")

    args = parser.parse_args()

    encSplitter(inputFile = args.input, outputImg1=args.image1, outputImg2=args.image2, encodeOrder='interleaf')
#%%
