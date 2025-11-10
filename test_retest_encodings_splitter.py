def encodingSplitter(inputFile, outputImg1, outputImg2, encodeOrder = "interleaf"):
    
    # Importing
    import h5py
    import numpy as np
    
    # Bringing the .h5 file in, call it using a string
    with h5py.File(inputFile, 'r') as hf:
        temp_img = hf['IMAGE'][:]
        temp_mag = hf['IMAGE_MAG'][:]
        temp_phase = hf['IMAGE_PHASE'][:]
    print(f"Loaded image shape: {temp_img.shape}")
    print(f"Loaded mag shape: {temp_mag.shape}")
    print(f"Loaded phase shape: {temp_phase.shape}")
    
    # Calling in the encodings into their own images
    enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8 = [temp_img[:, i, :, :, :] for i in range(8)]
    encs = np.stack([enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8], axis=1)

    if encodeOrder == "interleaf":
        img1 = np.concatenate((encs[:, [0]], encs[:,[5]], encs[:, [2]], encs[:,[7]]), axis=1)
        img2 = np.concatenate((encs[:, [4]], encs[:,[1]], encs[:, [6]], encs[:,[3]]), axis=1)
    elif encodeOrder == "regular":
        img1 = np.concatenate((encs[:, [0]], encs[:,[1]], encs[:, [2]], encs[:,[3]]), axis=1)
        img2 = np.concatenate((encs[:, [4]], encs[:,[5]], encs[:, [6]], encs[:,[7]]), axis=1)

    print(f"Image1 shape: {img1.shape}")
    print(f"Image2 shape: {img2.shape}")

    enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8 = [temp_mag[:, i, :, :, :] for i in range(8)]
    encs = np.stack([enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8], axis=1)

    if encodeOrder == "interleaf":
        mag1 = np.concatenate((encs[:, [0]], encs[:,[5]], encs[:, [2]], encs[:,[7]]), axis=1)
        mag2 = np.concatenate((encs[:, [4]], encs[:,[1]], encs[:, [6]], encs[:,[3]]), axis=1)
    elif encodeOrder == "regular":
        mag1 = np.concatenate((encs[:, [0]], encs[:,[1]], encs[:, [2]], encs[:,[3]]), axis=1)
        mag2 = np.concatenate((encs[:, [4]], encs[:,[5]], encs[:, [6]], encs[:,[7]]), axis=1)

    print(f"Magnitude1 shape: {mag1.shape}")
    print(f"Magnitude2 shape: {mag2.shape}")

    enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8 = [temp_phase[:, i, :, :, :] for i in range(8)]
    encs = np.stack([enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8], axis=1)

    if encodeOrder == "interleaf":
        phase1 = np.concatenate((encs[:, [0]], encs[:,[5]], encs[:, [2]], encs[:,[7]]), axis=1)
        phase2 = np.concatenate((encs[:, [4]], encs[:,[1]], encs[:, [6]], encs[:,[3]]), axis=1)
    elif encodeOrder == "regular":
        phase1 = np.concatenate((encs[:, [0]], encs[:,[1]], encs[:, [2]], encs[:,[3]]), axis=1)
        phase2 = np.concatenate((encs[:, [4]], encs[:,[5]], encs[:, [6]], encs[:,[7]]), axis=1)

    print(f"Phase1 shape: {phase1.shape}")
    print(f"Phase2 shape: {phase2.shape}")
    
    # Creating new images file, again, call using a string
    with h5py.File(outputImg1, 'w') as hf:  
        hf.create_dataset('IMAGE', data=img1)
        hf.create_dataset('IMAGE_MAG', data=mag1)
        hf.create_dataset('IMAGE_PHASE', data=phase1)
        print("Datasets:", list(hf.keys()))

    with h5py.File(outputImg2, 'w') as hf:  
        hf.create_dataset('IMAGE', data=img2)
        hf.create_dataset('IMAGE_MAG', data=mag2)
        hf.create_dataset('IMAGE_PHASE', data=phase2)
        print("Datasets:", list(hf.keys()))
        
#%% Printing shape/size of each dataset to ensure it was done proper
with h5py.File('/mounts/data/analyses/bawad2/testData/Image1.h5', 'r') as hf:
    print("Datasets:", list(hf.keys()))
    ds_tot = ["IMAGE", "IMAGE_MAG", "IMAGE_PHASE"]
    img = [[] for _ in range(8)]
    for name in ds_tot:
        if name in hf:
            ds = hf[name]
            print(f"Dimensions of dataset: '{name}': {ds.shape}")
        img[0]
