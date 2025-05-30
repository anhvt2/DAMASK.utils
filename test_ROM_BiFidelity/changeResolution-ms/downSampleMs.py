
"""
    This function 
        (1) takes in a finest mesh (high-fidelity), 
        (2) gradually coarsens the mesh,
        (3) progressively dumps finer meshes
        to study in a multi-fidelity setting.

    Adopt from
        - getCrystalTexture.py # dump material.config to pd.DataFrame
        - df2MatlConfig.py # write material.config from pd.DataFrame

    Options for down-sampling mesh:
        * NOT RECOMMEND: This function artificially introduces a 1-pixel air on the side of dogbone
        ```python
        from scipy.ndimage import zoom
        # Downscale by 50% in all dimensions
        downscaled = zoom(array, zoom=(0.5, 0.5, 0.5), order=1)  # order=1 = linear interpolation
        ```
        * 
        ```python
        from skimage.transform import resize
        # Resize to new shape (e.g., half size)
        rescaled = resize(array, output_shape=(new_x, new_y, new_z), mode='reflect', anti_aliasing=True)
        ```
        * 
        ```python
        from skimage.measure import block_reduce
        # Downscale by 2 in each dimension using mean
        rescaled = block_reduce(array, block_size=(2, 2, 2), func=np.mean)
        ```
        * PRODUCTION
        ```python
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        import torch
        import torch.nn.functional as F
        rescaled = F.interpolate(tensor, scale_factor=float, mode='nearest', align_corners=None)
        ```

"""

import scipy
import skimage
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import subprocess
import numpy as np
import re
import pandas as pd
import time
import logging

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'downSampleMs.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)

# Run getCrystalTexture.py to get crystal info at the high-fidelity level and start down-sampling
# python3 getCrystalTexture.py
df = pd.read_csv('material_config.csv') # .drop(columns=['grain_id'])
# High-fidelity: 50um
mhf = np.load('main.npy')
mhf.shape  # (120, 24, 200) 
# mhf = mhf[:, 4:-4, :] # need to remove air buffer layer
outFileName = 'mhf-50um'
# Save high-fidelity ms
np.save(outFileName + '.npy', mhf)
subprocess.run(["python3", "../../utils-pep8/npy2geom.py", "--npy",
               f"{outFileName}.npy"], check=True)
subprocess.run(["python3", "../../utils-pep8/npy2png.py", "--threshold", "1",
               "--npy", f"{outFileName}.npy"], check=True)
subprocess.run(["geom_check", f"{outFileName}.geom"], check=True)
subprocess.run(["getCrystalTexture"], check=True) # dump material_config.csv


zoomFactors = np.array([0.5, 0.25])  # 0.125 with 400um too coarse
resolutions = 50 / zoomFactors

def renumerate(geom, startIndex=0, isCluster=False, excludeList=None):
    ''' 
    Renumerates grain IDs starting from `startIndex`. Can cluster grains before renumerating.
    
    Inputs:
        geom : 3D NumPy array of grain IDs
        startIndex : integer to start numbering from
        cluster : if True, perform DBSCAN clustering
        excludeList : list of grain IDs to ignore (e.g. background)

    Outputs:
        renumeratedGeom : 3D NumPy array
        grain_map : list of tuples (old_id, new_id)
    '''
    grainIdxList = np.sort(np.unique(geom))

    if excludeList is not None:
        grainIdxList = np.setdiff1d(grainIdxList, excludeList)

    renumeratedGeom = np.copy(geom)
    grain_map = []

    maxGrainId = np.max(grainIdxList)

    if isCluster: # isCluster == True
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom == grainIdx)
            X = np.column_stack((x, y, z))

            clustering = DBSCAN(eps=2, min_samples=1).fit(X)
            clustering.labels_ -= np.min(clustering.labels_)  # Ensure labels start at 0

            for j in range(clustering.labels_.shape[0]):
                new_id = maxGrainId + clustering.labels_[j] + startIndex + 1
                renumeratedGeom[x[j], y[j], z[j]] = new_id

            logging.info(f'renumerate(): Segregating grains from grainId {grainIdx} to [{maxGrainId + np.min(clustering.labels_) +startIndex + 1}, {maxGrainId + np.max(clustering.labels_) + startIndex + 1}].')
            maxGrainId = np.max(renumeratedGeom)

        # After clustering, renumerate again
        grainIdxList = np.sort(np.unique(renumeratedGeom))
        for i, grainIdx in enumerate(grainIdxList):
            x, y, z = np.where(renumeratedGeom == grainIdx)
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex + i}.')
            renumeratedGeom[x, y, z] = startIndex + i
            grain_map.append((int(grainIdx), int(startIndex + i)))

    else: # isCluster == False
        for i, grainIdx in enumerate(grainIdxList):
            x, y, z = np.where(geom == grainIdx)
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex + i}.')
            renumeratedGeom[x, y, z] = startIndex + i
            grain_map.append((int(grainIdx), int(startIndex + i)))

    return renumeratedGeom, grain_map


def write_material_config(df, filename='material.config'):
    """
    Write a material.config file from a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing columns:
        ['grain_id', 'phi1', 'Phi', 'phi2', 'crystallite', 'phase', 'texture', 'fraction']

    filename : str
        The output filename (default: 'material.config')
    """
    with open(filename, 'w') as f:
        # Write <texture> section
        f.write("<texture>\n")
        for _, row in df.iterrows():
            f.write(f"[grain{int(row['grain_id'])}]\n")
            f.write(f"(gauss) phi1 {row['phi1']:.3f}   Phi {row['Phi']:.3f}    phi2 {row['phi2']:.3f}   scatter 0.0   fraction 1.0\n")
        f.write("\n")

        # Write <microstructure> section
        f.write("<microstructure>\n")
        for _, row in df.iterrows():
            f.write(f"[grain{int(row['grain_id'])}]\n")
            f.write(f"crystallite {int(row['crystallite'])}\n")
            f.write(f"(constituent)   phase {int(row['phase'])} texture {int(row['texture'])} fraction {row['fraction']:.6f}\n")
        f.write("\n")


# # Save microstructures WITH renumerate()
for zoomFactor, resolution in zip(zoomFactors, resolutions):
    # Down-sample microstructure
    # mlf = scipy.ndimage.zoom(mhf, zoom=(zoomFactor, zoomFactor, zoomFactor), mode='constant', order=0)
    mlf = F.interpolate(torch.tensor(mhf).to(torch.float32).unsqueeze(0).unsqueeze(0), scale_factor=zoomFactor).squeeze(0).squeeze(0).numpy().astype(int) # need to be 5d tensor - (batch, channel, x, y, z)
    # Renumerate microstructure geometry
    mlf, grain_map = renumerate(mlf, startIndex=1, isCluster=False, excludeList=[1]) # index start at 1: this step renumerate without clustering
    # Convert to DataFrame
    grain_map_df = pd.DataFrame(grain_map, columns=["old_id", "new_id"])
    # Filter df to keep only grain_ids in `new_id` column of grain_map
    filtered_df = df[df["grain_id"].isin(grain_map_df["old_id"])]
    # Merge to assign new grain_id
    merged_df = pd.merge(filtered_df, grain_map_df, left_on="grain_id", right_on="old_id")
    # Replace grain_id with new_id
    merged_df["grain_id"] = merged_df["new_id"]
    # Drop the mapping helper columns
    mlf_df = merged_df.drop(columns=["old_id", "new_id"])
    # Optionally sort by new grain_id
    mlf_df = mlf_df.sort_values(by="grain_id").reset_index(drop=True)
    # Save outputs
    outFileName = f'mlf-{resolution.astype(int)}um' + '-renumerated.npy'
    outCsvName = f'mlf-{resolution.astype(int)}um' + '-renumerated-df.csv'
    np.save(outFileName, mlf)
    # mlf_df.to_csv(outCsvName, index=False)
    write_material_config(df, filename=f'mlf-{resolution.astype(int)}um' + '-renumerated.config')
    # Dump 3d images
    subprocess.run(["python3", "../../utils-pep8/npy2png.py", "--threshold", "1",
                   "--npy", f"{outFileName}"], check=True)



