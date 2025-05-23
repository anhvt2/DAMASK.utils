
"""
    This function 
        (1) takes in a finest mesh (high-fidelity), 
        (2) gradually coarsens the mesh,
        (3) progressively dumps finer meshes
        to study in a multi-fidelity setting.

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
        * 
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

# High-fidelity: 50um
mhf = np.load('main.npy')
mhf.shape  # (120, 24, 200) 
# mhf = mhf[:, 4:-4, :] # need to remove air buffer layer
outFileName = 'mhf-50um'
# Save high-fidelity ms
np.save(outFileName + '.npy', mhf)
subprocess.run(["python3", "../../utils-pep8/npy2geom.py", "--npy",
               f"{outFileName}.npy"], check=True)
subprocess.run(["python3", "../../utils-pep8/npy2png.py", "--threshold",
               "1", "--npy", f"{outFileName}.npy"], check=True)
subprocess.run(["geom_check", f"{outFileName}.geom"], check=True)
subprocess.run(["getCrystalTexture"], check=True) # dump material_config.csv
df = pd.read_csv('material_config.csv')

zoomFactors = np.array([0.5, 0.25])  # 0.125 with 400um too coarse
resolutions = 50 / zoomFactors

def renumerate(geom, startIndex=0, cluster=False, excludeList=None):
    ''' 
    This function renumerates so that the DEFAULT grain index starts at ZERO (0) and (gradually) increases by 1. 
    Input
    -----
        3d npy array
        startIndex: starting index, usually at 0 or 1
        cluster: 'False' or 'True'
            * True: perform clustering algorithm using DBSCAN then renumerate
            * False: only renumerate
    Output
    ------
        3d npy array
    '''
    grainIdxList = np.sort(np.unique(geom))

    # Remove grains from excludeList from being segmented, e.g. air
    if excludeList is not None:
        grainIdxList = np.setdiff1d(grainIdxList, excludeList)

    renumeratedGeom = np.copy(geom) # make a deep copy
    maxGrainId = np.max(grainIdxList)
    if cluster==True:
        # Perform clustering algorithm to decluster many grains with same grain id: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)
            X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
            clustering = DBSCAN(eps=2, min_samples=1).fit(X)
            # Relabel grainId for every pixels needed relabel: Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
            clustering.labels_ -= np.min(clustering.labels_) # re-start at 0: no noisy samples
            for j in range(clustering.labels_.shape[0]):
                renumeratedGeom[x[j],y[j],z[j]] = maxGrainId + clustering.labels_[j] + startIndex + 1
            # Print diagnostics
            logging.info(f'clustering.labels_ = {clustering.labels_}')
            logging.info(f'np.min(clustering.labels_) = {np.min(clustering.labels_)}')
            logging.info(f'np.max(clustering.labels_) = {np.max(clustering.labels_)}')
            logging.info(f'maxGrainId = {maxGrainId}')
            logging.info(f'renumerate(): Segregating grains from grainId {grainIdx} to [{maxGrainId + np.min(clustering.labels_) +startIndex + 1}, {maxGrainId + np.max(clustering.labels_) + startIndex + 1}].')
            logging.info(f'\n')
            # Update maxGrainId
            # maxGrainId = np.max(renumeratedGeom)+np.max(clustering.labels_)+startIndex+1 # debug
            # maxGrainId += np.max(clustering.labels_)+startIndex+1 # debug
            maxGrainId = np.max(renumeratedGeom) # debug
        # run vanilla renumerate()
        grainIdxList = np.sort(np.unique(renumeratedGeom))
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)    
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex + i}.')
            for j in range(len(x)):
                renumeratedGeom[x[j],y[j],z[j]] = i+startIndex
    else:
        # run vanilla renumerate() without clustering grains
        for i in range(len(grainIdxList)):
            grainIdx = grainIdxList[i]
            x, y, z = np.where(geom==grainIdx)    
            logging.info(f'renumerate(): Mapping grain id from {grainIdx} to {startIndex + i}.')
            for j in range(len(x)):
                renumeratedGeom[x[j],y[j],z[j]] = i + startIndex
    return renumeratedGeom


# # Save microstructures WITH renumerate()
for zoomFactor, resolution in zip(zoomFactors, resolutions):
    # mlf = scipy.ndimage.zoom(mhf, zoom=(zoomFactor, zoomFactor, zoomFactor), mode='constant', order=0)
    mlf = F.interpolate(torch.tensor(mhf).to(torch.float32).unsqueeze(0).unsqueeze(0), scale_factor=zoomFactor).squeeze(0).squeeze(0).numpy().astype(int)
    mlf = renumerate(mlf, startIndex=1, cluster=False, excludeList=[1]) # index start at 1: this step renumerate without clustering
    outFileName = f'mlf-{resolution.astype(int)}um' + '-renumerated.npy'
    np.save(outFileName, mlf)
    subprocess.run(["python3", "../../utils-pep8/npy2png.py", "--threshold",
                   "0", "--npy", f"{outFileName}"], check=True)



