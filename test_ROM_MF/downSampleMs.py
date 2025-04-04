import scipy
import numpy as np
import os, sys
import subprocess

# High-fidelity: 50um
mhf = np.load('mhf-50um.npy')
mhf.shape # debug (120, 24, 200) # need to remove air buffer layer
mhf = mhf[:,2:-2, :]
outFileName = 'mhf-50um'
# Save high-fidelity ms
np.save(outFileName + '.npy', mhf)
subprocess.run(["python3", "../../npy2geom.py", "--npy", f"{outFileName}.npy"], check=True)
subprocess.run(["python3", "../../npy2png.py", "--threshold", "1", "--npy", f"{outFileName}.npy"], check=True)
subprocess.run(["geom_check", f"{outFileName}.geom"], check=True)

zoomFactors  = np.array([0.5, 0.25]) # 0.125 with 400um too coarse
resolutions  = 50 / zoomFactors

for zoomFactor, resolution in zip(zoomFactors, resolutions):
    mlf = scipy.ndimage.zoom(mhf, zoom=(zoomFactor, zoomFactor, zoomFactor), mode='constant', order=0)
    outFileName = f'mlf-{resolution.astype(int)}um'
    np.save(outFileName + '.npy', mlf)
    subprocess.run(["python3", "../../npy2geom.py", "--npy", f"{outFileName}.npy"], check=True)
    subprocess.run(["python3", "../../npy2png.py", "--threshold", "1", "--npy", f"{outFileName}.npy"], check=True)
    subprocess.run(["geom_check", f"{outFileName}.geom"], check=True)
