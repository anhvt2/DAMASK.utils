import scipy
import numpy as np
import os, sys
import subprocess

# High-fidelity: 50um
mhf = np.load('mhf-50um.npy')
zoomFactors  = np.array([0.5, 0.25]) # 0.125 with 400um too coarse
resolutions  = 50 / zoomFactors

for zoomFactor, resolution in zip(zoomFactors, resolutions):
    mlf = scipy.ndimage.zoom(mhf, zoom=(zoomFactor, zoomFactor, zoomFactor), mode='constant', order=0)
    outFileName = f'mlf-{resolution.astype(int)}um'
    np.save(outFileName + '.npy', mlf)
    # os.system(f'python3 ../../npy2geom.py --npy {outFileName}.npy')
    # os.system(f'python3 ../../npy2png.py --threshold 1 --npy {outFileName}.npy')
    # os.system(f'geom_check {outFileName}.geom')
    subprocess.run(["python3", "../../npy2geom.py", "--npy", f"{outFileName}.npy"], check=True)
    subprocess.run(["python3", "../../npy2png.py", "--threshold", "1", "--npy", f"{outFileName}.npy"], check=True)
    subprocess.run(["geom_check", f"{outFileName}.geom"], check=True)
