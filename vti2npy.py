
import pyvista
import numpy as np
import argparse
parser = argparse.ArgumentParser()

import vtk
from vtk.util.numpy_support import vtk_to_numpy

parser.add_argument("-vti", "--vti", help='.vti file', type=str, default='', required=True)
args = parser.parse_args()
fileName = args.vti

def load_vti_to_array(fileName):
    reader = pyvista.get_reader(fileName)
    msMesh = reader.read()
    ms = msMesh.get_array('Spin')
    x, y, z = int(msMesh.bounds[1]), int(msMesh.bounds[3]), int(msMesh.bounds[5])
    ms = ms.reshape(z,y,x).T
    return np.array(ms)

np_array = load_vti_to_array(fileName)
np.save(fileName[:-4] + '.npy', np_array)
