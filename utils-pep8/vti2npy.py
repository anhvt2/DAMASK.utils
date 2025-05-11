#!/usr/bin/env python3

from vtk.util.numpy_support import vtk_to_numpy
import vtk
import pyvista
import numpy as np
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-vti", "--vti", help='.vti file',
                    type=str, default='', required=True)
args = parser.parse_args()
filename = args.vti


def load_vti_to_array(filename):
    reader = pyvista.get_reader(filename)
    ms_mesh = reader.read()
    ms = ms_mesh.get_array('Spin')
    x, y, z = int(ms_mesh.bounds[1]), int(
        ms_mesh.bounds[3]), int(ms_mesh.bounds[5])
    ms = ms.reshape(z, y, x).T
    return np.array(ms)


np_array = load_vti_to_array(filename)
np.save(filename[:-4] + '.npy', np_array)
