#!/usr/bin/env python3

import argparse

import numpy as np
import pyvista

PARSER = argparse.ArgumentParser()


PARSER.add_argument("-vti", "--vti", help='.vti file', type=str, default='', required=True)

ARGS = PARSER.parse_args()
FILENAME = ARGS.vti


def _load_vti_to_array(filename):
    reader = pyvista.get_reader(filename)
    ms_mesh = reader.read()
    ms = ms_mesh.get_array('Spin')
    (x, y, z) = (int(ms_mesh.bounds[1]), int(ms_mesh.bounds[3]), int(ms_mesh.bounds[5]))
    ms = ms.reshape(z, y, x).T
    return np.array(ms)


NP_ARRAY = _load_vti_to_array(FILENAME)
np.save(FILENAME[:-4] + '.npy', NP_ARRAY)
