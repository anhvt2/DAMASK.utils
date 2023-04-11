#!/usr/bin/env python3

import numpy as np

import damask

r = damask.Result('star_tensionX.hdf5')

r.add_strain()
r.add_equivalent_Mises('epsilon_V^0.0(F)')
r.export_VTK('epsilon_V^0.0(F)_vM')
