

"""
	Convert a microstructure SPPARKS dump file to another modified SPPARKS dump file. 

	This script is to be used in concert with
		1. `geom_cad2phase.py` to model void.

	In the nutshell, it increases the spin/grain ID += 1 and assign void ID = 1 for empty space. 
	This script is inspired by SPPARKS.utils/spparks_logo/maskSpkVti.py

	Examples
	--------
		python3 geom_spk2spk.py --vti='potts_3d.*.vti' --phase='m_dump_12_out.npy'

	Parameters
	----------
		--vti: dump file from SPPARKS
		--phase (formatted in .npy): phase for dogbone modeling (could be generalized to internal void as well)
		--void_id (DEPRECATED) (adopted from geom_cad2phase.py): default void_id = np.inf, non void = -1 (see geom_cad2phase.py for more information)


"""

import numpy as np
import os, sys, time, glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import pyvista
from natsort import natsorted, ns

parser = argparse.ArgumentParser()
parser.add_argument("-vti", "--vti",     type=str, required=True)
parser.add_argument("-p"  , "--phase",   type=str, required=True)
# parser.add_argument("-p"  , "--void_id", type=str, False=True, default=np.inf)
args = parser.parse_args()

vtiFileNames  = args.vti # 'potts_3d.*.vti'
phaseFileName = args.phase # 'phase_dump_12_out.npy'

try:
	phase = np.load(phaseFileName)
except OSError:
	print(f"File {phaseFileName} does not exist.")

def getDumpMs(dumpFileName):
	"""
		This function return a 3d array 'm' microstructure from reading a SPPARKS dump file, specified by 'dumpFileName'.
	"""
	dumpFile = open(dumpFileName)
	dumptxt = dumpFile.readlines()
	dumpFile.close()
	for i in range(20): # look for header info in first 20 lines
		tmp = dumptxt[i]
		if 'BOX BOUNDS' in tmp:
			Nx = int(dumptxt[i+1].replace('\n','').replace('0 ', '').replace(' ', ''))
			Ny = int(dumptxt[i+2].replace('\n','').replace('0 ', '').replace(' ', ''))
			Nz = int(dumptxt[i+3].replace('\n','').replace('0 ', '').replace(' ', ''))
			break
	header = np.array(dumptxt[i+4].replace('\n','').replace('ITEM: ATOMS ', '').split(' '), dtype=str)
	d = np.loadtxt(dumpFileName, skiprows=9, dtype=int)
	num_grains = len(np.unique(d[:,1]))
	old_grain_ids = np.unique(d[:,1])
	new_grain_ids = range(len(np.unique(d[:,1])))
	m = np.zeros([Nx, Ny, Nz]) # initialize
	for ii in range(len(d)):
		i = int(d[ii,np.where(header=='x')[0][0]]) # 'x'
		j = int(d[ii,np.where(header=='y')[0][0]]) # 'y'
		k = int(d[ii,np.where(header=='z')[0][0]]) # 'z'
		grain_id = int(d[ii,1]) # or d[i,2] -- both are the same
		# option: DO re-enumerating
		lookup_idx = np.where(old_grain_ids == grain_id)[0][0]
		new_grain_id = new_grain_ids[lookup_idx]
		m[i,j,k] = new_grain_id
		# option: DO NOT re-enumerating
		# m[i,j,k] = grain_id # TODO: implement re-enumerate grain_id
		# print(f"finish ({x},{y}, {z})")
	return m, Nx, Ny, Nz, num_grains

t_start = time.time() # tic

for vtiFileName in natsorted(glob.glob(vtiFileNames)):
	outFileName = 'masked_' + vtiFileName
	vtiData = pyvista.read(vtiFileName)
	Nx, Ny, Nz = np.array(vtiData.dimensions) - 1
	spin = vtiData.get_array('Spin').reshape(Nz, Ny, Nx).T
	spin += 1
	print(f"Processing {vtiFileName}...")
	for i in range(Nx):
		for j in range(Ny):
			for k in range(Nz):
				if phase[i,j,k] == np.inf:
					spin[i,j,k] = 1 # assign void
	vtiData['Spin'] = spin.T.flatten()
	vtiData.save(outFileName)			
	print(f"Finished processing {vtiFileName}!\n")


elapsed = time.time() - t_start # toc
print("geom_spk2spk.py: finished in {:5.2f} seconds.\n".format(elapsed), end="")
