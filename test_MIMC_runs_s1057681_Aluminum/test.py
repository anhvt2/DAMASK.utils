# test: python3 test.py --index="(2, 3)"

""" 
PURPOSES:

This script "wrapperMIMC-DREAM3D-DAMASK.py":

1. Generates a multi-index SVE approximation of a microstructure realization by DREAM.3D
	* Each folder corresponds to a unique SVE mesh size
	-- This is done by calling "generateMsDream3d.sh"
2. Pre-process the microstructure geometry
	-- This is done by "geom_check *.geom"
3. Submit a DAMASK job on Solo with TWO QoIs
	* This includes the post-processing step
	* Dump out the quantities of interests (e.g. "output.dat")
	-- This is done by calling "ssubmit sbatch.damask.solo"

	* If index 0, then only return 1 QoIs (twice -- duplicated returns)
	* If index > 0, then return 2 QoIs: one at the particular index, and the one right below it, e.g. if submitting at index = 2, then return index = {2,1}
	-- See "damask-MIMC.jl" for more specifics:
		"return index == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multiindex difference and approximation at mesh index m"

4. Collect the QoIs and return to some Julia scripts based on "MultilevelEstimators.jl"

BUGS:

1. (Potentially -- not verified exist): fine mesh produces results, but coarse does not

NOTE: 

2. Notation: 0 = coarsest, len(dimCellList) - 1: highest -- from coarsest to finest

3. "generateMsDream3d.sh" and subsequently the .json file used by DREAM.3D can be automatically generated, but currently we assume they are fixed and hard-coded.

For example, the number of indexs can (but not yet) be adaptively changed and controlled as well as the size of SVEs

4. DREAM.3D automatically creates folder if it doesn't exist.

300: [1; 2; 3; 4; 5; 6; 10; 12; 15; 20; 25; 30; 50; 60; 75; 100; 150; 300]
320: [1; 2; 4; 5; 8; 10; 16; 20; 32; 40; 64; 80; 160; 320]

DREAM.3D takes 3 minutes to generate ALL microstructures
BENCHMARK on Solo: (using numProcessors = int(meshSize / 4.))
8x8x8: <1 minute
10x10x10: 7.51 minutes
16x16x16: 6 -- 10 minutes
20x20x20: 19 -- 20 minutes
32x32x32: 51 minutes

BENCHMARK on Solo: (using numProcessors = int(meshSize / 2.)) # unstable
8x8x8: 1.67 minutes
16x16x16: (unstable)
32x32x32: 41 minutes
64x64x64: > 4 hours (est. 320 minutes ~ 6 hours)

RUNNING COMMAND:
rm -rfv $(ls -1dv */); python3 wrapperMIMC-DREAM3D-DAMASK.py --index=1 
# deprecated: rm -rfv $(ls -1dv */); python3 wrapperMIMC-DREAM3D-DAMASK.py --index=1 --isNewMs="True"
# deprecated: rm -rfv $(ls -1dv */); python3 wrapperMIMC-DREAM3D-DAMASK.py --meshSize=32 --isNewMs="True"
"""

import numpy as np
import os, glob
import argparse
import time
import datetime

## write dimCellList.dat for "generateMsDream3d.sh" to pick up
dimCellFile = open('dimCellList.dat', 'w')
dimCellList = [8, 10, 16, 20, 32] 

for dimCell in dimCellList:
	dimCellFile.write('%d\n' % int(dimCell))

dimCellFile.close()


# adopt from Sandwich.py and Example.jl from Pieterjan Robbe (KU Leuven)

## python3 Sandwich.py --elemx $(hex) --elemy $(hey) --young $(Ed1) $(Ed2)
def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
# parser.add_argument("-meshSize", "--meshSize", type=int)
parser.add_argument("-index", "--index", type=str)
# https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
# parser.add_argument("-isNewMs", "--isNewMs", default="True", type=str) # deprecated: always generate new microstructure
# parser.add_argument("-baseSize", "--baseSize", default=320, type=int) # unnecessary -- unless generateMsDream3d.sh is adaptive then this parameter is fixed for now
# NOTE: note that "generateMsDream3d.sh" is hard-coded with a specific number of indices and a specific set of lines to change

args = parser.parse_args()
# meshSize = int(args.meshSize) # make sure meshSize is integer
index = np.array(args.index[1:-1].split(','), dtype=int) # reformat to dtype=int
meshSizeIndex = index[0]
meshSize = int(dimCellList[meshSizeIndex]) # get the meshSize from dimCellList[meshSizeIndex]
constitutiveModelIndex = index[1] # 0 = "Isotropic", 1 = "Phenopowerlaw", 2 = "Nonlocal"

for s in index:
	print(s)

# print(meshSize)
# print(constitutiveModelIndex)

def getAllQueryIndex(meshSizeIndex, constitutiveModelIndex):
	s = []
	s += [[meshSizeIndex, constitutiveModelIndex]]
	if meshSizeIndex > 0:
		s += [[meshSizeIndex - 1, constitutiveModelIndex]]
	if constitutiveModelIndex > 0:
		s += [[meshSizeIndex, constitutiveModelIndex - 1]]
	if meshSizeIndex > 0 and constitutiveModelIndex > 0:
		s += [[meshSizeIndex - 1, constitutiveModelIndex - 1]]
	return s

query_list = getAllQueryIndex(meshSizeIndex, constitutiveModelIndex)
print("Querying at")
print(query_list)

