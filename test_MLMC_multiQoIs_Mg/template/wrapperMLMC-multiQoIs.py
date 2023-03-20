## Usage: python3 wrapperMLMC-DREAM3D-DAMASK.py --level=1

""" 

REQUIREMENTS:

# $ python wrapper-DREAM3D-DAMASK.py --level 0
# Estimated Young modulus at 0 is -2.921755914475476
#
# $ python wrapper-DREAM3D-DAMASK.py --level 1
# Estimated Young modulus at 1 is 1.4565607955675095
# Estimated Young modulus at 0 is -2.4967721807515217
#
# $ python wrapper-DREAM3D-DAMASK.py --level 3 --nb_of_qoi 4
# Estimated Young modulus at 3 is 0.2761242362928689, -0.9839518678051921, -0.1033699430980116, 0.13297340048334058
# Estimated Young modulus at 2 is 0.8931224201628596, -0.42128388915439724, 0.3768735421579537, 0.24507689336645652
#
# $ python wrapper-DREAM3D-DAMASK.py --index (2, 1) --nb_of_qoi 3
# Estimated Young modulus at (2, 1) is -2.5457461714149145, -0.02181018669814895, -2.3552475764029435
# Estimated Young modulus at (1, 1) is 0.501231535869913, 0.5191660280513454, -2.1937281246076665
# Estimated Young modulus at (2, 0) is -0.10207593026854639, 1.547139317467864, -0.9352563652562738
# Estimated Young modulus at (1, 0) is -0.5373920568687983, -0.31172737038566456, 0.24969705569949627

PURPOSES:

This script "wrapperMLMC-DREAM3D-DAMASK.py":

1. Generates a multi-level SVE approximation of a microstructure realization by DREAM.3D
	* Each folder corresponds to a unique SVE mesh size
	-- This is done by calling "generateMsDream3d.sh"
2. Pre-process the microstructure geometry
	-- This is done by "geom_check *.geom"
3. Submit a DAMASK job on Solo with TWO levels of interests
	* This includes the post-processing step
	* Dump out the quantities of interests (e.g. "output.dat")
	-- This is done by calling "ssubmit sbatch.damask.solo"

	* If level 0, then only return 1 QoIs (twice -- duplicated returns)
	* If level > 0, then return 2 QoIs: one at the particular level, and the one right below it, e.g. if submitting at level = 2, then return level = {2,1}
	-- See "damask-MIMC.jl" for more specifics:
		"return level == 0 ? (Qf, Qf) : (Qf-Qc, Qf) # return multilevel difference and approximation at mesh level m"

4. Collect the QoIs and return to some Julia scripts based on "MultilevelEstimators.jl"

BUGS:

1. (Potentially -- not verified exist): fine mesh produces results, but coarse does not

NOTE: 

2. Notation: 0 = coarsest, len(dimCellList) - 1: highest -- from coarsest to finest

3. "generateMsDream3d.sh" and subsequently the .json file used by DREAM.3D can be automatically generated, but currently we assume they are fixed and hard-coded.

For example, the number of levels can (but not yet) be adaptively changed and controlled as well as the size of SVEs

4. DREAM.3D automatically creates folder if it doesn't exist.

RUNNING COMMAND:
rm -rfv $(ls -1dv */); python3 wrapperMLMC-DREAM3D-DAMASK.py --level=1 
# rm -rfv $(ls -1dv */); python3 wrapperMLMC-DREAM3D-DAMASK.py --index=1 # N/A: for MIMC
# deprecated: rm -rfv $(ls -1dv */); python3 wrapperMLMC-DREAM3D-DAMASK.py --index=1 --isNewMs="True"
# deprecated: rm -rfv $(ls -1dv */); python3 wrapperMLMC-DREAM3D-DAMASK.py --meshSize=32 --isNewMs="True"
"""

import numpy as np
import os, glob
import argparse
import time
import datetime
import socket

## write dimCellList.dat for "generateMsDream3d.sh" to pick up
dimCellFile = open('dimCellList.dat', 'w')
dimCellList = [2, 4, 8, 16, 32]

for dimCell in dimCellList:
	dimCellFile.write('%d\n' % int(dimCell))

dimCellFile.close()


# adopt from Sandwich.py and Example.jl from Pieterjan Robbe (KU Leuven)
# python3 Sandwich.py --elemx $(hex) --elemy $(hey) --young $(Ed1) $(Ed2)

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
parser.add_argument("-level", "--level", type=int)
# parser.add_argument("-isNewMs", "--isNewMs", default="True", type=str) # deprecated: always generate new microstructure
# parser.add_argument("-baseSize", "--baseSize", default=320, type=int) # unnecessary -- unless generateMsDream3d.sh is adaptive then this parameter is fixed for now
# NOTE: note that "generateMsDream3d.sh" is hard-coded with a specific number of levels and a specific set of lines to change

args = parser.parse_args()
# meshSize = int(args.meshSize) # make sure meshSize is integer
level = int(args.level); meshSize = int(dimCellList[level]) # get the meshSize from dimCellList[level]
# isNewMs = str2bool(args.isNewMs) # if true, then run DREAM.3D to get new microstructures

# generate all the meshSize but only run in the selected meshSize
# NOTE: meshSize must be divisible by the base
# for example: base 320 generate 320x320x320 SVE initially
# 			then the meshSize could be 64, 32, and so on

# for testing purposes: probably best to test with small size 8x8x8 (2 procs) or 16x16x16 (4 procs)


## generate ALL microstructure approximations
parentDirectory = os.getcwd() # get parentDirectory for reference
def generateMicrostructures(parentDirectory):
	# only generate if isNewMs is True (default = True) -- deprecated
	# if isNewMs:
	# clear folders before doing anything else
	os.chdir(parentDirectory)
	print("wrapperMLMC-DREAM3D-DAMASK.py: removing/cleaning up ?x?x? folders")
	for dimCell in dimCellList:
		os.system('rm -rfv %dx%dx%d' % (int(dimCell), int(dimCell), int(dimCell)))

	print("wrapperMLMC-DREAM3D-DAMASK.py: calling DREAM.3D to generate microstructures")
	os.system('bash generateMsDream3d.sh')


def run_DAMASK_offline(meshSize, parentDirectory, level):
	os.chdir(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize)) # go into subfolder "${meshSize}x${meshSize}x${meshSize}"

	startTime = datetime.datetime.now()
	os.system('bash run_damask.sh')

	### read outputs
	currentTime = datetime.datetime.now()
	feasible = 0
	if os.path.exists(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc/feasible.dat'):
		feasible = np.loadtxt(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc/feasible.dat')
		if feasible == 0:
			vmStrain = [np.nan] * np.ones([11,2]) # invalid condition
			vmStress = [np.nan] * np.ones([11,2]) # invalid condition
		elif feasible == 1:
			currentTime = datetime.datetime.now()
			outData = np.loadtxt(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc/output.dat')
			vmStrain = outData[:, 0]
			vmStress = outData[:, 1] # in MPa
			str2print = ', '.join(vmStress) # construct a string to print on screen
			print("Results available in %s" % (parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize)))
			print("\n Elapsed time = %.2f minutes on %s" % ((currentTime - startTime).total_seconds() / 60.), socket.gethostname())
			print("Collocated von Mises stresses at level %d is %s MPa" % (level, str2print))
			### write log
			f = open(parentDirectory + '/' + 'log.MultilevelEstimators-multiQoIs', 'a') # can be 'r', 'w', 'a', 'r+'
			f.write("Collocated von Mises stresses at level %d is %s MPa" % (level, str2print))
			f.close()
			### save results (the whole mesh hierarchy with computed outputs) for forensic analysis
			### TODO

	return feasible

def evaluate_DAMASK(meshSize, parentDirectory, level):
	# adaptive functional evaluation w.r.t. different platforms
	if 'solo' in socket.gethostname():
		feasible = submitDAMASK(meshSize, parentDirectory, level)
	else:
		feasible = run_DAMASK_offline(meshSize, parentDirectory, level)
	return feasible

## if level > 0 then submit a DAMASK job at [level - 1]
feasible = 0

# while feasible == 0:
generateMicrostructures(parentDirectory)
level = int(args.level); meshSize = int(dimCellList[level]) # get the meshSize from dimCellList[level]

feasible = evaluate_DAMASK(meshSize, parentDirectory, level)

if level > 0:
	level -= 1
	meshSize = int(dimCellList[level]) # get the meshSize from dimCellList[level - 1] -- coarser mesh
	feasible = evaluate_DAMASK(meshSize, parentDirectory, level)





