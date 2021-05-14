
""" 
PURPOSES:

This script "wrapper_DREAM3D-DAMASK.py":

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
rm -rfv $(ls -1dv */); python3 wrapper_DREAM3D-DAMASK.py --level=1 
# deprecated: rm -rfv $(ls -1dv */); python3 wrapper_DREAM3D-DAMASK.py --level=1 --isNewMs="True"
# deprecated: rm -rfv $(ls -1dv */); python3 wrapper_DREAM3D-DAMASK.py --meshSize=32 --isNewMs="True"
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
	print("wrapper_DREAM3D-DAMASK.py: removing/cleaning up ?x?x? folders")
	for dimCell in dimCellList:
		os.system('rm -rfv %dx%dx%d' % (int(dimCell), int(dimCell), int(dimCell)))

	print("wrapper_DREAM3D-DAMASK.py: calling DREAM.3D to generate microstructures")
	os.system('sh generateMsDream3d.sh')

## define a function to submit a DAMASK job with "meshSize" and "parentDirectory" and parameters
## WITHOUT generating a new microstructure
def submitDAMASK(meshSize, parentDirectory, level):
	os.chdir(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize)) # go into subfolder "${meshSize}x${meshSize}x${meshSize}"
	os.system('cp ../sbatch.damask.solo .')

	# write down numProcessors to be picked up later by sbatch.damask.solo
	numProcessors = np.floor(meshSize / 4.)
	if numProcessors > 36:
		numProcessors = 36 # threshold on Solo node

	f = open('numProcessors.dat', 'w') # can be 'r', 'w', 'a', 'r+'
	f.write('%d' % numProcessors)
	f.close()

	os.system('ssubmit sbatch.damask.solo')
	# os.chdir(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc')

	startTime = datetime.datetime.now()

	## get thresholdSlurmTime adaptively from sbatch.damask.solo (or just hard-code it, e.g. 48 * 24 * 3600)
	# return thresholdSlurmTime in seconds, read from sbatch.damask.solo
	slurmFile = open(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/sbatch.damask.solo')
	slurmSubmitText = slurmFile.readlines()
	slurmFile.close()
	thresholdSlurmTime = slurmSubmitText[3].split('=')[1].split("#")[0].replace(" ", "") # e.g. thresholdSlurmTime = "48:00:00"
	thresholdSlurmTime = thresholdSlurmTime.split(":")[0]
	thresholdSlurmTime = int(thresholdSlurmTime)
	thresholdSlurmTime *= (24*3600) # convert to seconds

	while not os.path.exists(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc/output.dat'):
		time.sleep(10)
		currentTime = datetime.datetime.now()
		if os.path.exists(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/log.feasible'):
			feasible = np.loadtxt(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/log.feasible')
			if feasible == 0:
				yieldStress = 0 # invalid condition
				break
			elif feasible == 1:
				currentTime = datetime.datetime.now()
				yieldData = np.loadtxt(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/postProc/output.dat')
				yieldStrain = float(yieldData[0])
				yieldStress = float(yieldData[1]) / 1e9 # in GPa
				print("Results available in %s" % (parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize)))
				print("\n Elapsed time = %.2f minutes on Solo" % ((currentTime - startTime).total_seconds() / 60.))
				print("Estimated Yield Stress (at level = %d) = %.16f GPa" % (level, yieldStress))

				f = open(parentDirectory + '/' + 'log.MultillevelEstimators-DAMASK-DREAM3D', 'a') # can be 'r', 'w', 'a', 'r+'
				f.write("Estimated Yield Stress (at level = %d, meshSize = %d) = %.16f GPa\n" % (level, meshSize, yieldStress))
				f.close()
				break

		if (currentTime - startTime).total_seconds() > thresholdSlurmTime:
			# if the previous loop does not return any output
			# it means the microstructure was not able to return any results with DAMASK within the time requested
			feasible = 0
			yieldStress = 0 # invalid condition
			break
			print("Time waited > time submitted: there is something wrong with the submission!")
			# print("Warning: Regenerating microstructure for another try!")
			# os.chdir(parentDirectory)
			# os.system('sh generateMsDream3d.sh')
			# os.chdir(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize)) # go into subfolder "${meshSize}x${meshSize}x${meshSize}"
			# os.system('cp ../sbatch.damask.solo .')

			# numProcessors = np.floor(meshSize / 4.)
			# if numProcessors > 36:
			# 	numProcessors = 36 # threshold on Solo node

			# f = open('numProcessors.dat', 'w') # can be 'r', 'w', 'a', 'r+'
			# f.write('%d' % numProcessors)
			# f.close()

			# os.system('ssubmit sbatch.damask.solo')
			# startTime = datetime.datetime.now()

			# slurmFile = open(parentDirectory + '/%dx%dx%d' % (meshSize, meshSize, meshSize) + '/sbatch.damask.solo')
			# slurmSubmitText = slurmFile.readlines()
			# slurmFile.close()
			# thresholdSlurmTime = slurmSubmitText[3].split('=')[1].split("#")[0].replace(" ", "") # e.g. thresholdSlurmTime = "48:00:00"
			# thresholdSlurmTime = thresholdSlurmTime.split(":")[0]
			# thresholdSlurmTime = int(thresholdSlurmTime)
			# thresholdSlurmTime *= (24*3600) # convert to seconds

	return feasible


## if level > 0 then submit a DAMASK job at [level - 1]
feasible = 0

while feasible == 0:
	generateMicrostructures(parentDirectory);
	level = int(args.level); meshSize = int(dimCellList[level]) # get the meshSize from dimCellList[level]
	feasible = submitDAMASK(meshSize, parentDirectory, level)
	if level > 0:
		level -= 1
		meshSize = int(dimCellList[level]) # get the meshSize from dimCellList[level - 1] -- coarser mesh
		feasible = submitDAMASK(meshSize, parentDirectory, level)





