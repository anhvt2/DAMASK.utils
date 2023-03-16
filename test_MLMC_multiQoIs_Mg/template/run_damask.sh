#!/bin/bash

source ~/.bashrc

hostName="$(echo $(hostname))"
if [[ ${hostName} == *"solo"* ]]; then
	echo "solo detected!"
	# export PETSC_DIR=/ascldap/users/anhtran/data/local/petsc-3.9.4
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 # DAMASK-2.0.2
	export PETSC_DIR=/usr/local/petsc-3.10.3 # DAMASK-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.10.5 # no longer at /data/ -- damask-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.13.6 # DAMASK-3.0.0-alpha
	export PETSC_ARCH=arch-linux2-c-opt # could be arch-linux2-c-debug
	export DAMASK_ROOT=/home/anhtran/scratch/DAMASK/damask-2.0.3/ # s1057681
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-2.0.3
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha
	# export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral
	export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral # s1057681
	export DAMASK_NUM_THREADS=4

	source ${DAMASK_ROOT}/env/DAMASK.sh
	# source /ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/DAMASK_env.sh
	# source /ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha/env/DAMASK.sh
elif [[ ${hostName} == *"s1057681"* ]]; then
	echo "s1057681 detected!"
	# export PETSC_DIR=/ascldap/users/anhtran/data/local/petsc-3.9.4
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 # DAMASK-2.0.2
	export PETSC_DIR=/usr/local/petsc-3.10.3 # DAMASK-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.10.5 # no longer at /data/ -- damask-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.13.6 # DAMASK-3.0.0-alpha
	export PETSC_ARCH=arch-linux2-c-opt # could be arch-linux2-c-debug
	export DAMASK_ROOT=/home/anhtran/Documents/DAMASK/damask-2.0.2/ # s1057681
	# export DAMASK_ROOT=/home/anhtran/Documents/DAMASK/damask-2.0.3/ # s1057681
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-2.0.3
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha
	# export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral
	export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral # s1057681
	export DAMASK_NUM_THREADS=4

	source ${DAMASK_ROOT}/env/DAMASK.sh
	# source /ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/DAMASK_env.sh
	# source /ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha/env/DAMASK.sh
elif [[ ${hostName} == *"strix"* ]]; then
	echo "Strix detected!"
	# export PETSC_DIR=/ascldap/users/anhtran/data/local/petsc-3.9.4
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 # DAMASK-2.0.2
	export PETSC_DIR=/usr/local/petsc-3.10.3 # DAMASK-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.10.5 # no longer at /data/ -- damask-2.0.3
	# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.13.6 # DAMASK-3.0.0-alpha
	export PETSC_ARCH=arch-linux2-c-opt # could be arch-linux2-c-debug
	export DAMASK_ROOT=/home/anhvt89/Documents/DAMASK/damask-2.0.2/ # asus strix
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-2.0.3
	# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha
	# export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral
	export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral
	export DAMASK_NUM_THREADS=4

	source ${DAMASK_ROOT}/env/DAMASK.sh
	# source /ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/DAMASK_env.sh
	# source /ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha/env/DAMASK.sh
else
	echo "run_damask.sh: Error: Please specify DAMASK executables!"
fi


# ---------------------------------- set DAMASK variables
# export PETSC_DIR=/ascldap/users/anhtran/data/local/petsc-3.9.4
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 # DAMASK-2.0.2
# export PETSC_DIR=/usr/local/petsc-3.10.3 # DAMASK-2.0.3
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.10.5 # no longer at /data/ -- damask-2.0.3
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.13.6 # DAMASK-3.0.0-alpha
# export PETSC_ARCH=arch-linux2-c-opt # could be arch-linux2-c-debug
# export DAMASK_ROOT=/home/anhtran/Documents/DAMASK/damask-2.0.3/ # s1057681
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-2.0.3
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha
# export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral
# export DAMASK_spectral=${DAMASK_ROOT}/bin/DAMASK_spectral # s1057681
# export DAMASK_NUM_THREADS=4

# source ${DAMASK_ROOT}/env/DAMASK.sh
# source /ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/DAMASK_env.sh
# source /ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha/env/DAMASK.sh

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Start running simulation at:"
echo $(date +%y-%m-%d-%H-%M-%S)
echo "from:"
echo $(pwd)
echo

### ---------------------------------- pre-set on Solo

# using openmpi-intel/1.10
# mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) /path/to/executable [--args...]
# sh generateMsDream3d.sh # call DREAM.3D for microstructure generation


### ---------------------------------- copy main file from the parent directory
# cp ../tension.load  .
# cp ../sigma.dat  .
# cp ../mu.dat  .
# cp ../msId.dat  .
# cp ../material.config.preamble  .
# cp ../dimCell.dat  .
# cp ../computeYieldStress.py  .
# cp ../computeYoungModulus.py .
# cp ../numerics.config .
# cp ../single_phase_equiaxed_${dimCell}x${dimCell}x${dimCell}.geom  .
# ln -sf *.geom single_phase_equiaxed.geom # assumption: suppose that there is only one *.geom file

### ---------------------------------- pre-process DAMASK
# rm -fv single_phase_equiaxed_tension* postProc/
# geom_check single_phase_equiaxed.geom

### ---------------------------------- run DAMASK
# mpirun -np 4 /media/anhvt89/seagateRepo/DAMASK/DAMASK-v2.0.2/bin/DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask
# mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) $DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask

if [ -f "numProcessors.dat" ]; then
	numProcessors=$(cat numProcessors.dat)
	mpirun -np ${numProcessors} $DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask
	# mpirun -np ${numProcessors} $DAMASK_spectral -g single_phase_equiaxed.geom -l tension.load 2>&1 > log.damask
else
	mpirun -np 32 $DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask
	# mpirun -np 32 $DAMASK_spectral -g single_phase_equiaxed.geom -l tension.load 2>&1 > log.damask
fi


### ---------------------------------- post-processing DAMASK
# sleep 10

filePrefix=$(ls *.spectralOut | cut -d. -f1)
postResults single_phase_equiaxed_tension.spectralOut --cr f,p

if [ -d "postProc" ]; then
	cd postProc/

	addStrainTensors ${filePrefix}.txt --left --logarithmic
	addCauchy ${filePrefix}.txt
	addMises ${filePrefix}.txt --strain 'ln(V)' --stress Cauchy
	filterTable < ${filePrefix}.txt --white inc,'Mises(ln(V))','Mises(Cauchy)' > stress_strain.log

	cp ../tension.load . 
	# check $1 argument in running this script, i.e. bash run_damask.sh $1
	if [[ $1 == "YieldStress" ]]; then 
		python3 ../computeYieldStress.py
	elif [[ $1 == "YoungModulus" ]]; then
		python3 ../computeYoungModulus.py
	else
		echo "run_damask_2.0.3.sh: \$1 argument is not detected in run_damask_2.0.3.sh"
	fi

	if [ -f "output.dat" ]; then
		echo 1 > ../log.feasible
		# needed in wrapper_DREAM3D-DAMASK.py
	fi
	cd ..
else
	echo "Simulation does not converge!!!"
	echo 0 > log.feasible
fi

echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo "Stop running simulation at"
echo $(date +%y-%m-%d-%H-%M-%S)
echo "from :"
echo $(pwd)
echo
