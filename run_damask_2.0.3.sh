#!/bin/bash

source ~/.bashrc

# ---------------------------------- set DAMASK variables
# export PETSC_DIR=/ascldap/users/anhtran/data/local/petsc-3.9.4
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.9.4 # DAMASK-2.0.2
export PETSC_DIR=/usr/local/petsc-3.10.3 # DAMASK-2.0.3
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.10.5 # no longer at /data/ -- damask-2.0.3
# export PETSC_DIR=/ascldap/users/anhtran/local/petsc-3.13.6 # DAMASK-3.0.0-alpha
export PETSC_ARCH=arch-linux2-c-opt # could be arch-linux2-c-debug
export DAMASK_ROOT=/home/anhtran/Documents/DAMASK/damask-2.0.3/ # s1057681
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-2.0.3
# export DAMASK_ROOT=/ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha
# export DAMASK_spectral=$DAMASK_ROOT/bin/DAMASK_spectral
export DAMASK_spectral=$DAMASK_ROOT/bin/DAMASK_spectral # s1057681

source $DAMASK_ROOT/env/DAMASK.sh
# source /ascldap/users/anhtran/data/DAMASK/DAMASK-2.0.2/DAMASK_env.sh
# source /ascldap/users/anhtran/data/DAMASK/damask-3.0.0-alpha/env/DAMASK.sh

echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Start running simulation at"
echo $(date +%y-%m-%d-%H-%M-%S)

### ---------------------------------- pre-set on Solo

# using openmpi-intel/1.10
# mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) /path/to/executable [--args...]
# sh generateMsDream3d.sh # call DREAM.3D for microstructure generation


### ---------------------------------- copy main file from the parent directory
cp ../tension.load  .
cp ../sigma.dat  .
cp ../mu.dat  .
cp ../msId.dat  .
cp ../material.config.preamble  .
cp ../dimCell.dat  .
cp ../computeYieldStress.py  .
cp ../computeYoungModulus.py .
# cp ../single_phase_equiaxed_${dimCell}x${dimCell}x${dimCell}.geom  .
ln -sf *.geom single_phase_equiaxed.geom # assumption: suppose that there is only one *.geom file

### ---------------------------------- pre-process DAMASK
rm -fv single_phase_equiaxed_tension* postProc/
geom_check single_phase_equiaxed.geom

### ---------------------------------- run DAMASK
# mpirun -np 4 /media/anhvt89/seagateRepo/DAMASK/DAMASK-v2.0.2/bin/DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask
# mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) $DAMASK_spectral --geom single_phase_equiaxed.geom --load tension.load 2>&1 > log.damask

if [ -f "numProcessors.dat" ]; then
	numProcessors=$(cat numProcessors.dat)
	mpirun -np ${numProcessors} $DAMASK_spectral -g single_phase_equiaxed.geom -l tension.load 2>&1 > log.damask
else
	mpirun -np 32 $DAMASK_spectral -g single_phase_equiaxed.geom -l tension.load 2>&1 > log.damask
fi


### ---------------------------------- post-processing DAMASK
sleep 10
postResults single_phase_equiaxed_tension.spectralOut --cr f,p
if [ -d "postProc" ]; then
	cd postProc/
	filterTable < single_phase_equiaxed_tension.txt --white inc,1_f,1_p > stress_strain.log
	cp ../tension.load . 
	# python3 ../computeYieldStress.py
	python3 ../computeYoungModulus.py
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
