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
echo "Start running simulation at"
echo $(date +%y-%m-%d-%H-%M-%S)
echo "from:"
echo $(pwd)
echo


### ---------------------------------- run DAMASK
# mpiexec --bind-to core --npernode $cores --n $(($cores*$nodes)) $DAMASK_spectral --geom ${filePrefix}.geom --load tension.load 2>&1 > log.damask

if [ -f "numProcessors.dat" ]; then
	numProcessors=$(cat numProcessors.dat)
	if [[ "$numProcessors" -eq 1 ]]
		$DAMASK_spectral --geom ${geomFileName}.geom --load tension.load 2>&1 > log.damask
	else
		mpirun -np ${numProcessors} $DAMASK_spectral --geom ${geomFileName}.geom --load tension.load 2>&1 > log.damask
	fi
else
	mpirun -np 36 $DAMASK_spectral --geom ${geomFileName}.geom --load tension.load 2>&1 > log.damask
fi


### ---------------------------------- post-processing DAMASK
sleep 10
postResults *.spectralOut --cr f,p
if [ -d "postProc" ]; then
	cd postProc/
	filterTable < ${filePrefix}*_tension.txt --white inc,1_f,1_p > stress_strain.log
	cd ..
	cd ..
	python3 computeLossFunction.py --f=40x40x40/
	if [ -f "output.dat" ]; then
		echo 1 > feasible.dat
		echo 1 > complete.dat
	fi
else
	echo "Simulation does not converge!!!"
	echo 0 > feasible.dat
	echo 1 > complete.dat
fi

echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo "Stop running simulation at"
echo $(date +%y-%m-%d-%H-%M-%S)
echo "from :"
echo $(pwd)
echo
