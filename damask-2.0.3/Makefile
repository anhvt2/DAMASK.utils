SHELL = /bin/sh
########################################################################################
# Makefile for the installation of DAMASK
########################################################################################
.PHONY: all
all: spectral FEM processing

.PHONY: spectral
spectral: build/spectral
	@(cd build/spectral;make -j4 --no-print-directory -ws all install;)

.PHONY: FEM
FEM: build/FEM
	@(cd build/FEM; make -j4 --no-print-directory -ws all install;)

.PHONY: build/spectral
build/spectral:
	@mkdir -p build/spectral
	@(cd build/spectral; cmake -Wno-dev -DDAMASK_SOLVER=SPECTRAL -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILDCMD_POST=${BUILDCMD_POST} -DBUILDCMD_PRE=${BUILDCMD_PRE} -DOPTIMIZATION=${OPTIMIZATION} -DOPENMP=${OPENMP} ../../;)

.PHONY: build/FEM
build/FEM:
	@mkdir -p build/FEM
	@(cd build/FEM; cmake -Wno-dev -DDAMASK_SOLVER=FEM -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILDCMD_POST=${BUILDCMD_POST} -DBUILDCMD_PRE=${BUILDCMD_PRE} -DOPTIMIZATION=${OPTIMIZATION} -DOPENMP=${OPENMP} ../../;)

.PHONY: clean
clean:
	@rm -rf build

.PHONY: processing
processing:
	@./installation/symlink_Processing.py ${MAKEFLAGS}

