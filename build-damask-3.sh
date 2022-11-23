#!/bin/bash

### PETSc-configure/build/install
# successfully built on Solo with modules:
#   1) cmake/3.22.3   2) mkl/16.0   3) fftw/2.1   4) openmpi-gnu/4.0   5) gnu/10.2.1

# unset PETSC_ARCH
# unset PETSC_DIR
# sudo ./configure \
#     --prefix=/usr/local/petsc-3.18.1 \
#     --with-fc=gfortran \
#     --with-cc=gcc \
#     --with-cxx=g++ \
#     --download-mpich \
#     --download-fftw \
#     --download-hdf5 \
#     --download-hdf5-fortran-bindings=1 \
#     --download-zlib \
#     --with-mpi-f90module-visibility=0 \
#     PETSC_DIR=$(pwd) PETSC_ARCH=arch-linux-c-debug

# sudo make PETSC_DIR=/home/anhvt89/Documents/DAMASK/petsc/petsc-3.18.1 PETSC_ARCH=arch-linux-c-debug all
# sudo make PETSC_DIR=/home/anhvt89/Documents/DAMASK/petsc/petsc-3.18.1 PETSC_ARCH=arch-linux-c-debug install
# export PETSC_DIR=/usr/local/petsc-3.18.1/
# cd $PETSC_DIR
# ln -sf /home/anhvt89/Documents/DAMASK/petsc/petsc-3.18.1/arch-linux-c-debug . 
# cd - 

### build DAMASK

rm -rfv build-grid/ build-mesh/
# damaskSrc="${damaskSrc}"
damaskSrc="DAMASK"
reset;
echo "Source directory: ${damaskSrc}"

cmake -S ${damaskSrc} -B build-grid -D DAMASK_SOLVER=grid -D CMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build-grid --target install
cmake -S ${damaskSrc} -B build-mesh -D DAMASK_SOLVER=mesh -D CMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build-mesh --target install
