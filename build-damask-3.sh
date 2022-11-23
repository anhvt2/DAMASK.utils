#!/bin/bash

### PETSc-configure/build/install

sudo ./configure \
    --prefix=/usr/local/petsc-3.18.1 \
    --with-fc=gfortran \
    --with-cc=gcc \
    --with-cxx=g++ \
    --download-mpich \
    --download-fftw \
    --download-hdf5 \
    --download-hdf5-fortran-bindings=1 \
    --download-zlib \
    --with-mpi-f90module-visibility=0 \
    PETSC_DIR=$(pwd)


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
