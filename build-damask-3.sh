#!/bin/bash

rm -rfv build-grid/ build-mesh/
# damaskSrc="${damaskSrc}"
damaskSrc="DAMASK"
reset;
echo "Source directory: ${damaskSrc}"

cmake -S ${damaskSrc} -B build-grid -D DAMASK_SOLVER=grid -D CMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build-grid --target install
cmake -S ${damaskSrc} -B build-mesh -D DAMASK_SOLVER=mesh -D CMAKE_INSTALL_PREFIX=$(pwd)
cmake --build build-mesh --target install
