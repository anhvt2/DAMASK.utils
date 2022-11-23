#!/bin/bash

rm -rfv build-grid/ build-mesh/
# damaskSrc="${damaskSrc}"
damaskSrc="DAMASK"
reset;
echo "Source directory: ${damaskSrc}"

cmake -S ${damaskSrc} -B build-grid -DDAMASK_SOLVER=grid
cmake --build build-grid --target install
cmake -S ${damaskSrc} -B build-mesh -DDAMASK_SOLVER=mesh
cmake --build build-mesh --target install
