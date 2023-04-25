#!/bin/bash
# pack into 1 single tarball

tag=$(basename $(pwd))
tar cvzf ${tag}.stress_strain.log.tar.gz *stress_strain.log
rm *stress_strain.log

tar cvzf ${tag}.output.dat.tar.gz *output.dat
rm *output.dat

cp log.MultilevelEstimators-multiQoIs ${tag}.log.MultilevelEstimators-multiQoIs
