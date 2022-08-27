#!/bin/bash

rm -v *cmaes* # clean up *cmaes* file
echo 0 > complete.dat
gnome-terminal -- timeout 2m bash -c "echo Hello, World; sleep 1m; echo 50 > output.dat; echo 1 > complete.dat" > log.timeout


