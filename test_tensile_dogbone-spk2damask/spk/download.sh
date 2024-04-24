#!/bin/bash
rm -rfv test/
mkdir test/
scp anhtran@inouye.sandia.gov:~/scratch/test.tar.gz .
cd test/
tar xvzf ../test.tar.gz