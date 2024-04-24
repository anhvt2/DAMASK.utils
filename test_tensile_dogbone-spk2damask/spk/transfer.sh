#!/bin/bash
rm -f test.tar.gz
tar cvzf test.tar.gz *jpg
scp test.tar.gz anhtran@inouye.sandia.gov:~/scratch/

