#!/bin/bash
ls -ltr output.dat.tar.gz
tar xvzf output.dat.tar.gz
tar cvzf output.dat.tar.gz *log
ls -ltr output.dat.tar.gz
rm *output.dat

