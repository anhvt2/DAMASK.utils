#!/bin/bash
ls -ltr stress_strain.log.tar.gz
tar xvzf stress_strain.log.tar.gz
tar cvzf stress_strain.log.tar.gz *log
ls -ltr stress_strain.log.tar.gz
rm *stress_strain.log
