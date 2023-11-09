#!/bin/bash

for fN in $(ls -1dv 128x128x128-mu-1.25*/); do
	cd $fN
	sdel
	cd ..
done
