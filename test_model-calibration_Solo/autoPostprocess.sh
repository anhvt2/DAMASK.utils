#!/bin/bash

for i in $(seq 20); do
	folderName="mlmcSS304L_Iter${i}"
	cd $folderName
	cp ../mlmcSS304L_Template/computeLossFunction.py . 
	python3 computeLossFunction.py --f=8x8x8/ -p 1
	cd ..
	echo "done $folderName"; echo; echo;
done
