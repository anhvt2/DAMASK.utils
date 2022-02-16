#!/bin/bash

for folderName in $(ls -1dv sg_input_*/); do
	cd $folderName

	ssubmit

	cd ..
done

