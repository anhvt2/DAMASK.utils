#!/bin/bash

# for i in $(seq -f "%03g" 1 100); do
for i in $(seq 1 100); do
	folderName="res-10um-run-${i}"
	cp -rfv res-50um/ $folderName
	cd $folderName
	ssubmit
	cd ..
done

