#!/bin/bash

# for folderName in $(ls -1dv res-50um-additive-run-*/); do
for i in $(cat tmp2.txt); do
	folderName="res-50um-additive-run-${i}"
	cd $folderName
	rm -fv *.additive_dogbone*.{jpg,png}
	for j in $(seq 2806); do
		rm -fv dump.additive_dogbone.${j}
	done
	echo "done $folderName"
	cd ..
done

