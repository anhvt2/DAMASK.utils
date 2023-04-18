#!/bin/bash

mkdir -p bak/
ln -sf ../tension.load bak/
ln -sf ../numerics.config bak/

timeStamp=$(date +%y-%m-%d-%H-%M-%S)
# for folderName in $(ls -1dv */); do
	# mv ${folderName} ${folderName}-${timeStamp}
# done

for dimCell in $(cat dimCellList.dat); do
	mv ${dimCell}x${dimCell}x${dimCell} bak/${dimCell}x${dimCell}x${dimCell}-${timeStamp}
done
