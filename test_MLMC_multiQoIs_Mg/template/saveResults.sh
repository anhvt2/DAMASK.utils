#!/bin/bash

timeStamp=$(date +%y-%m-%d-%H-%M-%S)
for folderName in $(ls -1dv */); do
	mv ${folderName} ${folderName}-${timeStamp}
done

