#!/bin/bash
numMs=30
for i in $(seq ${numMs}); do
	bash generateMsDream3d.sh
	mv material.config texture_${i}.config
	cp material.config.bak material.config
	echo; echo; echo;
	echo "done microstructure $i"
done
