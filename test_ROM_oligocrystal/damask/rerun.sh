#!/bin/bash
for i in $(seq 1000); do
	cd $i
	if [ ! -d "postProc" ]; then
		rm -v main_geom.*
		cp -L ../template/main.geom .
		ssubmit
		echo "$i/postProc does not exist. Rerun in $i"
	fi
	echo "done $i"
	cd ..
done

