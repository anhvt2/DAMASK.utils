#!/bin/bash
# for fN in $(ls -1dv ap*Run?/); do
for fN in $(ls -1dv parallel*Run?/); do
	cd $fN

	sh ../../utils/buildLogs.sh

	cd ..
	echo "done $fN"
done

