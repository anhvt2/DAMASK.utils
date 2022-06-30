#!/bin/bash
for fN in $(ls -1dv pBO*michal*Run?/); do
# for fN in $(ls -1dv pBO*Run?/); do
	cd $fN

	sh ../../utils/buildLogs_batchSeqPar.sh

	cd ..
	echo "done $fN"
done

