#!/bin/bash
for folderName in $(ls -1dv test-run-*-*/); do
	cd ${folderName}

	ln -sf padded*.geom main.geom # change geom fileName
    # cp -v ../../sbatch.damask.srn .
	rm sbatch.damask.srn
	rm sbatch.postProcDamaskLocalQoIs.srn
	# cp -v ../../sbatch.postProcDamaskLocalQoIs.srn . # Local QoIs
	cp -v ../../sbatch.postProcDamaskGlobalQoIs.srn . # Global QoIs
    ln -sf ../../tension.load .
    ln -sf ../../numerics.config .
    ln -sf ../../numProcessors.dat .
	ln -sf ../../gaugeFilter.txt .
	ssubmit

	cd ..
	echo "done ${folderName}"
done

