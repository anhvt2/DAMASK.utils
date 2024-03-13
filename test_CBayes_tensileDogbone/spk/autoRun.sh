#!/bin/bash
# for spkFolder in $(ls -1dv */); do
for i in $(seq 100); do
	spkFolder="res-50um-run-${i}"
	cd ${spkFolder}

	for geomFolder in $(ls -1dv */); do
		cd ${geomFolder}

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
		echo "done ${spkFolder}/${geomFolder}"
	done

	cd ..
	echo "done ${spkFolder}"
done

