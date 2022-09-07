#!/bin/bash

# workDir="/ascldap/users/anhtran/scratch/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64-15May20"
workDir="/ascldap/users/anhtran/scratch/DAMASK/DAMASK.utils/testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64"
targetDir="/ascldap/users/anhtran/scratch/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testCBayes-Damask-Phase_Dislotwin_TWIP-Steel-FeMnC-64x64x64-15May20/*/"

for folder in $(ls -1dv $targetDir); do
	cd $folder
	# ls -ltr $folder
	folderName=$(basename $(pwd))

	cd $workDir
	mkdir -p $folderName
	cd $folderName

	# begin copy
	cp -rfv $folder/tension.load .
	cp -rfv $folder/sigma.dat .
	cp -rfv $folder/mu.dat .
	cp -rfv $folder/msId.dat .
	cp -rfv $folder/material.config.preamble  .
	cp -rfv $folder/generateMsDream3d.sh .
	cp -rfv $folder/computeYoungModulus.py .
	cp -rfv $folder/computeYieldStress.py .
	cp -rfv $folder/sbatch.damask.solo .
	cp -rfv $folder/log.sbatch .
	cp -rfv $folder/test-adoptSingleCubicPhaseEquiaxed-DAMASK.json .
	cp -rfv $folder/single_phase_equiaxed.geom .
	cp -rfv $folder/dream3d.material.config .
	cp -rfv $folder/single_phase_equiaxed.vtr .
	cp -rfv $folder/grainSize.dat .
	cp -rfv $folder/material.config .
	cp -rfv $folder/log.damask .
	cp -rfv $folder/slurm-758772.out .
	cp -rfv $folder/postProc .
	# end copy

	cd ..

	echo "done $folder"
done

