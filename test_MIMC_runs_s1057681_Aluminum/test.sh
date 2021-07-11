#!/bin/sh

source ~/.bashrc # get geom_check environment variable

for dimCell in $(cat dimCellList.dat); do
	for constitutiveModel in Isotropic Phenopowerlaw Nonlocal; do
		mkdir -p ${dimCell}x${dimCell}x${dimCell}_${constitutiveModel}

		cd ${dimCell}x${dimCell}x${dimCell}_${constitutiveModel}
		echo ${dimCell} > dimCell.dat # update dimCell.dat
		cp ../${dimCell}x${dimCell}x${dimCell}/* . # copy all the geometry information

		cat ../material.config.${constitutiveModel}.preamble  | cat - material.config | sponge material.config
		geom_check single_phase_equiaxed_${dimCell}x${dimCell}x${dimCell}.geom
		sh ../getDream3dInfo.sh
		cd ..
	done
done

