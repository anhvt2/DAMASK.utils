#!/bin/bash
for dimCell in $(cat dimCellList.dat); do
	cd ${dimCell}x${dimCell}x${dimCell}
	echo ${dimCell} > dimCell.dat # update dimCell.dat

	cat ../material.config.preamble  | cat - material.config | sponge material.config
	geom_check single_phase_equiaxed_${dimCell}x${dimCell}x${dimCell}.geom
	sh ../getDream3dInfo.sh
	ln -sf ../tension.load
	cd ..
done

