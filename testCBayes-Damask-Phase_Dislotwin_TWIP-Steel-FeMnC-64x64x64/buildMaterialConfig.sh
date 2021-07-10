#!/bin/bash

for fN in $(ls -1dv *sve*/); do
	cd $fN


	o="material.config"
	cat ../Homogenization_None_Dummy.config > $o
	cat ../Crystallite_aLittleSomething.config >> $o
	cat ../Phase_DisloUCLA_Tungsten.config >> $o
	cat Dream3d.Texture-Microstructure.material.config >> $o


	cd ..
	echo "done $fN"
done



