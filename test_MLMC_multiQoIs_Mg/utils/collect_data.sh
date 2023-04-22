#!/bin/bash

rm -fv log.MultilevelEstimators-multiQoIs

for folderName in $(find . -name 'postProc'); do
	shortFolderName=$(echo $folderName | cut -d/ -f4)
	cp $folderName/output.dat ./$shortFolderName-output.dat
	cp $folderName/stress_strain.log ./$shortFolderName-stress_strain.log

	cat $folderName/../../../log.MultilevelEstimators-multiQoIs >> log.MultilevelEstimators-multiQoIs
	# echo $shortFolderName
done

# pack into 1 single tarball

tag=$(basename $(pwd))
tar cvzf ${tag}.stress_strain.log.tar.gz *stress_strain.log
rm *stress_strain.log

tar cvzf ${tag}.output.dat.tar.gz *output.dat
rm *output.dat

cp log.MultilevelEstimators-multiQoIs ../${tag}.log.MultilevelEstimators-multiQoIs
