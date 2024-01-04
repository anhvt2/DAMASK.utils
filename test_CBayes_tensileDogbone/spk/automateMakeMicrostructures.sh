#!/bin/bash
for i in $(seq 10); do
	cd res-50um-run-${i}
	ln -sf ../phase_dump_12_out.npy .
	ln -sf ../orientations.dat .
	bash ../makeMicrostructures.sh
	cd ..
done
