#!/bin/bash

echo "Run vizAM.py to produce highlighted_*.npy"

i=0 # Do not compare against the initial ms with itself

for npyFileName in $(ls -1v highlighted_*.npy); do
	if [ "$i" -ne "0" ]; then
		python3 ../../../npy2png.py --npy="${npyFileName}" --threshold=1
		echo "done ${npyFileName}"
	fi
	i=$((i+1))
done
# ffmpeg -framerate 1 -i highlighted_ms_%3d.png -c:v libx264 -r 30 output.mp4
