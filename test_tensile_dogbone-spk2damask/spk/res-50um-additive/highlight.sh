#!/bin/bash

echo "WARNING: Ignore the first runtime error as initial ms cannot be compared against itself."
echo
echo

for npyFileName in $(ls -1v highlighted_*.npy); do
	python3 ../../../npy2png.py --npy="${npyFileName}" --threshold=1
	echo "done ${npyFileName}"
done
# ffmpeg -framerate 1 -i highlighted_ms_%3d.png -c:v libx264 -r 30 output.mp4
