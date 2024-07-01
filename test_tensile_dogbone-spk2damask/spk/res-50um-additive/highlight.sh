#!/bin/bash
for npyFileName in $(ls -1v *.npy); do
	python3 ../../../npy2png.py --npy="${npyFileName}" --threshold=1
	echo "done ${npyFileName}"
done
# ffmpeg -framerate 1 -i highlighted_ms_%3d.png -c:v libx264 -r 30 output.mp4
