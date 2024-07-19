#!/bin/bash
# ffmpeg -framerate 1 -i highlighted_ms_%3d.png -c:v libx264 -r 30 output.mp4
# ffmpeg -framerate 1/5 -i highlighted_ms_%3d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
# ffmpeg -i highlighted_ms_%3d.png -c:v libx264  -r 5 -pix_fmt yuv420p out.mp4
ffmpeg -i highlighted_ms_%d.png -c:v libx264  -r 5 -pix_fmt yuv420p out.mp4

