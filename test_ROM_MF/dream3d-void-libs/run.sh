#!/bin/bash
dream3dPath="/home/anhvt89/Documents/DREAM.3D/DREAM3D-6.5.141-Linux-x86_64/bin/"
${dream3dPath}/PipelineRunner -p $(pwd)/voidDAMASK.json
geom_check voidEquiaxed.geom
geom2npy --geom voidEquiaxed.geom
geom2png --geom voidEquiaxed.geom
