#!/bin/bash
asuslocal=anhvt89@192.168.1.160
rsync -Wav --progress -e 'ssh -p 6900' ${asuslocal}:/home/anhvt89/Documents/DAMASK.utils/test_tensile_dogbone-spk2damask/spk/test/ ./
