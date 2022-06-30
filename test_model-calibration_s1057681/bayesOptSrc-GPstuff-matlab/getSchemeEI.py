import numpy as np 
import os, glob, time

# NOTE: don't write '\n' char to the acquisitionScheme.dat
# add safeguard to write until the file exists
while not os.path.exists('acquisitionScheme.dat'):
	f = open('acquisitionScheme.dat','w')
	f.write('EI');
	print('getAcquisitionScheme.py: EI is selected')
	f.close()
	time.sleep(0.25)

