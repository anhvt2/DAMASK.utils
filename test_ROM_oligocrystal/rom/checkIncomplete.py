
import os
import numpy as np
from natsort import natsorted, ns

f = open('IncompleteIdx.dat', 'w')

for i in range(1,1001):
    if not os.path.exists(f'../damask/{i:<d}/postProc/main_tension_inc19.txt'):
        print(f'In {i:<d}')
        try:
            LastFile = natsorted(glob.glob(f'../damask/{i:<d}/postProc/main_tension_inc??.txt'))[-1]
            print(f'Folder {i:<d}: LastFile = {LastFile}')
        except:
            print(f'There is no main_tension_inc??.txt in ../damask/{i:<d}/postProc/')
        f.write(f'{i-1:<d}\n')

f.close()
