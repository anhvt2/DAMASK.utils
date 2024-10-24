
import numpy as np
import os, glob
import time
import logging

"""
This script comes in and extracts parameters from each CPFEM, including
    (-) i: CPFEM simulation index, 
    (1) dotVareps: strain rate, 
    (-) loadingTime: total loading time, 
    (2) initialT: temperature,
    (3) time: (pseudo)-time 
            back calculated from the homogenized stress_strain.log
            assuming constant loading rate
    (4) strain (Mises(ln(V)): dotVareps * time
            instead of interpolating on time, we can also interpolate on strain
* and dump information to 'inputRom.dat'
"""

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'extractRomInput.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

t_start = time.time()
TrainIdx = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx  = np.loadtxt('TestIdx.dat', dtype=int)
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py
controlInfo = np.loadtxt('control.log', delimiter=',', skiprows=1)

for MlType in ['Train', 'Test']:
    # Write input for train/test dataset
    outFileName = 'inputRom_%s.dat' % MlType
    f = open(outFileName, 'w')
    f.write('dotVareps, initialT, vareps, sigma, DamaskIndex, PostProcIndex\n')
    # Switch between data type
    if MlType == 'Train':
        idx = TrainIdx
    elif MlType == 'Test':
        idx = TestIdx
    for i in idx: # for i in range(1,501):
        folderName = str(i+1) # taken from randomizeLoad.py
        logging.info(f'Processing ../damask/{int(i+1):<d}/')
        fileName = '../damask/%d/postProc/stress_strain.log' % i
        dotVareps = controlInfo[i,1]
        initialT  = controlInfo[i,3]
        # Check if 'stress_strain.log' exists
        if os.path.exists(fileName):
            fileHandler = open(fileName)
            txt = fileHandler.readlines()
            fileHandler.close()
            numHeaderRows = int(txt[0].split('\t')[0])
            oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
            data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
            # Extract relevant data
            inc = data[:,0]
            strain = data[:,1]
            stress = data[:,2]

            # Write to output file
            for j in range(1,len(strain)):
                f.write('%.8e, %.8e, %.8e, %.1f, %d, %d\n'% (dotVareps, initialT, strain[j], stress[j], i, inc[j]))

            # Copy the relevant portion in the same directory
            localOutFileName = '../damask/%d/inputRom.dat' % i
            lF.write('dotVareps, initialT, vareps, sigma, DamaskIndex, PostProcIndex\n')
            lF = open(localOutFileName, 'w')
            for j in range(1,len(strain)):
                lF.write('%.8e, %.8e, %.8e, %.1f, %d, %d\n'% (dotVareps, initialT, strain[j], stress[j], i, inc[j]))
            lF.close()

    f.close()

