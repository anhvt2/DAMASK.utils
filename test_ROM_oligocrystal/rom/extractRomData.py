
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
logging.info(f'Warning: Only run this file AFTER running computeCoefs.py!')
time.sleep(3) # add delay to read the warning message
TrainIdx = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx  = np.loadtxt('TestIdx.dat', dtype=int)
FoI = ['Mises(Cauchy)','Mises(ln(V))'] # from export2npy.py
controlInfo = np.loadtxt('control.log', delimiter=',', skiprows=1)

for MlType, idx in zip(['Train', 'Test'], [TrainIdx, TestIdx]):
    # Write input/output datasets for train/test dataset
    inputRomFileName = 'inputRom_%s.dat' % MlType
    iF = open(inputRomFileName, 'w') # input file handler
    iF.write('dotVareps, initialT, vareps, sigma, DamaskIndex, PostProcIndex\n')
    # 
    outputRomFileName = 'outputRom_%s.dat' % MlType
    oF = open(outputRomFileName, 'w') # output file handler
    oFheader = ['podCoef-MisesCauchy-%d' % i for i in range(1,5541)] + ['podCoef-MisesLnV-%d' % i for i in range(1,5541)] # output file header
    oF.write('%s\n' % ",".join(oFheader))
    for i in idx: # for i in range(1,501):
        folderName = str(i+1) # taken from randomizeLoad.py
        logging.info(f'Processing ../damask/{int(i):<d}/')
        fileName = '../damask/%d/postProc/stress_strain.log' % i
        dotVareps, initialT = controlInfo[i,1], controlInfo[i,3]
        # Check if 'stress_strain.log' exists
        if os.path.exists(fileName):
            fileHandler = open(fileName)
            txt = fileHandler.readlines()
            fileHandler.close()
            numHeaderRows = int(txt[0].split('\t')[0])
            oldHeader = txt[numHeaderRows].replace('\n', '').split('\t')
            data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
            # Extract relevant data
            inc, strain, stress = data[:,0], data[:,1], data[:,2]

            # Write to output file
            for j in range(1,len(strain)):
                podFileName = '../damask/%d/postProc/podCoefs_main_tension_inc%s.npy' % (i, str(j).zfill(2) )
                # Only write to global file if POD coefficients exists
                if os.path.exists(podFileName):
                    iF.write('%.8e, %.8e, %.8e, %.1f, %d, %d\n'% (dotVareps, initialT, strain[j], stress[j], i, inc[j]))
                    podCoefs = np.load(podFileName).ravel(order='F') # unravel in columns
                    oF.write(','.join(map(str, podCoefs)) + '\n')
                    logging.info(f'Processing {podFileName}\n')

            # Copy the relevant portion in the same directory
            localOutFileName = '../damask/%d/inputRom.dat' % i
            lF = open(localOutFileName, 'w')
            lF.write('dotVareps, initialT, vareps, sigma, DamaskIndex, PostProcIndex\n')
            for j in range(1,len(strain)):
                lF.write('%.8e, %.8e, %.8e, %.1f, %d, %d\n'% (dotVareps, initialT, strain[j], stress[j], i, inc[j]))
            lF.close()

    iF.close()
    oF.close()

logging.info(f'Elapsed time: {time.time() - t_start} seconds.')

