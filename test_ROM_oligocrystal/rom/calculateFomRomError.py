import glob, os, time
import numpy as np
import logging

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'calculateFomRomError.py.log'
os.system('rm -fv %s' % logFileName)
handlers = [logging.FileHandler(logFileName), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)

controlInfo = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = controlInfo[:,1]
loadingTime = controlInfo[:,2] # dependent - not an input
initialT = controlInfo[:,3]

TrainIdx   = np.loadtxt('TrainIdx.dat', dtype=int)
TestIdx    = np.loadtxt('TestIdx.dat', dtype=int)
TestIdxOOD = np.loadtxt('TestIdxOOD.dat', dtype=int)
TestIdxID  = np.loadtxt('TestIdxID.dat', dtype=int)
fois = ['MisesCauchy', 'MisesLnV'] # fields of interest
labels = [r'$\sigma_{vM}$', r'$\varepsilon_{vM}$']

x_test       = np.loadtxt('inputRom_Test.dat',  delimiter=',', skiprows=1)
DamaskIdxs   = x_test[:,5].astype(int)
PostProcIdxs = x_test[:,6].astype(int)
NumCases = len(DamaskIdxs)

t_start = time.time()

def calcMeanRelErr(true, pred):
    return np.mean(np.abs(true-pred)/true)*100

def calcMeanAbsErr(true, pred):
    return np.mean(np.abs(true-pred))

f = open('FomRomErrors.dat', 'w')
f.write(f'dotVareps, initialT, vareps, sigma, time, DamaskIndex, PostProcIndex, MeanRelError_MisesCauchy, MeanRelError_MisesLnV, MeanAbsError_MisesCauchy, MeanAbsError_MisesLnV\n')

for i in range(NumCases):
    # Extract local parameters
    _dotVareps, _initialT, _vareps, _sigma, _time, _DamaskIndex, _PostProcIndex = x_test[i,:]
    # Get predicted vs true
    predFileName = '../damask/%d/postProc/pred_main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
    trueFileName = '../damask/%d/postProc/main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
    y_pred = np.load(predFileName)
    y_true = np.load(trueFileName)
    # Calculate error
    MeanRelError_MisesCauchy = calcMeanRelErr(y_true[:,0], y_pred[:,0])
    MeanRelError_MisesLnV = calcMeanRelErr(y_true[:,1], y_pred[:,1])
    MeanAbsError_MisesCauchy = calcMeanAbsErr(y_true[:,0], y_pred[:,0])
    MeanAbsError_MisesLnV = calcMeanAbsErr(y_true[:,1], y_pred[:,1])
    # Print/log info
    logging.info(f'Processing damask/{DamaskIdxs[i]+1:<d}/inc{str(PostProcIdxs[i]).zfill(2)}: MeanRelError(MisesCauchy) = {MeanRelError_MisesCauchy:<.4e}; MeanRelError(MisesLnV) = {MeanRelError_MisesLnV:<.4e}; MeanAbsError(MisesCauchy) = {MeanAbsError_MisesCauchy:<.4e}; MeanAbsError(MisesLnV) = {MeanAbsError_MisesLnV:<.4e};')
    f.write(f'{_dotVareps:<.8e}, {_initialT:<.8e}, {_vareps::<.8e}, {_sigma::<.8e}, {_time::<.8e}, {int(_DamaskIndex)::<d}, {int(_PostProcIndex):<d}, {MeanRelError_MisesCauchy:<.8e}, {MeanRelError_MisesLnV:<.8e}, {MeanAbsError_MisesCauchy:<.8e}, {MeanAbsError_MisesLnV:<.8e}\n')

logging.info(f'calculateFomRomError.py: Total elapsed time: {time.time() - t_start} seconds.')



