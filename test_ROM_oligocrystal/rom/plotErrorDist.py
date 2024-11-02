import matplotlib.pyplot as plt
import glob, os, time
import numpy as np
import matplotlib as mpl
import logging

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

level    = logging.INFO
format   = '  %(message)s'
logFileName = 'plotErrorDist.py.log'
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

t_local = time.time()

def calcMeanRelErr(true, pred):
    return np.mean(np.abs(true-pred)/true)*100

def calcMeanAbsErr(true, pred):
    return np.mean(np.abs(true-pred))   

for i in range(NumCases):
    predFileName = '../damask/%d/postProc/pred_main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
    trueFileName = '../damask/%d/postProc/main_tension_inc%s.npy' % (DamaskIdxs[i], str(PostProcIdxs[i]).zfill(2))
    y_pred = np.load(predFileName)
    y_true = np.load(trueFileName)
    logging.info(f'Processing damask/{DamaskIdxs[i]+1:<d}/inc{str(PostProcIdxs[i]).zfill(2)}: MeanRelError(MisesCauchy) = {calcMeanRelErr(y_true[:,0], y_pred[:,0]):<.4e}; MeanRelError(MisesLnV) = {calcMeanRelErr(y_true[:,1], y_pred[:,1]):<.4e}; MeanAbsError(MisesCauchy) = {calcMeanAbsErr(y_true[:,0], y_pred[:,0]):<.4e}; MeanAbsError(MisesLnV) = {calcMeanAbsErr(y_true[:,1], y_pred[:,1]):<.4e};')

logging.info(f'plotErrorDist.py: Total elapsed time: {time.time() - t_start} seconds.')

    


# fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
# plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
# plt.scatter(dotVarEps_Test, initialT_Test, marker='o', s=30, c='tab:orange', label='test')
# plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=True, markerscale=3)

# plt.title('Input Distribution', fontsize=24)
# plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
# plt.ylabel(r'$T$ [K]', fontsize=24)
# plt.xscale('log',base=10) 
# # plt.show()
# plt.savefig('TrainTestDistribution', dpi=300, facecolor='w', edgecolor='w',
#     orientation='portrait', format=None,
#     transparent=False, bbox_inches='tight', pad_inches=0.1,
#     metadata=None)

# # Plot: train/test-OOD/test-ID
# fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
# plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
# plt.scatter(dotVarEps_Test_OOD, initialT_Test_OOD, marker='o', s=30, c='tab:orange', label='test (OOD)')
# plt.scatter(dotVarEps_Test_ID, initialT_Test_ID, marker='o', s=30, c='tab:green', label='test (ID)')
# plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=3)

# plt.title('Input Distribution', fontsize=24)
# plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
# plt.ylabel(r'$T$ [K]', fontsize=24)
# plt.xscale('log',base=10) 
# # plt.show()
# plt.savefig('TrainTestDistribution-OOD-ID', dpi=300, facecolor='w', edgecolor='w',
#     orientation='portrait', format=None,
#     transparent=False, bbox_inches='tight', pad_inches=0.1,
#     metadata=None)




