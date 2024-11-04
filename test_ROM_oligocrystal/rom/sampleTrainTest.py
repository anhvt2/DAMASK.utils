import pyvista
import matplotlib.pyplot as plt
import glob, os
import numpy as np
import matplotlib as mpl
np.random.seed(8)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

d = np.loadtxt('control.log', skiprows=1, delimiter=',')
dotVarEps = d[:,1]
loadingTime = d[:,2] # dependent - not an input
initialT = d[:,3]

# Define train/test fraction
trainFraction = 0.3
testFraction  = 1 - trainFraction
numDataPts = d.shape[0]
numTrainPts = int(numDataPts * trainFraction)
numTestPts = numDataPts - numTrainPts

# Sample train/test datasets: OOD first, then ID, then train indices
TestIdxOOD = [] # test indices out-of-distribution
for i in range(numDataPts):
    if (np.log10(dotVarEps[i]) < -3.5 or np.log10(dotVarEps[i]) > 1.5) or (initialT[i] < 350 or initialT[i] > 1000):
        TestIdxOOD += [i]

TestIdxOOD = np.sort(np.array(TestIdxOOD))

TestIdxID = [] # test indices in-distribution
TestIdxID = np.random.choice( np.setdiff1d(np.arange(numDataPts), TestIdxOOD), 
    size=int(testFraction*numDataPts - len(TestIdxOOD)), 
    replace=False)

TestIdx = np.sort(np.union1d(TestIdxID, TestIdxOOD))
TrainIdx = np.setdiff1d(np.arange(numDataPts), TestIdx)

TrainData   = d[TrainIdx]
TestData    = d[TestIdx]
TestDataOOD = d[TestIdxOOD]
TestDataID  = d[TestIdxID]

dotVarEps_Train, initialT_Train = TrainData[:,1], TrainData[:,3]
dotVarEps_Test, initialT_Test = TestData[:,1], TestData[:,3]
dotVarEps_Test_OOD, initialT_Test_OOD = TestDataOOD[:,1], TestDataOOD[:,3]
dotVarEps_Test_ID,  initialT_Test_ID  = TestDataID[:,1], TestDataID[:,3]

# Save train/test datasets
np.savetxt('TrainData.dat', TrainData, 
    fmt='%d, %.8e, %.8e, %.1f', 
    header='i, dotVareps, loadingTime, initialT',
    comments='')
np.savetxt('TrainIdx.dat', TrainIdx, fmt='%d')
np.savetxt('TestData.dat',  TestData,  
    fmt='%d, %.8e, %.8e, %.1f', 
    header='i, dotVareps, loadingTime, initialT',
    comments='')
np.savetxt('TestIdx.dat', TestIdx, fmt='%d')
np.savetxt('TestDataOOD.dat',  TestDataOOD,  
    fmt='%d, %.8e, %.8e, %.1f', 
    header='i, dotVareps, loadingTime, initialT',
    comments='')
np.savetxt('TestIdxOOD.dat', TestIdxOOD, fmt='%d')
np.savetxt('TestDataID.dat',  TestDataID,  
    fmt='%d, %.8e, %.8e, %.1f', 
    header='i, dotVareps, loadingTime, initialT',
    comments='')
np.savetxt('TestIdxID.dat', TestIdxID, fmt='%d')

IncompleteIdxType = []
if os.path.exists('IncompleteIdx.dat'):
    IncompleteIdx = np.loadtxt('IncompleteIdx.dat', dtype=int)
    print(f'Detected existing IncompleteIdx.dat.')
    print(f'Indices in IncompleteIdx.dat will be removed.')
    for i in IncompleteIdx:
        if i in TrainIdx:
            IncompleteIdxType += ['Train']
        elif i in TestIdxOOD:
            IncompleteIdxType += ['TestOOD']
        elif i in TestIdxID:
            IncompleteIdxType += ['TestID']
        else:
            print(f'Index {i} does not belong to any dataset! Check!')
    TrainIdx   = np.setdiff1d(TrainIdx, IncompleteIdx)
    TestIdx    = np.setdiff1d(TestIdx, IncompleteIdx)
    TestIdxOOD = np.setdiff1d(TestIdxOOD, IncompleteIdx)
    TestIdxID  = np.setdiff1d(TestIdxID, IncompleteIdx)

# Plot: train/test
fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
plt.scatter(dotVarEps_Test, initialT_Test, marker='o', s=30, c='tab:orange', label='test')
plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=True, markerscale=3)

plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T$ [K]', fontsize=24)
plt.xscale('log',base=10) 
# plt.show()
plt.savefig('TrainTestDistribution', dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', format=None,
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)

# Plot: train/test-OOD/test-ID
fig = plt.figure(num=None, figsize=(14, 12), dpi=300, facecolor='w', edgecolor='k')
plt.scatter(dotVarEps_Train, initialT_Train, marker='o', s=30, c='tab:blue', label='train')
plt.scatter(dotVarEps_Test_OOD, initialT_Test_OOD, marker='o', s=30, c='tab:orange', label='test (OOD)')
plt.scatter(dotVarEps_Test_ID, initialT_Test_ID, marker='o', s=30, c='tab:green', label='test (ID)')
plt.legend(fontsize=24, loc='upper left', bbox_to_anchor=(1.05, 1.0),frameon=False, markerscale=3)

plt.title('Input Distribution', fontsize=24)
plt.xlabel(r'$\dot{\varepsilon}$ [s$^{-1}$]', fontsize=24)
plt.ylabel(r'$T$ [K]', fontsize=24)
plt.xscale('log',base=10) 
# plt.show()
plt.savefig('TrainTestDistribution-OOD-ID', dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', format=None,
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)

# Diagnostics
print(f'Number of train points: {TrainIdx.shape[0]:<d}')
print(f'Number of test points: {TestIdx.shape[0]:<d}')
print(f'Number of test points (out-of-distribution or OOD): {TestIdxOOD.shape[0]:<d}')
print(f'Number of test points (in-distribution or ID): {TestIdxID.shape[0]:<d}')


