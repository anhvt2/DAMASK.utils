
import numpy as np
import pandas as pd


from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

fileName = '2.00/postProc/main_tension_inc16.txt'

fileHandler = open(fileName)
txt = fileHandler.readlines()
fileHandler.close()

### Pre-process
numHeaderRows = int(txt[0].split('\t')[0])
headers = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=headers)
resolution = 50
for selStr in ['1_pos', '2_pos', '3_pos']:
	df[selStr] -= (resolution/2.)
	df[selStr] /= (resolution)

# df.keys()
stress = df['Mises(Cauchy)']
strain = df['Mises(ln(V))']
X = df[['Mises(Cauchy)', 'Mises(ln(V))']]
X = np.atleast_2d(X)
# X = (X - X.min()) / (X.max() - X.min()) * 2 - 1
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler(feature_range=(-1, 1), copy=True, clip=False).fit_transform(X) # rescale to [-1,1]

### Read ground true & extract info from seedVoid.log - y3d: void true/false in 3d .npy; y: flatten 1d array
y3d = np.load('padded_voidSeeded_2.000pc_spk_dump_12_out.npy')
y3dVoid = (y3d>=2) & (y3d<= 4806) # extract void
y3dVoid = y3dVoid.astype(int)
numTotalVoxels = 120 * 24 * 200 # adjust from seedVoid.log due to padding air to break pbc
numSolidVoxels = 240260
numVoidVoxels = y3dVoid.sum() # 4805
y = np.zeros(numTotalVoxels)
for i in range(numTotalVoxels): # flatten 3d array
	y[i] = y3dVoid[ int(df['1_pos'][i]), int(df['2_pos'][i]), int(df['3_pos'][i]) ]

np.save('y.npy', y)
# y = np.load('y.npy') # quick n' dirty bypass

### Anomaly detection
# https://scikit-learn.org/0.20/auto_examples/plot_anomaly_comparison.html
n_samples = df.shape[0]
outliers_fraction = y3dVoid.sum() / X.shape[0] * 10 # 0.02 # percentage of void: soft parameter (to be studied)
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

anomaly_algorithms = [
	(0, "Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
	(1, "One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
									  gamma=0.1)),
	(2, "Isolation Forest", IsolationForest(contamination=outliers_fraction,
										 random_state=42)),
	(3, "Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))]

fileNames = ['RobustCovariance', 'OneClassSVM', 'IsolationForest', 'LOF']

y_pred = np.zeros([X.shape[0], len(anomaly_algorithms)])
for i, name, algorithm in anomaly_algorithms:
	t0 = time.time()
	algorithm.fit(X)
	t1 = time.time()
	# fit the data and tag outliers
	if name == "Local Outlier Factor":
		y_pred[:,i] = algorithm.fit_predict(X)
	else:
		y_pred[:,i] = algorithm.fit(X).predict(X)

y_pred = y_pred * (-1) / 2. + 0.5
np.save('y_pred.npy', y_pred)
# y_pred = np.load('y_pred.npy') # quick n' dirty bypass

### Plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
for i in range(len(anomaly_algorithms)):
    cm = confusion_matrix(y, y_pred[:,0], labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    plt.show()

### Create a masked anomalous microstructure with the predicted anomaly
'''
phase = 1: air
phase = 2: solid
phase = 3: anomaly
'''
# original microstructure
ms = np.load('padded_voidSeeded_2.000pc_spk_dump_12_out.npy')
ms[ms>=2] = 2 # if not air then assign phase 3 -- solid
for i in range(numTotalVoxels):
    xx = int(df['1_pos'][i])
    yy = int(df['2_pos'][i])
    zz = int(df['3_pos'][i])
    if y[i] == 1:
        ms[xx,yy,zz] = 3 # if detected anomaly, then assign phase 2
    np.save('origVoid' + '.npy', ms)

del(ms)

# anomaly detection microstructure
for j, fileName in zip(range(len(anomaly_algorithms)), fileNames):
    ms = np.load('padded_voidSeeded_2.000pc_spk_dump_12_out.npy')
    ms[ms>=2] = 2 # if not air then assign phase 3 -- solid
    for i in range(numTotalVoxels):
        xx = int(df['1_pos'][i])
        yy = int(df['2_pos'][i])
        zz = int(df['3_pos'][i])
        if y_pred[i] == 1:
            ms[xx,yy,zz] = 3 # if detected anomaly, then assign phase 2
        np.save(fileName + '.npy')
    del(ms)
