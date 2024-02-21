
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
y3d = (y3d>=2) & (y3d<= 4806) # extract void
y3d = y3d.astype(int)
numTotalVoxels = 120 * 24 * 200 # adjust from seedVoid.log due to padding air to break pbc
numSolidVoxels = 240260
numVoidVoxels = y3d.sum() # 4805
y = np.zeros(numTotalVoxels)
for i in range(numTotalVoxels): # flatten 3d array
	y[i] = y3d[ int(df['1_pos'][i]), int(df['2_pos'][i]), int(df['3_pos'][i]) ]

np.save('y.npy', y)
# y = np.load('y.npy') # quick n' dirty bypass

### Anomaly detection
# https://scikit-learn.org/0.20/auto_examples/plot_anomaly_comparison.html
n_samples = df.shape[0]
outliers_fraction = y3d.sum() / X.shape[0] # 0.02 # percentage of void: soft parameter (to be studied)
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

anomaly_algorithms = [
	(0, "Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
	(1, "One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
									  gamma=0.1)),
	(2, "Isolation Forest", IsolationForest(contamination=outliers_fraction,
										 random_state=42)),
	(3, "Local Outlier Factor", LocalOutlierFactor(
		n_neighbors=35, contamination=outliers_fraction))]

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

