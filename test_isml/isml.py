
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

numHeaderRows = int(txt[0].split('\t')[0])
headers = txt[numHeaderRows].replace('\n', '').split('\t')
data = np.loadtxt(fileName, skiprows=numHeaderRows+1)
df = pd.DataFrame(data, columns=headers)

# https://scikit-learn.org/0.20/auto_examples/plot_anomaly_comparison.html

n_samples = df.shape[0]
outliers_fraction = 0.02 # percentage of void: soft parameter (to be studied)
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
anomaly_algorithms = [
	("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
	("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
									  gamma=0.1)),
	("Isolation Forest", IsolationForest(contamination=outliers_fraction,
										 random_state=42)),
	("Local Outlier Factor", LocalOutlierFactor(
		n_neighbors=35, contamination=outliers_fraction))]

resolution = 50
for selStr in ['1_pos', '2_pos', '3_pos']:
	df[selStr] -= (resolution/2.)
	df[selStr] /= (resolution)

# df.keys()
stress = df['Mises(Cauchy)']
strain = df['Mises(ln(V))']

# Add outliers
X = np.concatenate([X, rng.uniform(low=-6, high=6,
				   size=(n_outliers, 2))], axis=0)

for name, algorithm in anomaly_algorithms:
	t0 = time.time()
	algorithm.fit(X)
	t1 = time.time()
	plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
	if i_dataset == 0:
		plt.title(name, size=18)

	# fit the data and tag outliers
	if name == "Local Outlier Factor":
		y_pred = algorithm.fit_predict(X)
	else:
		y_pred = algorithm.fit(X).predict(X)

	# plot the levels lines and the points
	if name != "Local Outlier Factor":  # LOF does not implement predict
		Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
