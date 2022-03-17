
import numpy as np

o = np.loadtxt('output.dat', delimiter=',')

dakotaTabFile = open('dakota_sparse_tabular.dat')
txt = dakotaTabFile.readlines()
dakotaTabFile.close()

tabFileInNp = np.loadtxt('dakota_sparse_tabular.dat',skiprows=1)
n = tabFileInNp.shape[0] # number of observations
x = tabFileInNp[:,2:] # read inputs (without weights)
d = x.shape[1] # dimensionality of the problem


### strain

outFile1 = open('dakota_sparse_tabular_strainYield.dat', 'w') # can be 'r', 'w', 'a', 'r+'
outFile1.write('{:8s}  {:10s}    {:16s}  {:16s}  {:16s}  {:16s}  {:16s}  {:16s}  {:16s}{:16s}\n' .format('%eval_id', 'interface', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'response_fn_1'))
for i in range(n):
	outFile1.write('{:8d}  {:10s}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12e}\n' .format(i+1, 'NO_ID', x[i,0], x[i,1], x[i,2], x[i,3], x[i,4], x[i,5], x[i,6], o[i,0]))
outFile1.close()
# clean up 0s
outFile1 = open('dakota_sparse_tabular_strainYield.dat')
t = outFile1.readlines()
outFile1.close()
for i in range(len(t)):
	t[i] = t[i].replace('0.000000000000', '             0')
outFile1 = open('dakota_sparse_tabular_strainYield.dat', 'w')
for i in range(len(t)):
	outFile1.write(t[i])
outFile1.close()

### stress

outFile2 = open('dakota_sparse_tabular_stressYield.dat', 'w') # can be 'r', 'w', 'a', 'r+'
outFile2.write('{:8s}  {:10s}    {:16s}  {:16s}  {:16s}  {:16s}  {:16s}  {:16s}  {:16s}{:16s}\n' .format('%eval_id', 'interface', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'response_fn_1'))
for i in range(n):
	outFile2.write('{:8d}  {:10s}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12f}  {:16.12e}\n' .format(i+1, 'NO_ID', x[i,0], x[i,1], x[i,2], x[i,3], x[i,4], x[i,5], x[i,6], o[i,1]))
outFile2.close()
# clean up 0s
outFile2 = open('dakota_sparse_tabular_stressYield.dat')
t = outFile2.readlines()
outFile2.close()
for i in range(len(t)):
	t[i] = t[i].replace('0.000000000000', '             0')
outFile2 = open('dakota_sparse_tabular_stressYield.dat', 'w')
for i in range(len(t)):
	outFile2.write(t[i])
outFile2.close()

# https://stackoverflow.com/questions/10837017/how-do-i-make-a-fixed-size-formatted-string-in-python
# print('{:10s} {:3d}  {:7.2f}'.format('xxx', 123, 98))
# print('{:10s} {:3d}  {:7.2f}'.format('yyyy', 3, 1.0))
# print('{:10s} {:3d}  {:7.2f}'.format('zz', 42, 123.34))



