
import numpy as np
import argparse
import glob, os, sys, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

loss = []
feasible = []

for folderName in glob.glob('rve*_8x8x8'):
	os.system('python3 computeLossFunction.py --f=%s' % folderName)
	loss += [np.loadtxt(folderName + '/' + 'output.dat')]
	feasible += [np.loadtxt(folderName + '/' + 'feasible.dat')]

loss = np.array(loss)
feasible = np.array(feasible)

avgLoss = np.sum(np.multiply(loss, feasible)) / np.sum(feasible)

print(loss)
print(feasible)
print(avgLoss)
# print(np.sum(np.array(feasible)))

print('\nWriting output.dat in folder: %s \n' % os.getcwd().split('/')[-1])
f = open('output.dat', 'w') # can be 'r', 'w', 'a', 'r+'
f.write('%.8e\n' % avgLoss) # example: 20097.859541889356 -- scale by a factor of 1e3
print('negative average loss = ', avgLoss)
print('Finished writing output.dat in folder: %s' % os.getcwd().split('/')[-1])
f.close()


f = open('feasible.dat', 'w') # can be 'r', 'w', 'a', 'r+'
f.write('%.d\n' % np.any(feasible))
f.close()

