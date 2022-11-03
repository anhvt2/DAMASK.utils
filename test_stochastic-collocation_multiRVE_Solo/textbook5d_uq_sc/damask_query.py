#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
adopted from dakota/6.15/dakota-6.15.0-public-src-cli/dakota-examples/official/global_sensitivity/Ishigami.py
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from io import open

from math import sin
import sys
import numpy as np

params_file, output_file = sys.argv[1:]

# print('%s' % params_file)

i_ = np.empty(5)
with open(params_file,'rt',encoding='utf8') as FF:
	d = FF.readlines() # Skip first line
	# print(len(d))
	# print(FF.readline())
	i_[0] = float(d[1].split()[0])
	i_[1] = float(d[2].split()[0])
	i_[2] = float(d[3].split()[0])
	i_[3] = float(d[4].split()[0])
	i_[4] = float(d[5].split()[0])

# print(i_)

# required
inputData  = np.loadtxt('dakota_sparse_tabular_template.dat',skiprows=1)[:,2:]
outputData = np.loadtxt('output.dat',delimiter=',')

def searchIndex(i_, inputData):
	n, d = inputData.shape
	tol = 1e-8
	index_ = np.where(np.linalg.norm(inputData - i_, axis=1) < tol)[0][0]
	return index_

index_ = searchIndex(i_, inputData)
o_ = outputData[index_, 0] # change the second index accordingly: 0 = strainYield, 1 = stressYield

print('debug:')
print(i_)
print(inputData[index_, :])
print(o_)
print(index_)

outFile = open(output_file, 'w')
outFile.write('%.12e' % (o_))
outFile.close()

