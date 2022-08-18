

# example syntax: python3 computeLossFunction.py  --f='64x64x64' 
# note: without '/'

import os
import numpy as np
import argparse
import os, sys, datetime
import pandas as pd

parser = argparse.ArgumentParser(description='parse folderName as <str> without /')
parser.add_argument("-folderName", "--folderName", type=str)
args = parser.parse_args()

folderName = args.folderName

refData = np.loadtxt('../datasets/true_Ta_polycrystal_SS_HCStack_CLC.dat', skiprows=1)
compData = np.loadtxt(os.getcwd() + '/' + folderName + '/postProc/single_phase_equiaxed_' + folderName + '_tension.txt', skiprows=3)

df = pd.DataFrame(compData, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
comp_vareps = [1] + list(df['1_f']) # d[:,1] # strain -- pad original
comp_sigma  = [0] + list(df['1_p']) # d[:,2] # stress -- pad original
_, uniq_idx = np.unique(np.array(comp_vareps), return_index=True)
comp_vareps = np.array(comp_vareps)[uniq_idx]
comp_sigma  = np.array(comp_sigma)[uniq_idx]



