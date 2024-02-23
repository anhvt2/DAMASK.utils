
import numpy as np
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt

fileName = 'main_tension.txt'

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
    # df[selStr] -= (resolution/2.)
    df[selStr] /= (resolution)
