
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import glob, os
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

from scipy.interpolate import interp1d

def getMetaInfo(StressStrainFile):
    """
    return 
    (1) number of lines for headers 
    (2) list of outputs for pandas dataframe
    """
    fileHandler = open(StressStrainFile)
    txtInStressStrainFile = fileHandler.readlines()
    fileHandler.close()
    numLinesHeader = int(txtInStressStrainFile[0].split('\t')[0])
    fieldsList = txtInStressStrainFile[numLinesHeader].split('\t')
    for i in range(len(fieldsList)):
        fieldsList[i] = fieldsList[i].replace('\n', '')
    print('numLinesHeader = ', numLinesHeader)
    print('fieldsList = ', fieldsList)
    return numLinesHeader, fieldsList

def readLoadFile(LoadFile):
    load_data = np.loadtxt(LoadFile, dtype=str)
    n_fields = len(load_data)
    # assume uniaxial:
    for i in range(n_fields):
        if load_data[i] == 'Fdot' or load_data[i] == 'fdot':
            print('Found *Fdot*!')
            Fdot11 = float(load_data[i+1])
        if load_data[i] == 'time':
            print('Found *totalTime*!')
            totalTime = float(load_data[i+1])
        if load_data[i] == 'incs':
            print('Found *totalIncrement*!')
            totalIncrement = float(load_data[i+1])
        if load_data[i] == 'freq':
            print('Found *freq*!')
            freq = float(load_data[i+1])
    return Fdot11, totalTime, totalIncrement


def getStressStrain(StressStrainFile):
    # d = np.loadtxt(StressStrainFile, skiprows=4)
    numLinesHeader, fieldsList = getMetaInfo(StressStrainFile)
    # d = np.loadtxt(StressStrainFile, skiprows=skiprows)
    d = np.loadtxt(StressStrainFile, skiprows=numLinesHeader+1)
    # df = pd.DataFrame(d, columns=['inc','elem','node','ip','grain','1_pos','2_pos','3_pos','1_f','2_f','3_f','4_f','5_f','6_f','7_f','8_f','9_f','1_p','2_p','3_p','4_p','5_p','6_p','7_p','8_p','9_p'])
    df = pd.DataFrame(d, columns=fieldsList)
    vareps = [0] + list(df['1_ln(V)']) # d[:,1]  # strain -- pad original
    sigma  = [0] + list(df['1_Cauchy']) # d[:,2]  # stress -- pad original
    # vareps = [1] + list(df['1_f']) # d[:,1]  # strain -- pad original
    # sigma  = [0] + list(df['1_p']) # d[:,2]  # stress -- pad original
    # vareps = list(df['Mises(ln(V))'])  # strain -- pad original
    # sigma  = list(df['Mises(Cauchy)']) # stress -- pad original
    _, uniq_idx = np.unique(np.array(vareps), return_index=True)
    vareps = np.array(vareps)[uniq_idx]
    sigma  = np.array(sigma)[uniq_idx]
    x = vareps # '1_ln(V)'
    # x = (vareps - 1) # '1_{f,p}'
    # x = (vareps) # 'Mises()'
    y = sigma / 1e6
    print(x,y)
    return x, y

class HandlerLineImage(HandlerBase):
    # def __init__(self, path, space=15, offset=10 ):
    def __init__(self, path, space=15, offset=-15):
        self.space=space
        self.offset=offset
        self.image_data = plt.imread(path)
        super(HandlerLineImage, self).__init__()
    #
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        l = matplotlib.lines.Line2D([xdescent+self.offset,xdescent+(width-self.space)/3.+self.offset],
                                     [ydescent+height/2., ydescent+height/2.])
        l.update_from(orig_handle)
        l.set_clip_on(False)
        l.set_transform(trans)
        bb = Bbox.from_bounds(xdescent +(width+self.space)/3.+self.offset,
                              ydescent,
                              height*self.image_data.shape[1]/self.image_data.shape[0],
                              height)
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        #
        self.update_prop(image, orig_handle, legend)
        return [l,image]


def Merge(dict1, dict2):
    """ 
    Merging two dictionaries
    https://www.geeksforgeeks.org/python-merging-two-dictionaries/
    """
    res = {**dict1, **dict2}
    return res

# plot
plt.figure(figsize=(4.8,3.2))
lineList = []; empty_list = [];
# for folderName in glob.glob('*/'):
counter = 0
tmpDict = {}
colorList = ['tab:blue', 
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyleList = [
    ('densely dashed',        (0, (5, 1))),
    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 1))),
    ('densely dotted',        (0, (1, 1))),
    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
     ]
# https://matplotlib.org/stable/api/markers_api.html
markerList = ["o",
"s",
"p",
"P",
"*",
"h",
"H",
"+",
"x",
"X",
"D",
"d",
"v",
"^",
"<",
">",
"1",
"2",
"3",
"4",
"8",
]
# 
stress_min = []
stress_max = []
tmp_stress = []
# for i in np.arange(1,10+1):
cellDimList = [4,8,10,16] #,20] # : #,40]: #,80]:
for i in np.arange(1,3+1):
    # for cellDim in [4,8,10,16,20]: #,40]: #,80]:
    for j in range(len(cellDimList)):
        cellDim = cellDimList[j]
        #
        folderName = 'sve%d_%dx%dx%d' % (i, cellDim, cellDim, cellDim)
        StressStrainFile = os.getcwd() + '/' + folderName + '/postProc/stress_strain.log'
        tmpx, tmpy = getStressStrain(StressStrainFile)
        print(len(tmpx))
        print(len(tmpy))
        stress_min += [tmpy]
        stress_max += [tmpy]
        # interp
        tmpx2 = np.linspace(tmpx.min(), tmpx.max(), num=1000)
        interpSpline = interp1d(tmpx, tmpy, kind='cubic', fill_value='extrapolate')
        tmpy2 = interpSpline(tmpx2)
        tmp_stress += [tmpy2]
        # plot
        tmp_line, = plt.plot(tmpx2, tmpy2, marker=markerList[j], markersize=4, linewidth=2.5, linestyle=linestyleList[j][1], color=colorList[i-1])
        lineList += [tmp_line]
        # 
        empty_list += [""]
        imgName = 'cropped_single_phase_equiaxed_%dx%dx%d_%s.png' % (cellDim, cellDim, cellDim, folderName.replace('/', ''))
        # imgName = 'cropped_compareExpComp_Ta_OptValidation_%s.png' % (folderName.replace('/', ''))
        print('%d: %s' % (counter, imgName))
        tmpDict = Merge(tmpDict, {lineList[counter]: HandlerLineImage(imgName)})
        counter += 1

plt.xlabel(r'$\varepsilon$ [-]', fontsize=30)
plt.ylabel(r'$\sigma$ [MPa]', fontsize=30)

## plot experimental dataset
refData = np.loadtxt('../datasets/true_Ta_polycrystal_SS_HCStack_CLC.bak2', skiprows=1)
exp_vareps = refData[:,0] # start at vareps = 0
exp_sigma  = refData[:,1]
max_interp_vareps = np.min([np.max(exp_vareps), 0.6]) 
interp_vareps = np.linspace(0, max_interp_vareps, 1000) # start at vareps = 0

# get interpolated exp. stress
interpSpline_exp = interp1d(exp_vareps, exp_sigma, kind='linear', fill_value='extrapolate')
interp_exp_sigma = interpSpline_exp(interp_vareps)
tmp_line, = plt.plot(interp_vareps, interp_exp_sigma, linestyle='-', marker='o', linewidth=2, color='black')
lineList += [tmp_line]
empty_list += [""]
imgName = 'cropped_exp.eps'
tmpDict = Merge(tmpDict, {lineList[counter]: HandlerLineImage(imgName)})

plt.fill_between(tmpx2, np.min(tmp_stress, axis=0), np.max(tmp_stress, axis=0), alpha=0.5, color='cyan')

# print(np.array(stress_min))
# print(np.array(stress_max))
# stress_min = stress_min.reshape(int(len(stress_min)/len(cellDimList)), int(len(cellDimList)))
# stress_max = stress_max.reshape(int(len(stress_max)/len(cellDimList)), int(len(cellDimList)))
# print(tmpx.shape)
# plt.fill_between(tmpx, np.min(stress_min, axis=1), np.max(stress_max, axis=1), alpha=0.2)
# print(np.array(stress_min).shape)

## alternate legend
leg = plt.legend(lineList, empty_list,
    handler_map=tmpDict,
    handlelength=2.5, labelspacing=0.0, fontsize=36, borderpad=0.15, 
    handletextpad=0.2, borderaxespad=0.15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

plt.xlim(left=0,right=4.239007884860644948e-01)
plt.ylim(bottom=0)
plt.title(r'Comparison of $\sigma-\varepsilon$ b/w exp. and comp.', fontsize=24)

plt.show()
