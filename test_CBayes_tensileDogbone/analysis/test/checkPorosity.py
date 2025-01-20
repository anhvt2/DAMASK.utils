
import numpy as np
from scipy.stats import norm # The standard Normal distribution
from scipy.stats import gaussian_kde as GKDE # A standard kernel density estimator
from natsort import natsorted, ns # natural-sort
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys, time, glob
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

testIdxs = [20,90,180,437]
labels = ['A','B','C','D']
folders = natsorted(glob.glob('*/'))
porosityInfo = np.loadtxt('porosity.txt')

porosities = []
# for folder in folders:
#     tmp = np.loadtxt(f'{folder}/porosity.txt')
#     porosities += [tmp]

# porosities = np.array(porosities) # Global, Local, Target

for folder in folders:
    tmp = np.loadtxt(f'{folder}/porosity.txt', dtype=float)
    testIdx = int(folder.split('-')[2])
    label = labels[np.where(np.array(testIdxs) == testIdx)[0][0]]
    porosities.append({
        'global': tmp[0], 
        'local': tmp[1], 
        'target': tmp[2], 
        'group': label,
        })

df = pd.DataFrame(porosities)
tmp = []
for index, row in df.iterrows():
    meanPoro = df[df['group'] == df.iloc[index]['group']]['global'].mean() * 100
    tmp += [meanPoro] # df[df['group'] == df.iloc[index]['group']]['local'].mean()

df[r'avg $\phi$'] = tmp
df[r'avg $\phi$'] = df[r'avg $\phi$'].map(lambda x: f"{x:.2f}%")

fig = plt.figure(num=None, figsize=(14, 14), dpi=300, facecolor='w', edgecolor='k')
plt.plot(df['global'], df['local'], 'bo', ms=10)
plt.xlabel(r'global $\phi$', fontsize=24)
plt.ylabel(r'local (gauge) $\phi$', fontsize=24)
plt.title(r'QoI porosity $\phi$ measured globally and locally', fontsize=24)
# plt.axis('equal')
plt.xlim(left=0,right=0.06)
plt.ylim(bottom=0,top=0.06)
# plt.show()
plt.savefig('porosities.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

# g = sns.JointGrid(data=df, x='global', y='local',)
# g.plot_joint(sns.scatterplot)

# for k in df.kind.unique():
#     data = df[df.kind == k]
#     sns.kdeplot(x=data.x, fill=True, label=k, ax=g.ax_marg_x)
#     sns.kdeplot(y=data.y, fill=True, label=k, ax=g.ax_marg_y)

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
plt.figure(num=None, figsize=(168, 100), dpi=300, facecolor='w', edgecolor='k')
sns.jointplot(data=df, x="global", y="local", hue=r'avg $\phi$', 
    marginal_kws={
    'fill': True,    # Fill the area under the KDE curve
    'color': 'purple',  # Set the color of the KDE curve
    'alpha': 0.6,    # Set the transparency of the filled area
    'linewidth': 2   # Set the line width of the KDE curve)
    })

# sns.jointplot(data=df, x="global", y="local", kind='scatter', 
#     marginal_kind='kde',
#     marginal_kws={
#         'fill': True,    # Fill the area under the KDE curve
#         'color': 'purple',  # Set the color of the KDE curve
#         'alpha': 0.6,    # Set the transparency of the filled area
#         'linewidth': 2   # Set the line width of the KDE curve
#     })
plt.xlabel(r'global $\phi$', fontsize=12)
plt.ylabel(r'local (gauge) $\phi$', fontsize=12)
plt.savefig('JointplotPorosities.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)
