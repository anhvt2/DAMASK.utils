
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

folders = natsorted(glob.glob('*/'))
# porosityInfo = np.loadtxt('porosity.txt')

porosities = []
# for folder in folders:
#     tmp = np.loadtxt(f'{folder}/porosity.txt')
#     porosities += [tmp]

# porosities = np.array(porosities) # Global, Local, Target

for folder in folders:
    tmp = np.loadtxt(f'{folder}/porosity.txt', dtype=float)
    porosities.append({
        'global': tmp[0], 
        'local': tmp[1], 
        'target': tmp[2], 
        })

df = pd.DataFrame(porosities)

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
g = sns.jointplot(data=df, x="global", y="local", kind='reg', 
    scatter_kws={'s': 3},
    joint_kws={'color': 'tab:gray'},
    marginal_kws={
    'bins': 23,
    'fill': True,    # Fill the area under the KDE curve
    'color': 'tab:gray',  # Set the color of the KDE curve
    'alpha': 0.2,    # Set the transparency of the filled area
    'linewidth': 2   # Set the line width of the KDE curve)
    },
    )

# g = sns.jointplot(data=df, x="global", y="local", kind='scatter', color='blue')
# g = sns.jointplot(data=df, x="global", y="local", kind='scatter', color='tab:blue', 
#     alpha=0.8,
#     marginal_kws=dict(bins=50, fill=True)
#     )
#     # marginal_kws={'fill': True, 'color': 'blue', 'alpha': 0.6, 'linewidth': 2})
# g.plot_marginals(sns.kdeplot, fill=True, color='tab:blue', alpha=0.8, linewidth=1, cut=0)

# g = sns.jointplot(
#     data=df, x="global", y="local", kind='resid', color='tab:blue', alpha=0.8
# )

# g.plot_marginals(sns.kdeplot, fill=True, color='tab:blue', alpha=0.8, linewidth=1, cut=0)


# sns.kdeplot(data=df, x="global", y="local", ax=g.ax_joint, color='blue', fill=True, alpha=0.8, linewidth=1)

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
plt.xlim(left=0,right=0.05)
plt.ylim(bottom=0,top=0.05)
plt.savefig('JointplotPorosities.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', format=None, transparent=False, bbox_inches='tight', pad_inches=0.1, metadata=None)

