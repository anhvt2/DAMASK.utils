
"""
This script 
    (1) takes a SPPARKS input script as an input
    (2) reads it
    (3) plots the (double ellipsoid) melt pool for both melt pool and HAZ 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import random
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

spkInp = 'in.potts_additive_dogbone' # SPPARKS input file

f = open(spkInp)
txt = f.readlines()
f.close()

searchResults = []
for i in range(len(txt)):
    for searchString in ['spotWidth','meltTailLength','meltDepth','capHeight','haz','tailHaz','depthHaz','capHaz']:
        if searchString in txt[i]:
            searchResults += [txt[i].replace('\n','').split()]

# print(searchResults) # debug

def getScalar(parameter, searchResults):
    """
    Return a scalar that matches with a parameter name
    Input
    -----
        parameter: a string
        searchResults: a short search database
    Output
    ------
        scalarParam: output
    """
    scalarParam = None
    # Conduct a search
    for i in range(len(searchResults)):
        tmpSentence = searchResults[i]
        if 'variable' == tmpSentence[0] and parameter == tmpSentence[1]:
            scalarParam = float(tmpSentence[3])
    # Return result
    return scalarParam

spotWidth = getScalar('spotWidth', searchResults) 
meltTailLength = getScalar('meltTailLength', searchResults) 
meltDepth = getScalar('meltDepth', searchResults) 
capHeight = getScalar('capHeight', searchResults) 
haz = getScalar('haz', searchResults) 
tailHaz = getScalar('tailHaz', searchResults) 
depthHaz = getScalar('depthHaz', searchResults) 
capHaz = getScalar('capHaz', searchResults) 

# Diagnostics
print(f'Diagnostics from reading {spkInp}:')
print(f'spotWidth = {spotWidth}')
print(f'meltTailLength = {meltTailLength}')
print(f'meltDepth = {meltDepth}')
print(f'capHeight = {capHeight}')
print(f'haz = {haz}')
print(f'tailHaz = {tailHaz}')
print(f'depthHaz = {depthHaz}')
print(f'capHaz = {capHaz}')
print('')

# Plot


def plotHalfEllipsoid(a,b,c,optLT,ax,optLR='both', alpha=0.9,**kwargs):
    '''
    See DAMASK.utils/test_tensile_dogbone-spk2damask/README.md for more info on parameterization
    * $\phi \in [-\frac{\pi}{2}, \frac{\pi}{2}]$ for leading half-ellipsoid
    * $\phi \in [\frac{\pi}{2}, \frac{3\pi}{2}]$ for trailing half-ellipsoid.
    Note:
    x - along pass direction (capHeight, meltTailLength, see https://spparks.github.io/doc/app_am_ellipsoid.html)
    y - traverse direction (spotWidth)
    z - top direction
    Parameters
    ----------
        a > b > c: major dimensions
        optLT: 'lead' or 'trail' (half ellipsoid) of the x-axis
        optLR: plot only 'left' or 'right' or 'both' of the y-axis
    '''
    Npts = 20
    if optLT == 'lead':
        phi = np.linspace(-1/2*np.pi,1/2*np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
        if optLR == 'left':
            phi = np.linspace(0*np.pi,1/2*np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
        elif optLR == 'right':
            phi = np.linspace(-1/2*np.pi,0*np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
    elif optLT == 'trail':
        phi = np.linspace(1/2*np.pi,3/2*np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
        if optLR == 'left':
            phi = np.linspace(1/2*np.pi,np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
        elif optLR == 'right':
            phi = np.linspace(np.pi,3/2*np.pi, Npts).reshape(Npts, 1) # the angle of the projection in the xy-plane
    else:
        print('vizMeltPool.py: optLT is not valid in plotHalfEllipsoid().')
    theta = np.linspace(0.5*np.pi, np.pi, Npts).reshape(-1, Npts) # the angle from the polar axis, ie the polar angle
    x = a*np.sin(theta)*np.cos(phi)
    y = b*np.sin(theta)*np.sin(phi)
    z = c*np.cos(theta)
    randomColor = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    ax.plot_surface(x, y, z, color=randomColor, alpha=alpha) # randomize colors: https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    return None

fig = plt.figure()  # Square figure
ax = fig.add_subplot(111, projection='3d')
# Plot metling pool
plotHalfEllipsoid(capHeight, spotWidth/2, meltDepth, 'lead', ax, optLR='both', alpha=0.75)
plotHalfEllipsoid(meltTailLength, spotWidth/2, meltDepth, 'trail', ax, optLR='both', alpha=0.75)
# Plot HAZ
plotHalfEllipsoid(capHaz, haz/2, depthHaz, 'lead', ax, optLR='both', alpha=0.5)
plotHalfEllipsoid(tailHaz, haz/2, depthHaz, 'trail', ax, optLR='both', alpha=0.5)

ax.scatter([0],[0],[0], marker='o', s=50, c="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) )

plt.axis('equal')
ax.set_xlabel(r'$x$', fontsize=24)
ax.set_ylabel(r'$y$', fontsize=24)
ax.set_zlabel(r'$z$', fontsize=24)
# # https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.voxels.html#mpl_toolkits.mplot3d.axes3d.Axes3D.voxels
# x,y,z = np.indices((int(tailHaz+capHaz+1), int(haz+1), int(depthHaz+1)))
# x -= int(tailHaz)
# y -= int(haz/2.)
# z -= int(depthHaz)
# ax.voxels(x,y,z, np.full((x.shape[0]-1,x.shape[1]-1,x.shape[2]-1), True), edgecolors=(0.0, 0.0, 0.0, 0.25))
plt.show()

