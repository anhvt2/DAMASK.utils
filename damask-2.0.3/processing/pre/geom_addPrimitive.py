#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-
# Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os,sys,math
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------
identifiers = {
        'grid':   ['a','b','c'],
        'size':   ['x','y','z'],
        'origin': ['x','y','z'],
          }
mappings = {
        'grid':            lambda x: int(x),
        'size':            lambda x: float(x),
        'origin':          lambda x: float(x),
        'homogenization':  lambda x: int(x),
        'microstructures': lambda x: int(x),
          }

parser = OptionParser(option_class=damask.extendableOption, usage='%prog option [geomfile(s)]', description = """
Positions a geometric object within the (three-dimensional) canvas of a spectral geometry description.
Depending on the sign of the dimension parameters, these objects can be boxes, cylinders, or ellipsoids.

""", version = scriptID)

parser.add_option('-c', '--center',     dest='center', 
                  type='float', nargs = 3, metavar=' '.join(['float']*3),
                  help='a,b,c origin of primitive %default')
parser.add_option('-d', '--dimension',  dest='dimension',
                  type='float', nargs = 3, metavar=' '.join(['float']*3),
                  help='a,b,c extension of hexahedral box; negative values are diameters')
parser.add_option('-e', '--exponent',  dest='exponent',
                  type='float', nargs = 3, metavar=' '.join(['float']*3),
                  help='i,j,k exponents for axes - 0 gives octahedron (|x|^(2^0) + |y|^(2^0) + |z|^(2^0) < 1), \
                  1 gives a sphere (|x|^(2^1) + |y|^(2^1) + |z|^(2^1) < 1), \
                  large values produce boxes, negative turns concave.')
parser.add_option('-f', '--fill',       dest='fill',
                  type='int', metavar = 'int',
                  help='grain index to fill primitive. "0" selects maximum microstructure index + 1 [%default]')
parser.add_option('-q', '--quaternion', dest='quaternion',
                  type='float', nargs = 4, metavar=' '.join(['float']*4),
                  help = 'rotation of primitive as quaternion')
parser.add_option('-a', '--angleaxis',  dest='angleaxis', type=float,
                  nargs = 4, metavar=' '.join(['float']*4),
                  help = 'axis and angle to rotate primitive')
parser.add_option(     '--degrees',     dest='degrees',
                  action='store_true',
                  help = 'angle is given in degrees [%default]')
parser.add_option(     '--nonperiodic', dest='periodic',
                  action='store_false',
                  help = 'wrap around edges [%default]')
parser.add_option(     '--realspace',  dest='realspace',
                  action='store_true',
                  help = '-c and -d span [origin,origin+size] instead of [0,grid] coordinates')
parser.set_defaults(center = (.0,.0,.0),
                    fill = 0,
                    degrees = False,
                    exponent = (20,20,20), # box shape by default
                    periodic = True,
                    realspace = False,
                   )

(options, filenames) = parser.parse_args()
if options.dimension is None:
  parser.error('no dimension specified.')   
if options.angleaxis is not None:
  rotation = damask.Rotation.fromAxisAngle(np.array(options.angleaxis),options.degrees,normalise=True)
elif options.quaternion is not None:
  rotation = damask.Rotation.fromQuaternion(options.quaternion)
else:
  rotation = damask.Rotation()

options.center = np.array(options.center)
options.dimension = np.array(options.dimension)
# undo logarithmic sense of exponent and generate ellipsoids for negative dimensions (backward compatibility)
options.exponent = np.where(np.array(options.dimension) > 0, np.power(2,options.exponent), 2)

# --- loop over input files -------------------------------------------------------------------------
if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False,
                                    labeled = False)
  except: continue
  damask.util.report(scriptName,name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()
  info,extra_header = table.head_getGeom()

  damask.util.croak(['grid     a b c:  %s'%(' x '.join(map(str,info['grid']))),
                     'size     x y z:  %s'%(' x '.join(map(str,info['size']))),
                     'origin   x y z:  %s'%(' : '.join(map(str,info['origin']))),
                     'homogenization:  %i'%info['homogenization'],
                     'microstructures: %i'%info['microstructures'],
              ])

  errors = []
  if np.any(info['grid'] < 1):    errors.append('invalid grid a b c.')
  if np.any(info['size'] <= 0.0): errors.append('invalid size x y z.')
  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

#--- read data ------------------------------------------------------------------------------------

  microstructure = table.microstructure_read(info['grid'])                                          # read microstructure

# --- do work ------------------------------------------------------------------------------------

  newInfo = {
             'microstructures': 0,
            }

  options.fill = microstructure.max()+1 if options.fill == 0 else options.fill
  
  microstructure = microstructure.reshape(info['grid'],order='F')
  
  # coordinates given in real space (default) vs voxel space
  if options.realspace:
    options.center    -= info['origin']
    options.center    *= np.array(info['grid']) / np.array(info['size'])
    options.dimension *= np.array(info['grid']) / np.array(info['size'])

  grid = microstructure.shape  

  # change to coordinate space where the primitive is the unit sphere/cube/etc
  if options.periodic: # use padding to achieve periodicity
    (X, Y, Z) = np.meshgrid(np.arange(-grid[0]/2, (3*grid[0])/2, dtype=np.float32), # 50% padding on each side 
                            np.arange(-grid[1]/2, (3*grid[1])/2, dtype=np.float32), 
                            np.arange(-grid[2]/2, (3*grid[2])/2, dtype=np.float32),
                            indexing='ij')
    # Padding handling
    X = np.roll(np.roll(np.roll(X,
            -grid[0]//2, axis=0), 
            -grid[1]//2, axis=1), 
            -grid[2]//2, axis=2)
    Y = np.roll(np.roll(np.roll(Y,
            -grid[0]//2, axis=0), 
            -grid[1]//2, axis=1), 
            -grid[2]//2, axis=2)
    Z = np.roll(np.roll(np.roll(Z,
            -grid[0]//2, axis=0), 
            -grid[1]//2, axis=1), 
            -grid[2]//2, axis=2)
  else: # nonperiodic, much lighter on resources
    # change to coordinate space where the primitive is the unit sphere/cube/etc
    (X, Y, Z) = np.meshgrid(np.arange(0, grid[0], dtype=np.float32), 
                            np.arange(0, grid[1], dtype=np.float32), 
                            np.arange(0, grid[2], dtype=np.float32),
                            indexing='ij')
    
  # first by translating the center onto 0, 0.5 shifts the voxel origin onto the center of the voxel
  X -= options.center[0] - 0.5
  Y -= options.center[1] - 0.5
  Z -= options.center[2] - 0.5
  # and then by applying the rotation
  (X, Y, Z) = rotation * (X, Y, Z)
  # and finally by scaling (we don't worry about options.dimension being negative, np.abs occurs on the microstructure = np.where... line)
  X /= options.dimension[0] * 0.5
  Y /= options.dimension[1] * 0.5
  Z /= options.dimension[2] * 0.5
    

 # High exponents can cause underflow & overflow - loss of precision is okay here, we just compare it to 1, so +infinity and 0 are fine
  old_settings = np.seterr()
  np.seterr(over='ignore', under='ignore')
  
  if options.periodic: # use padding to achieve periodicity
    inside = np.zeros(grid, dtype=bool)
    for i in range(2):
      for j in range(2):
        for k in range(2):
          inside = inside | ( # Most of this is handling the padding
                np.abs(X[grid[0] * i : grid[0] * (i+1),
                         grid[1] * j : grid[1] * (j+1),
                         grid[2] * k : grid[2] * (k+1)])**options.exponent[0] +
                np.abs(Y[grid[0] * i : grid[0] * (i+1),
                         grid[1] * j : grid[1] * (j+1),
                         grid[2] * k : grid[2] * (k+1)])**options.exponent[1] +
                np.abs(Z[grid[0] * i : grid[0] * (i+1),
                         grid[1] * j : grid[1] * (j+1),
                         grid[2] * k : grid[2] * (k+1)])**options.exponent[2] <= 1.0)
    
    microstructure = np.where(inside, options.fill, microstructure)   

  else: # nonperiodic, much lighter on resources
    microstructure = np.where(np.abs(X)**options.exponent[0] +
                              np.abs(Y)**options.exponent[1] +
                              np.abs(Z)**options.exponent[2] <= 1.0, options.fill, microstructure)

  np.seterr(**old_settings) # Reset warnings to old state
  newInfo['microstructures'] = microstructure.max()

# --- report ---------------------------------------------------------------------------------------
  if (newInfo['microstructures'] != info['microstructures']):
    damask.util.croak('--> microstructures: %i'%newInfo['microstructures'])


#--- write header ---------------------------------------------------------------------------------

  table.info_clear()
  table.info_append(extra_header+[
    scriptID + ' ' + ' '.join(sys.argv[1:]),
    "grid\ta {}\tb {}\tc {}".format(*info['grid']),
    "size\tx {}\ty {}\tz {}".format(*info['size']),
    "origin\tx {}\ty {}\tz {}".format(*info['origin']),
    "homogenization\t{}".format(info['homogenization']),
    "microstructures\t{}".format(newInfo['microstructures']),
    ])
  table.labels_clear()
  table.head_write()
  table.output_flush()

# --- write microstructure information ------------------------------------------------------------

  formatwidth = int(math.floor(math.log10(microstructure.max())+1))
  table.data = microstructure.reshape((info['grid'][0],info['grid'][1]*info['grid'][2]),order='F').transpose()
  table.data_writeArray('%%%ii'%(formatwidth),delimiter = ' ')

#--- output finalization --------------------------------------------------------------------------

  table.close()
