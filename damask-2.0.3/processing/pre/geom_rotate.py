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
import damask
from scipy import ndimage
from optparse import OptionParser

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [geomfile(s)]', description = """
Rotates spectral geometry description.

""", version=scriptID)

parser.add_option('-r', '--rotation',
                  dest='rotation',
                  type = 'float', nargs = 4, metavar = ' '.join(['float']*4),
                  help = 'rotation given as angle and axis')
parser.add_option('-e', '--eulers',
                  dest = 'eulers',
                  type = 'float', nargs = 3, metavar = ' '.join(['float']*3),
                  help = 'rotation given as Euler angles')
parser.add_option('-d', '--degrees',
                  dest = 'degrees',
                  action = 'store_true',
                  help = 'Euler angles are given in degrees [%default]')
parser.add_option('-m', '--matrix',
                  dest = 'matrix',
                  type = 'float', nargs = 9, metavar = ' '.join(['float']*9),
                  help = 'rotation given as matrix')
parser.add_option('-q', '--quaternion',
                  dest = 'quaternion',
                  type = 'float', nargs = 4, metavar = ' '.join(['float']*4),
                  help = 'rotation given as quaternion')
parser.add_option('-f', '--fill',
                  dest = 'fill',
                  type = 'int', metavar = 'int',
                  help = 'background grain index. "0" selects maximum microstructure index + 1 [%default]')

parser.set_defaults(degrees = False,
                    fill = 0)

(options, filenames) = parser.parse_args()

if sum(x is not None for x in [options.rotation,options.eulers,options.matrix,options.quaternion]) != 1:
  parser.error('not exactly one rotation specified...')

if options.quaternion is not None:
  eulers = damask.Rotation.fromQuaternion(np.array(options.quaternion)).asEulers(degrees=True)
if options.rotation is not None:
  eulers = damask.Rotation.fromAxisAngle(np.array(options.rotation,degrees=True)).asEulers(degrees=True)
if options.matrix is not None:
  eulers = damask.Rotation.fromMatrix(np.array(options.Matrix)).asEulers(degrees=True)
if options.eulers is not None:
  eulers = damask.Rotation.fromEulers(np.array(options.eulers),degrees=True).asEulers(degrees=True)

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False,
                              labeled = False)
  except: continue
  damask.util.report(scriptName,name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()
  info,extra_header = table.head_getGeom()

  damask.util.croak(['grid     a b c:  {}'.format(' x '.join(map(str,info['grid']))),
                     'size     x y z:  {}'.format(' x '.join(map(str,info['size']))),
                     'origin   x y z:  {}'.format(' : '.join(map(str,info['origin']))),
                     'homogenization:  {}'.format(info['homogenization']),
                     'microstructures: {}'.format(info['microstructures']),
                    ])

  errors = []
  if np.any(info['grid'] < 1):    errors.append('invalid grid a b c.')
  if np.any(info['size'] <= 0.0): errors.append('invalid size x y z.')
  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# --- read data ------------------------------------------------------------------------------------

  microstructure = table.microstructure_read(info['grid']).reshape(info['grid'],order='F')                    # read microstructure

  newGrainID = options.fill if options.fill != 0 else microstructure.max()+1
  microstructure = ndimage.rotate(microstructure,eulers[2],(0,1),order=0,prefilter=False,output=int,cval=newGrainID) # rotation around Z
  microstructure = ndimage.rotate(microstructure,eulers[1],(1,2),order=0,prefilter=False,output=int,cval=newGrainID) # rotation around X
  microstructure = ndimage.rotate(microstructure,eulers[0],(0,1),order=0,prefilter=False,output=int,cval=newGrainID) # rotation around Z

# --- do work ------------------------------------------------------------------------------------

  newInfo = {
             'size':   microstructure.shape*info['size']/info['grid'],
             'grid':   microstructure.shape,
             'microstructures': microstructure.max(),
            }


# --- report ---------------------------------------------------------------------------------------

  remarks = []
  if (any(newInfo['grid']            != info['grid'])):
    remarks.append('--> grid    a b c:  %s'%(' x '.join(map(str,newInfo['grid']))))
  if (any(newInfo['size']            != info['size'])):
    remarks.append('--> size    x y z:  %s'%(' x '.join(map(str,newInfo['size']))))
  if (    newInfo['microstructures'] != info['microstructures']): 
    remarks.append('--> microstructures: %i'%newInfo['microstructures'])
  if remarks != []: damask.util.croak(remarks)

# --- write header ---------------------------------------------------------------------------------

  table.labels_clear()
  table.info_clear()
  table.info_append(extra_header+[
    scriptID + ' ' + ' '.join(sys.argv[1:]),
    "grid\ta {grid[0]}\tb {grid[1]}\tc {grid[2]}".format(grid=newInfo['grid']),
    "size\tx {size[0]}\ty {size[1]}\tz {size[2]}".format(size=newInfo['size']),
    "origin\tx {origin[0]}\ty {origin[1]}\tz {origin[2]}".format(origin=info['origin']),
    "homogenization\t{homog}".format(homog=info['homogenization']),
    "microstructures\t{microstructures}".format(microstructures=newInfo['microstructures']),
    ])
  table.head_write()

# --- write microstructure information ------------------------------------------------------------

  formatwidth = int(math.floor(math.log10(microstructure.max())+1))
  table.data = microstructure.reshape((newInfo['grid'][0],np.prod(newInfo['grid'][1:])),order='F').transpose()
  table.data_writeArray('%%%ii'%(formatwidth),delimiter = ' ')

# --- output finalization --------------------------------------------------------------------------

  table.close()                                                                                     # close ASCII table
