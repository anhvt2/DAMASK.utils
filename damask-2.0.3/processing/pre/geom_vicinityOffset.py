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
from scipy import ndimage
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

def taintedNeighborhood(stencil,trigger=[],size=1):

  me = stencil[stencil.shape[0]//2]
  if len(trigger) == 0:
    return np.any(stencil != me)
  if me in trigger:
    trigger = set(trigger)
    trigger.remove(me)
    trigger = list(trigger)
  return np.any(np.in1d(stencil,np.array(trigger)))

#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Offset microstructure index for points which see a microstructure different from themselves
(or listed as triggers) within a given (cubic) vicinity, i.e. within the region close to a grain/phase boundary.

""", version = scriptID)

parser.add_option('-v', '--vicinity',
                  dest = 'vicinity',
                  type = 'int', metavar = 'int',
                  help = 'voxel distance checked for presence of other microstructure [%default]')
parser.add_option('-m', '--microstructureoffset',
                  dest='offset',
                  type = 'int', metavar = 'int',
                  help = 'offset (positive or negative) for tagged microstructure indices. '+
                         '"0" selects maximum microstructure index [%default]')
parser.add_option('-t', '--trigger',
                  action = 'extend', dest = 'trigger', metavar = '<int LIST>',
                  help = 'list of microstructure indices triggering a change')
parser.add_option('-n', '--nonperiodic',
                  dest = 'mode',
                  action = 'store_const', const = 'nearest',
                  help = 'assume geometry to be non-periodic')

parser.set_defaults(vicinity = 1,
                    offset   = 0,
                    trigger  = [],
                    mode     = 'wrap',
                   )

(options, filenames) = parser.parse_args()
options.trigger = np.array(options.trigger, dtype=int)


# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False, labeled = False)
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

# --- read data ------------------------------------------------------------------------------------

  microstructure = table.microstructure_read(info['grid']).reshape(info['grid'],order='F')          # read microstructure

# --- do work ------------------------------------------------------------------------------------

  newInfo = {
             'microstructures': 0,
            }

  if options.offset == 0: options.offset = microstructure.max()

  microstructure = np.where(ndimage.filters.generic_filter(microstructure,
                                                           taintedNeighborhood,
                                                           size=1+2*options.vicinity,mode=options.mode,
                                                           extra_arguments=(),
                                                           extra_keywords={"trigger":options.trigger,"size":1+2*options.vicinity}),
                            microstructure + options.offset,microstructure)

  newInfo['microstructures'] = microstructure.max()

# --- report ---------------------------------------------------------------------------------------

  if (newInfo['microstructures'] != info['microstructures']):
    damask.util.croak('--> microstructures: %i'%newInfo['microstructures'])

# --- write header ---------------------------------------------------------------------------------

  table.labels_clear()
  table.info_clear()
  table.info_append(extra_header+[
    scriptID + ' ' + ' '.join(sys.argv[1:]),
    "grid\ta {grid[0]}\tb {grid[1]}\tc {grid[2]}".format(grid=info['grid']),
    "size\tx {size[0]}\ty {size[1]}\tz {size[2]}".format(size=info['size']),
    "origin\tx {origin[0]}\ty {origin[1]}\tz {origin[2]}".format(origin=info['origin']),
    "homogenization\t{homog}".format(homog=info['homogenization']),
    "microstructures\t{microstructures}".format(microstructures=newInfo['microstructures']),
    ])
  table.head_write()
  
# --- write microstructure information ------------------------------------------------------------

  formatwidth = int(math.floor(math.log10(microstructure.max())+1))
  table.data = microstructure.reshape((info['grid'][0],info['grid'][1]*info['grid'][2]),order='F').transpose()
  table.data_writeArray('%%%ii'%(formatwidth),delimiter = ' ')
    
# --- output finalization --------------------------------------------------------------------------

  table.close()                                                                                     # close ASCII table
