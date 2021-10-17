#!/usr/bin/env python2.7
# -*- coding: UTF-8 no BOM -*-
# Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
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

import os,h5py
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog [dream3dfile[s]]', description = """
Convert DREAM3D file to ASCIItable. Works for 3D datasets, but, hey, its not called DREAM2D ;)

""", version = scriptID)

parser.add_option('-d','--data',
                  dest = 'data',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'data to extract from DREAM3D file')
parser.add_option('-c','--container',
                  dest = 'container', metavar = 'string',
                  help = 'root container(group) in which data is stored [%default]')

parser.set_defaults(container="ImageDataContainer",
                   )

(options, filenames) = parser.parse_args()

if options.data is None:
  parser.error('No data selected')

rootDir ='DataContainers/'+options.container

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: parser.error('no input file specified.')

for name in filenames:
  try:
    table = damask.ASCIItable(outname = os.path.splitext(name)[0]+'.txt',
                              buffered = False
                              )
  except: continue
  damask.util.report(scriptName,name)

  inFile = h5py.File(name, 'r')
  try:
    grid = inFile[rootDir+'/_SIMPL_GEOMETRY/DIMENSIONS'][...]
  except:
    damask.util.croak('Group {} not found'.format(options.container))
    table.close(dismiss = True)
    continue
  
# --- read comments --------------------------------------------------------------------------------
                    
  coords = (np.mgrid[0:grid[2], 0:grid[1], 0: grid[0]]).reshape(3, -1).T                             
  table.data = (np.fliplr(coords)*inFile[rootDir+'/_SIMPL_GEOMETRY/SPACING'][...] \
             + inFile[rootDir+'/_SIMPL_GEOMETRY/ORIGIN'][...] \
             + inFile[rootDir+'/_SIMPL_GEOMETRY/SPACING'][...]*0.5)
  labels = ['1_pos','2_pos','3_pos']
  for data in options.data:
    try:
      l = np.prod(inFile[rootDir+'/CellData/'+data].shape[3:])
      labels+=['{}_{}'.format(i+1,data.replace(' ','')) for i in range(l)] if l >1 else [data.replace(' ','')]
    except KeyError:
      damask.util.croak('Data {} not found'.format(data))
      pass
    table.data = np.hstack((table.data,
                            inFile[rootDir+'/CellData/'+data][...].reshape(grid.prod(),l)))
    
# ------------------------------------------ assemble header ---------------------------------------
  table.labels_clear()
  table.labels_append(labels,reset = True)
  table.head_write()
  
# ------------------------------------------ finalize output ---------------------------------------
  table.data_writeArray()
  table.close()
