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

import os,sys
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Rotate vector and/or tensor column data by given angle around given axis.

""", version = scriptID)

parser.add_option('-d', '--data',
                  dest = 'data',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'vector/tensor value(s) label(s)')
parser.add_option('-r', '--rotation',
                  dest = 'rotation',
                  type = 'float', nargs = 4, metavar = ' '.join(['float']*4),
                  help = 'axis and angle to rotate data [%default]')
parser.add_option('--degrees',
                  dest = 'degrees',
                  action = 'store_true',
                  help = 'angles are given in degrees')

parser.set_defaults(rotation = (1.,1.,1.,0),                                                        # no rotation about (1,1,1)
                    degrees = False,
                   )
                    
(options,filenames) = parser.parse_args()

if options.data is None:
  parser.error('no data column specified.')

r = damask.Rotation.fromAxisAngle(np.array(options.rotation),options.degrees,normalise=True)

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()

  errors  = []
  remarks = []
  active  = {'vector':[],'tensor':[]}

  for i,dim in enumerate(table.label_dimension(options.data)):
    label = options.data[i]
    if dim == -1:
      remarks.append('"{}" not found...'.format(label))
    elif dim ==  3:
      remarks.append('adding vector "{}"...'.format(label))
      active['vector'].append(label)
    elif dim ==  9:
      remarks.append('adding tensor "{}"...'.format(label))
      active['tensor'].append(label)

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data ------------------------------------------
  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    for v in active['vector']:
      column = table.label_index(v)
      table.data[column:column+3] = r * np.array(list(map(float,table.data[column:column+3])))
    for t in active['tensor']:
      column = table.label_index(t)
      table.data[column:column+9] = (r * np.array(list(map(float,table.data[column:column+9]))).reshape((3,3))).reshape(9)
      
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
