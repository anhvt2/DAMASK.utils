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
Blows up each value to a surrounding data block of size 'packing' thus increasing the former resolution
to resolution*packing.

""", version = scriptID)

parser.add_option('-c','--coordinates',
                  dest = 'pos', metavar = 'string',
                  help = 'column label of coordinates [%default]')
parser.add_option('-p','--packing',
                  dest = 'packing', type = 'int', nargs = 3, metavar = 'int int int',
                  help = 'dimension of packed group [%default]')
parser.add_option('-g','--grid',
                  dest = 'resolution', type = 'int', nargs = 3, metavar = 'int int int',
                  help = 'grid in x,y,z (optional)')
parser.add_option('-s','--size',
                  dest = 'dimension', type = 'float', nargs = 3, metavar = 'int int int',
                  help = 'size in x,y,z (optional)')
parser.set_defaults(pos  = 'pos',
                    packing = (2,2,2),
                    grid    = (0,0,0),
                    size    = (0.0,0.0,0.0),
                   )

(options,filenames) = parser.parse_args()

options.packing = np.array(options.packing)
prefix = 'blowUp{}x{}x{}_'.format(*options.packing)

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    outname = os.path.join(os.path.dirname(name),
                                                           prefix+os.path.basename(name)) if name else name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------
  
  errors  = []
  remarks = []
  
  if table.label_dimension(options.pos) != 3:  errors.append('coordinates "{}" are not a vector.'.format(options.pos))

  colElem = table.label_index('elem')
  
  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# --------------- figure out size and grid ---------------------------------------------------------

  table.data_readArray(options.pos)
  table.data_rewind()

  grid,size = damask.util.coordGridAndSize(table.data)
  
  packing = np.array(options.packing,'i')
  outSize = grid*packing
  
# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data -------------------------------------------

  data = np.zeros(outSize.tolist()+[len(table.labels(raw = True))])
  p = np.zeros(3,'i')
  
  for p[2] in range(grid[2]):
    for p[1] in range(grid[1]):
      for p[0] in range(grid[0]):
        d = p*packing
        table.data_read()
        data[d[0]:d[0]+packing[0],
             d[1]:d[1]+packing[1],
             d[2]:d[2]+packing[2],
             : ] = np.tile(np.array(table.data_asFloat(),'d'),packing.tolist()+[1])                 # tile to match blowUp voxel size
  elementSize = size/grid/packing
  elem = 1
  for c in range(outSize[2]):
    for b in range(outSize[1]):
      for a in range(outSize[0]):
        data[a,b,c,table.label_indexrange(options.pos)] = [a+0.5,b+0.5,c+0.5]*elementSize
        if colElem != -1: data[a,b,c,colElem] = elem
        table.data = data[a,b,c,:].tolist()
        outputAlive = table.data_write()                                                            # output processed line
        elem += 1

# ------------------------------------------ output finalization -----------------------------------

  table.close()                                                                                     # close input ASCII table (works for stdin)
