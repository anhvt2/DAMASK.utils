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
Sort rows by given (or all) column label(s).

Examples:
With coordinates in columns "x", "y", and "z"; sorting with x slowest and z fastest varying index: --label x,y,z.
""", version = scriptID)


parser.add_option('-l','--label',
                  dest   = 'keys',
                  action = 'extend', metavar = '<string LIST>',
                  help   = 'list of column labels (a,b,c,...)')
parser.add_option('-r','--reverse',
                  dest   = 'reverse',
                  action = 'store_true',
                  help   = 'sort in reverse')

parser.set_defaults(reverse = False,
                   )

(options,filenames) = parser.parse_args()


# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ assemble header ---------------------------------------  

  table.head_read()
  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.head_write()

# ------------------------------------------ process data ---------------------------------------  

  table.data_readArray()

  keys = table.labels(raw = True)[::-1] if options.keys is None else options.keys[::-1]             # numpy sorts with most significant column as last

  cols    = []
  remarks = []
  for i,column in enumerate(table.label_index(keys)):
    if column < 0: remarks.append('label "{}" not present...'.format(keys[i]))
    else:          cols += [table.data[:,column]]
  if remarks != []: damask.util.croak(remarks)
  
  ind = np.lexsort(cols) if cols != [] else np.arange(table.data.shape[0])
  if options.reverse: ind = ind[::-1]

# ------------------------------------------ output result ---------------------------------------  

  table.data = table.data[ind]
  table.data_writeArray()
  table.close()                                                                                     # close ASCII table
