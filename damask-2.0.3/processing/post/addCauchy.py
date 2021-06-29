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
Add column containing Cauchy stress based on deformation gradient and first Piola--Kirchhoff stress.

""", version = scriptID)

parser.add_option('-f','--defgrad',
                  dest = 'defgrad',
                  type = 'string', metavar = 'string',
                  help = 'heading of columns containing deformation gradient [%default]')
parser.add_option('-p','--stress',
                  dest = 'stress',
                  type = 'string', metavar = 'string',
                  help = 'heading of columns containing first Piola--Kirchhoff stress [%default]')

parser.set_defaults(defgrad = 'f',
                    stress  = 'p',
                   )

(options,filenames) = parser.parse_args()

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name, buffered = False)
  except:
    continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------

  errors = []
  column = {}
  
  for tensor in [options.defgrad,options.stress]:
    dim = table.label_dimension(tensor)
    if   dim <  0: errors.append('column {} not found.'.format(tensor))
    elif dim != 9: errors.append('column {} is not a tensor.'.format(tensor))
    else:
      column[tensor] = table.label_index(tensor)

  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_append(['{}_Cauchy'.format(i+1) for i in range(9)])                                  # extend ASCII header with new labels
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    F = np.array(list(map(float,table.data[column[options.defgrad]:column[options.defgrad]+9])),'d').reshape(3,3)
    P = np.array(list(map(float,table.data[column[options.stress ]:column[options.stress ]+9])),'d').reshape(3,3)
    table.data_append(list(1.0/np.linalg.det(F)*np.dot(P,F.T).reshape(9)))                          # [Cauchy] = (1/det(F)) * [P].[F_transpose]
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------

  table.close()                                                                                     # close input ASCII table (works for stdin)
