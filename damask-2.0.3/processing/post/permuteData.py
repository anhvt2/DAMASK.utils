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
Permute all values in given column(s).

""", version = scriptID)

parser.add_option('-l','--label',
                  dest = 'label',
                  action = 'extend', metavar = '<string LIST>',
                  help  ='column(s) to permute')
parser.add_option('-u', '--unique',
                  dest = 'unique',
                  action = 'store_true',
                  help = 'shuffle unique values as group')
parser.add_option('-r', '--rnd',
                  dest = 'randomSeed',
                  type = 'int', metavar = 'int',
                  help = 'seed of random number generator [%default]')

parser.set_defaults(label = [],
                    unnique = False,
                    randomSeed = None,
                   )

(options,filenames) = parser.parse_args()

if len(options.label) == 0:
  parser.error('no labels specified.')

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ process labels ---------------------------------------  

  errors  = []
  remarks = []
  columns = []
  dims    = []

  indices    = table.label_index    (options.label)
  dimensions = table.label_dimension(options.label)
  for i,index in enumerate(indices):
    if index == -1: remarks.append('label "{}" not present...'.format(options.label[i]))
    else:
      columns.append(index)
      dims.append(dimensions[i])

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue
       
# ------------------------------------------ assemble header ---------------------------------------

  randomSeed = int(os.urandom(4).encode('hex'), 16) if options.randomSeed is None else options.randomSeed         # random seed per file
  np.random.seed(randomSeed)

  table.info_append([scriptID + '\t' + ' '.join(sys.argv[1:]),
                     'random seed {}'.format(randomSeed),
                    ])
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  table.data_readArray()                                                                            # read all data at once
  for col,dim in zip(columns,dims):
    if options.unique:
      s = set(map(tuple,table.data[:,col:col+dim]))                                                 # generate set of (unique) values
      uniques = np.array(map(np.array,s))                                                           # translate set to np.array
      shuffler = dict(zip(s,np.random.permutation(len(s))))                                         # random permutation
      table.data[:,col:col+dim] = uniques[np.array(map(lambda x: shuffler[tuple(x)],
                                                       table.data[:,col:col+dim]))]                 # fill table with mapped uniques
    else:
      np.random.shuffle(table.data[:,col:col+dim])                                                  # independently shuffle every row

# ------------------------------------------ output result -----------------------------------------  

  table.data_writeArray()

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
