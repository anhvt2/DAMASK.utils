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
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Append data of ASCIItable(s).

""", version = scriptID)

parser.add_option('-a', '--add','--table',
                  dest = 'table',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'tables to add')

(options,filenames) = parser.parse_args()

if options.table is None:
  parser.error('no table specified.')


# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False)
  except: continue

  damask.util.report(scriptName,name)

  tables = []
  for addTable in options.table:
    try:    tables.append(damask.ASCIItable(name = addTable,
                                            buffered = False,
                                            readonly = True)
                         )
    except: continue

# ------------------------------------------ read headers ------------------------------------------

  table.head_read()
  for addTable in tables: addTable.head_read()

# ------------------------------------------ assemble header --------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))

  for addTable in tables: table.labels_append(addTable.labels(raw = True))                          # extend ASCII header with new labels

  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():
    for addTable in tables:
      outputAlive = addTable.data_read()                                                            # read next table's data
      if not outputAlive: break
      table.data_append(addTable.data)                                                              # append to master table
    if outputAlive:
      outputAlive = table.data_write()                                                              # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
  for addTable in tables:
    addTable.close()
