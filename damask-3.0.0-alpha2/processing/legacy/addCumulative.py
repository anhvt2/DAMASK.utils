#!/usr/bin/env python3
# Copyright 2011-20 Max-Planck-Institut für Eisenforschung GmbH
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

import os
import sys
from io import StringIO
from optparse import OptionParser

import numpy as np

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Add cumulative (sum of first to current row) values for given label(s).
""", version = scriptID)

parser.add_option('-l','--label',
                  dest='labels',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'columns to cumulate')
parser.add_option('-p','--product',
                  dest='product', action = 'store_true',
                  help = 'product of values instead of sum')

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

if options.labels is None:
  parser.error('no data column(s) specified.')

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.load(StringIO(''.join(sys.stdin.read())) if name is None else name)
    for label in options.labels:
        table = table.add('cum_{}({})'.format('prod'     if options.product else 'sum',label),
                          np.cumprod(table.get(label),0) if options.product else np.cumsum(table.get(label),0),
                          scriptID+' '+' '.join(sys.argv[1:]))

    table.save((sys.stdout if name is None else name),legacy=True)
