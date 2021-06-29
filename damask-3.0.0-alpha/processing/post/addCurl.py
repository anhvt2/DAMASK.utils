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
Add column(s) containing curl of requested column(s).
Operates on periodic ordered three-dimensional data sets of vector and tensor fields.
""", version = scriptID)

parser.add_option('-p','--pos','--periodiccellcenter',
                  dest = 'pos',
                  type = 'string', metavar = 'string',
                  help = 'label of coordinates [%default]')
parser.add_option('-l','--label',
                  dest = 'labels',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'label(s) of field values')

parser.set_defaults(pos = 'pos',
                   )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

if options.labels is None: parser.error('no data column specified.')

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)
    grid,size,origin = damask.grid_filters.cell_coord0_gridSizeOrigin(table.get(options.pos))

    for label in options.labels:
        field = table.get(label)
        shape = (3,) if np.prod(field.shape)//np.prod(grid) == 3 else (3,3)                         # vector or tensor
        field = field.reshape(tuple(grid)+(-1,),order='F').reshape(tuple(grid)+shape)
        curl  = damask.grid_filters.curl(size,field)
        table.add('curlFFT({})'.format(label),
                  curl.reshape(tuple(grid)+(-1,)).reshape(-1,np.prod(shape),order='F'),
                  scriptID+' '+' '.join(sys.argv[1:]))

    table.to_ASCII(sys.stdout if name is None else name)
