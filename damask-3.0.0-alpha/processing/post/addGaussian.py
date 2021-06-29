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

from scipy import ndimage

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog option [ASCIItable(s)]', description = """
Add column(s) containing Gaussian filtered values of requested column(s).
Operates on periodic and non-periodic ordered three-dimensional data sets.
For details see scipy.ndimage documentation.

""", version = scriptID)

parser.add_option('-p','--pos','--periodiccellcenter',
                  dest = 'pos',
                  type = 'string', metavar = 'string',
                  help = 'label of coordinates [%default]')
parser.add_option('-s','--scalar',
                  dest = 'labels',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'label(s) of scalar field values')
parser.add_option('-o','--order',
                  dest = 'order',
                  type = int,
                  metavar = 'int',
                  help = 'order of the filter [%default]')
parser.add_option('--sigma',
                  dest = 'sigma',
                  type = float,
                  metavar = 'float',
                  help = 'standard deviation [%default]')
parser.add_option('--periodic',
                  dest = 'periodic',
                  action = 'store_true',
                  help = 'assume periodic grain structure')



parser.set_defaults(pos = 'pos',
                    order = 0,
                    sigma = 1,
                   )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

if options.labels is None: parser.error('no data column specified.')

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)
    damask.grid_filters.coord0_check(table.get(options.pos))

    for label in options.labels:
        table.add('Gauss{}({})'.format(options.sigma,label),
                  ndimage.filters.gaussian_filter(table.get(label).reshape(-1),
                                                  options.sigma,options.order,
                                                  mode = 'wrap' if options.periodic else 'nearest'),
                  scriptID+' '+' '.join(sys.argv[1:]))

    table.to_ASCII(sys.stdout if name is None else name)
