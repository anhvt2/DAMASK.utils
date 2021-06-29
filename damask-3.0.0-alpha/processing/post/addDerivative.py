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

def derivative(coordinates,what):
  
  result = np.empty_like(what)
  
  # use differentiation by interpolation
  # as described in http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf

  result[1:-1,:] = + what[1:-1,:] * (2.*coordinates[1:-1]-coordinates[:-2]-coordinates[2:]) / \
                     ((coordinates[1:-1]-coordinates[:-2])*(coordinates[1:-1]-coordinates[2:])) \
                   + what[2:,:] * (coordinates[1:-1]-coordinates[:-2]) / \
                     ((coordinates[2:]-coordinates[1:-1])*(coordinates[2:]-coordinates[:-2])) \
                   + what[:-2,:] * (coordinates[1:-1]-coordinates[2:]) / \
                     ((coordinates[:-2]-coordinates[1:-1])*(coordinates[:-2]-coordinates[2:])) \

  result[0,:]    = (what[0,:] - what[1,:]) / \
                   (coordinates[0] - coordinates[1])
  result[-1,:]   = (what[-1,:] - what[-2,:]) / \
                   (coordinates[-1] - coordinates[-2])
  
  return result


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Add column(s) containing numerical derivative of requested column(s) with respect to given coordinates.

""", version = scriptID)

parser.add_option('-c','--coordinates',
                  dest = 'coordinates',
                  type = 'string', metavar='string',
                  help = 'heading of coordinate column')
parser.add_option('-l','--label',
                  dest = 'labels',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'heading of column(s) to differentiate')


(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

if options.coordinates is None:
    parser.error('no coordinate column specified.')
if options.labels is None:
    parser.error('no data column specified.')

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)
    for label in options.labels:
        table.add('d({})/d({})'.format(label,options.coordinates),
                  derivative(table.get(options.coordinates),table.get(label)),
                  scriptID+' '+' '.join(sys.argv[1:]))

    table.to_ASCII(sys.stdout if name is None else name)
