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
Transform X,Y,Z,F APS BeamLine 34 coordinates to x,y,z APS strain coordinates.

""", version = scriptID)

parser.add_option('-f','--frame',dest='frame', metavar='string',
                                 help='label of APS X,Y,Z coords')
parser.add_option('--depth',     dest='depth', metavar='string',
                                 help='depth')

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

if options.frame is None:
    parser.error('frame not specified')
if options.depth is None:
    parser.error('depth not specified')


rot_to_TSL = damask.Rotation.from_axis_angle([-1,0,0,.75*np.pi])

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)
    
    coord      = - table.get(options.frame)
    coord[:,2] += table.get(options.depth)[:,0]

    table.add('coord',rot_to_TSL.broadcast_to(coord.shape[0]) @ coord,scriptID+' '+' '.join(sys.argv[1:]))

    table.to_ASCII(sys.stdout if name is None else name)
