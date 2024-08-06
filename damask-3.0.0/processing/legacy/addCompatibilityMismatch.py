#!/usr/bin/env python3
# Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
from io import StringIO
from optparse import OptionParser

import numpy as np

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


def shapeMismatch(size,F,nodes,centres):
  """
  Routine to calculate the shape mismatch.

  shape mismatch is defined as difference between the vectors from the central point to
  the corners of reconstructed (combatible) volume element and the vectors calculated by deforming
  the initial volume element with the  current deformation gradient.
  """
  delta = size/grid*.5

  return + np.linalg.norm(nodes[:-1,:-1,:-1] -centres - np.dot(F,delta * np.array((-1,-1,-1))),axis=-1)\
         + np.linalg.norm(nodes[+1:,:-1,:-1] -centres - np.dot(F,delta * np.array((+1,-1,-1))),axis=-1)\
         + np.linalg.norm(nodes[+1:,+1:,:-1] -centres - np.dot(F,delta * np.array((+1,+1,-1))),axis=-1)\
         + np.linalg.norm(nodes[:-1,+1:,:-1] -centres - np.dot(F,delta * np.array((-1,+1,-1))),axis=-1)\
         + np.linalg.norm(nodes[:-1,:-1,+1:] -centres - np.dot(F,delta * np.array((-1,-1,+1))),axis=-1)\
         + np.linalg.norm(nodes[+1:,:-1,+1:] -centres - np.dot(F,delta * np.array((+1,-1,+1))),axis=-1)\
         + np.linalg.norm(nodes[+1:,+1:,+1:] -centres - np.dot(F,delta * np.array((+1,+1,+1))),axis=-1)\
         + np.linalg.norm(nodes[:-1,+1:,+1:] -centres - np.dot(F,delta * np.array((-1,+1,+1))),axis=-1)


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(usage='%prog options [ASCIItable(s)]', description = """
Add column(s) containing the shape and volume mismatch resulting from given deformation gradient.
Operates on periodic three-dimensional x,y,z-ordered data sets.

""", version = scriptID)


parser.add_option('-c','--coordinates',
                  dest = 'pos',
                  type = 'string', metavar = 'string',
                  help = 'column heading of coordinates [%default]')
parser.add_option('-f','--defgrad',
                  dest = 'defgrad',
                  type = 'string', metavar = 'string ',
                  help = 'column heading of deformation gradient [%default]')
parser.add_option('--no-shape','-s',
                  dest = 'shape',
                  action = 'store_false',
                  help = 'omit shape mismatch')
parser.set_defaults(pos     = 'pos',
                    defgrad = 'f',
                    shape   = True,
                    volume  = True,
                   )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]


for name in filenames:
  damask.util.report(scriptName,name)

  table = damask.Table.load(StringIO(''.join(sys.stdin.read())) if name is None else name)
  grid,size,origin = damask.grid_filters.cellsSizeOrigin_coordinates0_point(table.get(options.pos))

  F = table.get(options.defgrad).reshape(tuple(grid)+(-1,),order='F').reshape(tuple(grid)+(3,3))
  nodes = damask.grid_filters.coordinates_node(size,F)

  if options.shape:
    centers = damask.grid_filters.coordinates_point(size,F)
    shapeMismatch = shapeMismatch(size,F,nodes,centers)
    table = table.add('shapeMismatch(({}))'.format(options.defgrad),
                      shapeMismatch.reshape(-1,1,order='F'),
                      scriptID+' '+' '.join(sys.argv[1:]))

  table.save((sys.stdout if name is None else name))
