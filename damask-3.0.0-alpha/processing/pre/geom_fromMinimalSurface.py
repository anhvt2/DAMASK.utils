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
from optparse import OptionParser

import numpy as np

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


minimal_surfaces = ['primitive','gyroid','diamond']

surface = {
            'primitive': lambda x,y,z: np.cos(x)+np.cos(y)+np.cos(z),
            'gyroid':    lambda x,y,z: np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.cos(x)*np.sin(z),
            'diamond':   lambda x,y,z: np.cos(x-y)*np.cos(z)+np.sin(x+y)*np.sin(z),
          }


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [geomfile]', description = """
Generate a bicontinuous structure of given type.

""", version = scriptID)


parser.add_option('-t','--type',
                  dest = 'type',
                  choices = minimal_surfaces, metavar = 'string',
                  help = 'type of minimal surface [primitive] {%s}' %(','.join(minimal_surfaces)))
parser.add_option('-f','--threshold',
                  dest = 'threshold',
                  type = 'float', metavar = 'float',
                  help = 'threshold value defining minimal surface [%default]')
parser.add_option('-g', '--grid',
                  dest = 'grid',
                  type = 'int', nargs = 3, metavar = 'int int int',
                  help = 'a,b,c grid of hexahedral box [%default]')
parser.add_option('-s', '--size',
                  dest = 'size',
                  type = 'float', nargs = 3, metavar = 'float float float',
                  help = 'x,y,z size of hexahedral box [%default]')
parser.add_option('-p', '--periods',
                  dest = 'periods',
                  type = 'int', metavar = 'int',
                  help = 'number of repetitions of unit cell [%default]')
parser.add_option('--homogenization',
                  dest = 'homogenization',
                  type = 'int', metavar = 'int',
                  help = 'homogenization index to be used [%default]')
parser.add_option('--m',
                  dest = 'microstructure',
                  type = 'int', nargs = 2, metavar = 'int int',
                  help = 'two microstructure indices to be used [%default]')

parser.set_defaults(type = minimal_surfaces[0],
                    threshold = 0.0,
                    periods = 1,
                    grid = (16,16,16),
                    size = (1.0,1.0,1.0),
                    homogenization = 1,
                    microstructure = (1,2),
                   )

(options,filename) = parser.parse_args()


name = None if filename == [] else filename[0]
damask.util.report(scriptName,name)

x,y,z = np.meshgrid(options.periods*2.0*np.pi*(np.arange(options.grid[0])+0.5)/options.grid[0],
                    options.periods*2.0*np.pi*(np.arange(options.grid[1])+0.5)/options.grid[1],
                    options.periods*2.0*np.pi*(np.arange(options.grid[2])+0.5)/options.grid[2],
                    indexing='xy',sparse=True)

microstructure = np.where(options.threshold < surface[options.type](x,y,z),
                          options.microstructure[1],options.microstructure[0])

geom=damask.Geom(microstructure,options.size,
                 homogenization=options.homogenization,
                 comments=[scriptID + ' ' + ' '.join(sys.argv[1:])])
damask.util.croak(geom)

geom.to_file(sys.stdout if name is None else name,pack=False)
