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

import sys,os
import damask
from optparse import OptionParser

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------------
#Borland, D., & Taylor, R. M. (2007). Rainbow Color Map (Still) Considered Harmful. Computer Graphics and Applications, IEEE, 27(2), 14--17.
#Moreland, K. (2009). Diverging Color Maps for Scientific Visualization. In Proc. 5th Int. Symp. Visual Computing (pp. 92--103).
outtypes    = ['paraview','gmsh','raw','GOM']
extensions  = ['.json','.msh','.txt','.legend']
colormodels = ['RGB','HSL','XYZ','CIELAB','MSH']

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Produces perceptually linear diverging and sequential colormaps in formats suitable for visualization software
or simply as a list of interpolated colors.

""", version = scriptID)

parser.add_option('-N','--steps', dest='N', type='int', nargs=1, metavar='int',
                  help='number of interpolation steps [%default]')
parser.add_option('-t','--trim', dest='trim', type='float', nargs=2, metavar='float float',
                  help='relative trim of colormap range [%default]')
parser.add_option('-l','--left', dest='left', type='float', nargs=3, metavar='float float float',
                  help='left color [%default]')
parser.add_option('-r','--right', dest='right', type='float', nargs=3, metavar='float float float',
                  help='right color [%default]')
parser.add_option('-c','--colormodel', dest='colormodel', metavar='string',
                  help='colormodel: '+', '.join(colormodels)+' [%default]')
parser.add_option('-p','--predefined', dest='predefined', metavar='string',
                  help='predefined colormap')
parser.add_option('-f','--format', dest='format', metavar='string',
                  help='output format: '+', '.join(outtypes)+' [%default]')
parser.set_defaults(colormodel = 'RGB')
parser.set_defaults(predefined = None)
parser.set_defaults(basename = None)
parser.set_defaults(format = 'paraview')
parser.set_defaults(N = 10)
parser.set_defaults(trim  = (-1.0,1.0))
parser.set_defaults(left  = (1.0,1.0,1.0))
parser.set_defaults(right = (0.0,0.0,0.0))

(options,filename) = parser.parse_args()

if options.format not in outtypes:
  parser.error('invalid format: "{}" (choices: {}).'.format(options.format,', '.join(outtypes)))

if options.N < 2:
  parser.error('too few steps (need at least 2).')

if options.trim[0] < -1.0 or \
   options.trim[1] >  1.0 or \
   options.trim[0] >= options.trim[1]:
  parser.error('invalid trim range (-1 +1).')

name   = options.format if filename == [] \
                        else filename[0]
output = sys.stdout     if filename == [] \
                        else open(os.path.basename(filename[0])+extensions[outtypes.index(options.format)],'w')

colorLeft = damask.Color(options.colormodel.upper(), list(options.left))
colorRight = damask.Color(options.colormodel.upper(), list(options.right))
colormap = damask.Colormap(colorLeft, colorRight, predefined=options.predefined)

output.write(colormap.export(name,options.format,options.N,list(options.trim)))
output.close()
