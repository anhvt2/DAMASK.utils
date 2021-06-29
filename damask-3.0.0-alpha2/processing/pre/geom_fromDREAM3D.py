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
from optparse import OptionParser

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(usage='%prog options [DREAM.3Dfile(s)]', description = """
Converts DREAM.3D file. Input can be cell data (direct pointwise takeover) or grain data (individual
grains are segmented). Requires orientation data as quaternion.

""", version = scriptID)

parser.add_option('-b','--basegroup',
                  dest = 'basegroup',
                  metavar = 'string',
                  help = 'name of the group in "DataContainers" containing the pointwise (and, if applicable grain average) data')
parser.add_option('-p','--pointwise',
                  dest = 'pointwise',
                  metavar = 'string',
                  help = 'name of the group in "DataContainers/<basegroup>" containing pointwise data [%default]')
parser.add_option('-a','--average',
                  dest = 'average',
                  metavar = 'string',
                  help = 'name of the group in "DataContainers</basegroup>" containing grain average data. '\
                       + 'Leave empty for pointwise data')
parser.add_option('--phase',
                  dest = 'phase',
                  type = 'string',
                  metavar = 'string',
                  help = 'name of the dataset containing pointwise/average phase IDs [%default]')
parser.add_option('--microstructure',
                  dest = 'microstructure',
                  type = 'string',
                  metavar = 'string',
                  help = 'name of the dataset connecting pointwise and average data [%default]')
parser.add_option('-q', '--quaternion',
                  dest = 'quaternion',
                  type = 'string',
                  metavar='string',
                  help = 'name of the dataset containing pointwise/average orientation as quaternion [%default]')

parser.set_defaults(pointwise      = 'CellData',
                    quaternion     = 'Quats',
                    phase          = 'Phases',
                    microstructure = 'FeatureIds',
                   )

(options, filenames) = parser.parse_args()

if options.basegroup is None:
  parser.error('No base group selected')

if filenames == []: parser.error('no input file specified.')

for name in filenames:
    damask.util.report(scriptName,name)

    geom = damask.Grid.load_DREAM3D(name,options.basegroup,options.pointwise)
    damask.util.croak(geom)

    geom.save_ASCII(os.path.splitext(name)[0]+'.geom')
