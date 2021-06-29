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
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Add RGB color value corresponding to TSL-OIM scheme for inverse pole figures.

""", version = scriptID)

parser.add_option('-p',
                  '--pole',
                  dest = 'pole',
                  type = 'float', nargs = 3, metavar = 'float float float',
                  help = 'lab frame direction for inverse pole figure [%default]')
parser.add_option('-s',
                  '--symmetry',
                  dest = 'symmetry',
                  type = 'choice', choices = damask.Symmetry.lattices[1:], metavar='string',
                  help = 'crystal symmetry [%default] {{{}}} '.format(', '.join(damask.Symmetry.lattices[1:])))
parser.add_option('-o',
                  '--orientation',
                  dest = 'quaternion',
                  metavar = 'string',
                  help = 'label of crystal orientation given as unit quaternion [%default]')

parser.set_defaults(pole = (0.0,0.0,1.0),
                    quaternion = 'orientation',
                    symmetry = damask.Symmetry.lattices[-1],
                   )

(options, filenames) = parser.parse_args()

# damask.Orientation requires Bravais lattice, but we are only interested in symmetry
symmetry2lattice={'cubic':'bcc','hexagonal':'hex','tetragonal':'bct'}
lattice = symmetry2lattice[options.symmetry]

pole = np.array(options.pole)
pole /= np.linalg.norm(pole)

# --- loop over input files ------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------

  if not table.label_dimension(options.quaternion) == 4:
    damask.util.croak('input {} does not have dimension 4.'.format(options.quaternion))
    table.close(dismiss = True)                                                                     # close ASCIItable and remove empty file
    continue

  column = table.label_index(options.quaternion)

# ------------------------------------------ assemble header ---------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_append(['{}_IPF_{:g}{:g}{:g}_{sym}'.format(i+1,*options.pole,sym = options.symmetry.lower()) for i in range(3)])
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    o = damask.Orientation(np.array(list(map(float,table.data[column:column+4]))),
                           lattice   = lattice).reduced()

    table.data_append(o.IPFcolor(pole))
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
