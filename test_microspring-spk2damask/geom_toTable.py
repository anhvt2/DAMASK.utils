#!/usr/bin/env python2.7
# -*- coding: UTF-8 no BOM -*-
# Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
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


#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog [geomfile(s)]', description = """
Translate geom description into ASCIItable containing position and microstructure.

""", version = scriptID)

parser.add_option('--float',
                  dest = 'real',
                  action = 'store_true',
                  help = 'use float input')

parser.set_defaults(real = False,
                   )
(options, filenames) = parser.parse_args()

datatype = 'f' if options.real else 'i'

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    outname = os.path.splitext(name)[0]+'.txt' if name else name,
                                    buffered = False,
                                    labeled = False,
                                   )
  except: continue
  damask.util.report(scriptName,name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()
  info,extra_header = table.head_getGeom()
  
  damask.util.croak(['grid     a b c:  %s'%(' x '.join(map(str,info['grid']))),
                     'size     x y z:  %s'%(' x '.join(map(str,info['size']))),
                     'origin   x y z:  %s'%(' : '.join(map(str,info['origin']))),
                     'homogenization:  %i'%info['homogenization'],
                     'microstructures: %i'%info['microstructures'],
                    ])

  errors = []
  if np.any(info['grid'] < 1):    errors.append('invalid grid a b c.')
  if np.any(info['size'] <= 0.0): errors.append('invalid size x y z.')
  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# --- read data ------------------------------------------------------------------------------------

  microstructure = table.microstructure_read(info['grid'],datatype)

# ------------------------------------------ assemble header ---------------------------------------

  table.info_clear()
  table.info_append(extra_header + [scriptID + '\t' + ' '.join(sys.argv[1:])])
  table.labels_clear()
  table.labels_append(['{}_{}'.format(1+i,'pos') for i in range(3)]+['microstructure'])
  table.head_write()
  table.output_flush()

#--- generate grid --------------------------------------------------------------------------------

  x = (0.5 + np.arange(info['grid'][0],dtype=float))/info['grid'][0]*info['size'][0]+info['origin'][0]
  y = (0.5 + np.arange(info['grid'][1],dtype=float))/info['grid'][1]*info['size'][1]+info['origin'][1]
  z = (0.5 + np.arange(info['grid'][2],dtype=float))/info['grid'][2]*info['size'][2]+info['origin'][2]

  xx = np.tile(          x,                info['grid'][1]* info['grid'][2])
  yy = np.tile(np.repeat(y,info['grid'][0]                ),info['grid'][2])
  zz =         np.repeat(z,info['grid'][0]*info['grid'][1])

  table.data = np.squeeze(np.dstack((xx,yy,zz,microstructure)))
  table.data_writeArray()

# ------------------------------------------ finalize output ---------------------------------------

  table.close()
