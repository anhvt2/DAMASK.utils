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

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Create seeds file by poking at 45 degree through given geom file.
Mimics APS Beamline 34-ID-E DAXM poking.

""", version = scriptID)

parser.add_option('-N', '--points',
                  dest = 'N',
                  type = 'int', metavar = 'int',
                  help = 'number of poking locations [%default]')
parser.add_option('-b', '--box',
                  dest = 'box',
                  type = 'float', nargs = 6, metavar = ' '.join(['float']*6),
                  help = 'bounding box as fraction in x, y, and z directions')
parser.add_option('-x',
                  action = 'store_true',
                  dest   = 'x',
                  help   = 'poke 45 deg along x')
parser.add_option('-y',
                  action = 'store_true',
                  dest   = 'y',
                  help   = 'poke 45 deg along y')

parser.set_defaults(x = False,
                    y = False,
                    box = [0.0,1.0,0.0,1.0,0.0,1.0],
                    N = 16,
                   )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

options.box = np.array(options.box).reshape(3,2)

for name in filenames:
    damask.util.report(scriptName,name)
    geom = damask.Geom.from_file(StringIO(''.join(sys.stdin.read())) if name is None else name)

    offset =(np.amin(options.box, axis=1)*geom.grid/geom.size).astype(int)
    box    = np.amax(options.box, axis=1) \
           - np.amin(options.box, axis=1)

    Nx = int(options.N/np.sqrt(options.N*geom.size[1]*box[1]/geom.size[0]/box[0]))
    Ny = int(options.N/np.sqrt(options.N*geom.size[0]*box[0]/geom.size[1]/box[1]))
    Nz = int(box[2]*geom.grid[2])

    damask.util.croak('poking {} x {} x {} in box {} {} {}...'.format(Nx,Ny,Nz,*box))

    seeds = np.zeros((Nx*Ny*Nz,4))
    g     = np.zeros(3,dtype=np.int)

    n = 0
    for i in range(Nx):
        for j in range(Ny):
            g[0] = round((i+0.5)*box[0]*geom.grid[0]/Nx-0.5)+offset[0]
            g[1] = round((j+0.5)*box[1]*geom.grid[1]/Ny-0.5)+offset[1]
            for k in range(Nz):
                g[2] = k + offset[2]
                g %= geom.grid
                seeds[n,0:3] = (g+0.5)/geom.grid                                                    # normalize coordinates to box
                seeds[n,  3] = geom.microstructure[g[0],g[1],g[2]]
                if options.x: g[0] += 1
                if options.y: g[1] += 1
                n += 1


    comments = geom.comments \
             + [scriptID + ' ' + ' '.join(sys.argv[1:]),
                'poking\ta {}\tb {}\tc {}'.format(Nx,Ny,Nz),
                'grid\ta {}\tb {}\tc {}'.format(*geom.grid),
                'size\tx {}\ty {}\tz {}'.format(*geom.size),
                'origin\tx {}\ty {}\tz {}'.format(*geom.origin),
                'homogenization\t{}'.format(geom.homogenization)]

    table = damask.Table(seeds,{'pos':(3,),'microstructure':(1,)},comments)
    table.set('microstructure',table.get('microstructure').astype(np.int))
    table.to_ASCII(sys.stdout if name is None else \
                   os.path.splitext(name)[0]+'_poked_{}.seeds'.format(options.N))
