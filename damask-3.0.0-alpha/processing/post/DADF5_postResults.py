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
import argparse

import numpy as np

import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('filenames', nargs='+',
                    help='DADF5 files')
parser.add_argument('-d','--dir', dest='dir',default='postProc',metavar='string',
                    help='name of subdirectory relative to the location of the DADF5 file to hold output')
parser.add_argument('--mat', nargs='+',
                    help='labels for materialpoint',dest='mat')
parser.add_argument('--con', nargs='+',
                    help='labels for constituent',dest='con')

options = parser.parse_args()

if options.mat is None: options.mat=[]
if options.con is None: options.con=[]

for filename in options.filenames:
    results = damask.Result(filename)

    if not results.structured: continue
    coords = damask.grid_filters.cell_coord0(results.grid,results.size,results.origin).reshape(-1,3,order='F')

    N_digits = int(np.floor(np.log10(int(results.increments[-1][3:]))))+1
    N_digits = 5 # hack to keep test intact
    for inc in damask.util.show_progress(results.iterate('increments'),len(results.increments)):
        table = damask.Table(np.ones(np.product(results.grid),dtype=int)*int(inc[3:]),{'inc':(1,)})
        table.add('pos',coords.reshape(-1,3))

        results.pick('materialpoints',False)
        results.pick('constituents',  True)
        for label in options.con:
            x = results.get_dataset_location(label)
            if len(x) != 0:
                table.add(label,results.read_dataset(x,0,plain=True).reshape(results.grid.prod(),-1))

        results.pick('constituents',  False)
        results.pick('materialpoints',True)
        for label in options.mat:
            x = results.get_dataset_location(label)
            if len(x) != 0:
                table.add(label,results.read_dataset(x,0,plain=True).reshape(results.grid.prod(),-1))

        dirname  = os.path.abspath(os.path.join(os.path.dirname(filename),options.dir))
        if not os.path.isdir(dirname):
            os.mkdir(dirname,0o755)
        file_out = '{}_inc{}.txt'.format(os.path.splitext(os.path.split(filename)[-1])[0],
                                         inc[3:].zfill(N_digits))
        table.to_ASCII(os.path.join(dirname,file_out))
