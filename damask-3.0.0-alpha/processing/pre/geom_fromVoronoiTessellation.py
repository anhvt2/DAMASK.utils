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
import multiprocessing
from io import StringIO
from functools import partial
from optparse import OptionParser,OptionGroup

import numpy as np
from scipy import spatial

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


def findClosestSeed(seeds, weights, point):
    return np.argmin(np.sum((np.broadcast_to(point,(len(seeds),3))-seeds)**2,axis=1) - weights)


def Laguerre_tessellation(grid, size, seeds, weights, origin = np.zeros(3), periodic = True, cpus = 2):

    if periodic:
        weights_p = np.tile(weights.squeeze(),27)                                                   # Laguerre weights (1,2,3,1,2,3,...,1,2,3)
        seeds_p = np.vstack((seeds  -np.array([size[0],0.,0.]),seeds,  seeds  +np.array([size[0],0.,0.])))
        seeds_p = np.vstack((seeds_p-np.array([0.,size[1],0.]),seeds_p,seeds_p+np.array([0.,size[1],0.])))
        seeds_p = np.vstack((seeds_p-np.array([0.,0.,size[2]]),seeds_p,seeds_p+np.array([0.,0.,size[2]])))
        coords = damask.grid_filters.cell_coord0(grid*3,size*3,-origin-size).reshape(-1,3)
    else:
        weights_p = weights.squeeze()
        seeds_p   = seeds
        coords = damask.grid_filters.cell_coord0(grid,size,-origin).reshape(-1,3)

    if cpus > 1:
        pool = multiprocessing.Pool(processes = cpus)
        result = pool.map_async(partial(findClosestSeed,seeds_p,weights_p), [coord for coord in coords])
        pool.close()
        pool.join()
        closest_seed = np.array(result.get()).reshape(-1,3)
    else:
        closest_seed= np.array([findClosestSeed(seeds_p,weights_p,coord) for coord in coords])

    if periodic:
        closest_seed = closest_seed.reshape(grid*3)
        return closest_seed[grid[0]:grid[0]*2,grid[1]:grid[1]*2,grid[2]:grid[2]*2]%seeds.shape[0]
    else:
        return closest_seed


def Voronoi_tessellation(grid, size, seeds, origin = np.zeros(3), periodic = True):

    coords = damask.grid_filters.cell_coord0(grid,size,-origin).reshape(-1,3)
    KDTree = spatial.cKDTree(seeds,boxsize=size) if periodic else spatial.cKDTree(seeds)
    devNull,closest_seed = KDTree.query(coords)

    return closest_seed


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog option(s) [seedfile(s)]', description = """
Generate geometry description and material configuration by tessellation of given seeds file.

""", version = scriptID)


group = OptionGroup(parser, "Tessellation","")

group.add_option('-l',
                 '--laguerre',
                 dest = 'laguerre',
                 action = 'store_true',
                 help = 'use Laguerre (weighted Voronoi) tessellation')
group.add_option('--cpus',
                 dest = 'cpus',
                 type = 'int', metavar = 'int',
                 help = 'number of parallel processes to use for Laguerre tessellation [%default]')
group.add_option('--nonperiodic',
                 dest = 'periodic',
                 action = 'store_false',
                 help = 'nonperiodic tessellation')

parser.add_option_group(group)

group = OptionGroup(parser, "Geometry","")

group.add_option('-g',
                 '--grid',
                 dest = 'grid',
                 type = 'int', nargs = 3, metavar = ' '.join(['int']*3),
                 help = 'a,b,c grid of hexahedral box')
group.add_option('-s',
                 '--size',
                 dest = 'size',
                 type = 'float', nargs = 3, metavar=' '.join(['float']*3),
                 help = 'x,y,z size of hexahedral box [1.0 1.0 1.0]')
group.add_option('-o',
                 '--origin',
                 dest = 'origin',
                 type = 'float', nargs = 3, metavar=' '.join(['float']*3),
                 help = 'origin of grid [0.0 0.0 0.0]')

parser.add_option_group(group)

group = OptionGroup(parser, "Seeds","")

group.add_option('-p',
                 '--pos', '--seedposition',
                  dest = 'pos',
                  type = 'string', metavar = 'string',
                  help = 'label of coordinates [%default]')
group.add_option('-w',
                 '--weight',
                 dest = 'weight',
                 type = 'string', metavar = 'string',
                 help = 'label of weights [%default]')
group.add_option('-m',
                 '--microstructure',
                 dest = 'microstructure',
                 type = 'string', metavar = 'string',
                 help = 'label of microstructures [%default]')
group.add_option('-e',
                 '--eulers',
                 dest = 'eulers',
                 type = 'string', metavar = 'string',
                 help = 'label of Euler angles [%default]')
group.add_option('--axes',
                 dest = 'axes',
                 type = 'string', nargs = 3, metavar = ' '.join(['string']*3),
                 help = 'orientation coordinate frame in terms of position coordinate frame')

parser.add_option_group(group)

group = OptionGroup(parser, "Configuration","")

group.add_option('--without-config',
                 dest = 'config',
                 action = 'store_false',
                 help = 'omit material configuration header')
group.add_option('--homogenization',
                 dest = 'homogenization',
                 type = 'int', metavar = 'int',
                 help = 'homogenization index to be used [%default]')
group.add_option('--phase',
                 dest = 'phase',
                 type = 'int', metavar = 'int',
                 help = 'phase index to be used [%default]')

parser.add_option_group(group)

parser.set_defaults(pos            = 'pos',
                    weight         = 'weight',
                    microstructure = 'microstructure',
                    eulers         = 'euler',
                    homogenization = 1,
                    phase          = 1,
                    cpus           = 2,
                    laguerre       = False,
                    periodic       = True,
                    config         = True,
                  )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)

    size   = np.ones(3)
    origin = np.zeros(3)
    for line in table.comments:
        items = line.lower().strip().split()
        key = items[0] if items else ''
        if   key == 'grid':
            grid   = np.array([  int(dict(zip(items[1::2],items[2::2]))[i]) for i in ['a','b','c']])
        elif key == 'size':
            size   = np.array([float(dict(zip(items[1::2],items[2::2]))[i]) for i in ['x','y','z']])
        elif key == 'origin':
            origin = np.array([float(dict(zip(items[1::2],items[2::2]))[i]) for i in ['x','y','z']])
    if options.grid:   grid   = np.array(options.grid)
    if options.size:   size   = np.array(options.size)
    if options.origin: origin = np.array(options.origin)

    seeds  = table.get(options.pos)

    grains = table.get(options.microstructure) if options.microstructure in table.labels else np.arange(len(seeds))+1
    grainIDs  = np.unique(grains).astype('i')

    if options.eulers in table.labels:
        eulers = table.get(options.eulers)

    if options.laguerre:
        indices = grains[Laguerre_tessellation(grid,size,seeds,table.get(options.weight),origin,
                                               options.periodic,options.cpus)]
    else:
        indices = grains[Voronoi_tessellation (grid,size,seeds,origin,options.periodic)]

    config_header = []
    if options.config:

        if options.eulers in table.labels:
            config_header += ['<texture>']
            for ID in grainIDs:
                eulerID = np.nonzero(grains == ID)[0][0]                                            # find first occurrence of this grain id
                config_header += ['[Grain{}]'.format(ID),
                                  '(gauss)\tphi1 {:.2f}\tPhi {:.2f}\tphi2 {:.2f}'.format(*eulers[eulerID])
                                 ]
                if options.axes: config_header += ['axes\t{} {} {}'.format(*options.axes)]

        config_header += ['<microstructure>']
        for ID in grainIDs:
            config_header += ['[Grain{}]'.format(ID),
                              '(constituent)\tphase {}\ttexture {}\tfraction 1.0'.format(options.phase,ID)
                             ]

        config_header += ['<!skip>']

    header = [scriptID + ' ' + ' '.join(sys.argv[1:])]\
           + config_header
    geom = damask.Geom(indices.reshape(grid),size,origin,
                       homogenization=options.homogenization,comments=header)
    damask.util.croak(geom)

    geom.to_file(sys.stdout if name is None else os.path.splitext(name)[0]+'.geom',pack=False)
