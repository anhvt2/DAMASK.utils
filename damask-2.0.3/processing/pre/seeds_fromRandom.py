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

import os,sys,math,random
import numpy as np
import damask
from optparse import OptionParser,OptionGroup
from scipy import spatial


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# ------------------------------------------ aux functions ---------------------------------

def kdtree_search(cloud, queryPoints):
  """Find distances to nearest neighbor among cloud (N,d) for each of the queryPoints (n,d)"""
  n = queryPoints.shape[0]
  distances = np.zeros(n,dtype=float)
  tree = spatial.cKDTree(cloud)
  
  for i in range(n):
    distances[i], index = tree.query(queryPoints[i])

  return distances
    
# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options', description = """
Distribute given number of points randomly within (a fraction of) the three-dimensional cube [0.0,0.0,0.0]--[1.0,1.0,1.0].
Reports positions with random crystal orientations in seeds file format to STDOUT.

""", version = scriptID)

parser.add_option('-N',
                  dest = 'N',
                  type = 'int', metavar = 'int',
                  help = 'number of seed points [%default]')
parser.add_option('-f',
                  '--fraction',
                  dest = 'fraction',
                  type = 'float', nargs = 3, metavar = 'float float float',
                  help='fractions along x,y,z of unit cube to fill %default')
parser.add_option('-g',
                  '--grid',
                  dest = 'grid',
                  type = 'int', nargs = 3, metavar = 'int int int',
                  help='min a,b,c grid of hexahedral box %default')
parser.add_option('-m',
                  '--microstructure',
                  dest = 'microstructure',
                  type = 'int', metavar = 'int',
                  help = 'first microstructure index [%default]')
parser.add_option('-r',
                  '--rnd',
                  dest = 'randomSeed', type = 'int', metavar = 'int',
                  help = 'seed of random number generator [%default]')
parser.add_option('--format',
                  dest = 'format', type = 'string', metavar = 'string',
                  help = 'output number format [auto]')

group = OptionGroup(parser, "Laguerre Tessellation",
                   "Parameters determining shape of weight distribution of seed points"
                   )
group.add_option( '-w',
                  '--weights',
                  action = 'store_true',
                  dest   = 'weights',
                  help   = 'assign random weights to seed points for Laguerre tessellation [%default]')
group.add_option( '--max',
                  dest = 'max',
                  type = 'float', metavar = 'float',
                  help = 'max of uniform distribution for weights [%default]')
group.add_option( '--mean',
                  dest = 'mean',
                  type = 'float', metavar = 'float',
                  help = 'mean of normal distribution for weights [%default]')
group.add_option( '--sigma',
                  dest = 'sigma',
                  type = 'float', metavar = 'float',
                  help='standard deviation of normal distribution for weights [%default]')
parser.add_option_group(group)

group = OptionGroup(parser, "Selective Seeding",
                    "More uniform distribution of seed points using Mitchell's Best Candidate Algorithm"
                   )
group.add_option( '-s',
                  '--selective',
                  action = 'store_true',
                  dest   = 'selective',
                  help   = 'selective picking of seed points from random seed points')
group.add_option( '--distance',
                  dest = 'distance',
                  type = 'float', metavar = 'float',
                  help = 'minimum distance to next neighbor [%default]')
group.add_option( '--numCandidates',
                  dest = 'numCandidates',
                  type = 'int', metavar = 'int',
                  help = 'size of point group to select best distance from [%default]')    
parser.add_option_group(group)

parser.set_defaults(randomSeed = None,
                    grid = (16,16,16),
                    fraction = (1.0,1.0,1.0),
                    N = 20,
                    weights = False,
                    max = 0.0,
                    mean = 0.2,
                    sigma = 0.05,
                    microstructure = 1,
                    selective = False,
                    distance = 0.2,
                    numCandidates = 10,
                    format = None,
                   )

(options,filenames) = parser.parse_args()

options.fraction = np.array(options.fraction)
options.grid = np.array(options.grid)
gridSize = options.grid.prod()

if options.randomSeed is None: options.randomSeed = int(os.urandom(4).hex(), 16)
np.random.seed(options.randomSeed)                                                                  # init random generators
random.seed(options.randomSeed)


# --- loop over output files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(outname = name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# --- sanity checks -------------------------------------------------------------------------

  remarks = []
  errors  = []
  if gridSize == 0:
    errors.append('zero grid dimension for {}.'.format(', '.join([['a','b','c'][x] for x in np.where(options.grid == 0)[0]])))
  if options.N > gridSize/10.:
    remarks.append('seed count exceeds 0.1 of grid points.')
  if options.selective and 4./3.*math.pi*(options.distance/2.)**3*options.N > 0.5:
    remarks.append('maximum recommended seed point count for given distance is {}.{}'.
                   format(int(3./8./math.pi/(options.distance/2.)**3)))

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    sys.exit()

# --- do work ------------------------------------------------------------------------------------
 
  grainEuler = np.random.rand(3,options.N)                                                          # create random Euler triplets
  grainEuler[0,:] *= 360.0                                                                          # phi_1    is uniformly distributed
  grainEuler[1,:] = np.degrees(np.arccos(2*grainEuler[1,:]-1))                                      # cos(Phi) is uniformly distributed
  grainEuler[2,:] *= 360.0                                                                          # phi_2    is uniformly distributed

  if not options.selective:
    n = np.maximum(np.ones(3),np.array(options.grid*options.fraction),
                   dtype=int,casting='unsafe')                                                      # find max grid indices within fraction
    meshgrid = np.meshgrid(*map(np.arange,n),indexing='ij')                                         # create a meshgrid within fraction
    coords = np.vstack((meshgrid[0],meshgrid[1],meshgrid[2])).reshape(3,n.prod()).T                 # assemble list of 3D coordinates
    seeds = ((random.sample(list(coords),options.N)+np.random.random(options.N*3).reshape(options.N,3))\
              / \
             (n/options.fraction)).T                                                                # pick options.N of those, rattle position,
                                                                                                    # and rescale to fall within fraction
  else:
    seeds = np.zeros((options.N,3),dtype=float)                                                     # seed positions array
    seeds[0] = np.random.random(3)*options.grid/max(options.grid)
    i = 1                                                                                           # start out with one given point
    if i%(options.N/100.) < 1: damask.util.croak('.',False)

    while i < options.N:
      candidates = np.random.random(options.numCandidates*3).reshape(options.numCandidates,3)
      distances  = kdtree_search(seeds[:i],candidates)
      best = distances.argmax()
      if distances[best] > options.distance:                                                        # require minimum separation
        seeds[i] = candidates[best]                                                                 # maximum separation to existing point cloud
        i += 1
        if i%(options.N/100.) < 1: damask.util.croak('.',False)

    damask.util.croak('')
    seeds = seeds.T                                                                                 # prepare shape for stacking

  if options.weights:
    weights = [np.random.uniform(low = 0, high = options.max, size = options.N)] if options.max > 0.0 \
         else [np.random.normal(loc = options.mean, scale = options.sigma, size = options.N)]
  else:
    weights = []
  seeds = np.transpose(np.vstack(tuple([seeds,
                                        grainEuler,
                                        np.arange(options.microstructure,
                                                  options.microstructure + options.N),
                                       ] + weights
                                 )))

# ------------------------------------------ assemble header ---------------------------------------

  table.info_clear()
  table.info_append([
    scriptID + ' ' + ' '.join(sys.argv[1:]),
    "grid\ta {}\tb {}\tc {}".format(*options.grid),
    "microstructures\t{}".format(options.N),
    "randomSeed\t{}".format(options.randomSeed),
    ])
  table.labels_clear()
  table.labels_append( ['{dim}_{label}'.format(dim = 1+k,label = 'pos')   for k in range(3)] +
                       ['{dim}_{label}'.format(dim = 1+k,label = 'euler') for k in range(3)] + 
                       ['microstructure'] +
                      (['weight'] if options.weights else []))
  table.head_write()
  table.output_flush()
  
# --- write seeds information ------------------------------------------------------------

  table.data = seeds
  table.data_writeArray(fmt = options.format)
    
# --- output finalization --------------------------------------------------------------------------

  table.close()                                                                                     # close ASCII table
