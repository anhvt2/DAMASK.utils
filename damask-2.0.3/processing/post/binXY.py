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
Produces a binned grid of two columns from an ASCIItable, i.e. a two-dimensional probability density map.

""", version = scriptID)

parser.add_option('-d','--data',
                  dest = 'data',
                  type = 'string', nargs = 2, metavar = 'string string',
                  help = 'column labels containing x and y ')
parser.add_option('-w','--weight',
                  dest = 'weight',
                  type = 'string', metavar = 'string',
                  help = 'column label containing weight of (x,y) point')
parser.add_option('-b','--bins',
                  dest = 'bins',
                  type = 'int', nargs = 2, metavar = 'int int',
                  help = 'number of bins in x and y direction [%default]')
parser.add_option('-t','--type',
                  dest = 'type',
                  type = 'string', nargs = 3, metavar = 'string string string',
                  help = 'type (linear/log) of x, y, and z axis [%default]')
parser.add_option('-x','--xrange',
                  dest = 'xrange',
                  type = 'float', nargs = 2, metavar = 'float float',
                  help = 'min max limits in x direction (optional)')
parser.add_option('-y','--yrange',
                  dest = 'yrange',
                  type = 'float', nargs = 2, metavar = 'float float',
                  help = 'min max limits in y direction (optional)')
parser.add_option('-z','--zrange',
                  dest = 'zrange',
                  type = 'float', nargs = 2, metavar = 'float float',
                  help = 'min max limits in z direction (optional)')
parser.add_option('-i','--invert',
                  dest = 'invert',
                  action = 'store_true',
                  help = 'invert probability density')
parser.add_option('-r','--rownormalize',
                  dest = 'normRow',
                  action = 'store_true',
                  help = 'normalize probability density in each row')
parser.add_option('-c','--colnormalize',
                  dest = 'normCol',
                  action = 'store_true',
                  help = 'normalize probability density in each column')

parser.set_defaults(bins = (10,10),
                    type = ('linear','linear','linear'),
                    xrange = (0.0,0.0),
                    yrange = (0.0,0.0),
                    zrange = (0.0,0.0),
                   )

(options,filenames) = parser.parse_args()

minmax = np.array([np.array(options.xrange),
                   np.array(options.yrange),
                   np.array(options.zrange)])
grid   = np.zeros(options.bins,'f')
result = np.zeros((options.bins[0],options.bins[1],3),'f')

if options.data is None: parser.error('no data columns specified.')

labels = list(options.data)


if options.weight is not None: labels += [options.weight]                                               # prevent character splitting of single string value

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    outname = os.path.join(os.path.dirname(name),
                                                           'binned-{}-{}_'.format(*options.data) +
                                                          ('weighted-{}_'.format(options.weight) if options.weight else '') +
                                                           os.path.basename(name)) if name else name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------

  missing_labels = table.data_readArray(labels)

  if len(missing_labels) > 0:
    damask.util.croak('column{} {} not found.'.format('s' if len(missing_labels) > 1 else '',', '.join(missing_labels)))
    table.close(dismiss = True)
    continue

  for c in (0,1):                                                                                   # check data minmax for x and y (i = 0 and 1)
    if (minmax[c] == 0.0).all(): minmax[c] = [table.data[:,c].min(),table.data[:,c].max()]
    if options.type[c].lower() == 'log':                                                            # if log scale
      table.data[:,c] = np.log(table.data[:,c])                                                     # change x,y coordinates to log
      minmax[c] = np.log(minmax[c])                                                                 # change minmax to log, too

  delta = minmax[:,1]-minmax[:,0]
  (grid,xedges,yedges) = np.histogram2d(table.data[:,0],table.data[:,1],
                                        bins=options.bins,
                                        range=minmax[:2],
                                        weights=None if options.weight is None else table.data[:,2])

  if options.normCol:
    for x in range(options.bins[0]):
      sum = np.sum(grid[x,:])
      if sum > 0.0:
        grid[x,:] /= sum
  if options.normRow:
    for y in range(options.bins[1]):
      sum = np.sum(grid[:,y])
      if sum > 0.0:
        grid[:,y] /= sum

  if (minmax[2] == 0.0).all(): minmax[2] = [grid.min(),grid.max()]                                   # auto scale from data
  if minmax[2,0] == minmax[2,1]:
    minmax[2,0] -= 1.
    minmax[2,1] += 1.
  if (minmax[2] == 0.0).all():                                                                       # no data in grid?
    damask.util.croak('no data found on grid...')
    minmax[2,:] = np.array([0.0,1.0])                                                                # making up arbitrary z minmax
  if options.type[2].lower() == 'log':
    grid = np.log(grid)
    minmax[2] = np.log(minmax[2])

  delta[2] = minmax[2,1]-minmax[2,0]

  for x in range(options.bins[0]):
    for y in range(options.bins[1]):
      result[x,y,:] = [minmax[0,0]+delta[0]/options.bins[0]*(x+0.5),
                       minmax[1,0]+delta[1]/options.bins[1]*(y+0.5),
                       min(1.0,max(0.0,(grid[x,y]-minmax[2,0])/delta[2]))]

  for c in (0,1):
    if options.type[c].lower() == 'log': result[:,:,c] = np.exp(result[:,:,c])

  if options.invert: result[:,:,2] = 1.0 - result[:,:,2]

# --- assemble header -------------------------------------------------------------------------------

  table.info_clear()
  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_clear()
  table.labels_append(['bin_%s'%options.data[0],'bin_%s'%options.data[1],'z'])
  table.head_write()

# --- output result ---------------------------------------------------------------------------------

  table.data = result.reshape(options.bins[0]*options.bins[1],3)
  table.data_writeArray()

  table.close()
