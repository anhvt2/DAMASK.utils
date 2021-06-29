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
from PIL import Image
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Generate PNG image from data in given column (or 2D data of overall table).

""", version = scriptID)

parser.add_option('-l','--label',
                  dest = 'label',
                  type = 'string', metavar = 'string',
                  help = 'column containing data [all]')
parser.add_option('-r','--range',
                  dest = 'range',
                  type = 'float', nargs = 2, metavar = 'float float',
                  help = 'data range (min max) [auto]')
parser.add_option('--gap', '--transparent',
                  dest = 'gap',
                  type = 'float', metavar = 'float',
                  help = 'value to treat as transparent [%default]')
parser.add_option('-d','--dimension',
                  dest = 'dimension',
                  type = 'int', nargs = 2, metavar = 'int int',
                  help = 'data dimension (width height) [native]')
parser.add_option('--color',
                  dest = 'color',
                  type = 'string', metavar = 'string',
                  help = 'color scheme [%default]')
parser.add_option('--invert',
                  dest = 'invert',
                  action = 'store_true',
                  help = 'invert color scheme')
parser.add_option('--abs',
                  dest = 'abs',
                  action = 'store_true',
                  help = 'magnitude of values')
parser.add_option('--log',
                  dest = 'log',
                  action = 'store_true',
                  help = 'log10 of values')
parser.add_option('--fliplr',
                  dest = 'flipLR',
                  action = 'store_true',
                  help = 'flip around vertical axis')
parser.add_option('--flipud',
                  dest = 'flipUD',
                  action = 'store_true',
                  help = 'flip around horizontal axis')
parser.add_option('--crop',
                  dest = 'crop',
                  type = 'int', nargs = 4, metavar = 'int int int int',
                  help = 'pixels cropped on left, right, top, bottom')
parser.add_option('-N','--pixelsize',
                  dest = 'pixelsize',
                  type = 'int', metavar = 'int',
                  help = 'pixel per data point')
parser.add_option('-x','--pixelsizex',
                  dest = 'pixelsizex',
                  type = 'int', metavar = 'int',
                  help = 'pixel per data point along x')
parser.add_option('-y','--pixelsizey',
                  dest = 'pixelsizey',
                  type = 'int', metavar = 'int',
                  help = 'pixel per data point along y')
parser.add_option('--show',
                  dest = 'show',
                  action = 'store_true',
                  help = 'show resulting image')

parser.set_defaults(label = None,
                    range = [0.0,0.0],
                    gap = None,
                    dimension = [],
                    abs = False,
                    log = False,
                    flipLR = False,
                    flipUD = False,
                    color = "gray",
                    invert = False,
                    crop = [0,0,0,0],
                    pixelsize  = 1,
                    pixelsizex = 1,
                    pixelsizey = 1,
                    show = False,
                   )

(options,filenames) = parser.parse_args()

if options.pixelsize > 1: (options.pixelsizex,options.pixelsizey) = [options.pixelsize]*2

# --- color palette ---------------------------------------------------------------------------------

theMap = damask.Colormap(predefined = options.color)
if options.invert: theMap = theMap.invert()
theColors = np.uint8(np.array(theMap.export(format = 'list',steps = 256))*255)

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              buffered = False,
                              labeled = options.label is not None,
                              readonly = True)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ process data ------------------------------------------

  missing_labels = table.data_readArray(options.label)
  if len(missing_labels) > 0:
    damask.util.croak('column {} not found.'.format(options.label))
    table.close(dismiss = True)                                                                     # close ASCIItable and remove empty file
    continue
# convert data to values between 0 and 1 and arrange according to given options
  if options.dimension != []: table.data = table.data.reshape(options.dimension[1],options.dimension[0])
  if options.abs:             table.data = np.abs(table.data)
  if options.log:             table.data = np.log10(table.data);options.range = np.log10(options.range)
  if options.flipLR:          table.data = np.fliplr(table.data)
  if options.flipUD:          table.data = np.flipud(table.data)

  mask = np.logical_or(table.data == options.gap, np.isnan(table.data))\
                                          if options.gap else np.logical_not(np.isnan(table.data))  # mask gap and NaN (if gap present)
  if np.all(np.array(options.range) == 0.0):
    options.range = [table.data[mask].min(),
                     table.data[mask].max()]
    damask.util.croak('data range: {0} – {1}'.format(*options.range))

  delta =      max(options.range) - min(options.range)
  avg   = 0.5*(max(options.range) + min(options.range))

  if delta * 1e8 <= avg:                                                                           # delta around numerical noise
    options.range = [min(options.range) - 0.5*avg, max(options.range) + 0.5*avg]                   # extend range to have actual data centered within

  table.data =         (table.data - min(options.range)) / \
               (max(options.range) - min(options.range))
  
  table.data = np.clip(table.data,0.0,1.0).\
                  repeat(options.pixelsizex,axis = 1).\
                  repeat(options.pixelsizey,axis = 0)

  mask =       mask.\
                  repeat(options.pixelsizex,axis = 1).\
                  repeat(options.pixelsizey,axis = 0)

  (height,width) = table.data.shape
  damask.util.croak('image dimension: {0} x {1}'.format(width,height))

  im = Image.fromarray(np.dstack((theColors[np.array(255*table.data,dtype = np.uint8)],
                                  255*mask.astype(np.uint8))), 'RGBA').\
             crop((       options.crop[0],
                          options.crop[2],
                   width -options.crop[1],
                   height-options.crop[3]))

# ------------------------------------------ output result -----------------------------------------

  im.save(sys.stdout if not name else
          os.path.splitext(name)[0]+ \
          ('' if options.label is None else '_'+options.label)+ \
          '.png',
          format = "PNG")

  table.close()                                                                                     # close ASCII table
  if options.show: im.show()
