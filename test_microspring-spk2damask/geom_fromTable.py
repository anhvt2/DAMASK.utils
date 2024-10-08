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

import os,sys,math,types,time
import scipy.spatial, numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])



# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog option(s) [ASCIItable(s)]', description = """
Generate geometry description and material configuration from position, phase, and orientation (or microstructure) data.

""", version = scriptID)

parser.add_option('--coordinates',
                  dest = 'pos',
                  type = 'string', metavar = 'string',
                  help = 'coordinates label (%default)')
parser.add_option('--phase',
                  dest = 'phase',
                  type = 'string', metavar = 'string',
                  help = 'phase label')
parser.add_option('--microstructure',
                  dest = 'microstructure',
                  type = 'string', metavar = 'string',
                  help = 'microstructure label')
parser.add_option('-t', '--tolerance',
                  dest = 'tolerance',
                  type = 'float', metavar = 'float',
                  help = 'angular tolerance for orientation squashing [%default]')
parser.add_option('-e', '--eulers',
                  dest = 'eulers',
                  type = 'string', metavar = 'string',
                  help = 'Euler angles label')
parser.add_option('-d', '--degrees',
                  dest = 'degrees',
                  action = 'store_true',
                  help = 'angles are given in degrees [%default]')
parser.add_option('-m', '--matrix',
                  dest = 'matrix',
                  type = 'string', metavar = 'string',
                  help = 'orientation matrix label')
parser.add_option('-a',
                  dest='a',
                  type = 'string', metavar = 'string',
                  help = 'crystal frame a vector label')
parser.add_option('-b',
                  dest='b',
                  type = 'string', metavar = 'string',
                  help = 'crystal frame b vector label')
parser.add_option('-c',
                  dest = 'c',
                  type =  'string', metavar='string',
                  help = 'crystal frame c vector label')
parser.add_option('-q', '--quaternion',
                  dest = 'quaternion',
                  type = 'string', metavar='string',
                  help = 'quaternion label')
parser.add_option('--axes',
                  dest = 'axes',
                  type = 'string', nargs = 3, metavar = ' '.join(['string']*3),
                  help = 'orientation coordinate frame in terms of position coordinate frame [same]')
parser.add_option('-s', '--symmetry',
                  dest = 'symmetry',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'crystal symmetry %default {{{}}} '.format(', '.join(damask.Symmetry.lattices[1:])))
parser.add_option('--homogenization',
                  dest = 'homogenization',
                  type = 'int', metavar = 'int',
                  help = 'homogenization index to be used [%default]')
parser.add_option('--crystallite',
                  dest = 'crystallite',
                  type = 'int', metavar = 'int',
                  help = 'crystallite index to be used [%default]')
parser.add_option('--verbose',
                  dest = 'verbose', action = 'store_true',
                  help = 'output extra info')

parser.set_defaults(symmetry       = [damask.Symmetry.lattices[-1]],
                    tolerance      = 0.0,
                    degrees        = False,
                    homogenization = 1,
                    crystallite    = 1,
                    verbose        = False,
                    pos            = 'pos',
                   )

(options,filenames) = parser.parse_args()

input = [options.eulers     is not None,
         options.a          is not None and \
         options.b          is not None and \
         options.c          is not None,
         options.matrix     is not None,
         options.quaternion is not None,
         options.microstructure is not None,
        ]

if np.sum(input) != 1:
  parser.error('need either microstructure label or exactly one orientation input format.')
if options.axes is not None and not set(options.axes).issubset(set(['x','+x','-x','y','+y','-y','z','+z','-z'])):
  parser.error('invalid axes {} {} {}.'.format(*options.axes))

(label,dim,inputtype) = [(options.eulers,3,'eulers'),
                         ([options.a,options.b,options.c],[3,3,3],'frame'),
                         (options.matrix,9,'matrix'),
                         (options.quaternion,4,'quaternion'),
                         (options.microstructure,1,'microstructure'),
                        ][np.where(input)[0][0]]                                                    # select input label that was requested
toRadians = math.pi/180.0 if options.degrees else 1.0                                               # rescale all angles to radians
threshold = np.cos(options.tolerance/2.*toRadians)                                                  # cosine of (half of) tolerance angle

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name,
                              outname = os.path.splitext(name)[-2]+'.geom' if name else name,
                              buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read head ---------------------------------------  

  table.head_read()                                                                                 # read ASCII header info

# ------------------------------------------ sanity checks ---------------------------------------  

  coordDim = table.label_dimension(options.pos)

  errors = []
  if not 3 >= coordDim >= 2:
    errors.append('coordinates "{}" need to have two or three dimensions.'.format(options.pos))
  if not np.all(table.label_dimension(label) == dim):
    errors.append('input "{}" needs to have dimension {}.'.format(label,dim))
  if options.phase and table.label_dimension(options.phase) != 1:
    errors.append('phase column "{}" is not scalar.'.format(options.phase))
  
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

  table.data_readArray([options.pos] \
                       + ([label] if isinstance(label, types.StringTypes) else label) \
                       + ([options.phase] if options.phase else []))
  
  if coordDim == 2:
    table.data = np.insert(table.data,2,np.zeros(len(table.data)),axis=1)                           # add zero z coordinate for two-dimensional input
    if options.verbose: damask.util.croak('extending to 3D...')
  if options.phase is None:
    table.data = np.column_stack((table.data,np.ones(len(table.data))))                             # add single phase if no phase column given
    if options.verbose: damask.util.croak('adding dummy phase info...')

# --------------- figure out size and grid ---------------------------------------------------------

  coords = [np.unique(table.data[:,i]) for i in range(3)]
  mincorner = np.array(map(min,coords))
  maxcorner = np.array(map(max,coords))
  grid   = np.array(map(len,coords),'i')
  size   = grid/np.maximum(np.ones(3,'d'), grid-1.0) * (maxcorner-mincorner)                        # size from edge to edge = dim * n/(n-1) 
  size   = np.where(grid > 1, size, min(size[grid > 1]/grid[grid > 1]))                             # spacing for grid==1 set to smallest among other spacings
  delta  = size/np.maximum(np.ones(3,'d'), grid)
  origin = mincorner - 0.5*delta                                                                    # shift from cell center to corner

  N = grid.prod()

  if  N != len(table.data):
    errors.append('data count {} does not match grid {}.'.format(len(table.data),' x '.join(map(repr,grid))))
  if   np.any(np.abs(np.log10((coords[0][1:]-coords[0][:-1])/delta[0])) > 0.01) \
    or np.any(np.abs(np.log10((coords[1][1:]-coords[1][:-1])/delta[1])) > 0.01) \
    or np.any(np.abs(np.log10((coords[2][1:]-coords[2][:-1])/delta[2])) > 0.01):
    errors.append('regular grid spacing {} violated.'.format(' x '.join(map(repr,delta))))

  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue
  
# ------------------------------------------ process data ------------------------------------------

  colOri = table.label_index(label)+(3-coordDim)                                                    # column(s) of orientation data followed by 3 coordinates

  if inputtype == 'microstructure':

    grain = table.data[:,colOri]
    nGrains = len(np.unique(grain))

  else:

    if options.verbose: bg = damask.util.backgroundMessage(); bg.start()                            # start background messaging

    colPhase = -1                                                                                   # column of phase data comes last
    if options.verbose: bg.set_message('sorting positions...')
    index  = np.lexsort((table.data[:,0],table.data[:,1],table.data[:,2]))                          # index of position when sorting x fast, z slow
    if options.verbose: bg.set_message('building KD tree...')
    KDTree = scipy.spatial.KDTree((table.data[index,:3]-mincorner) / delta)                         # build KDTree with dX = dY = dZ = 1 and origin 0,0,0
  
    statistics = {'global': 0, 'local': 0}
    grain = -np.ones(N,dtype = 'int32')                                                             # initialize empty microstructure
    orientations = []                                                                               # orientations
    multiplicity = []                                                                               # orientation multiplicity (number of group members)
    phases       = []                                                                               # phase info
    nGrains = 0                                                                                     # counter for detected grains
    existingGrains = np.arange(nGrains)
    myPos   = 0                                                                                     # position (in list) of current grid point

    tick = time.clock()
    if options.verbose: bg.set_message('assigning grain IDs...')

    for z in range(grid[2]):
      for y in range(grid[1]):
        for x in range(grid[0]):
          if (myPos+1)%(N/500.) < 1:
            time_delta = (time.clock()-tick) * (N - myPos) / myPos
            if options.verbose: bg.set_message('(%02i:%02i:%02i) processing point %i of %i (grain count %i)...'
                                            %(time_delta//3600,time_delta%3600//60,time_delta%60,myPos,N,nGrains))

          myData = table.data[index[myPos]]                                                         # read data for current grid point
          myPhase = int(myData[colPhase])
          mySym = options.symmetry[min(myPhase,len(options.symmetry))-1]                            # take last specified option for all with higher index

          if inputtype == 'eulers':
            o = damask.Orientation(Eulers = myData[colOri:colOri+3]*toRadians,
                                   symmetry = mySym)
          elif inputtype == 'matrix':
            o = damask.Orientation(matrix = myData[colOri:colOri+9].reshape(3,3).transpose(),
                                   symmetry = mySym)
          elif inputtype == 'frame':
            o = damask.Orientation(matrix = np.hstack((myData[colOri[0]:colOri[0]+3],
                                                       myData[colOri[1]:colOri[1]+3],
                                                       myData[colOri[2]:colOri[2]+3],
                                                      )).reshape(3,3),
                                   symmetry = mySym)
          elif inputtype == 'quaternion':
            o = damask.Orientation(quaternion = myData[colOri:colOri+4],
                                   symmetry = mySym)
          
          cos_disorientations = -np.ones(1,dtype='f')                                               # largest possible disorientation
          closest_grain = -1                                                                        # invalid neighbor

          if options.tolerance > 0.0:                                                               # only try to compress orientations if asked to
            neighbors = np.array(KDTree.query_ball_point([x,y,z], 3))                               # point indices within radius
# filter neighbors: skip myself, anyone further ahead (cannot yet have a grain ID), and other phases
            neighbors = neighbors[(neighbors < myPos) & \
                                  (table.data[index[neighbors],colPhase] == myPhase)]               
            grains = np.unique(grain[neighbors])                                                    # unique grain IDs among valid neighbors

            if len(grains) > 0:                                                                     # check immediate neighborhood first
              cos_disorientations = np.array([o.disorientation(orientations[grainID],
                                                               SST = False)[0].quaternion.w \
                                              for grainID in grains])                               # store disorientation per grainID
              closest_grain = np.argmax(cos_disorientations)                                        # grain among grains with closest orientation to myself
              match = 'local'

            if cos_disorientations[closest_grain] < threshold:                                      # orientation not close enough?
              grains = existingGrains[np.atleast_1d( (np.array(phases) == myPhase ) & \
                                                     (np.in1d(existingGrains,grains,invert=True)))] # other already identified grains (of my phase)

              if len(grains) > 0:
                cos_disorientations = np.array([o.disorientation(orientations[grainID],
                                                                 SST = False)[0].quaternion.w \
                                                for grainID in grains])                             # store disorientation per grainID
                closest_grain = np.argmax(cos_disorientations)                                      # grain among grains with closest orientation to myself
                match = 'global'

          if cos_disorientations[closest_grain] >= threshold:                                       # orientation now close enough?
            grainID = grains[closest_grain]
            grain[myPos] = grainID                                                                  # assign myself to that grain ...
            orientations[grainID] = damask.Orientation.average([orientations[grainID],o],
                                                               [multiplicity[grainID],1])           # update average orientation of best matching grain
            multiplicity[grainID] += 1
            statistics[match] += 1
          else:
            grain[myPos] = nGrains                                                                  # assign new grain to me ...
            nGrains += 1                                                                            # ... and update counter
            orientations.append(o)                                                                  # store new orientation for future comparison
            multiplicity.append(1)                                                                  # having single occurrence so far
            phases.append(myPhase)                                                                  # store phase info for future reporting
            existingGrains = np.arange(nGrains)                                                     # update list of existing grains

          myPos += 1

    if options.verbose:
      bg.stop()
      bg.join()
      damask.util.croak("{} seconds total.\n{} local and {} global matches.".\
                        format(time.clock()-tick,statistics['local'],statistics['global']))
    
    grain += 1                                                                                      # offset from starting index 0 to 1
     
# --- generate header ----------------------------------------------------------------------------

  info = {
          'grid':    grid,
          'size':    size,
          'origin':  origin,
          'microstructures': nGrains,
          'homogenization':  options.homogenization,
         }

  damask.util.croak(['grid     a b c:  {}'.format(' x '.join(map(str,info['grid']))),
                     'size     x y z:  {}'.format(' x '.join(map(str,info['size']))),
                     'origin   x y z:  {}'.format(' : '.join(map(str,info['origin']))),
                     'homogenization:  {}'.format(info['homogenization']),
                     'microstructures: {}'.format(info['microstructures']),
                    ])
    
# --- write header ---------------------------------------------------------------------------------

  formatwidth = 1+int(math.log10(info['microstructures']))

  if inputtype == 'microstructure':
    config_header = []
  else:
    config_header = ['<microstructure>']
    for i,phase in enumerate(phases):
      config_header += ['[Grain%s]'%(str(i+1).zfill(formatwidth)),
                        'crystallite %i'%options.crystallite,
                        '(constituent)\tphase %i\ttexture %s\tfraction 1.0'%(phase,str(i+1).rjust(formatwidth)),
                       ]
  
    config_header += ['<texture>']
    for i,orientation in enumerate(orientations):
      config_header += ['[Grain%s]'%(str(i+1).zfill(formatwidth)),
                        'axes\t%s %s %s'%tuple(options.axes) if options.axes is not None else '',
                        '(gauss)\tphi1 %g\tPhi %g\tphi2 %g\tscatter 0.0\tfraction 1.0'%tuple(orientation.asEulers(degrees = True)),
                       ]
  
  table.labels_clear()
  table.info_clear()
  table.info_append([scriptID + ' ' + ' '.join(sys.argv[1:])])
  table.head_putGeom(info)
  table.info_append(config_header)
  table.head_write()
  
# --- write microstructure information ------------------------------------------------------------

  table.data = grain.reshape(info['grid'][1]*info['grid'][2],info['grid'][0])
  table.data_writeArray('%%%ii'%(formatwidth),delimiter=' ')
  
#--- output finalization --------------------------------------------------------------------------

  table.close()
