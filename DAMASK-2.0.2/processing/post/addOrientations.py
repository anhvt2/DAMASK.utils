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

import os,sys,math
import numpy as np
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Add quaternion and/or Bunge Euler angle representation of crystal lattice orientation.
Orientation is given by quaternion, Euler angles, rotation matrix, or crystal frame coordinates
(i.e. component vectors of rotation matrix).
Additional (globally fixed) rotations of the lab frame and/or crystal frame can be applied.

""", version = scriptID)

outputChoices = ['quaternion','rodrigues','eulers']
parser.add_option('-o', '--output',
                  dest = 'output',
                  action = 'extend', metavar = '<string LIST>',
                  help = 'output orientation formats {{{}}}'.format(', '.join(outputChoices)))
parser.add_option('-s', '--symmetry',
                  dest = 'symmetry',
                  type = 'choice', choices = damask.Symmetry.lattices[1:], metavar='string',
                  help = 'crystal symmetry [%default] {{{}}} '.format(', '.join(damask.Symmetry.lattices[1:])))
parser.add_option('-d', '--degrees',
                  dest = 'degrees',
                  action = 'store_true',
                  help = 'angles are given in degrees [%default]')
parser.add_option('-R', '--labrotation',
                  dest='labrotation',
                  type = 'float', nargs = 4, metavar = ' '.join(['float']*4),
                  help = 'angle and axis of additional lab frame rotation')
parser.add_option('-r', '--crystalrotation',
                  dest='crystalrotation',
                  type = 'float', nargs = 4, metavar = ' '.join(['float']*4),
                  help = 'angle and axis of additional crystal frame rotation')
parser.add_option(      '--eulers',
                  dest = 'eulers',
                  type = 'string', metavar = 'string',
                  help = 'Euler angles label')
parser.add_option(      '--rodrigues',
                  dest = 'rodrigues',
                  type = 'string', metavar = 'string',
                  help = 'Rodrigues vector label')
parser.add_option(      '--matrix',
                  dest = 'matrix',
                  type = 'string', metavar = 'string',
                  help = 'orientation matrix label')
parser.add_option(       '--quaternion',
                  dest = 'quaternion',
                  type = 'string', metavar = 'string',
                  help = 'quaternion label')
parser.add_option('-a',
                  dest = 'a',
                  type = 'string', metavar = 'string',
                  help = 'crystal frame a vector label')
parser.add_option('-b',
                  dest = 'b',
                  type = 'string', metavar = 'string',
                  help = 'crystal frame b vector label')
parser.add_option('-c',
                  dest = 'c',
                  type = 'string', metavar = 'string',
                  help = 'crystal frame c vector label')

parser.set_defaults(output = [],
                    symmetry = damask.Symmetry.lattices[-1],
                    labrotation     = (0.,1.,1.,1.),                                                # no rotation about 1,1,1
                    crystalrotation = (0.,1.,1.,1.),                                                # no rotation about 1,1,1
                    degrees = False,
                   )

(options, filenames) = parser.parse_args()

options.output = map(lambda x: x.lower(), options.output)
if options.output == [] or (not set(options.output).issubset(set(outputChoices))):
  parser.error('output must be chosen from {}.'.format(', '.join(outputChoices)))

input = [options.eulers     is not None,
         options.rodrigues  is not None,
         options.a          is not None and \
         options.b          is not None and \
         options.c          is not None,
         options.matrix     is not None,
         options.quaternion is not None,
        ]

if np.sum(input) != 1: parser.error('needs exactly one input format.')

(label,dim,inputtype) = [(options.eulers,3,'eulers'),
                         (options.rodrigues,3,'rodrigues'),
                         ([options.a,options.b,options.c],[3,3,3],'frame'),
                         (options.matrix,9,'matrix'),
                         (options.quaternion,4,'quaternion'),
                        ][np.where(input)[0][0]]                                                    # select input label that was requested
toRadians = math.pi/180.0 if options.degrees else 1.0                                               # rescale degrees to radians
r = damask.Quaternion().fromAngleAxis(toRadians*options.crystalrotation[0],options.crystalrotation[1:]) # crystal frame rotation
R = damask.Quaternion().fromAngleAxis(toRadians*options.    labrotation[0],options.    labrotation[1:]) #     lab frame rotation

# --- loop over input files ------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:    table = damask.ASCIItable(name = name,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,name)

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks -----------------------------------------

  errors  = []
  remarks = []
  
  if not np.all(table.label_dimension(label) == dim):  errors.append('input {} does not have dimension {}.'.format(label,dim))
  else:  column = table.label_index(label)

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

# ------------------------------------------ assemble header ---------------------------------------

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  for output in options.output:
    if   output == 'quaternion': table.labels_append(['{}_{}_{}({})'.format(i+1,'quat',options.symmetry,label) for i in range(4)])
    elif output == 'rodrigues':  table.labels_append(['{}_{}_{}({})'.format(i+1,'rodr',options.symmetry,label) for i in range(3)])
    elif output == 'eulers':     table.labels_append(['{}_{}_{}({})'.format(i+1,'eulr',options.symmetry,label) for i in range(3)])
  table.head_write()

# ------------------------------------------ process data ------------------------------------------

  outputAlive = True
  while outputAlive and table.data_read():                                                          # read next data line of ASCII table
    if inputtype == 'eulers':
      o = damask.Orientation(Eulers   = np.array(map(float,table.data[column:column+3]))*toRadians,
                             symmetry = options.symmetry).reduced()
    elif inputtype == 'rodrigues':
      o = damask.Orientation(Rodrigues= np.array(map(float,table.data[column:column+3])),
                             symmetry = options.symmetry).reduced()
    elif inputtype == 'matrix':
      o = damask.Orientation(matrix   = np.array(map(float,table.data[column:column+9])).reshape(3,3).transpose(),
                             symmetry = options.symmetry).reduced()
    elif inputtype == 'frame':
      o = damask.Orientation(matrix = np.array(map(float,table.data[column[0]:column[0]+3] + \
                                                         table.data[column[1]:column[1]+3] + \
                                                         table.data[column[2]:column[2]+3])).reshape(3,3),
                             symmetry = options.symmetry).reduced()
    elif inputtype == 'quaternion':
      o = damask.Orientation(quaternion = np.array(map(float,table.data[column:column+4])),
                             symmetry   = options.symmetry).reduced()

    o.quaternion = r*o.quaternion*R                                                                 # apply additional lab and crystal frame rotations

    for output in options.output:
      if   output == 'quaternion': table.data_append(o.asQuaternion())
      elif output == 'rodrigues':  table.data_append(o.asRodrigues())
      elif output == 'eulers':     table.data_append(o.asEulers('Bunge', degrees=options.degrees))
    outputAlive = table.data_write()                                                                # output processed line

# ------------------------------------------ output finalization -----------------------------------  

  table.close()                                                                                     # close ASCII tables
