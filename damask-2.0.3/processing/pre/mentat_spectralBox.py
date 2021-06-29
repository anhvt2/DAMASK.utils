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
sys.path.append(damask.solver.Marc().libraryPath())

#-------------------------------------------------------------------------------------------------
def outMentat(cmd,locals):
  if cmd[0:3] == '(!)':
    exec(cmd[3:])
  elif cmd[0:3] == '(?)':
    cmd = eval(cmd[3:])
    py_mentat.py_send(cmd)
  else:
    py_mentat.py_send(cmd)
  return

#-------------------------------------------------------------------------------------------------
def outFile(cmd,locals,dest):
  if cmd[0:3] == '(!)':
    exec(cmd[3:])
  elif cmd[0:3] == '(?)':
    cmd = eval(cmd[3:])
    dest.write(cmd+'\n')
  else:
    dest.write(cmd+'\n')
  return

#-------------------------------------------------------------------------------------------------
def output(cmds,locals,dest):
  for cmd in cmds:
    if isinstance(cmd,list):
      output(cmd,locals,dest)
    else:
      if dest == 'Mentat':
        outMentat(str(cmd),locals)
      else:
        outFile(str(cmd),locals,dest)
  return


  
#-------------------------------------------------------------------------------------------------
def init():
  return [
    "|"+' '.join([scriptID] + sys.argv[1:]),
    "*draw_manual",              # prevent redrawing in Mentat, should be much faster
    "*new_model yes",
    "*reset",
    "*select_clear",
    "*set_element_class hex8",
    "*set_nodes off",
    "*elements_solid",
    "*show_view 4",
    "*reset_view",
    "*view_perspective",
    "*redraw",
    ]


#-------------------------------------------------------------------------------------------------
def mesh(r,d):
    return [
  "*add_nodes",
  "%f %f %f"%(0.0,0.0,0.0),
  "%f %f %f"%(0.0,0.0,d[2]),
  "%f %f %f"%(0.0,d[1],d[2]),
  "%f %f %f"%(0.0,d[1],0.0),
  "%f %f %f"%(-d[0],0.0,0.0),
  "%f %f %f"%(-d[0],0.0,d[2]),
  "%f %f %f"%(-d[0],d[1],d[2]),
  "%f %f %f"%(-d[0],d[1],0.0),
  "*add_elements",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "*sub_divisions",
  "%i %i %i"%(r[2],r[1],r[0]),
  "*subdivide_elements",
  "all_existing",
  "*set_sweep_tolerance",
  "%f"%(float(min(d))/max(r)/2.0),
  "*sweep_all",
  "*renumber_all",
  "*set_move_scale_factor x -1",
  "*move_elements",
  "all_existing",
  "*flip_elements",
  "all_existing",
  "*fill_view",
    ]


#-------------------------------------------------------------------------------------------------
def material():
  cmds = [\
  "*new_mater standard",
  "*mater_option general:state:solid",
  "*mater_option structural:type:hypo_elast",
  "*mater_name",
  "hypela2",
  "*add_mater_elements",
  "all_existing",
  "*geometry_type mech_three_solid",
#  "*geometry_option red_integ_capacity:on", reduced integration with one IP gave trouble being always OUTDATED...
  "*add_geometry_elements",
  "all_existing",
  ]
  
  return cmds
  

#-------------------------------------------------------------------------------------------------
def geometry():
  cmds = [\
  "*geometry_type mech_three_solid",
#  "*geometry_option red_integ_capacity:on",
  "*add_geometry_elements",
  "all_existing",
# we are NOT using reduced integration (type 117) but opt for /elementhomogeneous/ in the respective phase description (material.config)
  "*element_type 7",
  "all_existing",
  ]
  
  return cmds
  

#-------------------------------------------------------------------------------------------------
def initial_conditions(homogenization,microstructures):
  elements = []
  element = 0
  for id in microstructures:
    element += 1 
    if len(elements) < id:
      for i in range(id-len(elements)):
        elements.append([])
    elements[id-1].append(element)

  cmds = [\
    "*new_icond",
    "*icond_name _temperature",
    "*icond_type state_variable",
    "*icond_param_value state_var_id 1",
    "*icond_dof_value var 300",
    "*add_icond_elements",
    "all_existing",
    "*new_icond",
    "*icond_name _homogenization",
    "*icond_type state_variable",
    "*icond_param_value state_var_id 2",
    "*icond_dof_value var %i"%homogenization,
    "*add_icond_elements",
    "all_existing",
    ]

  for grain,elementList in enumerate(elements):
    cmds.append([\
            "*new_icond",
            "*icond_name microstructure_%i"%(grain+1),
            "*icond_type state_variable",
            "*icond_param_value state_var_id 3",
            "*icond_dof_value var %i"%(grain+1),
            "*add_icond_elements",
            elementList,
            "#",
                ])
  return cmds


#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Generate MSC.Marc FE hexahedral mesh from geom file.

""", version = scriptID)

parser.add_option('-p', '--port',
                  dest = 'port',
                  type = 'int', metavar = 'int',
                  help = 'Mentat connection port [%default]')
parser.add_option('--homogenization',
                  dest = 'homogenization',
                  type = 'int', metavar = 'int',
                  help = 'homogenization index to be used [auto]')

parser.set_defaults(port           = None,
                    homogenization = None,
)

(options, filenames) = parser.parse_args()

if options.port:
  try:
    import py_mentat
  except:
    parser.error('no valid Mentat release found.')

# --- loop over input files ------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name    = name,
                              outname = os.path.splitext(name)[0]+'.proc' if name else name,
                              buffered = False, labeled = False)
  except: continue
  damask.util.report(scriptName,name)

# --- interpret header ----------------------------------------------------------------------------

  table.head_read()
  info,extra_header = table.head_getGeom()
  if options.homogenization: info['homogenization'] = options.homogenization
    
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

  microstructure = table.microstructure_read(info['grid']).reshape(info['grid'].prod(),order='F')   # read microstructure

  cmds = [\
    init(),
    mesh(info['grid'],info['size']),
    material(),
    geometry(),
    initial_conditions(info['homogenization'],microstructure),
    '*identify_sets',
    '*show_model',
    '*redraw',
    '*draw_automatic',
  ]
  
  outputLocals = {}
  if options.port:
    py_mentat.py_connect('',options.port)
    output(cmds,outputLocals,'Mentat')
    py_mentat.py_disconnect()
  else:
    output(cmds,outputLocals,table.__IO__['out'])                                                   # bad hack into internals of table class...

  table.close()
