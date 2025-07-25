#!/usr/bin/env python3
# Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
from io import StringIO
from optparse import OptionParser

import damask

script_name = os.path.splitext(os.path.basename(__file__))[0]
script_id   = ' '.join([script_name,damask.version])

#-------------------------------------------------------------------------------------------------
def outMentat(cmd,locals):
  if cmd[0:3] == '(!)':
    exec(cmd[3:])
  elif cmd[0:3] == '(?)':
    cmd = eval(cmd[3:])
    py_mentat.py_send(cmd)
  else:
    py_mentat.py_send(cmd)

#-------------------------------------------------------------------------------------------------
def outFile(cmd,locals,dest):
  if cmd[0:3] == '(!)':
    exec(cmd[3:])
  elif cmd[0:3] == '(?)':
    cmd = eval(cmd[3:])
    dest.write(cmd+'\n')
  else:
    dest.write(cmd+'\n')

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


#-------------------------------------------------------------------------------------------------
def init():
  return [
    "|"+' '.join([script_id] + sys.argv[1:]),
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
  "%f %f %f"%(d[0],0.0,0.0),
  "%f %f %f"%(d[0],d[1],0.0),
  "%f %f %f"%(0.0,d[1],0.0),
  "%f %f %f"%(0.0,0.0,d[2]),
  "%f %f %f"%(d[0],0.0,d[2]),
  "%f %f %f"%(d[0],d[1],d[2]),
  "%f %f %f"%(0.0,d[1],d[2]),
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
  "%i %i %i"%(r[0],r[1],r[2]),
  "*subdivide_elements",
  "all_existing",
  "*set_sweep_tolerance",
  "%f"%(float(min(d))/max(r)/2.0),
  "*sweep_all",
  "*remove_unused_nodes",
  "*set_renumber_direction",
  "%i %i %i"%(1,r[0],r[0]*r[1]),
  "*renumber_elements_directed",
  "*renumber_nodes_directed",
    ]


#-------------------------------------------------------------------------------------------------
def materials():
  return [\
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


#-------------------------------------------------------------------------------------------------
def geometry():
  return [\
  "*geometry_type mech_three_solid",
#  "*geometry_option red_integ_capacity:on",
  "*add_geometry_elements",
  "all_existing",
# we are NOT using reduced integration (type 117) but opt for /elementhomogeneous/ in the respective phase description (material.config)
  "*element_type 7",
  "all_existing",
  ]


#-------------------------------------------------------------------------------------------------
def initial_conditions(material):
  elements = []
  element = 0
  for id in material:
    element += 1
    if len(elements) < id:
      for i in range(id-len(elements)):
        elements.append([])
    elements[id-1].append(element)

  cmds = [\
    "*new_icond",
    "*icond_name _temperature",
    "*icond_type elem_temperature_state",
    "*icond_dof_value var 300",
    "*add_icond_elements",
    "all_existing",
    ]

  for grain,elementList in enumerate(elements):
    cmds.append([\
            "*new_icond",
            "*icond_name material_%i"%(grain+1),
            "*icond_type elem_user_state",
            "*icond_param_value state_var_id 2",
            "*icond_dof_value var %i"%(grain),
            "*add_icond_elements",
            elementList,
            "#",
                ])
  return cmds


#--------------------------------------------------------------------------------------------------
#                                MAIN
#--------------------------------------------------------------------------------------------------

parser = OptionParser(usage='%prog options [file[s]]', description = """
Generate MSC.Marc FE hexahedral mesh from geom file.

""", version = script_id)

parser.add_option('-p', '--port',
                  dest = 'port',
                  type = 'int', metavar = 'int',
                  help = 'Mentat connection port [%default]')

parser.set_defaults(port = None,
                   )

(options, filenames) = parser.parse_args()

if options.port is not None:
    try:
        sys.path.append(str(damask.solver.Marc().library_path))
        import py_mentat
    except ImportError:
        parser.error('no valid Mentat release found')

# --- loop over input files ------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
    print(script_name+': '+name)

    geom = damask.GeomGrid.load(StringIO(''.join(sys.stdin.read())) if name is None else name)
    material = geom.material.flatten(order='F')

    cmds = [\
      init(),
      mesh(geom.cells,geom.size),
      materials(),
      geometry(),
      initial_conditions(material),
      '*identify_sets',
      '*show_model',
      '*redraw',
      '*draw_automatic',
      '*fill_view',
    ]

    output_locals = {}
    if options.port:
        py_mentat.py_connect('',options.port)
        output(cmds,output_locals,'Mentat')
        py_mentat.py_disconnect()
    else:
        with sys.stdout if name is None else open(os.path.splitext(name)[0]+'.proc','w') as f:
            output(cmds,output_locals,f)
