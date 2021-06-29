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


import damask.solver

class Solver():
  """
  General class for solver specific functionality.

  Sub-classed by the individual solvers.
  """
  
  def __init__(self,solver=''):
    solverClass = {
                      'spectral': damask.solver.Spectral,
                      'marc':     damask.solver.Marc,
                    }
    if solver.lower() in list(solverClass.keys()):
      self.__class__=solverClass[solver.lower()]
      self.__init__()

