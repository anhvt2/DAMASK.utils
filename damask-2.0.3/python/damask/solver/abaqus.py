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

from .solver import Solver
import damask
import subprocess

class Abaqus(Solver):

  def __init__(self,version=''):                                                                    # example version string: 2017
    self.solver='Abaqus'
    if version =='':
      version = damask.Environment().options['ABAQUS_VERSION']
    else:
      self.version = version

  def return_run_command(self,model):
    env=damask.Environment()
    try:
      cmd='abq'+self.version
      subprocess.check_output([cmd,'information=release'])
    except OSError:                                                                                 # link to abqXXX not existing
      cmd='abaqus'
      process = subprocess.Popen(['abaqus','information=release'],stdout = subprocess.PIPE,stderr = subprocess.PIPE)
      detectedVersion = process.stdout.readlines()[1].split()[1].decode('utf-8')
      if self.version != detectedVersion:
        raise Exception('found Abaqus version {}, but requested {}'.format(detectedVersion,self.version))
    return '{} -job {} -user {}/src/DAMASK_abaqus interactive'.format(cmd,model,env.rootDir())
