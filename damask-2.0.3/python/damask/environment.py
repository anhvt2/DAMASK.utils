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

import os,subprocess,shlex,re

class Environment():
  __slots__ = [ \
                'options',
              ]

  def __init__(self):
    self.options = {}
    self.get_options()

  def relPath(self,relative = '.'):
    return os.path.join(self.rootDir(),relative)

  def rootDir(self):
    return os.path.normpath(os.path.join(os.path.realpath(__file__),'../../../'))

  def get_options(self):
    with open(self.relPath(self.rootDir()+'/CONFIG')) as configFile:
      for line in configFile:
        l = re.sub('^set ', '', line).strip()                                                       # remove "set" (tcsh) when setting variables
        if l and not l.startswith('#'):
          items = re.split(r'\s*=\s*',l)
          if len(items) == 2: 
            self.options[items[0].upper()] = \
              re.sub('\$\{*DAMASK_ROOT\}*',self.rootDir(),os.path.expandvars(items[1]))             # expand all shell variables and DAMASK_ROOT
      
  def isAvailable(self,software,Nneeded =-1):
    licensesNeeded = {'abaqus'  :5,
                      'standard':5
                      }
    if Nneeded == -1: Nneeded = licensesNeeded[software]
    try:
      cmd = """ ssh mulicense2 "/lm-status | grep 'Users of %s: ' | cut -d' ' -f7,13" """%software
      process = subprocess.Popen(shlex.split(cmd),stdout = subprocess.PIPE,stderr = subprocess.PIPE)
      licenses = list(map(int, process.stdout.readline().split()))
      try:
        if licenses[0]-licenses[1] >= Nneeded:
          return 0
        else:
          print('%s missing licenses for %s'%(licenses[1] + Nneeded - licenses[0],software))
          return licenses[1] + Nneeded - licenses[0]
      except IndexError:
        print('Could not retrieve license information for %s'%software)
        return 127
    except:
      return 126
