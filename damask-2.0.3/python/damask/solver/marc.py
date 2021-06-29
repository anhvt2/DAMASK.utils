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


class Marc(Solver):

  def __init__(self):
    self.solver = 'Marc'


#--------------------------
  def version(self):
    import damask.environment

    return damask.environment.Environment().options['MARC_VERSION']
      
    
#--------------------------
  def libraryPath(self,release = ''):
    import os,damask.environment

    MSCpath     = damask.environment.Environment().options['MSC_ROOT']
    if len(release) == 0: release = self.version()
    
    path = '{}/mentat{}/shlib/linux64'.format(MSCpath,release)

    return path if os.path.exists(path) else ''


#--------------------------
  def toolsPath(self,release = ''):
    import os,damask.environment

    MSCpath = damask.environment.Environment().options['MSC_ROOT']
    if len(release) == 0: release = self.version()

    path = '%s/marc%s/tools'%(MSCpath,release)

    return path if os.path.exists(path) else ''
  

#--------------------------
  def submit_job(self,
                 release      = '',
                 model        = 'model',
                 job          = 'job1',
                 logfile      = None,
                 compile      = False,
                 optimization ='',
                ):

    import os,damask.environment
    import subprocess,shlex
    
    if len(release) == 0: release = self.version()
    damaskEnv = damask.environment.Environment()
    
    user = 'not found'

    if compile:
      if os.path.isfile(os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc{}.f90'.format(release))):
        user = os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc{}'.format(release))
    else:
      if os.path.isfile(os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc{}.marc'.format(release))):
        user = os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc{}'.format(release))

    # Define options [see Marc Installation and Operation Guide, pp 23]
    script = 'run_damask_{}mp'.format(optimization)
    
    cmd = os.path.join(self.toolsPath(release),script) + \
          ' -jid ' + model + '_' + job + \
          ' -nprocd 1  -autorst 0 -ci n  -cr n  -dcoup 0 -b no -v no'

    if compile: cmd += ' -u ' + user+'.f90' + ' -save y'
    else:       cmd += ' -prog ' + user

    print('job submission with{} compilation: {}'.format({False:'out',True:''}[compile],user))
    if logfile:
      log = open(logfile, 'w')
    print(cmd)
    process = subprocess.Popen(shlex.split(cmd),stdout = log,stderr = subprocess.STDOUT)
    log.close()
    process.wait()
      
#--------------------------
  def exit_number_from_outFile(self,outFile=None):
    import string
    exitnumber = -1
    fid_out = open(outFile,'r')
    for line in fid_out:
      if (string.find(line,'tress iteration') is not -1):
        print(line)
      elif (string.find(line,'Exit number') is not -1):
        substr = line[string.find(line,'Exit number'):len(line)]
        exitnumber = int(substr[12:16])

    fid_out.close()
    return exitnumber
