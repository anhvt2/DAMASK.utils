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


from .solver import Solver


class Marc(Solver):

  def __init__(self):
    self.solver = 'Marc'
    self.releases = { \
              '2016':  ['linux64',''],
              '2015':  ['linux64',''],
              '2014.2':['linux64',''],
              '2014'  :['linux64',''],
             }


#--------------------------
  def version(self):
    import os,damask.environment

    MSCpath = damask.environment.Environment().options['MSC_ROOT']
    
    for release,subdirs in sorted(list(self.releases.items()),reverse=True):
      for subdir in subdirs:
        path = '%s/mentat%s/shlib/%s'%(MSCpath,release,subdir)
        if os.path.exists(path): return release
        else: continue
    
    return ''
      
    
#--------------------------
  def libraryPath(self,releases = []):
    import os,damask.environment

    MSCpath = damask.environment.Environment().options['MSC_ROOT']
    
    if len(releases) == 0: releases = list(self.releases.keys())
    if type(releases) is not list: releases = [releases]
    for release in sorted(releases,reverse=True):
      if release not in self.releases: continue
      for subdir in self.releases[release]:
        libPath = '%s/mentat%s/shlib/%s'%(MSCpath,release,subdir)
        if os.path.exists(libPath): return libPath
        else: continue
    
    return ''
  

#--------------------------
  def toolsPath(self,release = ''):
    import os,damask.environment

    MSCpath = damask.environment.Environment().options['MSC_ROOT']
    
    if len(release) == 0: release = self.version()
    path = '%s/marc%s/tools'%(MSCpath,release)
    if os.path.exists(path): return path
    else: return ''
  

#--------------------------
  def submit_job(self,
                 release      = '',
                 model        = 'model',
                 job          = 'job1',
                 logfile      = None,
                 compile      = False,
                 optimization ='',
                 openMP       = False
                ):

    import os,damask.environment
    import subprocess,shlex
    
    if len(release) == 0: release = self.version()

    if release not in self.releases:
      raise Exception("Unknown MSC.Marc Version %s"%release)
      

    damaskEnv = damask.environment.Environment()
    
    user = os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc')                                   # might be updated if special version (symlink) is found
    if compile:
      if os.path.isfile(os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc%s.f90'%release)):
        user = os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc%s'%release)
    else:
      if os.path.isfile(os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc%s.marc'%release)):
        user = os.path.join(damaskEnv.relPath('src/'),'DAMASK_marc%s'%release)

    # Define options [see Marc Installation and Operation Guide, pp 23]
    script = 'run_damask%s'%({False:'',True:'_'}[optimization!='' or openMP])
    script =  script+'%s%s'%({False:'',True:optimization}[optimization!=''],{False:'',True:'mp'}[openMP])
    
    cmd = os.path.join(self.toolsPath(release),script) + \
          ' -jid ' + model + '_' + job + \
          ' -nprocd 1  -autorst 0 -ci n  -cr n  -dcoup 0 -b no -v no'

    if compile: cmd += ' -u ' + user+'.f90' + ' -save y'
    else:       cmd += ' -prog ' + user

    print('job submission with%s compilation: %s'%({False:'out',True:''}[compile],user))
    if logfile:
      log = open(logfile, 'w')
    print(cmd)
    self.p = subprocess.Popen(shlex.split(cmd),stdout = log,stderr = subprocess.STDOUT)
    log.close()
    self.p.wait()
      
#--------------------------
  def exit_number_from_outFile(self,outFile=None):
    import string
    exitnumber = -1
    fid_out = open(outFile,'r')
    for ln in fid_out:
      if (string.find(ln,'tress iteration') is not -1):
        print(ln)
      elif (string.find(ln,'Exit number') is not -1):
        substr = ln[string.find(ln,'Exit number'):len(ln)]
        exitnumber = int(substr[12:16])

    fid_out.close()
    return exitnumber
