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

# Makes postprocessing routines acessible from everywhere.
import os,sys
import damask

damaskEnv = damask.Environment()
baseDir = damaskEnv.relPath('processing/')
binDir = damaskEnv.options['DAMASK_BIN']

if not os.path.isdir(binDir):
  os.mkdir(binDir)

#define ToDo list
processing_subDirs    = ['pre',
                         'post',
                         'misc',
                        ]
processing_extensions = ['.py',
                         '.sh',
                        ]

sys.stdout.write('\nsymbolic linking...\n')

for subDir in processing_subDirs:
  theDir = os.path.abspath(os.path.join(baseDir,subDir))

  sys.stdout.write('\n'+binDir+' ->\n'+theDir+damask.util.deemph(' ...')+'\n')

  for theFile in os.listdir(theDir):
    theName,theExt = os.path.splitext(theFile)
    if theExt in processing_extensions:                                                             # only consider files with proper extensions

      src      = os.path.abspath(os.path.join(theDir,theFile))
      sym_link = os.path.abspath(os.path.join(binDir,theName))

      if os.path.lexists(sym_link):
        os.remove(sym_link)
        output = theName+damask.util.deemph(theExt)
      else:
        output = damask.util.emph(theName)+damask.util.deemph(theExt)

      sys.stdout.write(damask.util.deemph('... ')+output+'\n')
      os.symlink(src,sym_link)


sys.stdout.write('\npruning broken links...\n')

brokenLinks = 0

for filename in os.listdir(binDir):
  path = os.path.join(binDir,filename)
  if os.path.islink(path) and not os.path.exists(path):
    sys.stdout.write(' '+damask.util.delete(path)+'\n')
    os.remove(path)
    brokenLinks += 1

sys.stdout.write(('none.' if brokenLinks == 0 else '')+'\n')
