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

import sys,os,re
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

# -----------------------------
def ParseOutputFormat(filename,what,me):
  format = {'outputs':{},'specials':{'brothers':[]}}

  outputmetafile = filename+'.output'+what
  try:
    myFile = open(outputmetafile)
  except:
    print('Could not open file %s'%outputmetafile)
    raise
  else: 
    content = myFile.readlines()
    myFile.close()

  tag = ''
  tagID = 0
  for line in content:
    if re.match("\s*$",line) or re.match("#",line):     # skip blank lines and comments
      continue
    m = re.match("\[(.+)\]",line)             # look for block indicator
    if m:                         # next section
      tag = m.group(1)
      tagID += 1
      format['specials']['brothers'].append(tag)
      if tag == me or (me.isdigit() and tagID == int(me)):
        format['specials']['_id'] = tagID
        format['outputs'] = []
        tag = me
    else:                         # data from section
      if tag == me:
        (output,length) = line.split()
        output.lower()
        if length.isdigit():
          length = int(length)
        if re.match("\((.+)\)",output):         # special data, (e.g. (Ngrains)
          format['specials'][output] = length
        elif length > 0:
          format['outputs'].append([output,length])
  return format


parser = OptionParser(option_class=damask.extendableOption, usage='%prog [option(s)] Abaqus.Inputfile(s)', description = """
Transfer the output variables requested in the material.config to
properly labelled user-defined variables within the Abaqus input file (*.inp).

Requires the files 
<modelname_jobname>.output<Homogenization/Crystallite/Constitutive>
that are written during the first run of the model.

Specify which user block format you want to apply by stating the homogenization, crystallite, and phase identifiers.
Or have an existing set of user variables copied over from another *.inp file.

""", version = scriptID)

parser.add_option('-m', dest='number', type='int', metavar = 'int',
                  help='maximum requested User Defined Variable [%default]')
parser.add_option('--homogenization', dest='homog', metavar = 'string',
                  help='homogenization name or index [%default]')
parser.add_option('--crystallite', dest='cryst', metavar = 'string',
                  help='crystallite identifier name or index [%default]')
parser.add_option('--phase', dest='phase', metavar = 'string',
                  help='phase identifier name or index [%default]')
parser.add_option('--use', dest='useFile', metavar = 'string',
                  help='optionally parse output descriptors from '+
                       'outputXXX files of given name')
parser.add_option('--option', dest='damaskOption', metavar = 'string',
                  help='Add DAMASK option to input file, e.g. "periodic x z"')
 
parser.set_defaults(number = 0,
                    homog = '1',
                    cryst = '1',
                    phase = '1')

(options, files) = parser.parse_args()

if not files:
  parser.error('no file(s) specified.')

me = {  'Homogenization':   options.homog,
        'Crystallite':      options.cryst,
        'Constitutive':     options.phase,
     }


for myFile in files:
  damask.util.report(scriptName,myFile)
  if options.useFile is not None:
    formatFile = os.path.splitext(options.useFile)[0]
  else:
    formatFile = os.path.splitext(myFile)[0]
  myFile = os.path.splitext(myFile)[0]+'.inp'
  if not os.path.lexists(myFile):
    print('{} not found'.format(myFile))
    continue
    
  print('Scanning format files of: {}'.format(formatFile))

  if options.number < 1:
    outputFormat = {}

    for what in me:
      outputFormat[what] = ParseOutputFormat(formatFile,what,me[what])
      if '_id' not in outputFormat[what]['specials']:
        print("'{}' not found in <{}>".format(me[what],what))
        print('\n'.join(map(lambda x:'  '+x,outputFormat[what]['specials']['brothers'])))
        sys.exit(1)

    UserVars = ['HomogenizationCount']
    for var in outputFormat['Homogenization']['outputs']:
      if var[1] > 1:
        UserVars += ['%i_%s'%(i+1,var[0]) for i in range(var[1])]
      else:
        UserVars += ['%s'%(var[0]) for i in range(var[1])]

    UserVars += ['GrainCount']

    for grain in range(outputFormat['Homogenization']['specials']['(ngrains)']):
      UserVars += ['%i_CrystalliteCount'%(grain+1)]
      for var in outputFormat['Crystallite']['outputs']:
        if var[1] > 1:
          UserVars += ['%i_%i_%s'%(grain+1,i+1,var[0]) for i in range(var[1])]
        else:
          UserVars += ['%i_%s'%(grain+1,var[0]) for i in range(var[1])]

      UserVars += ['%i_ConstitutiveCount'%(grain+1)]
      for var in outputFormat['Constitutive']['outputs']:
        if var[1] > 1:
          UserVars += ['%i_%i_%s'%(grain+1,i+1,var[0]) for i in range(var[1])]
        else:
          UserVars += ['%i_%s'%(grain+1,var[0]) for i in range(var[1])]

# Now change *.inp file(s)        
  print('Adding labels to:         {}'.format(myFile))
  inFile = open(myFile)
  input = inFile.readlines()
  inFile.close()
  output = open(myFile,'w')
  thisSection = ''
  if options.damaskOption is not None:
    output.write('$damask {0}\n'.format(options.damaskOption))
  for line in input:
    #Abaqus keyword line begins with: *keyword, argument1, ...
    m = re.match('([*]\w+)\s',line)
    if m:
      lastSection = thisSection
      thisSection = m.group(1)
      if (lastSection.upper() == '*DEPVAR' and thisSection.upper() == '*USER'):      # Abaqus keyword can be upper or lower case
        if options.number > 0:
          output.write('{}\n'.format(options.number))                                # Abaqus needs total number of SDVs in the line after *Depvar keyword
        else:
          output.write('{}\n'.format(len(UserVars)))
          
          for i in range(len(UserVars)): 
             output.write('%i,"%i%s","%i%s"\n'%(i+1,0,UserVars[i],0,UserVars[i]))    #index,output variable key,output variable description
    if (thisSection.upper() != '*DEPVAR' or not re.match('\s*\d',line)):
      output.write(line)
  output.close()
