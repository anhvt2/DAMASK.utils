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

import os,sys,math,re,time,struct
import damask
from optparse import OptionParser, OptionGroup

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


fileExtensions = { \
                   'marc': ['.t16',],
                   'spectral': ['.spectralOut',],
                 }


# -----------------------------
class vector:   # mimic py_post node object
  x,y,z = [None,None,None]

  def __init__(self,coords):
    self.x = coords[0]
    self.y = coords[1]
    self.z = coords[2]

# -----------------------------
class element:     # mimic py_post element object
  items = []
  type = None

  def __init__(self,nodes,type):
    self.items = nodes
    self.type = type

# -----------------------------
class elemental_scalar:   # mimic py_post element_scalar object
  id = None
  value = None

  def __init__(self,node,value):
    self.id = node
    self.value = value


# -----------------------------
class MPIEspectral_result:    # mimic py_post result object

  file = None
  dataOffset = 0
  N_elemental_scalars = 0
  grid = [0,0,0]
  size = [0.0,0.0,0.0]
  theTitle = ''
  wd = ''
  geometry = ''
  extrapolate = ''
  N_loadcases = 0
  N_increments = 0
  N_positions = 0
  _frequencies = []
  _increments = []
  _times = []
  increment = 0
  startingIncrement = 0
  position = 0
# this is a dummy at the moment, we need to parse the load file and figure out what time a particular increment corresponds to
  time = 0.0
  N_nodes = 0
  N_node_scalars = 0
  N_elements = 0
  N_element_scalars = 0
  N_element_tensors = 0

  def __init__(self,filename):

    self.file = open(filename, 'rb')
    self.filesize = os.path.getsize(filename)
    self.dataOffset = 0
    while self.dataOffset < self.filesize:
      self.file.seek(self.dataOffset)
      if self.file.read(3) == 'eoh': break
      self.dataOffset += 1
    self.dataOffset   += 7
#search first for the new keywords with ':', if not found try to find the old ones
    self.theTitle     = self._keyedString('load:')
    if self.theTitle is None:
      self.theTitle     = self._keyedString('load')

    self.wd           = self._keyedString('workingdir:')
    if self.wd is None:
      self.wd           = self._keyedString('workingdir')

    self.geometry     = self._keyedString('geometry:')
    if self.geometry is None:
      self.geometry     = self._keyedString('geometry')

    self.N_loadcases  = self._keyedPackedArray('loadcases:',count=1,type='i')[0]
    if self.N_loadcases is None:
      self.N_loadcases  = self._keyedPackedArray('loadcases',count=1,type='i')[0]

    self._frequencies = self._keyedPackedArray('frequencies:',count=self.N_loadcases,type='i')
    if all ( i is None for i in self._frequencies):
      self._frequencies = self._keyedPackedArray('frequencies',count=self.N_loadcases,type='i')

    self._increments  = self._keyedPackedArray('increments:',count=self.N_loadcases,type='i')
    if all (i is None for i in self._increments):
      self._increments  = self._keyedPackedArray('increments',count=self.N_loadcases,type='i')

    self.startingIncrement = self._keyedPackedArray('startingIncrement:',count=1,type='i')[0]
    if self.startingIncrement is None:
      self.startingIncrement = self._keyedPackedArray('startingIncrement',count=1,type='i')[0]


    self._times       = self._keyedPackedArray('times:',count=self.N_loadcases,type='d')
    if all (i is None for i in self._times):
      self._times       = self._keyedPackedArray('times',count=self.N_loadcases,type='d')

    self._logscales   = self._keyedPackedArray('logscales:',count=self.N_loadcases,type='i')
    if all (i is None for i in self._logscales):
      self._logscales   = self._keyedPackedArray('logscales',count=self.N_loadcases,type='i')

    self.size         = self._keyedPackedArray('size:',count=3,type='d')
    if self.size == [None,None,None]:                                                               # no 'size' found, try legacy alias 'dimension'
      self.size       = self._keyedPackedArray('dimension',count=3,type='d')

    self.grid         = self._keyedPackedArray('grid:',count=3,type='i')
    if self.grid == [None,None,None]:                                                               # no 'grid' found, try legacy alias 'resolution'
      self.grid         = self._keyedPackedArray('resolution',count=3,type='i')

    self.N_nodes      = (self.grid[0]+1)*(self.grid[1]+1)*(self.grid[2]+1)
    self.N_elements   =  self.grid[0]   * self.grid[1]   * self.grid[2]

    self.N_element_scalars = self._keyedPackedArray('materialpoint_sizeResults:',count=1,type='i')[0]
    if self.N_element_scalars is None:
      self.N_element_scalars = self._keyedPackedArray('materialpoint_sizeResults',count=1,type='i')[0]

    self.N_positions  = (self.filesize-self.dataOffset)/(self.N_elements*self.N_element_scalars*8)
    self.N_increments = 1                                                    # add zero'th entry
    for i in range(self.N_loadcases):
      self.N_increments += self._increments[i]//self._frequencies[i]

# parameters for file handling depending on output format

    if options.legacy:
       self.tagLen=8
       self.fourByteLimit = 2**31 -1 -8
    else:
       self.tagLen=0
    self.expectedFileSize = self.dataOffset+self.N_increments*(self.tagLen+self.N_elements*self.N_element_scalars*8)
    if options.legacy: self.expectedFileSize+=self.expectedFileSize//self.fourByteLimit*8             # add extra 8 bytes for additional headers at 4 GB limits
    if self.expectedFileSize != self.filesize:
      print('\n**\n* Unexpected file size. Incomplete simulation or file corrupted!\n**')

  def __str__(self):
    """Summary of results file"""
    return '\n'.join([
      'workdir: %s'%self.wd,
      'geometry: %s'%self.geometry,
      'loadcases: %i'%self.N_loadcases,
      'grid: %s'%(','.join(map(str,self.grid))),
      'size: %s'%(','.join(map(str,self.size))),
      'header size: %i'%self.dataOffset,
      'actual   file size: %i'%self.filesize,
      'expected file size: %i'%self.expectedFileSize,
      'positions in file : %i'%self.N_positions,
      'starting increment: %i'%self.startingIncrement,
      ]
    )


  def locateKeyValue(self,identifier):

    key = {'name':None,'pos':None}

    name = ''
    filepos=0                                                           # start at the beginning
    while name != identifier and filepos < self.dataOffset:             # stop searching when found or when reached end of header
      self.file.seek(filepos)
# read the starting tag in front of the keyword (Fortran indicates start and end of writing by a 4 byte tag indicating the length of the following data)
      dataLen=struct.unpack('i',self.file.read(4))[0]
      name = self.file.read(len(identifier))                                                        # anticipate identifier
      start=filepos+(4+len(identifier))                                                             # position of the values for the found key
      filepos=filepos+(4+dataLen+4)                                                                 # forward to next keyword

    if name==identifier:                                                                            # found the correct name
      key['pos']  = start                                                                           # save position
      key['name'] = name
    return key

  def _keyedPackedArray(self,identifier,count = 3,type = 'd',default = None):
    bytecount = {'d': 8,'i': 4}
    values = [default]*count
    key = self.locateKeyValue(identifier)
    if key['name'] == identifier and key['pos'] is not None:
      self.file.seek(key['pos'])
      for i in range(count):
        values[i] = struct.unpack(type,self.file.read(bytecount[type]))[0]
    return values


  def _keyedString(self,identifier,default=None):
    value = default
    self.file.seek(0)
    m = re.search(r'(.{4})%s(.*?)\1'%identifier,self.file.read(self.dataOffset),re.DOTALL)
    if m:
      value = m.group(2)
    return value

  def title(self):
    return self.theTitle

  def moveto(self,pos):
    self.position = pos
    self.increment = 0
    self.time = 0.0
    p = pos
    for l in range(self.N_loadcases):
      if p <= self._increments[l]//self._frequencies[l]:
        break
      else:
        self.increment += self._increments[l]
        self.time      += self._times[l]
        p -= self._increments[l]//self._frequencies[l]

    self.increment += self._frequencies[l] * p

    if self._logscales[l] > 0:                                                                                            # logarithmic time scale
      if l == 0: self.time  = 2**(self._increments[l] - (1+self._frequencies[l]*p)) * self._times[l]                      # first loadcase
      else:      self.time *= ((self.time + self._times[l])/self.time)**((1+self._frequencies[l]*p)/self._increments[l])  # any subsequent loadcase
    else:                                                                                                                 # linear time scale
      self.time += self._times[l]/self._increments[l] * self._frequencies[l] * p

  def extrapolation(self,value):
    self.extrapolate = value

  def node_sequence(self,n):
    return n-1

  def node_id(self,n):
    return n+1

  def node(self,n):
    a = self.grid[0]+1
    b = self.grid[1]+1
    c = self.grid[2]+1
    return vector([self.size[0] *       (n%a) / self.grid[0],
                   self.size[1] *   ((n/a)%b) / self.grid[1],
                   self.size[2] * ((n/a/b)%c) / self.grid[2],
            ])

  def element_sequence(self,e):
    return e-1

  def element_id(self,e):
    return e+1

  def element(self,e):
    a = self.grid[0]+1
    b = self.grid[1]+1
    basenode = 1 + e+e/self.grid[0] + e/self.grid[0]/self.grid[1]*a
    basenode2 = basenode+a*b
    return (element([basenode ,basenode +1,basenode +a+1,basenode +a,
                    basenode2 ,basenode2+1,basenode2+a+1,basenode2+a,
                   ],117))

  def increments(self):
    return self.N_positions

  def nodes(self):
    return self.N_nodes

  def node_scalars(self):
    return self.N_node_scalars

  def elements(self):
    return self.N_elements

  def element_scalars(self):
    return self.N_element_scalars

  def element_scalar(self,e,idx):
    if not options.legacy:
      incStart =  self.dataOffset \
               +  self.position*8*self.N_elements*self.N_element_scalars
      where = (e*self.N_element_scalars + idx)*8
      try:
        self.file.seek(incStart+where)
        value = struct.unpack('d',self.file.read(8))[0]
      except:
        print('seeking {}'.format(incStart+where))
        print('e {} idx {}'.format(e,idx))
        sys.exit(1)

    else:
      self.fourByteLimit = 2**31 -1 -8
# header & footer + extra header and footer for 4 byte int range (Fortran)
# values
      incStart =  self.dataOffset \
               +  self.position*8*( 1 + self.N_elements*self.N_element_scalars*8//self.fourByteLimit \
                                    + self.N_elements*self.N_element_scalars)

      where = (e*self.N_element_scalars + idx)*8
      try:
        if where%self.fourByteLimit + 8 >= self.fourByteLimit:                       # danger of reading into fortran record footer at 4 byte limit
          data=''
          for i in range(8):
            self.file.seek(incStart+where+(where//self.fourByteLimit)*8+4)
            data  += self.file.read(1)
            where += 1
          value = struct.unpack('d',data)[0]
        else:
          self.file.seek(incStart+where+(where//self.fourByteLimit)*8+4)
          value = struct.unpack('d',self.file.read(8))[0]
      except:
        print('seeking {}'.format(incStart+where+(where//self.fourByteLimit)*8+4))
        print('e {} idx {}'.format(e,idx))
        sys.exit(1)

    return [elemental_scalar(node,value) for node in self.element(e).items]

  def element_scalar_label(elem,idx):
    return 'User Defined Variable %i'%(idx+1)

  def element_tensors(self):
    return self.N_element_tensors

# -----------------------------
def ipCoords(elemType, nodalCoordinates):
  """Returns IP coordinates for a given element"""
  nodeWeightsPerNode =  {
              7:    [ [27.0,  9.0,  3.0,  9.0,  9.0,  3.0,  1.0,  3.0],
                      [ 9.0, 27.0,  9.0,  3.0,  3.0,  9.0,  3.0,  1.0],
                      [ 3.0,  9.0, 27.0,  9.0,  1.0,  3.0,  9.0,  3.0],
                      [ 9.0,  3.0,  9.0, 27.0,  3.0,  1.0,  3.0,  9.0],
                      [ 9.0,  3.0,  1.0,  3.0, 27.0,  9.0,  3.0,  9.0],
                      [ 3.0,  9.0,  3.0,  1.0,  9.0, 27.0,  9.0,  3.0],
                      [ 1.0,  3.0,  9.0,  3.0,  3.0,  9.0, 27.0,  9.0],
                      [ 3.0,  1.0,  3.0,  9.0,  9.0,  3.0,  9.0, 27.0] ],
              57:   [ [27.0,  9.0,  3.0,  9.0,  9.0,  3.0,  1.0,  3.0],
                      [ 9.0, 27.0,  9.0,  3.0,  3.0,  9.0,  3.0,  1.0],
                      [ 3.0,  9.0, 27.0,  9.0,  1.0,  3.0,  9.0,  3.0],
                      [ 9.0,  3.0,  9.0, 27.0,  3.0,  1.0,  3.0,  9.0],
                      [ 9.0,  3.0,  1.0,  3.0, 27.0,  9.0,  3.0,  9.0],
                      [ 3.0,  9.0,  3.0,  1.0,  9.0, 27.0,  9.0,  3.0],
                      [ 1.0,  3.0,  9.0,  3.0,  3.0,  9.0, 27.0,  9.0],
                      [ 3.0,  1.0,  3.0,  9.0,  9.0,  3.0,  9.0, 27.0] ],
              117:  [ [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0] ],
              125:  [ [ 3.0,  0.0,  0.0,  4.0,  1.0,  4.0],
                      [ 0.0,  3.0,  0.0,  4.0,  4.0,  1.0],
                      [ 0.0,  0.0,  3.0,  1.0,  4.0,  4.0],],
              127:  [ [ 45.0, 17.0, 17.0, 17.0],
                      [ 17.0, 45.0, 17.0, 17.0],
                      [ 17.0, 17.0, 45.0, 17.0],
                      [ 17.0, 17.0, 17.0, 45.0],],
              136:  [ [42.0, 15.0, 15.0, 14.0,  5.0,  5.0],
                      [15.0, 42.0, 15.0,  5.0, 14.0,  5.0],
                      [15.0, 15.0, 42.0,  5.0,  5.0, 14.0],
                      [14.0,  5.0,  5.0, 42.0, 15.0, 15.0],
                      [ 5.0, 14.0,  5.0, 15.0, 42.0, 15.0],
                      [ 5.0,  5.0, 14.0, 15.0, 15.0, 42.0] ],
            }

  Nips = len(nodeWeightsPerNode[elemType])
  ipCoordinates = [[0.0,0.0,0.0] for i in range(Nips)]
  for ip in range(Nips):
    for node in range(len(nodeWeightsPerNode[elemType][ip])):
      for i in range(3):
        ipCoordinates[ip][i] += nodeWeightsPerNode[elemType][ip][node] * nodalCoordinates[node][i]
    for i in range(3):
      ipCoordinates[ip][i] /= sum(nodeWeightsPerNode[elemType][ip])

  return ipCoordinates



# -----------------------------
def ipIDs(elemType):
  """Returns IP numbers for given element type"""
  ipPerNode =  {
              7:    [ 1, 2, 4, 3, 5, 6, 8, 7 ],
              57:   [ 1, 2, 4, 3, 5, 6, 8, 7 ],
              117:  [ 1 ],
              125:  [ 1, 2, 3 ],
              127:  [ 1, 2, 3, 4 ],
              136:  [ 1, 2, 3, 4, 5, 6 ],
            }

  return ipPerNode[elemType]



# -----------------------------
def substituteLocation(string, mesh, coords):
  """Do variable interpolation in group and filter strings"""
  substitute = string
  substitute = substitute.replace('elem', str(mesh[0]))
  substitute = substitute.replace('node', str(mesh[1]))
  substitute = substitute.replace('ip',   str(mesh[2]))
  substitute = substitute.replace('grain', str(mesh[3]))
  substitute = substitute.replace('x', '%.6g'%coords[0])
  substitute = substitute.replace('y', '%.6g'%coords[1])
  substitute = substitute.replace('z', '%.6g'%coords[2])
  return substitute



# -----------------------------
def heading(glue,parts):
  """Joins pieces from parts by glue. second to last entry in pieces tells multiplicity"""
  header = []
  for pieces in parts:
    if pieces[-2] == 0:
      del pieces[-2]
    header.append(glue.join(map(str,pieces)))

  return header


# -----------------------------
def mapIncremental(label, mapping, N, base, new):
  """
  Applies the function defined by "mapping"

  (can be either 'min','max','avg', 'sum', or user specified)
  to a list of data
  """
  theMap =  { 'min': lambda n,b,a: a if n==0 else min(b,a),
              'max': lambda n,b,a: a if n==0 else max(b,a),
              'avg': lambda n,b,a: (n*b+a)/(n+1),
              'avgabs': lambda n,b,a: (n*b+abs(a))/(n+1),
              'sum': lambda n,b,a: a if n==0 else b+a,
              'sumabs': lambda n,b,a: abs(a) if n==0 else b+abs(a),
              'unique': lambda n,b,a: a if n==0 or b==a else 'nan'
            }
  if mapping in theMap:
    mapped = map(theMap[mapping],[N]*len(base),base,new)                        # map one of the standard functions to data
    if label.lower() == 'orientation':                                          # orientation is special case:...
      orientationNorm = math.sqrt(sum([q*q for q in mapped]))                   # ...calc norm of average quaternion
      mapped = map(lambda x: x/orientationNorm, mapped)                         # ...renormalize quaternion
  else:
    try:
      mapped = eval('map(%s,[N]*len(base),base,new)'%mapping)                   # map user defined function to colums in chunks
    except:
      mapped = ['nan']*len(base)

  return mapped



# -----------------------------
def OpenPostfile(name,type,nodal = False):
  """Open postfile with extrapolation mode 'translate'"""
  p = {
         'spectral': MPIEspectral_result,
         'marc':     post_open,
      }[type](name)
  p.extrapolation({True:'linear',False:'translate'}[nodal])
  p.moveto(1)

  return p


# -----------------------------
def ParseOutputFormat(filename,what,me):
  """Parse .output* files in order to get a list of outputs"""
  content = []
  format = {'outputs':{},'specials':{'brothers':[]}}
  for prefix in ['']+map(str,range(1,17)):
    if os.path.exists(prefix+filename+'.output'+what):
      try:
        file = open(prefix+filename+'.output'+what)
        content = file.readlines()
        file.close()
        break
      except:
        pass

  if content == []: return format                                         # nothing found...

  tag = ''
  tagID = 0
  for line in content:
    if re.match("\s*$",line) or re.match("#",line):                       # skip blank lines and comments
      continue
    m = re.match("\[(.+)\]",line)                                         # look for block indicator
    if m:                                                                 # next section
      tag = m.group(1)
      tagID += 1
      format['specials']['brothers'].append(tag)
      if tag == me or (me.isdigit() and tagID == int(me)):
        format['specials']['_id'] = tagID
        format['outputs'] = []
        tag = me
    else:                                           # data from section
      if tag == me:
        (output,length) = line.split()
        output.lower()
        if length.isdigit():
          length = int(length)
        if re.match("\((.+)\)",output):                           # special data, e.g. (Ngrains)
          format['specials'][output] = length
        elif length > 0:
          format['outputs'].append([output,length])
  return format


# -----------------------------
def ParsePostfile(p,filename, outputFormat):
  """
  Parse postfile in order to get position and labels of outputs

  needs "outputFormat" for mapping of output names to postfile output indices
  """
  stat = {
  'IndexOfLabel': {},
  'Title': p.title(),
  'Extrapolation': p.extrapolate,
  'NumberOfIncrements': p.increments(),
  'NumberOfNodes': p.nodes(),
  'NumberOfNodalScalars': p.node_scalars(),
  'LabelOfNodalScalar': [None]*p.node_scalars(),
  'NumberOfElements': p.elements(),
  'NumberOfElementalScalars': p.element_scalars(),
  'LabelOfElementalScalar': [None]*p.element_scalars(),
  'NumberOfElementalTensors': p.element_tensors(),
  'LabelOfElementalTensor': [None]*p.element_tensors(),
  }

# --- find labels

  for labelIndex in range(stat['NumberOfNodalScalars']):
    label =  p.node_scalar_label(labelIndex)
    stat['IndexOfLabel'][label] = labelIndex
    stat['LabelOfNodalScalar'][labelIndex] = label

  for labelIndex in range(stat['NumberOfElementalScalars']):
    label =  p.element_scalar_label(labelIndex)
    stat['IndexOfLabel'][label] = labelIndex
    stat['LabelOfElementalScalar'][labelIndex] = label

  for labelIndex in range(stat['NumberOfElementalTensors']):
    label =  p.element_tensor_label(labelIndex)
    stat['IndexOfLabel'][label] = labelIndex
    stat['LabelOfElementalTensor'][labelIndex] = label

  if 'User Defined Variable 1' in stat['IndexOfLabel']:       # output format without dedicated names?
    stat['IndexOfLabel']['HomogenizationCount'] = stat['IndexOfLabel']['User Defined Variable 1']  # adjust first named entry

  if 'HomogenizationCount' in stat['IndexOfLabel']:          # does the result file contain relevant user defined output at all?
    startIndex = stat['IndexOfLabel']['HomogenizationCount']
    stat['LabelOfElementalScalar'][startIndex] = 'HomogenizationCount'

# We now have to find a mapping for each output label as defined in the .output* files to the output position in the post file
# Since we know where the user defined outputs start ("startIndex"), we can simply assign increasing indices to the labels
# given in the .output* file

    offset = 1
    for (name,N) in outputFormat['Homogenization']['outputs']:
      for i in range(N):
        label = {False:   '%s'%(    name),
                  True:'%i_%s'%(i+1,name)}[N > 1]
        stat['IndexOfLabel'][label] = startIndex + offset
        stat['LabelOfElementalScalar'][startIndex + offset] = label
        offset += 1
    stat['IndexOfLabel']['GrainCount'] = startIndex + offset
    stat['LabelOfElementalScalar'][startIndex + offset] = 'GrainCount'                             # add GrainCount
    offset += 1

    if '(ngrains)' in outputFormat['Homogenization']['specials']:
      for grain in range(outputFormat['Homogenization']['specials']['(ngrains)']):

        stat['IndexOfLabel']['%i_CrystalliteCount'%(grain+1)] = startIndex + offset                 # report crystallite count
        stat['LabelOfElementalScalar'][startIndex + offset] = '%i_CrystalliteCount'%(grain+1)       # add GrainCount
        offset += 1

        for (name,N) in outputFormat['Crystallite']['outputs']:                                     # add crystallite outputs
          for i in range(N):
            label = '%i_'%(grain+1) + ('%i_'%(i+1) if N>1 else '') + name
            stat['IndexOfLabel'][label] = startIndex + offset
            stat['LabelOfElementalScalar'][startIndex + offset] = label
            offset += 1

        stat['IndexOfLabel']['%i_ConstitutiveCount'%(grain+1)] = startIndex + offset               # report constitutive count
        stat['LabelOfElementalScalar'][startIndex + offset] = '%i_ConstitutiveCount'%(grain+1)     # add GrainCount
        offset += 1

        for (name,N) in outputFormat['Constitutive']['outputs']:                                   # add constitutive outputs
          for i in range(N):
            label = '%i_'%(grain+1) + ('%i_'%(i+1) if N>1 else '') + name
            stat['IndexOfLabel'][label] = startIndex + offset
            try:
              stat['LabelOfElementalScalar'][startIndex + offset] = label
            except IndexError:
              print('trying to assign {} at position {}+{}'.format(label,startIndex,offset))
              sys.exit(1)
            offset += 1

  return stat


# -----------------------------
def SummarizePostfile(stat,where=sys.stdout,format='marc'):

  where.write('\n\n')
  where.write('title:\t%s'%stat['Title'] + '\n\n')
  where.write('extraplation:\t%s'%stat['Extrapolation'] + '\n\n')
  where.write('increments:\t%i'%(stat['NumberOfIncrements']) + '\n\n')
  where.write('nodes:\t%i'%stat['NumberOfNodes'] + '\n\n')
  where.write('elements:\t%i'%stat['NumberOfElements'] + '\n\n')
  where.write('nodal scalars:\t%i'%stat['NumberOfNodalScalars'] + '\n\n  '\
              +'\n  '.join(stat['LabelOfNodalScalar']) + '\n\n')
  where.write('elemental scalars:\t%i'%stat['NumberOfElementalScalars'] + '\n\n  '\
              + '\n  '.join(stat['LabelOfElementalScalar']) + '\n\n')
  where.write('elemental tensors:\t%i'%stat['NumberOfElementalTensors'] + '\n\n  '\
              + '\n  '.join(stat['LabelOfElementalTensor']) + '\n\n')

  return True


# -----------------------------
# MAIN FUNCTION STARTS HERE
# -----------------------------

# --- input parsing

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Extract data from a .t16 (MSC.Marc) or .spectralOut results file.

List of output variables is given by options '--ns','--es','--et','--ho','--cr','--co'.

Filters and separations use 'elem','node','ip','grain', and 'x','y','z' as key words.
Example:
1) get averaged results in slices perpendicular to x for all negative y coordinates
--filter 'y < 0.0' --separation x --map 'avg'
2) global sum of squared data falling into first quadrant arc between R1 and R2
--filter 'x >= 0.0 and y >= 0.0 and x*x + y*y >= R1*R1 and x*x + y*y <= R2*R2'
--map 'lambda n,b,a: n*b+a*a'

User mappings need to be formulated in an incremental fashion for each new data point, a(dd),
and may use the current (incremental) result, b(ase), as well as the number, n(umber),
of already processed data points for evaluation.

""", version = scriptID)

parser.add_option('-i','--info', action='store_true', dest='info',
                  help='list contents of resultfile')
parser.add_option('-l','--legacy', action='store_true', dest='legacy',
                  help='data format of spectral solver is in legacy format (no MPI out)')
parser.add_option('-n','--nodal', action='store_true', dest='nodal',
                  help='data is extrapolated to nodal value')
parser.add_option(    '--prefix', dest='prefix',
                  metavar='string',
                  help='prefix to result file name')
parser.add_option(    '--suffix', dest='suffix',
                  metavar='string',
                  help='suffix to result file name')
parser.add_option('-d','--dir', dest='dir',
                  metavar='string',
                  help='name of subdirectory to hold output [%default]')
parser.add_option('-s','--split', action='store_true', dest='separateFiles',
                  help='split output per increment')
parser.add_option('-r','--range', dest='range', type='int', nargs=3,
                  metavar='int int int',
                  help='range of positions (or increments) to output (start, end, step) [all]')
parser.add_option('--increments', action='store_true', dest='getIncrements',
                  help='switch to increment range')
parser.add_option('-m','--map', dest='func',
                  metavar='string',
                  help='data reduction mapping [%default] out of min, max, avg, avgabs, sum, sumabs or user-lambda')
parser.add_option('-p','--type', dest='filetype',
                  metavar = 'string',
                  help = 'type of result file [auto]')
parser.add_option('-q','--quiet', dest='verbose',
                  action = 'store_false',
                  help = 'suppress verbose output')

group_material = OptionGroup(parser,'Material identifier')

group_material.add_option('--homogenization', dest='homog',
                          help='homogenization identifier (as string or integer [%default])', metavar='string')
group_material.add_option('--crystallite', dest='cryst',
                          help='crystallite identifier (as string or integer [%default])', metavar='string')
group_material.add_option('--phase', dest='phase',
                          help='phase identifier (as string or integer [%default])', metavar='string')

group_special  = OptionGroup(parser,'Special outputs')

group_special.add_option('-t','--time', action='store_true', dest='time',
                         help='output time of increment [%default]')
group_special.add_option('-f','--filter', dest='filter',
                         help='condition(s) to filter results [%default]', metavar='string')
group_special.add_option('--separation', action='extend', dest='sep',
                         help='properties to separate results [%default]', metavar='<string LIST>')
group_special.add_option('--sort', action='extend', dest='sort',
                         help='properties to sort results [%default]', metavar='<string LIST>')

group_general  = OptionGroup(parser,'General outputs')

group_general.add_option('--ns', action='extend', dest='nodalScalar',
                         help='nodal scalars to extract', metavar='<string LIST>')
group_general.add_option('--es', action='extend', dest='elemScalar',
                         help='elemental scalars to extract', metavar='<string LIST>')
group_general.add_option('--et', action='extend', dest='elemTensor',
                         help='elemental tensors to extract', metavar='<string LIST>')
group_general.add_option('--ho', action='extend', dest='homogenizationResult',
                         help='homogenization results to extract', metavar='<string LIST>')
group_general.add_option('--cr', action='extend', dest='crystalliteResult',
                         help='crystallite results to extract', metavar='<string LIST>')
group_general.add_option('--co', action='extend', dest='constitutiveResult',
                         help='constitutive results to extract', metavar='<string LIST>')

parser.add_option_group(group_material)
parser.add_option_group(group_general)
parser.add_option_group(group_special)

parser.set_defaults(info = False,
                    verbose = True,
                    legacy = False,
                    nodal = False,
                    prefix = '',
                    suffix = '',
                    dir = 'postProc',
                    filetype = None,
                    func = 'avg',
                    homog = '1',
                    cryst = '1',
                    phase = '1',
                    filter = '',
                    sep = [],
                    sort = [],
                    inc = False,
                    time = False,
                    separateFiles = False,
                    getIncrements= False,
                   )

(options, files) = parser.parse_args()

# --- basic sanity checks


if files == []:
  parser.print_help()
  parser.error('no file specified...')

if not os.path.exists(files[0]):
  parser.print_help()
  parser.error('invalid file "%s" specified...'%files[0])

# --- figure out filetype

if options.filetype is None:
  ext = os.path.splitext(files[0])[1]
  for theType in fileExtensions.keys():
    if ext in fileExtensions[theType]:
      options.filetype = theType
      break

if options.filetype is not None: options.filetype = options.filetype.lower()

if options.filetype == 'marc':  offset_pos = 1
else:                           offset_pos = 0


# --- more sanity checks

if options.filetype not in ['marc','spectral']:
  parser.print_help()
  parser.error('file type "%s" not supported...'%options.filetype)

if options.filetype == 'marc':
  sys.path.append(damask.solver.Marc().libraryPath())

  try:
    from py_post import post_open
  except:
    print('error: no valid Mentat release found')
    sys.exit(-1)
else:
  def post_open():
    return

if options.constitutiveResult and not options.phase:
  parser.print_help()
  parser.error('constitutive results require phase...')

if options.nodalScalar and (   options.elemScalar or options.elemTensor\
                            or options.homogenizationResult or options.crystalliteResult or options.constitutiveResult ):
  parser.print_help()
  parser.error('not allowed to mix nodal with elemental results...')

if not options.nodalScalar:           options.nodalScalar   = []
if not options.elemScalar:            options.elemScalar    = []
if not options.elemTensor:            options.elemTensor    = []
if not options.homogenizationResult:  options.homogenizationResult   = []
if not options.crystalliteResult:     options.crystalliteResult   = []
if not options.constitutiveResult:    options.constitutiveResult = []

options.sort.reverse()
options.sep.reverse()

# --- start background messaging

if options.verbose:
  bg = damask.util.backgroundMessage()
  bg.start()

# --- parse .output and .t16 files

if os.path.splitext(files[0])[1] == '':
  filename = files[0]
  extension = fileExtensions[options.filetype]
else:
  filename = os.path.splitext(files[0])[0]
  extension = os.path.splitext(files[0])[1]

outputFormat = {}
me = {
      'Homogenization': options.homog,
      'Crystallite':    options.cryst,
      'Constitutive':   options.phase,
     }

if options.verbose: bg.set_message('parsing .output files...')

for what in me:
  outputFormat[what] = ParseOutputFormat(filename, what, me[what])
  if '_id' not in outputFormat[what]['specials']:
    print("\nsection '{}' not found in <{}>".format(me[what], what))
    print('\n'.join(map(lambda x:'  [%s]'%x, outputFormat[what]['specials']['brothers'])))

if options.verbose: bg.set_message('opening result file...')

p = OpenPostfile(filename+extension,options.filetype,options.nodal)
if options.verbose: bg.set_message('parsing result file...')
stat = ParsePostfile(p, filename, outputFormat)
if options.filetype == 'marc':
  stat['NumberOfIncrements'] -= 1             # t16 contains one "virtual" increment (at 0)

# --- sanity check for output variables
# for mentat variables (nodalScalar,elemScalar,elemTensor) we simply have to check whether the label
# is found in the stat[indexOfLabel] dictionary for user defined variables (homogenizationResult,
# crystalliteResult,constitutiveResult) we have to check the corresponding outputFormat, since the
# namescheme in stat['IndexOfLabel'] is different

for opt in ['nodalScalar','elemScalar','elemTensor','homogenizationResult','crystalliteResult','constitutiveResult']:
  if eval('options.%s'%opt):
    for label in eval('options.%s'%opt):
      if (opt in ['nodalScalar','elemScalar','elemTensor'] and label not in stat['IndexOfLabel'] and label not in ['elements',]) \
           or (opt in ['homogenizationResult','crystalliteResult','constitutiveResult'] \
               and (not outputFormat[opt[:-6].capitalize()]['outputs'] \
                  or label not in zip(*outputFormat[opt[:-6].capitalize()]['outputs'])[0])):
        parser.error('%s "%s" unknown...'%(opt,label))


# --- output info

if options.info:
  if options.filetype == 'marc':
    print('\n\nMentat release {}'.format(damask.solver.Marc().version()))
  if options.filetype == 'spectral':
    print('\n\n{}'.format(p))

  SummarizePostfile(stat)

  print('\nUser Defined Outputs')
  for what in me:
    print('\n {}:'.format(what))
    for output in outputFormat[what]['outputs']:
      print('  {}'.format(output))

  sys.exit(0)


# --- build connectivity maps

elementsOfNode = {}
for e in range(stat['NumberOfElements']):
  if options.verbose and e%1000 == 0: bg.set_message('connect elem %i...'%e)
  for n in map(p.node_sequence,p.element(e).items):
    if n not in elementsOfNode:
      elementsOfNode[n] = [p.element_id(e)]
    else:
      elementsOfNode[n] += [p.element_id(e)]

maxCountElementsOfNode = 0
for l in elementsOfNode.values():
  maxCountElementsOfNode = max(maxCountElementsOfNode,len(l))


# ---------------------------   build group membership   --------------------------------

p.moveto(offset_pos)
index = {}
groups = []
groupCount = 0
memberCount = 0

if options.nodalScalar:
  for n in range(stat['NumberOfNodes']):
    if options.verbose and n%1000 == 0: bg.set_message('scan node %i...'%n)
    myNodeID = p.node_id(n)
    myNodeCoordinates = [p.node(n).x, p.node(n).y, p.node(n).z]
    myElemID = 0
    myIpID = 0
    myGrainID = 0

    # generate an expression that is only true for the locations specified by options.filter
    filter = substituteLocation(options.filter, [myElemID,myNodeID,myIpID,myGrainID], myNodeCoordinates)
    if filter != '' and not eval(filter):                                                              # for all filter expressions that are not true:...
      continue                                                                                         # ... ignore this data point and continue with next

    # --- group data locations
    # generate a unique key for a group of separated data based on the separation criterium for the location
    grp = substituteLocation('#'.join(options.sep), [myElemID,myNodeID,myIpID,myGrainID], myNodeCoordinates)

    if grp not in index:                                                                               # create a new group if not yet present
      index[grp] = groupCount
      groups.append([[0,0,0,0,0.0,0.0,0.0]])                                                           # initialize with avg location
      groupCount += 1

    groups[index[grp]][0][:4] = mapIncremental('','unique',
                                               len(groups[index[grp]])-1,
                                               groups[index[grp]][0][:4],
                                               [myElemID,myNodeID,myIpID,myGrainID])                   # keep only if unique average location
    groups[index[grp]][0][4:] = mapIncremental('','avg',
                                               len(groups[index[grp]])-1,
                                               groups[index[grp]][0][4:],
                                               myNodeCoordinates)                                      # incrementally update average location
    groups[index[grp]].append([myElemID,myNodeID,myIpID,myGrainID,0])                                  # append a new list defining each group member
    memberCount += 1

else:
  for e in range(stat['NumberOfElements']):
    if options.verbose and e%1000 == 0: bg.set_message('scan elem %i...'%e)
    myElemID = p.element_id(e)
    myIpCoordinates = ipCoords(p.element(e).type, map(lambda node: [node.x, node.y, node.z],
                                                         map(p.node, map(p.node_sequence, p.element(e).items))))
    myIpIDs = ipIDs(p.element(e).type)
    Nips = len(myIpIDs)
    myNodeIDs = p.element(e).items[:Nips]
    for n in range(Nips):
      myIpID = myIpIDs[n]
      myNodeID = myNodeIDs[n]
      for g in range(('GrainCount' in stat['IndexOfLabel']\
                       and int(p.element_scalar(e, stat['IndexOfLabel']['GrainCount'])[0].value))\
                       or 1):
        myGrainID = g + 1

        # --- filter valid locations
        # generates an expression that is only true for the locations specified by options.filter
        filter = substituteLocation(options.filter, [myElemID,myNodeID,myIpID,myGrainID], myIpCoordinates[n])
        if filter != '' and not eval(filter):                                                       # for all filter expressions that are not true:...
          continue                                                                                  # ... ignore this data point and continue with next

        # --- group data locations
        # generates a unique key for a group of separated data based on the separation criterium for the location
        grp = substituteLocation('#'.join(options.sep), [myElemID,myNodeID,myIpID,myGrainID], myIpCoordinates[n])

        if grp not in index:                                                                        # create a new group if not yet present
          index[grp] = groupCount
          groups.append([[0,0,0,0,0.0,0.0,0.0]])                                                    # initialize with avg location
          groupCount += 1

        groups[index[grp]][0][:4] = mapIncremental('','unique',
                                                   len(groups[index[grp]])-1,
                                                   groups[index[grp]][0][:4],
                                                   [myElemID,myNodeID,myIpID,myGrainID])            # keep only if unique average location
        groups[index[grp]][0][4:] = mapIncremental('','avg',
                                                   len(groups[index[grp]])-1,
                                                   groups[index[grp]][0][4:],
                                                   myIpCoordinates[n])                              # incrementally update average location
        groups[index[grp]].append([myElemID,myNodeID,myIpID,myGrainID,n])                           # append a new list defining each group member
        memberCount += 1


# ---------------------------   sort groups   --------------------------------

where = {
    'elem': 0,
    'node': 1,
    'ip': 2,
    'grain': 3,
    'x': 4,
    'y': 5,
    'z': 6,
    }

sortProperties = []
for item in options.sep:
  if item not in options.sort:
    sortProperties.append(item)

theKeys = []
if 'none' not in map(str.lower, options.sort):
  for criterium in options.sort + sortProperties:
    if criterium in where:
      theKeys.append('x[0][%i]'%where[criterium])

sortKeys = eval('lambda x:(%s)'%(','.join(theKeys)))
if options.verbose: bg.set_message('sorting groups...')
groups.sort(key = sortKeys)                                                                         # in-place sorting to save mem


# ---------------------------   create output dir   --------------------------------

dirname = os.path.abspath(os.path.join(os.path.dirname(filename),options.dir))
if not os.path.isdir(dirname):
  os.mkdir(dirname,0o755)

fileOpen = False
assembleHeader = True
header = []
standard = ['inc'] + \
           (['time'] if options.time else []) + \
           ['elem','node','ip','grain','1_pos','2_pos','3_pos']

# ---------------------------   loop over positions   --------------------------------

if options.verbose: bg.set_message('getting map between positions and increments...')

incAtPosition = {}
positionOfInc = {}

for position in range(stat['NumberOfIncrements']):
  p.moveto(position+offset_pos)
  incAtPosition[position] = p.increment            # remember "real" increment at this position
  positionOfInc[p.increment] = position            # remember position of "real" increment

if not options.range:
  options.getIncrements = False
  locations = range(stat['NumberOfIncrements'])    # process all positions
else:
  options.range = list(options.range)              # convert to list
  if options.getIncrements:
    locations = [positionOfInc[x] for x in range(options.range[0],options.range[1]+1,options.range[2])
                                   if x in positionOfInc]
  else:
    locations = range( max(0,options.range[0]),
                       min(stat['NumberOfIncrements'],options.range[1]+1),
                       options.range[2] )

increments = [incAtPosition[x] for x in locations] # build list of increments to process

time_start = time.time()

for incCount,position in enumerate(locations):     # walk through locations

  p.moveto(position+offset_pos)                    # wind to correct position

# ---------------------------   file management   --------------------------------

  if options.separateFiles:
    if fileOpen:
      file.close()                                                                                  # noqa
      fileOpen = False
    outFilename = eval('"'+eval("'%%s_inc%%0%ii%%s.txt'%(math.log10(max(increments+[1]))+1)")\
                      +'"%(dirname + os.sep + options.prefix + os.path.split(filename)[1],increments[incCount],options.suffix)')
  else:
    outFilename = '%s.txt'%(dirname + os.sep + options.prefix + os.path.split(filename)[1] + options.suffix)

  if not fileOpen:
    file = open(outFilename,'w')
    fileOpen = True
    file.write('2\theader\n')
    file.write(scriptID + '\t' + ' '.join(sys.argv[1:]) + '\n')
    headerWritten = False

  file.flush()

# ---------------------------   read and map data per group   --------------------------------

  member = 0
  for group in groups:

    N = 0                                                                                           # group member counter
    for (e,n,i,g,n_local) in group[1:]:                                                             # loop over group members
      member += 1
      if member%1000 == 0:
        time_delta = ((len(locations)*memberCount)/float(member+incCount*memberCount)-1.0)*(time.time()-time_start)
        if options.verbose: bg.set_message('(%02i:%02i:%02i) processing point %i of %i from increment %i (position %i)...'
          %(time_delta//3600,time_delta%3600//60,time_delta%60,member,memberCount,increments[incCount],position))

      newby = []                                                                                    # current member's data

      if options.nodalScalar:
        for label in options.nodalScalar:
          if label == 'elements':
            length = maxCountElementsOfNode
            content = elementsOfNode[p.node_sequence(n)]+[0]*(length-len(elementsOfNode[p.node_sequence(n)]))
          else:
            length = 1
            content = [ p.node_scalar(p.node_sequence(n),stat['IndexOfLabel'][label]) ]
          if assembleHeader:
            header += heading('_',[[component,''.join( label.split() )]
                      for component in range(int(length>1),length+int(length>1))])
          newby.append({'label':label,
                        'len':length,
                        'content':content })

      if options.elemScalar:
        for label in options.elemScalar:
          if assembleHeader:
            header += [''.join( label.split() )]
          newby.append({'label':label,
                        'len':1,
                        'content':[ p.element_scalar(p.element_sequence(e),stat['IndexOfLabel'][label])[n_local].value ]})

      if options.elemTensor:
        for label in options.elemTensor:
          if assembleHeader:
            header += heading('.',[[''.join( label.split() ),component]
                      for component in ['intensity','t11','t22','t33','t12','t23','t13']])
          myTensor = p.element_tensor(p.element_sequence(e),stat['IndexOfLabel'][label])[n_local]
          newby.append({'label':label,
                        'len':7,
                        'content':[ myTensor.intensity,
                                    myTensor.t11, myTensor.t22, myTensor.t33,
                                    myTensor.t12, myTensor.t23, myTensor.t13,
                                  ]})

      if options.homogenizationResult or \
         options.crystalliteResult or \
         options.constitutiveResult:
        for (label,resultType) in zip(options.homogenizationResult +
                                      options.crystalliteResult +
                                      options.constitutiveResult,
                                      ['Homogenization']*len(options.homogenizationResult) +
                                      ['Crystallite']*len(options.crystalliteResult) +
                                      ['Constitutive']*len(options.constitutiveResult)
                                      ):
          outputIndex = list(zip(*outputFormat[resultType]['outputs'])[0]).index(label)             # find the position of this output in the outputFormat
          length = int(outputFormat[resultType]['outputs'][outputIndex][1])
          thisHead = heading('_',[[component,''.join( label.split() )] for component in range(int(length>1),length+int(length>1))])
          if assembleHeader: header += thisHead
          if resultType != 'Homogenization':
            thisHead = heading('_',[[g,component,label] for component in range(int(length>1),length+int(length>1))])
          try:
            newby.append({'label':label,
                          'len':length,
                          'content':[ p.element_scalar(p.element_sequence(e),stat['IndexOfLabel'][head])[n_local].value
                                      for head in thisHead ]})
          except KeyError:
            print('\nDAMASK outputs seem missing from "post" section of the *.dat file!')
            sys.exit()

      assembleHeader = False

      if N == 0:
        mappedResult = [float(x) for x in range(len(header))]                                       # init with debug data (should get deleted by *N at N=0)

      pos = 0
      for chunk in newby:
        mappedResult[pos:pos+chunk['len']] = mapIncremental(chunk['label'],options.func,
                                                            N,mappedResult[pos:pos+chunk['len']],chunk['content'])
        pos += chunk['len']

      N += 1

    # --- write data row to file ---

    if not headerWritten:
      file.write('\t'.join(standard + header) + '\n')
      headerWritten = True

    file.write('\t'.join(map(str,[p.increment] + \
                                 {True:[p.time],False:[]}[options.time] + \
                                 group[0] + \
                                 mappedResult)
                        ) + '\n')

if fileOpen:
  file.close()


# ---------------------------       DONE     --------------------------------
