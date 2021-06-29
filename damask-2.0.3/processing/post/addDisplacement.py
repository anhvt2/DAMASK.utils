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

import os,sys,math
import numpy as np
import scipy.ndimage
from optparse import OptionParser
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


#--------------------------------------------------------------------------------------------------
def cell2node(cellData,grid):

  nodeData = 0.0
  datalen = np.array(cellData.shape[3:]).prod()
  
  for i in range(datalen):
    node = scipy.ndimage.convolve(cellData.reshape(tuple(grid[::-1])+(datalen,))[...,i],
                                  np.ones((2,2,2))/8.,                                              # 2x2x2 neighborhood of cells
                                  mode = 'wrap',
                                  origin = -1,                                                      # offset to have cell origin as center
                                 )                                                                  # now averaged at cell origins
    node = np.append(node,node[np.newaxis,0,:,:,...],axis=0)                                        # wrap along z
    node = np.append(node,node[:,0,np.newaxis,:,...],axis=1)                                        # wrap along y
    node = np.append(node,node[:,:,0,np.newaxis,...],axis=2)                                        # wrap along x

    nodeData = node[...,np.newaxis] if i==0 else np.concatenate((nodeData,node[...,np.newaxis]),axis=-1)

  return nodeData

#--------------------------------------------------------------------------------------------------
def displacementAvgFFT(F,grid,size,nodal=False,transformed=False):
  """Calculate average cell center (or nodal) displacement for deformation gradient field specified in each grid cell"""
  if nodal:
    x, y, z = np.meshgrid(np.linspace(0,size[2],1+grid[2]),
                          np.linspace(0,size[1],1+grid[1]),
                          np.linspace(0,size[0],1+grid[0]),
                          indexing = 'ij')
  else:
    x, y, z = np.meshgrid(np.linspace(0,size[2],grid[2],endpoint=False),
                          np.linspace(0,size[1],grid[1],endpoint=False),
                          np.linspace(0,size[0],grid[0],endpoint=False),
                          indexing = 'ij')

  origCoords = np.concatenate((z[:,:,:,None],y[:,:,:,None],x[:,:,:,None]),axis = 3) 

  F_fourier = F if transformed else np.fft.rfftn(F,axes=(0,1,2))                                    # transform or use provided data
  Favg = np.real(F_fourier[0,0,0,:,:])/grid.prod()                                                  # take zero freq for average
  avgDisplacement = np.einsum('ml,ijkl->ijkm',Favg-np.eye(3),origCoords)                            # dX = Favg.X

  return avgDisplacement

#--------------------------------------------------------------------------------------------------
def displacementFluctFFT(F,grid,size,nodal=False,transformed=False):
  """Calculate cell center (or nodal) displacement for deformation gradient field specified in each grid cell"""
  integrator = 0.5j * size / math.pi

  kk, kj, ki = np.meshgrid(np.where(np.arange(grid[2])>grid[2]//2,np.arange(grid[2])-grid[2],np.arange(grid[2])),
                           np.where(np.arange(grid[1])>grid[1]//2,np.arange(grid[1])-grid[1],np.arange(grid[1])),
                                    np.arange(grid[0]//2+1),
                           indexing = 'ij')
  k_s = np.concatenate((ki[:,:,:,None],kj[:,:,:,None],kk[:,:,:,None]),axis = 3) 
  k_sSquared = np.einsum('...l,...l',k_s,k_s)
  k_sSquared[0,0,0] = 1.0                                                                           # ignore global average frequency

#--------------------------------------------------------------------------------------------------
# integration in Fourier space

  displacement_fourier = -np.einsum('ijkml,ijkl,l->ijkm',
                                    F if transformed else np.fft.rfftn(F,axes=(0,1,2)),
                                    k_s,
                                    integrator,
                                   ) / k_sSquared[...,np.newaxis]

#--------------------------------------------------------------------------------------------------
# backtransformation to real space

  displacement = np.fft.irfftn(displacement_fourier,grid[::-1],axes=(0,1,2))

  return cell2node(displacement,grid) if nodal else displacement


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Add displacments resulting from deformation gradient field.
Operates on periodic three-dimensional x,y,z-ordered data sets.
Outputs at cell centers or cell nodes (into separate file).

""", version = scriptID)

parser.add_option('-f',
                  '--defgrad',
                  dest    = 'defgrad',
                  metavar = 'string',
                  help    = 'label of deformation gradient [%default]')
parser.add_option('-p',
                  '--pos', '--position',
                  dest    = 'pos',
                  metavar = 'string',
                  help    = 'label of coordinates [%default]')
parser.add_option('--nodal',
                  dest    = 'nodal',
                  action  = 'store_true',
                  help    = 'output nodal (instead of cell-centered) displacements')

parser.set_defaults(defgrad = 'f',
                    pos     = 'pos',
                   )

(options,filenames) = parser.parse_args()

# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  outname = (os.path.splitext(name)[0] +
             '_nodal' +
             os.path.splitext(name)[1]) if (options.nodal and name) else None
  try:    table = damask.ASCIItable(name = name,
                                    outname = outname,
                                    buffered = False)
  except: continue
  damask.util.report(scriptName,'{}{}'.format(name if name else '',
                                              ' --> {}'.format(outname) if outname else ''))

# ------------------------------------------ read header ------------------------------------------

  table.head_read()

# ------------------------------------------ sanity checks ----------------------------------------

  errors  = []
  remarks = []
  
  if table.label_dimension(options.defgrad) != 9:
    errors.append('deformation gradient "{}" is not a 3x3 tensor.'.format(options.defgrad))

  coordDim = table.label_dimension(options.pos)
  if not 3 >= coordDim >= 1:
    errors.append('coordinates "{}" need to have one, two, or three dimensions.'.format(options.pos))
  elif coordDim < 3:
    remarks.append('appending {} dimension{} to coordinates "{}"...'.format(3-coordDim,
                                                                            's' if coordDim < 2 else '',
                                                                            options.pos))

  if remarks != []: damask.util.croak(remarks)
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss=True)
    continue

# --------------- figure out size and grid ---------------------------------------------------------

  table.data_readArray([options.defgrad,options.pos])
  table.data_rewind()

  if len(table.data.shape) < 2: table.data.shape += (1,)                                            # expand to 2D shape
  if table.data[:,9:].shape[1] < 3:
    table.data = np.hstack((table.data,
                            np.zeros((table.data.shape[0],
                                      3-table.data[:,9:].shape[1]),dtype='f')))                     # fill coords up to 3D with zeros

  grid,size = damask.util.coordGridAndSize(table.data[:,9:12])
  N = grid.prod()

  if N != len(table.data): errors.append('data count {} does not match grid {}x{}x{}.'.format(N,*grid))
  if errors  != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue
  
# ------------------------------------------ process data ------------------------------------------

  F_fourier = np.fft.rfftn(table.data[:,:9].reshape(grid[2],grid[1],grid[0],3,3),axes=(0,1,2))      # perform transform only once...

  fluctDisplacement = displacementFluctFFT(F_fourier,grid,size,options.nodal,transformed=True)
  avgDisplacement   = displacementAvgFFT  (F_fourier,grid,size,options.nodal,transformed=True)

# ------------------------------------------ assemble header ---------------------------------------

  if options.nodal:
    table.info_clear()
    table.labels_clear()

  table.info_append(scriptID + '\t' + ' '.join(sys.argv[1:]))
  table.labels_append((['{}_pos'         .format(i+1)   for i in range(3)] if options.nodal else []) +
                       ['{}_avg({}).{}'  .format(i+1,options.defgrad,options.pos) for i in range(3)] +
                       ['{}_fluct({}).{}'.format(i+1,options.defgrad,options.pos) for i in range(3)] )
  table.head_write()

# ------------------------------------------ output data -------------------------------------------

  Zrange = np.linspace(0,size[2],1+grid[2]) if options.nodal else range(grid[2])
  Yrange = np.linspace(0,size[1],1+grid[1]) if options.nodal else range(grid[1])
  Xrange = np.linspace(0,size[0],1+grid[0]) if options.nodal else range(grid[0])

  for i,z     in enumerate(Zrange):
    for j,y   in enumerate(Yrange):
      for k,x in enumerate(Xrange):
        if options.nodal: table.data_clear()
        else:             table.data_read()
        table.data_append([x,y,z] if options.nodal else [])
        table.data_append(list(  avgDisplacement[i,j,k,:]))
        table.data_append(list(fluctDisplacement[i,j,k,:]))
        table.data_write()                       

# ------------------------------------------ output finalization -----------------------------------

  table.close()                                                                                     # close ASCII tables
