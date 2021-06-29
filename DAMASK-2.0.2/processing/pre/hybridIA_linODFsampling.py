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

from optparse import OptionParser
import damask
import os,sys,math,random
import numpy as np

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


# --- helper functions ---
def integerFactorization(i):

  j = int(math.floor(math.sqrt(float(i))))
  while j>1 and int(i)%j != 0:
    j -= 1
  return j

def binAsBins(bin,intervals):
  """Explode compound bin into 3D bins list"""
  bins = [0]*3
  bins[0] = (bin//(intervals[1] * intervals[2])) % intervals[0]
  bins[1] = (bin//intervals[2]) % intervals[1]
  bins[2] = bin % intervals[2]
  return bins
  
def binsAsBin(bins,intervals):
  """Implode 3D bins into compound bin"""
  return (bins[0]*intervals[1] + bins[1])*intervals[2] + bins[2]

def EulersAsBins(Eulers,intervals,deltas,center):
  """Return list of Eulers translated into 3D bins list"""
  return [int((euler+(0.5-center)*delta)//delta)%interval \
                  for euler,delta,interval in zip(Eulers,deltas,intervals) \
         ]

def binAsEulers(bin,intervals,deltas,center):
  """Compound bin number translated into list of Eulers"""
  Eulers = [0.0]*3
  Eulers[2] = (bin%intervals[2] + center)*deltas[2]
  Eulers[1] = (bin//intervals[2]%intervals[1] + center)*deltas[1]
  Eulers[0] = (bin//(intervals[2]*intervals[1]) + center)*deltas[0]
  return Eulers

def directInvRepetitions(probability,scale):
  """Calculate number of samples drawn by direct inversion"""
  nDirectInv = 0
  for bin in range(len(probability)):                                                               # loop over bins
    nDirectInv += int(round(probability[bin]*scale))                                                # calc repetition
  return nDirectInv


# ---------------------- sampling methods -----------------------------------------------------------------------

# ----- efficient algorithm ---------
def directInversion (ODF,nSamples):
  """ODF contains 'dV_V' (normalized to 1), 'center', 'intervals', 'limits' (in radians)"""
  nOptSamples = max(ODF['nNonZero'],nSamples)                                                       # random subsampling if too little samples requested

  nInvSamples = 0
  repetition = [None]*ODF['nBins']

  scaleLower = 0.0
  nInvSamplesLower = 0
  scaleUpper = float(nOptSamples)
  incFactor = 1.0
  nIter = 0
  nInvSamplesUpper = directInvRepetitions(ODF['dV_V'],scaleUpper)
  while (\
      (scaleUpper-scaleLower > scaleUpper*1e-15 or nInvSamplesUpper < nOptSamples) and \
      nInvSamplesUpper != nOptSamples \
      ):                                                                                            # closer match required?
    if nInvSamplesUpper < nOptSamples:
      scaleLower,scaleUpper = scaleUpper,scaleUpper+incFactor*(scaleUpper-scaleLower)/2.0
      incFactor *= 2.0
      nInvSamplesLower,nInvSamplesUpper = nInvSamplesUpper,directInvRepetitions(ODF['dV_V'],scaleUpper)
    else:
      scaleUpper = (scaleLower+scaleUpper)/2.0
      incFactor = 1.0
      nInvSamplesUpper = directInvRepetitions(ODF['dV_V'],scaleUpper)
    nIter += 1
    damask.util.croak('%i:(%12.11f,%12.11f) %i <= %i <= %i'%(nIter,scaleLower,scaleUpper,
                                                         nInvSamplesLower,nOptSamples,nInvSamplesUpper))
  nInvSamples = nInvSamplesUpper
  scale = scaleUpper
  damask.util.croak('created set of %i samples (%12.11f) with scaling %12.11f delivering %i'%(nInvSamples,
                                                                                        float(nInvSamples)/nOptSamples-1.0,
                                                                                        scale,nSamples))
  repetition = [None]*ODF['nBins']                                                                  # preallocate and clear
    
  for bin in range(ODF['nBins']):                                                                   # loop over bins
    repetition[bin] = int(round(ODF['dV_V'][bin]*scale))                                            # calc repetition

# build set
  set = [None]*nInvSamples
  i = 0
  for bin in range(ODF['nBins']):
    set[i:i+repetition[bin]] = [bin]*repetition[bin]                                                # fill set with bin, i.e. orientation
    i += repetition[bin]                                                                            # advance set counter
  
  orientations     = np.zeros((nSamples,3),'f')
  reconstructedODF = np.zeros(ODF['nBins'],'f')
  unitInc = 1.0/nSamples
  for j in range(nSamples):
    if (j == nInvSamples-1): ex = j
    else: ex = int(round(random.uniform(j+0.5,nInvSamples-0.5)))
    bin = set[ex]
    Eulers = binAsEulers(bin,ODF['interval'],ODF['delta'],ODF['center'])
    orientations[j] = np.degrees(Eulers)
    reconstructedODF[bin] += unitInc
    set[ex] = set[j]                                                                                # exchange orientations
  
  return orientations, reconstructedODF


# ----- trial and error algorithms ---------

def MonteCarloEulers (ODF,nSamples):
  """ODF contains 'dV_V' (normalized to 1), 'center', 'intervals', 'limits' (in radians)"""
  countMC = 0
  maxdV_V = max(ODF['dV_V'])
  orientations     = np.zeros((nSamples,3),'f')
  reconstructedODF = np.zeros(ODF['nBins'],'f')
  unitInc = 1.0/nSamples
  
  for j in range(nSamples):
    MC = maxdV_V*2.0
    bin = 0
    while MC > ODF['dV_V'][bin]:
      countMC += 1
      MC = maxdV_V*random.random()
    Eulers = [limit*random.random() for limit in ODF['limit']]
    bins = EulersAsBins(Eulers,ODF['interval'],ODF['delta'],ODF['center'])
    bin = binsAsBin(bins,ODF['interval'])
    orientations[j] = np.degrees(Eulers)
    reconstructedODF[bin] += unitInc

  return orientations, reconstructedODF, countMC


def MonteCarloBins (ODF,nSamples):
  """ODF contains 'dV_V' (normalized to 1), 'center', 'intervals', 'limits' (in radians)"""
  countMC = 0
  maxdV_V = max(ODF['dV_V'])
  orientations     = np.zeros((nSamples,3),'f')
  reconstructedODF = np.zeros(ODF['nBins'],'f')
  unitInc = 1.0/nSamples
  
  for j in range(nSamples):
    MC = maxdV_V*2.0
    bin = 0
    while MC > ODF['dV_V'][bin]:
      countMC += 1
      MC  = maxdV_V*random.random()
      bin = int(ODF['nBins'] * random.random())
    Eulers = binAsEulers(bin,ODF['interval'],ODF['delta'],ODF['center'])
    orientations[j] = np.degrees(Eulers)
    reconstructedODF[bin] += unitInc

  return orientations, reconstructedODF


def TothVanHoutteSTAT (ODF,nSamples):
  """ODF contains 'dV_V' (normalized to 1), 'center', 'intervals', 'limits' (in radians)"""
  orientations     = np.zeros((nSamples,3),'f')
  reconstructedODF = np.zeros(ODF['nBins'],'f')
  unitInc = 1.0/nSamples
  
  selectors = [random.random() for i in range(nSamples)]
  selectors.sort()
  indexSelector = 0
  
  cumdV_V = 0.0
  countSamples = 0
  
  for bin in range(ODF['nBins']) :
    cumdV_V += ODF['dV_V'][bin]
    while indexSelector < nSamples and selectors[indexSelector] < cumdV_V:
      Eulers = binAsEulers(bin,ODF['interval'],ODF['delta'],ODF['center'])
      orientations[countSamples] = np.degrees(Eulers)
      reconstructedODF[bin] += unitInc
      countSamples += 1
      indexSelector += 1

  damask.util.croak('created set of %i when asked to deliver %i'%(countSamples,nSamples))
  
  return orientations, reconstructedODF


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------
parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description ="""
Transform linear binned ODF data into given number of orientations.
IA: integral approximation, STAT: Van Houtte, MC: Monte Carlo

""", version = scriptID)

algorithms = ['IA', 'STAT','MC']
parser.add_option('-n', '--nsamples',
                  dest = 'number',
                  type = 'int', metavar = 'int',
                  help = 'number of orientations to be generated [%default]')
parser.add_option('-a','--algorithm',
                  dest = 'algorithm',
                  choices = algorithms, metavar = 'string',
                  help = 'sampling algorithm {%s} [IA]'%(', '.join(algorithms)))
parser.add_option('-p','--phase',
                  dest = 'phase',
                  type = 'int', metavar = 'int',
                  help = 'phase index to be used [%default]')
parser.add_option('--crystallite',
                  dest = 'crystallite',
                  type = 'int', metavar = 'int',
                  help = 'crystallite index to be used [%default]')
parser.add_option('-r', '--rnd',
                  dest = 'randomSeed',
                  type = 'int', metavar = 'int', \
                  help = 'seed of random number generator [%default]')
parser.set_defaults(randomSeed = None,
                    number      = 500,
                    algorithm   = 'IA',
                    phase       = 1,
                    crystallite = 1,
                    ang  = True,
                   )

(options,filenames) = parser.parse_args()

nSamples       = options.number
methods        = [options.algorithm]


# --- loop over input files -------------------------------------------------------------------------

if filenames == []: filenames = [None]

for name in filenames:
  try:
    table = damask.ASCIItable(name = name, buffered = False, readonly=True)
  except:
    continue
  damask.util.report(scriptName,name)

  randomSeed = int(os.urandom(4).encode('hex'), 16)  if options.randomSeed is None else options.randomSeed         # random seed per file for second phase
  random.seed(randomSeed)

# ------------------------------------------ read header and data ---------------------------------
  table.head_read()

  errors = []
  labels = ['1_euler','2_euler','3_euler','intensity']
  for i,index in enumerate(table.label_index(labels)):
    if index < 0: errors.append('label {} not present.'.format(labels[i]))
  
  if errors != []:
    damask.util.croak(errors)
    table.close(dismiss = True)
    continue

  table.data_readArray(labels)

# --------------- figure out limits (left/right), delta, and interval -----------------------------

  ODF = {}
  limits = np.array([np.min(table.data[:,0:3],axis=0),
                     np.max(table.data[:,0:3],axis=0)])                                             # min/max euler angles in degrees
  ODF['limit'] = np.radians(limits[1,:])                                                            # right hand limits in radians
  ODF['center'] = 0.0 if all(limits[0,:]<1e-8) else 0.5                                             # vertex or cell centered

  ODF['interval'] = np.array(map(len,[np.unique(table.data[:,i]) for i in range(3)]),'i')           # steps are number of distict values
  ODF['nBins'] = ODF['interval'].prod()
  ODF['delta'] = np.radians(np.array(limits[1,0:3]-limits[0,0:3])/(ODF['interval']-1))              # step size

  if table.data.shape[0] != ODF['nBins']:
    damask.util.croak('expecting %i values but got %i'%(ODF['nBins'],table.data.shape[0]))
    continue
  
# ----- build binnedODF array and normalize ------------------------------------------------------
  sumdV_V = 0.0
  ODF['dV_V'] = [None]*ODF['nBins']
  ODF['nNonZero'] = 0
  dg = ODF['delta'][0]*2.0*math.sin(ODF['delta'][1]/2.0)*ODF['delta'][2]
  for b in range(ODF['nBins']):
    ODF['dV_V'][b] = \
      max(0.0,table.data[b,table.label_index('intensity')]) * dg * \
      math.sin(((b//ODF['interval'][2])%ODF['interval'][1]+ODF['center'])*ODF['delta'][1])
    if ODF['dV_V'][b] > 0.0:
      sumdV_V += ODF['dV_V'][b]
      ODF['nNonZero'] += 1
  
  for b in range(ODF['nBins']):
    ODF['dV_V'][b] /= sumdV_V                                                            # normalize dV/V

  damask.util.croak(['non-zero fraction: %12.11f (%i/%i)'%(float(ODF['nNonZero'])/ODF['nBins'],
                                                     ODF['nNonZero'],
                                                     ODF['nBins']),
               'Volume integral of ODF: %12.11f\n'%sumdV_V,
               'Reference Integral: %12.11f\n'%(ODF['limit'][0]*ODF['limit'][2]*(1-math.cos(ODF['limit'][1]))),
               ])
                                                     
# call methods
  Functions = {'IA': 'directInversion', 'STAT': 'TothVanHoutteSTAT', 'MC': 'MonteCarloBins'}
  method = Functions[options.algorithm]

  Orientations, ReconstructedODF = (globals()[method])(ODF,nSamples)
  
# calculate accuracy of sample
  squaredDiff     = {'orig':0.0,method:0.0}
  squaredRelDiff  = {'orig':0.0,method:0.0}
  mutualProd      = {'orig':0.0,method:0.0}
  indivSum        = {'orig':0.0,method:0.0}
  indivSquaredSum = {'orig':0.0,method:0.0}
  
  for bin in range(ODF['nBins']):
    squaredDiff[method] += (ODF['dV_V'][bin] - ReconstructedODF[bin])**2
    if ODF['dV_V'][bin] > 0.0:
      squaredRelDiff[method] += (ODF['dV_V'][bin] - ReconstructedODF[bin])**2/ODF['dV_V'][bin]**2
    mutualProd[method] += ODF['dV_V'][bin]*ReconstructedODF[bin]
    indivSum[method] += ReconstructedODF[bin]
    indivSquaredSum[method] += ReconstructedODF[bin]**2
  indivSum['orig'] += ODF['dV_V'][bin]
  indivSquaredSum['orig'] += ODF['dV_V'][bin]**2
  
  damask.util.croak(['sqrt(N*)RMSD of ODFs:\t %12.11f'% math.sqrt(nSamples*squaredDiff[method]),
               'RMSrD of ODFs:\t %12.11f'%math.sqrt(squaredRelDiff[method]),
               'rMSD of ODFs:\t %12.11f'%(squaredDiff[method]/indivSquaredSum['orig']),
               'nNonZero correlation slope:\t %12.11f'\
                      %((ODF['nNonZero']*mutualProd[method]-indivSum['orig']*indivSum[method])/\
                        (ODF['nNonZero']*indivSquaredSum['orig']-indivSum['orig']**2)),
               'nNonZero correlation confidence:\t %12.11f'\
                      %((mutualProd[method]-indivSum['orig']*indivSum[method]/ODF['nNonZero'])/\
                        (ODF['nNonZero']*math.sqrt((indivSquaredSum['orig']/ODF['nNonZero']-(indivSum['orig']/ODF['nNonZero'])**2)*\
                        (indivSquaredSum[method]/ODF['nNonZero']-(indivSum[method]/ODF['nNonZero'])**2)))),
              ])
  
  if method == 'IA' and nSamples < ODF['nNonZero']:
    strOpt = '(%i)'%ODF['nNonZero']
  
  formatwidth = 1+int(math.log10(nSamples))

  materialConfig = [
      '#' + scriptID + ' ' + ' '.join(sys.argv[1:]),
      '# random seed %i'%randomSeed,
      '#-------------------#',
      '<microstructure>',
      '#-------------------#',
      ]
  
  for i,ID in enumerate(range(nSamples)):
    materialConfig += ['[Grain%s]'%(str(ID+1).zfill(formatwidth)),
                      'crystallite %i'%options.crystallite,
                      '(constituent)   phase %i   texture %s   fraction 1.0'%(options.phase,str(ID+1).rjust(formatwidth)),
                     ]
  
  materialConfig += [
      '#-------------------#',
      '<texture>',
      '#-------------------#',
      ]

  for ID in range(nSamples):
    eulers = Orientations[ID]
  
    materialConfig += ['[Grain%s]'%(str(ID+1).zfill(formatwidth)),
                     '(gauss)   phi1 {}   Phi {}   phi2 {}   scatter 0.0   fraction 1.0'.format(*eulers),
                     ]

#--- output finalization -------------------------------------------------------------------------- 

  with (open(os.path.splitext(name)[0]+'_'+method+'_'+str(nSamples)+'_material.config','w')) as outfile:
    outfile.write('\n'.join(materialConfig)+'\n')

  table.close()
