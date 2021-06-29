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

import threading,time,os,sys,random
import numpy as np
from optparse import OptionParser
from cStringIO import StringIO
import damask

scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])

mismatch = None
currentSeedsName = None

#---------------------------------------------------------------------------------------------------
class myThread (threading.Thread):
  """perturbes seed in seed file, performes Voronoi tessellation, evaluates, and updates best match"""

  def __init__(self, threadID):
    threading.Thread.__init__(self)
    self.threadID = threadID

  def run(self):
    global bestSeedsUpdate
    global bestSeedsVFile
    global nMicrostructures
    global delta
    global points
    global target
    global match
    global baseFile
    global maxSeeds

    s.acquire()
    bestMatch = match
    s.release()
    
    random.seed(options.randomSeed+self.threadID)                                                   # initializes to given seeds
    knownSeedsUpdate = bestSeedsUpdate -1.0                                                         # trigger update of local best seeds
    randReset = True                                                                                # aquire new direction
    
    myBestSeedsVFile    = StringIO()                                                                # store local copy of best seeds file
    perturbedSeedsVFile = StringIO()                                                                # perturbed best seeds file
    perturbedGeomVFile  = StringIO()                                                                # tessellated geom file

#--- still not matching desired bin class ----------------------------------------------------------
    while bestMatch < options.threshold:
      s.acquire()                                                                                   # ensure only one thread acces global data
      if bestSeedsUpdate > knownSeedsUpdate:                                                        # write best fit to virtual file
        knownSeedsUpdate = bestSeedsUpdate
        bestSeedsVFile.reset()
        myBestSeedsVFile.close()
        myBestSeedsVFile = StringIO()
        i=0
        for line in bestSeedsVFile:
          myBestSeedsVFile.write(line)
      s.release()
      
      if randReset:                                                                                 # new direction because current one led to worse fit

        randReset = False

        NmoveGrains = random.randrange(1,maxSeeds)
        selectedMs = []
        direction = []
        for i in range(NmoveGrains):
          selectedMs.append(random.randrange(1,nMicrostructures))

          direction.append(np.array(((random.random()-0.5)*delta[0],
                                     (random.random()-0.5)*delta[1],
                                     (random.random()-0.5)*delta[2])))
        
      perturbedSeedsVFile.close()                                                                   # reset virtual file
      perturbedSeedsVFile = StringIO()
      myBestSeedsVFile.reset()

      perturbedSeedsTable = damask.ASCIItable(myBestSeedsVFile,perturbedSeedsVFile,labeled=True)    # write best fit to perturbed seed file
      perturbedSeedsTable.head_read()
      perturbedSeedsTable.head_write()
      outputAlive=True
      ms = 1
      i = 0
      while outputAlive and perturbedSeedsTable.data_read():                                        # perturbe selected microstructure
        if ms in selectedMs:
          newCoords=np.array(tuple(map(float,perturbedSeedsTable.data[0:3]))+direction[i])
          newCoords=np.where(newCoords>=1.0,newCoords-1.0,newCoords)                                # ensure that the seeds remain in the box
          newCoords=np.where(newCoords <0.0,newCoords+1.0,newCoords)
          perturbedSeedsTable.data[0:3]=[format(f, '8.6f') for f in newCoords]
          direction[i]*=2.
          i+= 1
        ms+=1
        perturbedSeedsTable.data_write()
#--- do tesselation with perturbed seed file ----------------------------------------------------------
      perturbedGeomVFile.close()
      perturbedGeomVFile = StringIO()
      perturbedSeedsVFile.reset()
      perturbedGeomVFile.write(damask.util.execute('geom_fromVoronoiTessellation '+
                     ' -g '+' '.join(map(str, options.grid)),streamIn=perturbedSeedsVFile)[0])
      perturbedGeomVFile.reset()

#--- evaluate current seeds file ----------------------------------------------------------------------
      perturbedGeomTable = damask.ASCIItable(perturbedGeomVFile,None,labeled=False,readonly=True)
      perturbedGeomTable.head_read()
      for i in perturbedGeomTable.info:
        if i.startswith('microstructures'): myNmicrostructures = int(i.split('\t')[1])
      perturbedGeomTable.data_readArray()
      perturbedGeomTable.output_flush()
      currentData=np.bincount(perturbedGeomTable.data.astype(int).ravel())[1:]/points
      currentError=[]
      currentHist=[]
      for i in range(nMicrostructures):                                                             # calculate the deviation in all bins per histogram
        currentHist.append(np.histogram(currentData,bins=target[i]['bins'])[0])
        currentError.append(np.sqrt(np.square(np.array(target[i]['histogram']-currentHist[i])).sum()))

# as long as not all grains are within the range of the target, use the deviation to left and right as error
      if currentError[0]>0.0:                                                                       
        currentError[0] *=((target[0]['bins'][0]-np.min(currentData))**2.0+                        
                           (target[0]['bins'][1]-np.max(currentData))**2.0)**0.5                    # norm of deviations by number of usual bin deviation
      s.acquire()                                                                                   # do the evaluation serially
      bestMatch = match
#--- count bin classes with no mismatch ----------------------------------------------------------------------
      myMatch=0
      for i in range(nMicrostructures):
        if currentError[i] > 0.0: break
        myMatch = i+1

      if myNmicrostructures == nMicrostructures:
        for i in range(min(nMicrostructures,myMatch+options.bins)):
          if currentError[i] > target[i]['error']:                                                  # worse fitting, next try
            randReset = True
            break
          elif currentError[i] < target[i]['error']:                                                # better fit
            bestSeedsUpdate = time.time()                                                           # save time of better fit
            damask.util.croak('Thread {:d}: Better match ({:d} bins, {:6.4f} --> {:6.4f})'\
                                          .format(self.threadID,i+1,target[i]['error'],currentError[i]))
            damask.util.croak('          target: '+np.array_str(target[i]['histogram']))
            damask.util.croak('          best:   '+np.array_str(currentHist[i]))
            currentSeedsName = baseFile+'_'+str(bestSeedsUpdate).replace('.','-')                   # name of new seed file (use time as unique identifier)
            perturbedSeedsVFile.reset()
            bestSeedsVFile.close()
            bestSeedsVFile = StringIO()
            sys.stdout.flush()
            with open(currentSeedsName+'.seeds','w') as currentSeedsFile:                           # write to new file
              for line in perturbedSeedsVFile:
                currentSeedsFile.write(line)
                bestSeedsVFile.write(line)
            for j in range(nMicrostructures):                                                       # save new errors for all bins
              target[j]['error'] = currentError[j]
            if myMatch > match:                                                                     # one or more new bins have no deviation
              damask.util.croak( 'Stage {:d}  cleared'.format(myMatch))
              match=myMatch
              sys.stdout.flush()
            break
          if i == min(nMicrostructures,myMatch+options.bins)-1:                                     # same quality as before: take it to keep on moving
            bestSeedsUpdate = time.time()
            perturbedSeedsVFile.reset()
            bestSeedsVFile.close()
            bestSeedsVFile = StringIO()
            for line in perturbedSeedsVFile:
              bestSeedsVFile.write(line)
            for j in range(nMicrostructures):
              target[j]['error'] = currentError[j]
            randReset = True
      else:                                                                                         #--- not all grains are tessellated
        damask.util.croak('Thread {:d}: Microstructure mismatch ({:d} microstructures mapped)'\
                                                         .format(self.threadID,myNmicrostructures))
        randReset = True

      
      s.release()


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [file[s]]', description = """
Monte Carlo simulation to produce seed file that gives same size distribution like given geometry file.

""", version = scriptID)

parser.add_option('-s','--seeds',    dest='seedFile', metavar='string',
                                     help='name of the intial seed file. If not found, a new one is created [%default]')
parser.add_option('-g','--grid',     dest='grid', type='int', nargs=3, metavar='int int int',
                                     help='a,b,c grid of hexahedral box [%default]')
parser.add_option('-t','--threads',  dest='threads', type='int', metavar='int',
                                     help='number of parallel executions [%default]')
parser.add_option('-r', '--rnd',     dest='randomSeed', type='int', metavar='int',
                                     help='seed of random number generator [%default]')
parser.add_option('--target',        dest='target', metavar='string',
                                     help='name of the geom file with target distribution [%default]')
parser.add_option('--tolerance',     dest='threshold', type='int', metavar='int',
                                     help='stopping criterion (bin number) [%default]')
parser.add_option('--scale',         dest='scale',type='float', metavar='float',                                     
                                     help='maximum moving distance of perturbed seed in pixel [%default]')
parser.add_option('--bins',          dest='bins', type='int', metavar='int',
                                     help='bins to sort beyond current best fit [%default]')
parser.add_option('--maxseeds',      dest='maxseeds', type='int', metavar='int',
                                     help='maximum number of seeds to move simulateneously [number of seeds]')

parser.set_defaults(seedFile    = 'seeds',
                    grid        = (64,64,64),
                    threads     = 2,
                    randomSeed  = None,
                    target      = 'geom',
                    threshold   = 20,
                    bins        = 15,
                    scale       = 1.0,
                    maxseeds    = 0)

options = parser.parse_args()[0]

damask.util.report(scriptName,options.seedFile)

if options.randomSeed is None:
  options.randomSeed = int(os.urandom(4).encode('hex'), 16)
damask.util.croak(options.randomSeed)
delta = (options.scale/options.grid[0],options.scale/options.grid[1],options.scale/options.grid[2])
baseFile=os.path.splitext(os.path.basename(options.seedFile))[0]
points = np.array(options.grid).prod().astype('float')

# ----------- calculate target distribution and bin edges
targetGeomFile = os.path.splitext(os.path.basename(options.target))[0]+'.geom'
targetGeomTable = damask.ASCIItable(targetGeomFile,None,labeled=False,readonly=True)
targetGeomTable.head_read()
info,devNull =  targetGeomTable.head_getGeom()
nMicrostructures = info['microstructures']
targetVolFrac = np.bincount(targetGeomTable.microstructure_read(info['grid']))[1:nMicrostructures+1]/\
                                                                       float(info['grid'].prod())
target=[]
for i in range(1,nMicrostructures+1):
  targetHist,targetBins = np.histogram(targetVolFrac,bins=i) #bin boundaries
  target.append({'histogram':targetHist,'bins':targetBins})

# ----------- create initial seed file or open existing one
bestSeedsVFile = StringIO()
if os.path.isfile(os.path.splitext(options.seedFile)[0]+'.seeds'):
  with open(os.path.splitext(options.seedFile)[0]+'.seeds') as initialSeedFile:
    for line in initialSeedFile: bestSeedsVFile.write(line)
else:
  bestSeedsVFile.write(damask.util.execute('seeds_fromRandom'+\
                                ' -g '+' '.join(map(str, options.grid))+\
                                ' -r {:d}'.format(options.randomSeed)+\
                                ' -N '+str(nMicrostructures))[0])
bestSeedsUpdate = time.time()

# ----------- tessellate initial seed file to get and evaluate geom file
bestSeedsVFile.reset()
initialGeomVFile = StringIO()
initialGeomVFile.write(damask.util.execute('geom_fromVoronoiTessellation '+
                               ' -g '+' '.join(map(str, options.grid)),bestSeedsVFile)[0])
initialGeomVFile.reset()
initialGeomTable = damask.ASCIItable(initialGeomVFile,None,labeled=False,readonly=True)
initialGeomTable.head_read()
info,devNull =  initialGeomTable.head_getGeom()

if info['microstructures'] != nMicrostructures: damask.util.croak('error. Microstructure count mismatch')

initialData = np.bincount(initialGeomTable.microstructure_read(info['grid']))/points
for i in range(nMicrostructures):
  initialHist = np.histogram(initialData,bins=target[i]['bins'])[0]
  target[i]['error']=np.sqrt(np.square(np.array(target[i]['histogram']-initialHist)).sum())

# as long as not all grain sizes are within the range, the error is the deviation to left and right 
if target[0]['error'] > 0.0:
  target[0]['error'] *=((target[0]['bins'][0]-np.min(initialData))**2.0+
                        (target[0]['bins'][1]-np.max(initialData))**2.0)**0.5
match=0
for i in range(nMicrostructures):
  if target[i]['error'] > 0.0: break
  match = i+1


if options.maxseeds < 1: 
  maxSeeds = info['microstructures']
else:
  maxSeeds = options.maxseeds

if match >0: damask.util.croak('Stage {:d} cleared'.format(match))
sys.stdout.flush()
initialGeomVFile.close()

# start mulithreaded monte carlo simulation
threads=[]
s=threading.Semaphore(1)

for i in range(options.threads):
  threads.append(myThread(i))
  threads[i].start()
for i in range(options.threads):
  threads[i].join()
