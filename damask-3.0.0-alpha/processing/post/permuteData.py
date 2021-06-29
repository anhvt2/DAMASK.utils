#!/usr/bin/env python3
# Copyright 2011-20 Max-Planck-Institut für Eisenforschung GmbH
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

import os
import sys
from io import StringIO
from optparse import OptionParser

import numpy as np

import damask


scriptName = os.path.splitext(os.path.basename(__file__))[0]
scriptID   = ' '.join([scriptName,damask.version])


# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------

parser = OptionParser(option_class=damask.extendableOption, usage='%prog options [ASCIItable(s)]', description = """
Permute all values in given column(s).

""", version = scriptID)

parser.add_option('-l','--label',
                  dest = 'label',
                  action = 'extend', metavar = '<string LIST>',
                  help  ='column(s) to permute')
parser.add_option('-u', '--unique',
                  dest = 'unique',
                  action = 'store_true',
                  help = 'shuffle unique values as group')
parser.add_option('-r', '--rnd',
                  dest = 'randomSeed',
                  type = 'int', metavar = 'int',
                  help = 'seed of random number generator [%default]')

parser.set_defaults(label = [],
                    unique = False,
                    randomSeed = None,
                   )

(options,filenames) = parser.parse_args()
if filenames == []: filenames = [None]

for name in filenames:
    damask.util.report(scriptName,name)

    table = damask.Table.from_ASCII(StringIO(''.join(sys.stdin.read())) if name is None else name)

    randomSeed = int(os.urandom(4).hex(), 16) if options.randomSeed is None else options.randomSeed # random seed per file
    rng = np.random.default_rng(randomSeed)

    for label in options.label:
        data = table.get(label)
        uniques,inverse  = np.unique(data,return_inverse=True,axis=0) if options.unique else (data,np.arange(len(data)))
        rng.shuffle(uniques)
        table.set(label,uniques[inverse], scriptID+' '+' '.join(sys.argv[1:]))

    table.to_ASCII(sys.stdout if name is None else name)
