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

# Makes postprocessing routines accessible from everywhere.
import sys
from pathlib import Path

import damask

env = damask.Environment()
bin_dir = env.root_dir/Path('bin')

if not bin_dir.exists():
    bin_dir.mkdir()


sys.stdout.write('\nsymbolic linking...\n')
for sub_dir in ['pre','post']:
    the_dir = env.root_dir/Path('processing')/Path(sub_dir)

    for the_file in the_dir.glob('*.py'):
        src = the_dir/the_file
        dst = bin_dir/Path(the_file.with_suffix('').name)
        if dst.is_file(): dst.unlink() # dst.unlink(True) for Python >3.8
        dst.symlink_to(src)


sys.stdout.write('\npruning broken links...\n')
for filename in bin_dir.glob('*'):
    if not filename.is_file():
        filename.unlink
