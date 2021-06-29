# Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
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
import subprocess
import shlex
import re
import os
from pathlib import Path

class Marc:
    """Wrapper to run DAMASK with MSCMarc."""

    def __init__(self,version=os.environ['MSC_VERSION']):
        """
        Create a Marc solver object.

        Parameters
        ----------
        version : float
            Marc version

        """
        self.solver  = 'Marc'
        self.version = version

    @property
    def library_path(self):

        path_lib = Path(f'{os.environ["MSC_ROOT"]}/mentat{self.version}/shlib/linux64')
        if not path_lib.is_dir():
            raise FileNotFoundError(f'library path "{path_lib}" not found')

        return path_lib


    @property
    def tools_path(self):

        path_tools = Path(f'{os.environ["MSC_ROOT"]}/marc{self.version}/tools')
        if not path_tools.is_dir():
            raise FileNotFoundError(f'tools path "{path_tools}" not found')

        return path_tools


    def submit_job(self, model, job,
                   compile      = False,
                   optimization = ''):

        usersub = Path(os.environ['DAMASK_ROOT'])/'src/DAMASK_Marc'
        usersub = usersub.parent/(usersub.name + ('.f90' if compile else '.marc'))
        if not usersub.is_file():
            raise FileNotFoundError(f'subroutine ({"source" if compile else "binary"}) "{usersub}" not found')

        # Define options [see Marc Installation and Operation Guide, pp 23]
        script = f'run_damask_{optimization}mp'

        cmd = str(self.tools_path/script) + \
              ' -jid ' + model+'_'+job + \
              ' -nprocd 1 -autorst 0 -ci n -cr n -dcoup 0 -b no -v no'
        cmd += ' -u ' + str(usersub) + ' -save y' if compile else \
               ' -prog ' + str(usersub.with_suffix(''))
        print(cmd)

        ret = subprocess.run(shlex.split(cmd),capture_output=True)

        try:
            v = int(re.search('Exit number ([0-9]+)',ret.stderr.decode()).group(1))
            if 3004 != v:
                print(ret.stderr.decode())
                print(ret.stdout.decode())
                raise RuntimeError(f'Marc simulation failed ({v})')
        except (AttributeError,ValueError):
            print(ret.stderr.decode())
            print(ret.stdout.decode())
            raise RuntimeError('Marc simulation failed (unknown return value)')

