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
import subprocess
import shlex
import re
import io
from pathlib import Path

from .. import environment

class Marc:
    """Wrapper to run DAMASK with MSCMarc."""

    def __init__(self,version=environment.options['MSC_VERSION']):
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

        path_MSC = environment.options['MSC_ROOT']
        path_lib = Path(f'{path_MSC}/mentat{self.version}/shlib/linux64')

        if not path_lib.is_dir():
            raise FileNotFoundError(f'library path "{path_lib}" not found')

        return path_lib


    @property
    def tools_path(self):

        path_MSC   = environment.options['MSC_ROOT']
        path_tools = Path(f'{path_MSC}/marc{self.version}/tools')

        if not path_tools.is_dir():
            raise FileNotFoundError(f'tools path "{path_tools}" not found')

        return path_tools


    def submit_job(self,
                   model,
                   job          = 'job1',
                   logfile      = False,
                   compile      = False,
                   optimization = '',
                  ):

        usersub = environment.root_dir/'src/DAMASK_marc'
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

        if logfile is not None:
            try:
                f = open(logfile,'w+')
            except TypeError:
                f = logfile
        else:
            f = io.StringIO()

        proc = subprocess.Popen(shlex.split(cmd),stdout=f,stderr=subprocess.STDOUT)
        proc.wait()
        f.seek(0)

        try:
            v = int(re.search('Exit number ([0-9]+)',''.join(f.readlines())).group(1))
        except (AttributeError,ValueError):
            raise RuntimeError('Marc simulation failed (unknown return value)')

        if v != 3004:
            raise RuntimeError(f'Marc simulation failed ({v})')
