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
from pathlib import Path

_msc_version = 2021.2
_msc_root = '/opt/msc'
_damask_root = str(Path(__file__).parents[3])

class Marc:
    """Wrapper to run DAMASK with MSC.Marc."""

    def __init__(self,msc_version=_msc_version,msc_root=_msc_root,damask_root=_damask_root):
        """
        Create a Marc solver object.

        Parameters
        ----------
        version : float
            Marc version

        """
        self.msc_version = msc_version
        self.msc_root    = Path(msc_root)
        self.damask_root = Path(damask_root)

    @property
    def library_path(self):

        path_lib = self.msc_root/f'mentat{self.msc_version}/shlib/linux64'
        if not path_lib.is_dir():
            raise FileNotFoundError(f'library path "{path_lib}" not found')

        return path_lib


    @property
    def tools_path(self):

        path_tools = self.msc_root/f'marc{self.msc_version}/tools'
        if not path_tools.is_dir():
            raise FileNotFoundError(f'tools path "{path_tools}" not found')

        return path_tools


    def submit_job(self, model, job,
                   compile      = False,
                   optimization = ''):
        """
        Assemble command line arguments and call Marc executable.

        Parameters
        ----------
        model : str
            Name of model.
        job : str
            Name of job.
        compile : bool, optional
            Compile DAMASK_Marc user subroutine (and save for future use).
            Defaults to False.
        optimization : str, optional
            Optimization level '' (-O0), 'l' (-O1), or 'h' (-O3).
            Defaults to ''.

        """
        usersub = (self.damask_root/'src/DAMASK_Marc').with_suffix('.f90' if compile else '.marc')
        if not usersub.is_file():
            raise FileNotFoundError(f'subroutine ({"source" if compile else "binary"}) "{usersub}" not found')

        # Define options [see Marc Installation and Operation Guide, pp 23]
        script = f'run_damask_{optimization}mp'

        cmd = f'{self.tools_path/script} -jid {model}_{job} -nprocd 1 -autorst 0 -ci n -cr n -dcoup 0 -b no -v no ' \
            + (f'-u {usersub} -save y' if compile else f'-prog {usersub.with_suffix("")}')

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
