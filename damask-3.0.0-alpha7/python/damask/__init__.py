# Copyright 2011-2022 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Tools for managing DAMASK simulations."""

from pathlib import Path as _Path
import re as _re

name = 'damask'
with open(_Path(__file__).parent/_Path('VERSION')) as _f:
    version = _re.sub(r'^v','',_f.readline().strip())
    __version__ = version

from .                 import _typehints       # noqa
from .                 import util             # noqa
from .                 import seeds            # noqa
from .                 import tensor           # noqa
from .                 import mechanics        # noqa
from .                 import solver           # noqa
from .                 import grid_filters     # noqa
# Modules that contain only one class (of the same name), are prefixed by a '_'.
# For example, '_colormap' contains a class called 'Colormap' which is imported as 'damask.Colormap'.
from ._rotation        import Rotation         # noqa
from ._crystal         import Crystal          # noqa
from ._orientation     import Orientation      # noqa
from ._table           import Table            # noqa
from ._colormap        import Colormap         # noqa
from ._vtk             import VTK              # noqa
from ._config          import Config           # noqa
from ._configmaterial  import ConfigMaterial   # noqa
from ._grid            import Grid             # noqa
from ._result          import Result           # noqa
