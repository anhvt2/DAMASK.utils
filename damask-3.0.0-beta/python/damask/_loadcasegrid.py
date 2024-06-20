# Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
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
from typing import Optional, Union, Dict, Any, List

from numpy import ma
import yaml

from ._typehints import FileHandle
from ._yaml import NiceDumper
from . import util
from . import YAML


class MaskedMatrixDumper(NiceDumper):
    """Format masked matrices."""

    def represent_data(self, data: Any):
        return super().represent_data(data.astype(object).filled('x') if isinstance(data, ma.core.MaskedArray) else data) # type: ignore[attr-defined]


class LoadcaseGrid(YAML):
    """Load case for grid solver."""

    def __init__(self,
                 config: Optional[Union[str,Dict[str,Any]]] = None,
                 *,
                 solver: Optional[Dict[str,str]] = None,
                 loadstep: Optional[List[Dict[str,Any]]] = None):
        """
        New grid solver load case.

        Parameters
        ----------
        config : dict or str, optional
            Grid solver load case. String needs to be valid YAML.
        solver : dict, optional
            Solver configuration.
            Defaults to an empty dict if 'config' is not given.
        loadstep : list of dict, optional
            Load step configuration.
            Defaults to an empty list if 'config' is not given.

        """
        kwargs: Dict[str,Union[Dict[str,str],List[Dict[str,Any]]]] = {}
        default: Union[List,Dict]
        for arg,value,default in [('solver',solver,{}),('loadstep',loadstep,[])]: # type: ignore[assignment]
            if value is not None:
                kwargs[arg] = value
            elif config is None:
                kwargs[arg] = default

        super().__init__(config,**kwargs)


    def save(self,
             fname: FileHandle,
             **kwargs):
        """
        Save to YAML file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            Filename or file to write.
        **kwargs : dict
            Keyword arguments parsed to yaml.dump.

        """
        for key,default in dict(width=256,
                                default_flow_style=None,
                                sort_keys=False).items():
            if key not in kwargs:
                kwargs[key] = default

        with util.open_text(fname,'w') as fhandle:
            try:
                fhandle.write(yaml.dump(self,Dumper=MaskedMatrixDumper,**kwargs))
            except TypeError:                                                                       # compatibility with old pyyaml
                del kwargs['sort_keys']
                fhandle.write(yaml.dump(self,Dumper=MaskedMatrixDumper,**kwargs))
