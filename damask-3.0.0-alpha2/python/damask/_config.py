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
from io import StringIO
import abc

import numpy as np
import yaml

class NiceDumper(yaml.SafeDumper):
    """Make YAML readable for humans."""

    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

    def represent_data(self, data):
        """Cast Config objects and its subclasses to dict."""
        return self.represent_data(dict(data)) if isinstance(data, dict) and type(data) != dict else \
               super().represent_data(data)


class Config(dict):
    """YAML-based configuration."""

    def __repr__(self):
        """Show as in file."""
        output = StringIO()
        self.save(output)
        output.seek(0)
        return ''.join(output.readlines())

    @classmethod
    def load(cls,fname):
        """
        Load from yaml file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            Filename or file for writing.

        """
        try:
            fhandle = open(fname)
        except TypeError:
            fhandle = fname
        return cls(yaml.safe_load(fhandle))

    def save(self,fname,**kwargs):
        """
        Save to yaml file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            Filename or file for writing.
        **kwargs : dict
            Keyword arguments parsed to yaml.dump.

        """
        try:
            fhandle = open(fname,'w')
        except TypeError:
            fhandle = fname

        if 'width' not in kwargs:
            kwargs['width'] = 256
        if 'default_flow_style' not in kwargs:
            kwargs['default_flow_style'] = None
        if 'sort_keys' not in kwargs:
            kwargs['sort_keys'] = False

        def array_representer(dumper, data):
            """Convert numpy array to list of native types."""
            return dumper.represent_list([d.item() for d in data])

        NiceDumper.add_representer(np.ndarray, array_representer)

        try:
            fhandle.write(yaml.dump(self,Dumper=NiceDumper,**kwargs))
        except TypeError:                                                                           # compatibility with old pyyaml
            del kwargs['sort_keys']
            fhandle.write(yaml.dump(self,Dumper=NiceDumper,**kwargs))


    @property
    @abc.abstractmethod
    def is_complete(self):
        """Check for completeness."""
        pass

    @property
    @abc.abstractmethod
    def is_valid(self):
        """Check for valid file layout."""
        pass
