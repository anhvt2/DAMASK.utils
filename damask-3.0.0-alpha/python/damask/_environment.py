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
from pathlib import Path

class Environment:

    def __init__(self):
        """Do Nothing."""
        pass

    @property
    def screen_size(self):
        width  = 1024
        height =  768
        try:
            import wx
            _ = wx.App(False)                                                                       # noqa
            width, height = wx.GetDisplaySize()
        except ImportError:
            try:
                import tkinter
                tk = tkinter.Tk()
                width  = tk.winfo_screenwidth()
                height = tk.winfo_screenheight()
                tk.destroy()
            except Exception as e:
                pass
        return (width,height)


    @property
    def options(self):
        options = {}
        for item in ['DAMASK_NUM_THREADS',
                     'MSC_ROOT',
                     'MARC_VERSION',
                     ]:
            options[item] = os.environ[item] if item in os.environ else None

        return options


    @property
    def root_dir(self):
        """Return DAMASK root path."""
        return Path(__file__).parents[2]


    # for compatibility
    def rootDir(self):
        return str(self.root_dir)
