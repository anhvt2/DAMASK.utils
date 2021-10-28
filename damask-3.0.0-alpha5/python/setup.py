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
import setuptools
from pathlib import Path
import re

# https://www.python.org/dev/peps/pep-0440
with open(Path(__file__).parent/'damask/VERSION') as f:
    version = re.sub(r'(-([^-]*)).*$',r'.\2',re.sub(r'^v(\d+\.\d+(\.\d+)?)',r'\1',f.readline().strip()))

setuptools.setup(
    name='damask',
    version=version,
    author='The DAMASK team',
    author_email='damask@mpie.de',
    description='DAMASK processing tools',
    long_description='Pre- and post-processing tools for DAMASK',
    url='https://damask.mpie.de',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires = '>=3.7',
    install_requires = [
        'pandas>=0.24',                                                                             # requires numpy
        'numpy>=1.17',                                                                              # needed for default_rng
        'scipy>=1.2',
        'h5py>=2.9',                                                                                # requires numpy
        'vtk>=8.1',
        'matplotlib>=3.0',                                                                          # requires numpy, pillow
        'pyyaml>=3.12'
    ],
    classifiers = [
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
)
