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
"""Functionality for typehints."""

from typing import Sequence, Union, Literal, TextIO, Collection
from pathlib import Path

import numpy as np


FloatSequence = Union[np.ndarray,Sequence[float]]
IntSequence = Union[np.ndarray,Sequence[int]]
IntCollection = Union[np.ndarray,Collection[int]]
FileHandle = Union[TextIO, str, Path]
CrystalFamily = Union[None,Literal['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'hexagonal', 'cubic']]
CrystalLattice = Union[None,Literal['aP', 'mP', 'mS', 'oP', 'oS', 'oI', 'oF', 'tP', 'tI', 'hP', 'cP', 'cI', 'cF']]
CrystalKinematics = Literal['slip', 'twin']
NumpyRngSeed = Union[int, IntSequence, np.random.SeedSequence, np.random.Generator]
# BitGenerator does not exists in older numpy versions
#NumpyRngSeed = Union[int, IntSequence, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator]
