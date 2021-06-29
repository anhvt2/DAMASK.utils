#!/usr/bin/env bash
# Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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
if [ $1x != 3to2x ]; then
  echo 'python2.7 to python3'
  find . -name '*.py' -type f | xargs sed -i 's/usr\/bin\/env python2.7/usr\/bin\/env python3/g'
else
  echo 'python3 to python2.7'
  find . -name '*.py' -type f | xargs sed -i 's/usr\/bin\/env python3/usr\/bin\/env python2.7/g'
fi
