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
# sets up an environment for DAMASK on zsh
# usage:  source DAMASK.zsh

function canonicalPath {
  python3 -c "import os,sys; print(os.path.normpath(os.path.realpath(os.path.expanduser(sys.argv[1]))))" $1
}

function blink {
  echo -e "\033[2;5m$1\033[0m"
}

ENV_ROOT=$(canonicalPath "${0:a:h}")
DAMASK_ROOT=$(canonicalPath "${0:a:h}'/..")

# add BRANCH if DAMASK_ROOT is a git repository
cd $DAMASK_ROOT >/dev/null; BRANCH=$(git branch 2>/dev/null| grep -E '^\* '); cd - >/dev/null

PATH=${DAMASK_ROOT}/bin:$PATH

SOLVER=$(which DAMASK_grid || true 2>/dev/null)
[[ "x$SOLVER" == "x" ]] && SOLVER=$(blink 'Not found!')


# currently, there is no information that unlimited stack size causes problems
# still, http://software.intel.com/en-us/forums/topic/501500 suggest to fix it
# more info https://jblevins.org/log/segfault
#           https://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap
#           http://superuser.com/questions/220059/what-parameters-has-ulimit
ulimit -s unlimited 2>/dev/null # maximum stack size (kB)

# disable output in case of scp
if [ ! -z "$PS1" ]; then
  echo
  echo Düsseldorf Advanced Materials Simulation Kit --- DAMASK
  echo Max-Planck-Institut für Eisenforschung GmbH, Düsseldorf
  echo https://damask.mpie.de
  echo
  echo "Using environment with ..."
  echo "DAMASK             $DAMASK_ROOT $BRANCH"
  echo "Grid Solver        $SOLVER"
  if [ "x$PETSC_DIR" != "x" ]; then
    echo -n "PETSc location     "
    [ -d $PETSC_DIR ] && echo $PETSC_DIR || blink $PETSC_DIR
    [[ $(canonicalPath "$PETSC_DIR") == $PETSC_DIR ]] \
    || echo "               ~~> "$(canonicalPath "$PETSC_DIR")
  fi
  [[ "x$PETSC_ARCH" != "x" ]] && echo "PETSc architecture $PETSC_ARCH"
  [[ "x$OMP_NUM_THREADS" == "x" ]] && export OMP_NUM_THREADS=4
  echo "Multithreading     OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo -n "heap  size         "
   [[ "$(ulimit -d)" == "unlimited" ]] \
   && echo "unlimited" \
   || echo $(python3 -c \
          "import math; \
           size=$(( 1024*$(ulimit -d) )); \
           print('{:.4g} {}'.format(size / (1 << ((int(math.log(size,2) / 10) if size else 0) * 10)), \
           ['bytes','KiB','MiB','GiB','TiB','EiB','ZiB'][int(math.log(size,2) / 10) if size else 0]))")
  echo -n "stack size         "
   [[ "$(ulimit -s)" == "unlimited" ]] \
   && echo "unlimited" \
   || echo $(python3 -c \
          "import math; \
           size=$(( 1024*$(ulimit -s) )); \
           print('{:.4g} {}'.format(size / (1 << ((int(math.log(size,2) / 10) if size else 0) * 10)), \
           ['bytes','KiB','MiB','GiB','TiB','EiB','ZiB'][int(math.log(size,2) / 10) if size else 0]))")
  echo
fi

export DAMASK_ROOT
export PYTHONPATH=$DAMASK_ROOT/python:$PYTHONPATH

for var in SOLVER BRANCH; do
  unset "${var}"
done
unset "ENV_ROOT"
