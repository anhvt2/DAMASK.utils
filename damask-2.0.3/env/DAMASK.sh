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
# sets up an environment for DAMASK on bash
# usage:  source DAMASK.sh

function canonicalPath {
  python -c "import os,sys; print(os.path.normpath(os.path.realpath(os.path.expanduser(sys.argv[1]))))" $1
}

function blink {
  echo -e "\033[2;5m$1\033[0m"
}

if [ "$OSTYPE" == "linux-gnu" ] || [ "$OSTYPE" == 'linux' ]; then
  DAMASK_ROOT=$(dirname $BASH_SOURCE)
else
  [[ "${BASH_SOURCE::1}" == "/" ]] && BASE="" || BASE="$(pwd)/"
  STAT=$(stat "$(dirname $BASE$BASH_SOURCE)")
  DAMASK_ROOT=${STAT##* }
fi

DAMASK_ROOT=$(canonicalPath "$DAMASK_ROOT/../")


# shorthand command to change to DAMASK_ROOT directory
eval "function DAMASK_root() { cd $DAMASK_ROOT; }"

# defining set() allows to source the same file for tcsh and bash, with and without space around =
set() {
    export $1$2$3
 }
source $DAMASK_ROOT/CONFIG
unset -f set

# add BRANCH if DAMASK_ROOT is a git repository
cd $DAMASK_ROOT >/dev/null; BRANCH=$(git branch 2>/dev/null| grep -E '^\* '); cd - >/dev/null

# add DAMASK_BIN if present
[ "x$DAMASK_BIN" != "x" ] && PATH=$DAMASK_BIN:$PATH

SOLVER=$(type -p DAMASK_spectral || true 2>/dev/null)
[ "x$SOLVER" == "x" ] && SOLVER=$(blink 'Not found!')

PROCESSING=$(type -p postResults || true 2>/dev/null)
[ "x$PROCESSING" == "x" ] && PROCESSING=$(blink 'Not found!')

[ "x$DAMASK_NUM_THREADS" == "x" ] && DAMASK_NUM_THREADS=1

# currently, there is no information that unlimited causes problems
# still, http://software.intel.com/en-us/forums/topic/501500 suggest to fix it
# more info https://jblevins.org/log/segfault
#           https://stackoverflow.com/questions/79923/what-and-where-are-the-stack-and-heap
# http://superuser.com/questions/220059/what-parameters-has-ulimit
ulimit -d unlimited 2>/dev/null # maximum  heap size (kB)
ulimit -s unlimited 2>/dev/null # maximum stack size (kB)
ulimit -v unlimited 2>/dev/null # maximum virtual memory size
ulimit -m unlimited 2>/dev/null # maximum physical memory size

# disable output in case of scp
if [ ! -z "$PS1" ]; then
  echo
  echo Düsseldorf Advanced Materials Simulation Kit --- DAMASK
  echo Max-Planck-Institut für Eisenforschung GmbH, Düsseldorf
  echo https://damask.mpie.de
  echo
  echo Using environment with ...
  echo "DAMASK             $DAMASK_ROOT $BRANCH"
  echo "Spectral Solver    $SOLVER" 
  echo "Post Processing    $PROCESSING"
  if [ "x$PETSC_DIR"   != "x" ]; then
    echo -n "PETSc location     "
    [ -d $PETSC_DIR ] && echo $PETSC_DIR || blink $PETSC_DIR
    [[ $(canonicalPath "$PETSC_DIR") == $PETSC_DIR ]] \
    || echo "               ~~> "$(canonicalPath "$PETSC_DIR")
  fi
  echo -n "MSC.Marc/Mentat    "
  [ -d $MSC_ROOT ] && echo $MSC_ROOT || blink $MSC_ROOT
  echo
  echo "Multithreading     DAMASK_NUM_THREADS=$DAMASK_NUM_THREADS"
  echo -n "heap  size         "
   [[ "$(ulimit -d)" == "unlimited" ]] \
   && echo "unlimited" \
   || echo $(python -c \
          "import math; \
           size=$(( 1024*$(ulimit -d) )); \
           print('{:.4g} {}'.format(size / (1 << ((int(math.log(size,2) / 10) if size else 0) * 10)), \
           ['bytes','KiB','MiB','GiB','TiB','EiB','ZiB'][int(math.log(size,2) / 10) if size else 0]))")
  echo -n "stack size         "
   [[ "$(ulimit -s)" == "unlimited" ]] \
   && echo "unlimited" \
   || echo $(python -c \
          "import math; \
           size=$(( 1024*$(ulimit -s) )); \
           print('{:.4g} {}'.format(size / (1 << ((int(math.log(size,2) / 10) if size else 0) * 10)), \
           ['bytes','KiB','MiB','GiB','TiB','EiB','ZiB'][int(math.log(size,2) / 10) if size else 0]))")
  echo
fi

export DAMASK_NUM_THREADS
export PYTHONPATH=$DAMASK_ROOT/python:$PYTHONPATH

for var in BASE STAT SOLVER PROCESSING FREE DAMASK_BIN BRANCH; do
  unset "${var}"
done
for var in DAMASK MSC; do
  unset "${var}_ROOT"
done
for var in ABAQUS MARC; do
  unset "${var}_VERSION"
done
