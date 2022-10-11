#!/usr/bin/env bash
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

#==================================================================================================
# Execute this script (type './DAMASK_prerequisites.sh') 
# and send system_report.txt to damask@mpie.de for support
#==================================================================================================

OUTFILE="system_report.txt"
echo ===========================================
echo +  Generating $OUTFILE                    
echo +  Send to damask@mpie.de for support
echo +  view with \'cat $OUTFILE\'
echo ===========================================

function firstLevel {
echo -e '\n\n=============================================================================================='
echo $1
echo ==============================================================================================
}

function secondLevel {
echo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
echo $1
echo ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}

function thirdLevel {
echo -e '\n----------------------------------------------------------------------------------------------'
echo $1
echo ----------------------------------------------------------------------------------------------
}

function getDetails {
if which $1 &> /dev/null; then
  secondLevel $1:
  echo + location:
  which $1
  echo + $1 $2:
  $1 $2
else
  echo $1 not found
fi
echo
}


# redirect STDOUT and STDERR to logfile
# https://stackoverflow.com/questions/11229385/redirect-all-output-in-a-bash-script-when-using-set-x^
exec > $OUTFILE 2>&1
 
# directory, file is not a symlink by definition
# https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
DAMASK_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
echo System report for \'$(hostname)\' created on $(date '+%Y-%m-%d %H:%M:%S') by \'$(whoami)\'
echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

firstLevel "DAMASK"
secondLevel "DAMASK_ROOT"
echo $DAMASK_ROOT
echo
secondLevel "Version"
cat  VERSION

firstLevel "System"
uname -a
echo
echo PATH: $PATH
echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH
echo PYTHONPATH: $PYTHONPATH
echo SHELL: $SHELL
echo PETSC_ARCH: $PETSC_ARCH
echo PETSC_DIR: $PETSC_DIR
echo
echo $PETSC_DIR/$PETSC_ARCH/lib:
ls $PETSC_DIR/$PETSC_ARCH/lib
echo
echo $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables:
cat $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables


firstLevel "Python"
DEFAULT_PYTHON=python3
for EXECUTABLE in python python3; do
  getDetails $EXECUTABLE '--version'
done
secondLevel "Details on $DEFAULT_PYTHON:"
echo $(ls -la $(which $DEFAULT_PYTHON))
for MODULE in numpy scipy pandas matplotlib yaml h5py;do
  thirdLevel $module
  $DEFAULT_PYTHON -c "import $MODULE; \
                      print('Version: {}'.format($MODULE.__version__)); \
                      print('Location: {}'.format($MODULE.__file__))"
done
thirdLevel vtk
$DEFAULT_PYTHON -c "import vtk; \
                    print('Version: {}'.format(vtk.vtkVersion.GetVTKVersion())); \
                    print('Location: {}'.format(vtk.__file__))"

firstLevel "GNU Compiler Collection"
for EXECUTABLE in gcc g++ gfortran ;do
  getDetails $EXECUTABLE '--version'
done

firstLevel "Intel Compiler Suite (classic)"
for EXECUTABLE in icc icpc ifort ;do
  getDetails $EXECUTABLE '--version'
done

firstLevel "Intel Compiler Suite (LLVM)"
for EXECUTABLE in icx icpx ifx ;do
  getDetails $EXECUTABLE '--version'
done

firstLevel "MPI Wrappers"
for EXECUTABLE in mpicc mpiCC mpiicc   mpic++ mpiicpc mpicxx    mpifort mpiifort mpif90 mpif77; do
  getDetails $EXECUTABLE '-show'
done

firstLevel "MPI Launchers"
for EXECUTABLE in mpirun mpiexec; do
  getDetails $EXECUTABLE '--version'
done

firstLevel "CMake"
getDetails cmake --version

