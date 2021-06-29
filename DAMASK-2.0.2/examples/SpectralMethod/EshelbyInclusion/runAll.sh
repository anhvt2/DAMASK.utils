#!/usr/bin/env bash
# Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
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

for geom in $(ls geom/*.geom)
do
  base=${geom%.geom}
  base=${base#geom/}
  name=${base}_thermal
  vtr=${base}.vtr

  [[ -f ${name}.spectralOut ]] || \
  DAMASK_spectral \
    --workingdir ./ \
    --load thermal.load \
    --geom $geom \
    > ${name}.out
  
  if [ ! -f postProc/${name}_inc10.txt ]
  then
    postResults ${name}.spectralOut \
      --ho temperature \
      --cr f,fe,fi,fp,p \
      --split \
      --separation x,y,z \

    addCauchy postProc/${name}_inc*.txt \

    addDeviator postProc/${name}_inc*.txt \
      --spherical \
      --tensor p,Cauchy \

    addDisplacement postProc/${name}_inc*.txt \
      --nodal \

  fi

  geom_check ${geom}
  
  for inc in {00..10}
  do
    echo "generating postProc/${name}_inc${inc}.vtr"
     cp geom/${vtr} postProc/${name}_inc${inc}.vtr
     vtk_addRectilinearGridData \
       postProc/${name}_inc${inc}.txt \
       --inplace \
       --vtk postProc/${name}_inc${inc}.vtr \
       --data 'sph(p)','sph(Cauchy)',temperature \
       --tensor f,fe,fi,fp,p,Cauchy \
      
    vtk_addRectilinearGridData \
      postProc/${name}_inc${inc}_nodal.txt \
      --inplace \
      --vtk postProc/${name}_inc${inc}.vtr \
      --data 'avg(f).pos','fluct(f).pos' \

  done
done
