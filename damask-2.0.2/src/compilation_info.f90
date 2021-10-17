! Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program. If not, see <http://www.gnu.org/licenses/>.
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
  write(6,*) 'Compiled with ', compiler_version()
  write(6,*) 'With options  ', compiler_options()
#else
  write(6,'(a,i4.4,a,i8.8)') ' Compiled with Intel fortran version ', __INTEL_COMPILER,&
                                                     ', build date ', __INTEL_COMPILER_BUILD_DATE
#endif
write(6,*) 'Compiled on ', __DATE__,' at ',__TIME__
write(6,*)
flush(6)
