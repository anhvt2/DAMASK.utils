! Copyright 2011-20 Max-Planck-Institut für Eisenforschung GmbH
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
!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Reads in the material, numerics & debug configuration from their respective file
!> @details Reads the material configuration file, where solverJobName.yaml takes
!! precedence over material.yaml.
!--------------------------------------------------------------------------------------------------
module config
  use prec
  use DAMASK_interface
  use IO
  use YAML_parse
  use YAML_types

#ifdef PETSc
#include <petsc/finclude/petscsys.h>
   use petscsys
#endif

  implicit none
  private

  class(tNode), pointer, public :: &
    config_material, &
    config_numerics, &
    config_debug

  public :: &
    config_init, &
    config_deallocate

contains

!--------------------------------------------------------------------------------------------------
!> @brief Real *.yaml configuration files.
!--------------------------------------------------------------------------------------------------
subroutine config_init

  print'(/,a)', ' <<<+-  config init  -+>>>'; flush(IO_STDOUT)

  call parse_material
  call parse_numerics
  call parse_debug

end subroutine config_init


!--------------------------------------------------------------------------------------------------
!> @brief Read material.yaml or <jobname>.yaml.
!--------------------------------------------------------------------------------------------------
subroutine parse_material

  logical :: fileExists
  character(len=:), allocatable :: fname

  fname = getSolverJobName()//'.yaml'
  inquire(file=fname,exist=fileExists)
  if(.not. fileExists) then
    fname = 'material.yaml'
    inquire(file=fname,exist=fileExists)
    if(.not. fileExists) call IO_error(100,ext_msg=fname)
  endif
  print*, 'reading '//fname; flush(IO_STDOUT)
  config_material => YAML_parse_file(fname)

end subroutine parse_material


!--------------------------------------------------------------------------------------------------
!> @brief Read numerics.yaml.
!--------------------------------------------------------------------------------------------------
subroutine parse_numerics

  logical :: fexist

  config_numerics => emptyDict
  inquire(file='numerics.yaml', exist=fexist)
  if (fexist) then
    print*, 'reading numerics.yaml'; flush(IO_STDOUT)
    config_numerics => YAML_parse_file('numerics.yaml')
  endif

end subroutine parse_numerics


!--------------------------------------------------------------------------------------------------
!> @brief Read debug.yaml.
!--------------------------------------------------------------------------------------------------
subroutine parse_debug

  logical :: fexist

  config_debug => emptyDict
  inquire(file='debug.yaml', exist=fexist)
  fileExists: if (fexist) then
    print*, 'reading debug.yaml'; flush(IO_STDOUT)
    config_debug => YAML_parse_file('debug.yaml')
  endif fileExists

end subroutine parse_debug


!--------------------------------------------------------------------------------------------------
!> @brief Deallocate config_material.
!ToDo: deallocation of numerics debug (optional)
!--------------------------------------------------------------------------------------------------
subroutine config_deallocate

  deallocate(config_material)

end subroutine config_deallocate

end module config
