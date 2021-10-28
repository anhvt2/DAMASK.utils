! Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Dummy plasticity for purely elastic material
!--------------------------------------------------------------------------------------------------
submodule(phase:plastic) none

contains

!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function plastic_none_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: &
    ph
  class(tNode), pointer :: &
    phases


  myPlasticity = plastic_active('none')
  if(count(myPlasticity) == 0) return

  print'(/,a)', ' <<<+-  phase:mechanical:plastic:none init  -+>>>'
  print'(a,i0)', ' # phases: ',count(myPlasticity); flush(IO_STDOUT)

  phases => config_material%get('phase')
  do ph = 1, phases%length
    if(.not. myPlasticity(ph)) cycle
    call phase_allocateState(plasticState(ph),count(material_phaseID == ph),0,0,0)
  enddo

end function plastic_none_init


end submodule none
