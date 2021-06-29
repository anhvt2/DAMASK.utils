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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Dummy plasticity for purely elastic material
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_plastic) plastic_none

contains

!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function plastic_none_init()  result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: &
    Ninstance, &
    p, &
    NipcMyPhase
  class(tNode), pointer :: &
    phases, &
    phase, &
    pl

  write(6,'(/,a)') ' <<<+-  plastic_none init  -+>>>'

  phases => material_root%get('phase')
  allocate(myPlasticity(phases%length), source = .false. )
  do p = 1, phases%length
    phase => phases%get(p)
    pl => phase%get('plasticity')
    if(pl%get_asString('type') == 'none') myPlasticity(p) = .true.
  enddo

  Ninstance = count(myPlasticity)  
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
  if(Ninstance == 0) return
  
  do p = 1, phases%length
    phase => phases%get(p)
    if(.not. myPlasticity(p)) cycle
    NipcMyPhase = count(material_phaseAt == p) * discretization_nIP
    call constitutive_allocateState(plasticState(p),NipcMyPhase,0,0,0)
  enddo

end function plastic_none_init


end submodule plastic_none
