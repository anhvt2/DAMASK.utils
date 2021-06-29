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
!> @brief dummy homogenization homogenization scheme for 1 constituent per material point
!--------------------------------------------------------------------------------------------------
submodule(homogenization) homogenization_mech_none

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all necessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
module subroutine mech_none_init

  integer :: &
    Ninstances, &
    h, &
    Nmaterialpoints

  print'(/,a)', ' <<<+-  homogenization_mech_none init  -+>>>'

  Ninstances = count(homogenization_type == HOMOGENIZATION_NONE_ID)
  print'(a,i2)', ' # instances: ',Ninstances; flush(IO_STDOUT)

  do h = 1, size(homogenization_type)
    if(homogenization_type(h) /= HOMOGENIZATION_NONE_ID) cycle

    if(homogenization_Nconstituents(h) /= 1) &
      call IO_error(211,ext_msg='N_constituents (mech_none)')
    
    Nmaterialpoints = count(material_homogenizationAt == h)
    homogState(h)%sizeState = 0
    allocate(homogState(h)%state0   (0,Nmaterialpoints))
    allocate(homogState(h)%subState0(0,Nmaterialpoints))
    allocate(homogState(h)%state    (0,Nmaterialpoints))

  enddo

end subroutine mech_none_init

end submodule homogenization_mech_none
