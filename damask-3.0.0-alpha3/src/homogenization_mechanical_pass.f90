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
!> @brief dummy homogenization homogenization scheme for 1 constituent per material point
!--------------------------------------------------------------------------------------------------
submodule(homogenization:mechanical) mechanical_pass

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all necessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
module subroutine pass_init

  integer :: &
    ho, &
    Nmaterialpoints

  print'(/,a)', ' <<<+-  homogenization:mechanical:pass init  -+>>>'

  print'(a,i2)', ' # instances: ',count(homogenization_type == HOMOGENIZATION_NONE_ID)
  flush(IO_STDOUT)

  do ho = 1, size(homogenization_type)
    if(homogenization_type(ho) /= HOMOGENIZATION_NONE_ID) cycle

    if(homogenization_Nconstituents(ho) /= 1) &
      call IO_error(211,ext_msg='N_constituents (pass)')

    Nmaterialpoints = count(material_homogenizationAt == ho)
    homogState(ho)%sizeState = 0
    allocate(homogState(ho)%state0(0,Nmaterialpoints))
    allocate(homogState(ho)%state (0,Nmaterialpoints))

  enddo

end subroutine pass_init

end submodule mechanical_pass
