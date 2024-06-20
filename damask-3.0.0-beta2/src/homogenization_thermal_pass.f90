! Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as
! published by the Free Software Foundation, either version 3 of the
! License, or (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Affero General Public License for more details.
! 
! You should have received a copy of the GNU Affero General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, KU Leuven
!> @brief Dummy homogenization scheme for 1 constituent per material point
!--------------------------------------------------------------------------------------------------
submodule(homogenization:thermal) thermal_pass

contains

module subroutine pass_init()

  integer :: &
    ho

  print'(/,1x,a)', '<<<+-  homogenization:thermal:pass init  -+>>>'

  do ho = 1, size(thermal_type)

    if (thermal_type(ho) /= THERMAL_PASS_ID) cycle

    if (homogenization_Nconstituents(ho) /= 1) &
      call IO_error(211,ext_msg='(pass) with N_constituents !=1')

  end do

end subroutine pass_init

end submodule thermal_pass
