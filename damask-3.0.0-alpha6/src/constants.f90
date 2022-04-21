! Copyright 2011-2022 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
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
!> @brief  physical constants
!--------------------------------------------------------------------------------------------------
module constants
  use prec

  implicit none
  public

  real(pReal), parameter :: &
    T_ROOM = 293.15_pReal, &                                                                        !< Room temperature in K (20°C)
    K_B = 1.380649e-23_pReal, &                                                                     !< Boltzmann constant in J/Kelvin (https://doi.org/10.1351/goldbook)
    N_A = 6.02214076e23_pReal                                                                       !< Avogadro constant in 1/mol (https://doi.org/10.1351/goldbook)

end module constants
