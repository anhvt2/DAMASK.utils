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
!> @brief Constants.
!--------------------------------------------------------------------------------------------------
module constants
  use prec

  implicit none(type,external)
  public

  real(pREAL), parameter :: &
    T_ROOM = 293.15_pREAL, &                                                                        !< Room temperature (20°C) in K (https://en.wikipedia.org/wiki/ISO_1)
    K_B = 1.380649e-23_pREAL, &                                                                     !< Boltzmann constant in J/Kelvin (https://doi.org/10.1351/goldbook)
    N_A = 6.02214076e23_pREAL                                                                       !< Avogadro constant in 1/mol (https://doi.org/10.1351/goldbook)

  character, parameter :: &
    CR = achar(13), &
    LF = new_line('DAMASK')

  character(len=*),          parameter :: LOWER = 'abcdefghijklmnopqrstuvwxyz'
  character(len=len(LOWER)), parameter :: UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  enum, bind(c); enumerator :: &
    STATUS_OK, &
    STATUS_ITERATING, &
    STATUS_FAIL_PHASE_MECHANICAL, &
    STATUS_FAIL_PHASE_MECHANICAL_STATE, &
    STATUS_FAIL_PHASE_MECHANICAL_DELTASTATE, &
    STATUS_FAIL_PHASE_MECHANICAL_STRESS, &
    STATUS_FAIL_PHASE_DAMAGE, &
    STATUS_FAIL_PHASE_DAMAGE_STATE, &
    STATUS_FAIL_PHASE_DAMAGE_DELTASTATE, &
    STATUS_FAIL_PHASE_THERMAL, &
    STATUS_FAIL_PHASE_THERMAL_DOTSTATE
  end enum

end module constants
