! Copyright 2011-2022 Max-Planck-Institut für Eisenforschung GmbH
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
!> @brief Isotemperature homogenization
!--------------------------------------------------------------------------------------------------
submodule(homogenization:thermal) isotemperature

contains

module subroutine isotemperature_init()

  print'(/,1x,a)', '<<<+-  homogenization:thermal:isotemperature init  -+>>>'

end subroutine isotemperature_init

end submodule isotemperature
