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
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine for isothermal temperature field
!--------------------------------------------------------------------------------------------------
module thermal_isothermal
  use config
  use material

  implicit none
  public

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
subroutine thermal_isothermal_init

  integer :: h,Nmaterialpoints

  print'(/,a)',   ' <<<+-  thermal_isothermal init  -+>>>'; flush(6)

  do h = 1, size(material_name_homogenization)
    if (thermal_type(h) /= THERMAL_isothermal_ID) cycle

    Nmaterialpoints = count(material_homogenizationAt == h)
    thermalState(h)%sizeState = 0
    allocate(thermalState(h)%state0   (0,Nmaterialpoints))
    allocate(thermalState(h)%subState0(0,Nmaterialpoints))
    allocate(thermalState(h)%state    (0,Nmaterialpoints))

    deallocate(temperature    (h)%p)
    allocate  (temperature    (h)%p(1), source=thermal_initialT(h))
    deallocate(temperatureRate(h)%p)
    allocate  (temperatureRate(h)%p(1))

  enddo

end subroutine thermal_isothermal_init

end module thermal_isothermal
