! Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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

 implicit none
 private
 
 public :: &
   thermal_isothermal_init

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
subroutine thermal_isothermal_init()
 use prec, only: &
   pReal
 use config, only: &
   material_Nhomogenization
 use material
 
 implicit none
 integer :: &
   homog, &
   NofMyHomog

 write(6,'(/,a)')   ' <<<+-  thermal_'//THERMAL_isothermal_label//' init  -+>>>'

 initializeInstances: do homog = 1, material_Nhomogenization
   
   if (thermal_type(homog) /= THERMAL_isothermal_ID) cycle
   NofMyHomog = count(material_homogenizationAt == homog)
   thermalState(homog)%sizeState = 0
   thermalState(homog)%sizePostResults = 0
   allocate(thermalState(homog)%state0   (0,NofMyHomog), source=0.0_pReal)
   allocate(thermalState(homog)%subState0(0,NofMyHomog), source=0.0_pReal)
   allocate(thermalState(homog)%state    (0,NofMyHomog), source=0.0_pReal)
     
   deallocate(temperature    (homog)%p)
   allocate  (temperature    (homog)%p(1), source=thermal_initialT(homog))
   deallocate(temperatureRate(homog)%p)
   allocate  (temperatureRate(homog)%p(1), source=0.0_pReal)

 enddo initializeInstances


end subroutine thermal_isothermal_init

end module thermal_isothermal
