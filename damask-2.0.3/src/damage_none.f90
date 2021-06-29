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
!> @brief material subroutine for constant damage field
!--------------------------------------------------------------------------------------------------
module damage_none

  implicit none
  private
  
  public :: &
    damage_none_init

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
subroutine damage_none_init()
  use config, only: &
    config_homogenization
  use material, only: &
    damage_initialPhi, &
    damage, &
    damage_type, &
    material_homogenizationAt, &
    damageState, &
    DAMAGE_NONE_LABEL, &
    DAMAGE_NONE_ID
 
  implicit none
  integer :: &
    homog, &
    NofMyHomog

  write(6,'(/,a)')   ' <<<+-  damage_'//DAMAGE_NONE_LABEL//' init  -+>>>'

   initializeInstances: do homog = 1, size(config_homogenization)
    
    myhomog: if (damage_type(homog) == DAMAGE_NONE_ID) then
      NofMyHomog = count(material_homogenizationAt == homog)
      damageState(homog)%sizeState = 0
      damageState(homog)%sizePostResults = 0
      allocate(damageState(homog)%state0   (0,NofMyHomog))
      allocate(damageState(homog)%subState0(0,NofMyHomog))
      allocate(damageState(homog)%state    (0,NofMyHomog))
      
      deallocate(damage(homog)%p)
      allocate  (damage(homog)%p(1), source=damage_initialPhi(homog))
      
    endif myhomog
  enddo initializeInstances

end subroutine damage_none_init

end module damage_none
