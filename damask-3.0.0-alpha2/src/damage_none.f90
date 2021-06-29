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
!> @brief material subroutine for constant damage field
!--------------------------------------------------------------------------------------------------
module damage_none
  use config
  use material

  implicit none
  public

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
subroutine damage_none_init

  integer :: h,Nmaterialpoints

  print'(/,a)', ' <<<+-  damage_none init  -+>>>'; flush(6)

  do h = 1, size(material_name_homogenization)
    if (damage_type(h) /= DAMAGE_NONE_ID) cycle

    Nmaterialpoints = count(material_homogenizationAt == h)
    damageState(h)%sizeState = 0
    allocate(damageState(h)%state0   (0,Nmaterialpoints))
    allocate(damageState(h)%subState0(0,Nmaterialpoints))
    allocate(damageState(h)%state    (0,Nmaterialpoints))

    deallocate(damage(h)%p)
    allocate  (damage(h)%p(1), source=damage_initialPhi(h))

  enddo

end subroutine damage_none_init

end module damage_none
