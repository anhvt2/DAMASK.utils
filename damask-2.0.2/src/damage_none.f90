! Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
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
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif
 use prec, only: &
   pInt 
 use IO, only: &
   IO_timeStamp
 use material
 
 implicit none
 integer(pInt) :: &
   homog, &
   NofMyHomog

 write(6,'(/,a)')   ' <<<+-  damage_'//DAMAGE_none_label//' init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

  initializeInstances: do homog = 1_pInt, material_Nhomogenization
   
   myhomog: if (damage_type(homog) == DAMAGE_none_ID) then
     NofMyHomog = count(material_homog == homog)
     damageState(homog)%sizeState = 0_pInt
     damageState(homog)%sizePostResults = 0_pInt
     allocate(damageState(homog)%state0   (0_pInt,NofMyHomog))
     allocate(damageState(homog)%subState0(0_pInt,NofMyHomog))
     allocate(damageState(homog)%state    (0_pInt,NofMyHomog))
     
     deallocate(damage(homog)%p)
     allocate  (damage(homog)%p(1), source=damage_initialPhi(homog))
     
   endif myhomog
 enddo initializeInstances


end subroutine damage_none_init

end module damage_none
