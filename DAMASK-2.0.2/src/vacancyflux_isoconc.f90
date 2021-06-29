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
!> @brief material subroutine for constant vacancy concentration
!--------------------------------------------------------------------------------------------------
module vacancyflux_isoconc

 implicit none
 private
 
 public :: &
   vacancyflux_isoconc_init

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
subroutine vacancyflux_isoconc_init()
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif
 use prec, only: &
   pReal, &
   pInt 
 use IO, only: &
   IO_timeStamp
 use material
 
 implicit none
 integer(pInt) :: &
   homog, &
   NofMyHomog

 write(6,'(/,a)')   ' <<<+-  vacancyflux_'//VACANCYFLUX_isoconc_label//' init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

 initializeInstances: do homog = 1_pInt, material_Nhomogenization
   
   myhomog: if (vacancyflux_type(homog) == VACANCYFLUX_isoconc_ID) then
     NofMyHomog = count(material_homog == homog)
     vacancyfluxState(homog)%sizeState = 0_pInt
     vacancyfluxState(homog)%sizePostResults = 0_pInt
     allocate(vacancyfluxState(homog)%state0   (0_pInt,NofMyHomog))
     allocate(vacancyfluxState(homog)%subState0(0_pInt,NofMyHomog))
     allocate(vacancyfluxState(homog)%state    (0_pInt,NofMyHomog))
     
     deallocate(vacancyConc    (homog)%p)
     allocate  (vacancyConc    (homog)%p(1), source=vacancyflux_initialCv(homog))
     deallocate(vacancyConcRate(homog)%p)
     allocate  (vacancyConcRate(homog)%p(1), source=0.0_pReal)

   endif myhomog
 enddo initializeInstances


end subroutine vacancyflux_isoconc_init

end module vacancyflux_isoconc
