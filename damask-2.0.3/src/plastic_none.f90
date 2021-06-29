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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Dummy plasticity for purely elastic material
!--------------------------------------------------------------------------------------------------
module plastic_none

 implicit none
 private

 public :: &
   plastic_none_init

contains

!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine plastic_none_init
 use debug, only: &
   debug_level, &
   debug_constitutive, &
   debug_levelBasic
 use material, only: &
   phase_plasticity, &
   material_allocatePlasticState, &
   PLASTICITY_NONE_label, &
   PLASTICITY_NONE_ID, &
   material_phase, &
   plasticState

 implicit none
 integer :: &
   Ninstance, &
   p, &
   NipcMyPhase

 write(6,'(/,a)')   ' <<<+-  plastic_'//PLASTICITY_NONE_label//' init  -+>>>'

 Ninstance = count(phase_plasticity == PLASTICITY_NONE_ID)
 if (iand(debug_level(debug_constitutive),debug_levelBasic) /= 0) &
   write(6,'(a16,1x,i5,/)') '# instances:',Ninstance

 do p = 1, size(phase_plasticity)
   if (phase_plasticity(p) /= PLASTICITY_NONE_ID) cycle

!--------------------------------------------------------------------------------------------------
! allocate state arrays
   NipcMyPhase = count(material_phase == p)
   call material_allocatePlasticState(p,NipcMyPhase,0,0,0, &
                                      0,0,0)
   plasticState(p)%sizePostResults = 0

 enddo

end subroutine plastic_none_init

end module plastic_none
