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
!> @brief material subroutine incorporating kinematics resulting from thermal expansion
!> @details to be done
!--------------------------------------------------------------------------------------------------
module kinematics_thermal_expansion
 use prec, only: &
   pReal, &
   pInt

 implicit none
 private
 integer(pInt),                       dimension(:),           allocatable,         public, protected :: &
   kinematics_thermal_expansion_sizePostResults, &                                                                !< cumulative size of post results
   kinematics_thermal_expansion_offset, &                                                                         !< which kinematics is my current damage mechanism?
   kinematics_thermal_expansion_instance                                                                          !< instance of damage kinematics mechanism

 integer(pInt),                       dimension(:,:),         allocatable, target, public  :: &
   kinematics_thermal_expansion_sizePostResult                                                                    !< size of each post result output

 character(len=64),                   dimension(:,:),         allocatable, target, public  :: &
   kinematics_thermal_expansion_output                                                                            !< name of each post result output
   
 integer(pInt),                       dimension(:),           allocatable, target, public  :: &
   kinematics_thermal_expansion_Noutput                                                                           !< number of outputs per instance of this damage 

! enum, bind(c)                                                                                                   ! ToDo kinematics need state machinery to deal with sizePostResult
!   enumerator :: undefined_ID, &                                                                                 ! possible remedy is to decouple having state vars from having output
!                 thermalexpansionrate_ID                                                                         ! which means to separate user-defined types tState + tOutput...
! end enum
 public :: &
   kinematics_thermal_expansion_init, &
   kinematics_thermal_expansion_initialStrain, &
   kinematics_thermal_expansion_LiAndItsTangent

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine kinematics_thermal_expansion_init(fileUnit)
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif
 use debug, only: &
   debug_level,&
   debug_constitutive,&
   debug_levelBasic
 use IO, only: &
   IO_read, &
   IO_lc, &
   IO_getTag, &
   IO_isBlank, &
   IO_stringPos, &
   IO_stringValue, &
   IO_floatValue, &
   IO_intValue, &
   IO_warning, &
   IO_error, &
   IO_timeStamp, &
   IO_EOF
 use material, only: &
   phase_kinematics, &
   phase_Nkinematics, &
   phase_Noutput, &
   KINEMATICS_thermal_expansion_label, &
   KINEMATICS_thermal_expansion_ID, &
   material_Nphase, &
   MATERIAL_partPhase

 implicit none
 integer(pInt), intent(in) :: fileUnit

 integer(pInt), allocatable, dimension(:) :: chunkPos
 integer(pInt) :: maxNinstance,phase,instance,kinematics
 character(len=65536) :: &
   tag     = '', &
   line    = ''

 write(6,'(/,a)')   ' <<<+-  kinematics_'//KINEMATICS_thermal_expansion_LABEL//' init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

 maxNinstance = int(count(phase_kinematics == KINEMATICS_thermal_expansion_ID),pInt)
 if (maxNinstance == 0_pInt) return
 
 if (iand(debug_level(debug_constitutive),debug_levelBasic) /= 0_pInt) &
   write(6,'(a16,1x,i5,/)') '# instances:',maxNinstance
 
 allocate(kinematics_thermal_expansion_offset(material_Nphase), source=0_pInt)
 allocate(kinematics_thermal_expansion_instance(material_Nphase), source=0_pInt)
 do phase = 1, material_Nphase
   kinematics_thermal_expansion_instance(phase) = count(phase_kinematics(:,1:phase) == kinematics_thermal_expansion_ID)
   do kinematics = 1, phase_Nkinematics(phase)
     if (phase_kinematics(kinematics,phase) == kinematics_thermal_expansion_ID) &
       kinematics_thermal_expansion_offset(phase) = kinematics
   enddo    
 enddo
   
 allocate(kinematics_thermal_expansion_sizePostResults(maxNinstance),                     source=0_pInt)
 allocate(kinematics_thermal_expansion_sizePostResult(maxval(phase_Noutput),maxNinstance),source=0_pInt)
 allocate(kinematics_thermal_expansion_output(maxval(phase_Noutput),maxNinstance))
          kinematics_thermal_expansion_output = ''
 allocate(kinematics_thermal_expansion_Noutput(maxNinstance),                             source=0_pInt) 

 rewind(fileUnit)
 phase = 0_pInt
 do while (trim(line) /= IO_EOF .and. IO_lc(IO_getTag(line,'<','>')) /= MATERIAL_partPhase)         ! wind forward to <phase>
   line = IO_read(fileUnit)
 enddo
 
 parsingFile: do while (trim(line) /= IO_EOF)                                                       ! read through sections of phase part
   line = IO_read(fileUnit)
   if (IO_isBlank(line)) cycle                                                                      ! skip empty lines
   if (IO_getTag(line,'<','>') /= '') then                                                          ! stop at next part
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit                                                                                           
   endif   
   if (IO_getTag(line,'[',']') /= '') then                                                          ! next phase section
     phase = phase + 1_pInt                                                                         ! advance phase section counter
     cycle                                                                                          ! skip to next line
   endif
   if (phase > 0_pInt ) then; if (any(phase_kinematics(:,phase) == KINEMATICS_thermal_expansion_ID)) then         ! do not short-circuit here (.and. with next if statemen). It's not safe in Fortran
     instance = kinematics_thermal_expansion_instance(phase)                                                         ! which instance of my damage is present phase
     chunkPos = IO_stringPos(line)
     tag = IO_lc(IO_stringValue(line,chunkPos,1_pInt))                                             ! extract key...
     select case(tag)
!       case ('(output)')
!         output = IO_lc(IO_stringValue(line,chunkPos,2_pInt))                                       ! ...and corresponding output
!         select case(output)
!           case ('thermalexpansionrate')
!             kinematics_thermal_expansion_Noutput(instance) = kinematics_thermal_expansion_Noutput(instance) + 1_pInt
!             kinematics_thermal_expansion_outputID(kinematics_thermal_expansion_Noutput(instance),instance) = &
!               thermalexpansionrate_ID
!             kinematics_thermal_expansion_output(kinematics_thermal_expansion_Noutput(instance),instance) = output
! ToDo add sizePostResult loop afterwards...

     end select
   endif; endif
 enddo parsingFile

end subroutine kinematics_thermal_expansion_init

!--------------------------------------------------------------------------------------------------
!> @brief  report initial thermal strain based on current temperature deviation from reference
!--------------------------------------------------------------------------------------------------
pure function kinematics_thermal_expansion_initialStrain(ipc, ip, el)
 use material, only: &
   material_phase, &
   material_homog, &
   temperature, &
   thermalMapping
 use lattice, only: &
   lattice_thermalExpansion33, &
   lattice_referenceTemperature
 
 implicit none
 integer(pInt), intent(in) :: &
   ipc, &                                                                                           !< grain number
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal), dimension(3,3) :: &
   kinematics_thermal_expansion_initialStrain                                                       !< initial thermal strain (should be small strain, though)
 integer(pInt) :: &
   phase, &
   homog, offset
   
 phase = material_phase(ipc,ip,el)
 homog = material_homog(ip,el)
 offset = thermalMapping(homog)%p(ip,el)
 
 kinematics_thermal_expansion_initialStrain = &
   (temperature(homog)%p(offset) - lattice_referenceTemperature(phase))**1 / 1. * &
   lattice_thermalExpansion33(1:3,1:3,1,phase) + &                                                  ! constant  coefficient
   (temperature(homog)%p(offset) - lattice_referenceTemperature(phase))**2 / 2. * &
   lattice_thermalExpansion33(1:3,1:3,2,phase) + &                                                  ! linear    coefficient
   (temperature(homog)%p(offset) - lattice_referenceTemperature(phase))**3 / 3. * &
   lattice_thermalExpansion33(1:3,1:3,3,phase)                                                      ! quadratic coefficient
  
end function kinematics_thermal_expansion_initialStrain

!--------------------------------------------------------------------------------------------------
!> @brief  contains the constitutive equation for calculating the velocity gradient  
!--------------------------------------------------------------------------------------------------
subroutine kinematics_thermal_expansion_LiAndItsTangent(Li, dLi_dTstar3333, ipc, ip, el)
 use material, only: &
   material_phase, &
   material_homog, &
   temperature, &
   temperatureRate, &
   thermalMapping
 use lattice, only: &
   lattice_thermalExpansion33, &
   lattice_referenceTemperature
 
 implicit none
 integer(pInt), intent(in) :: &
   ipc, &                                                                                           !< grain number
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal),   intent(out), dimension(3,3) :: &
   Li                                                                                               !< thermal velocity gradient
 real(pReal),   intent(out), dimension(3,3,3,3) :: &
   dLi_dTstar3333                                                                                   !< derivative of Li with respect to Tstar (4th-order tensor defined to be zero)
 integer(pInt) :: &
   phase, &
   homog, offset
 real(pReal) :: &
   T, TRef, TDot  
   
 phase = material_phase(ipc,ip,el)
 homog = material_homog(ip,el)
 offset = thermalMapping(homog)%p(ip,el)
 T = temperature(homog)%p(offset)
 TDot = temperatureRate(homog)%p(offset)
 TRef = lattice_referenceTemperature(phase)
 
 Li = TDot * ( &
               lattice_thermalExpansion33(1:3,1:3,1,phase)*(T - TRef)**0 &                           ! constant  coefficient
             + lattice_thermalExpansion33(1:3,1:3,2,phase)*(T - TRef)**1 &                           ! linear    coefficient
             + lattice_thermalExpansion33(1:3,1:3,3,phase)*(T - TRef)**2 &                           ! quadratic coefficient
             ) / &
      (1.0_pReal \
            + lattice_thermalExpansion33(1:3,1:3,1,phase)*(T - TRef)**1 / 1. &
            + lattice_thermalExpansion33(1:3,1:3,2,phase)*(T - TRef)**2 / 2. &
            + lattice_thermalExpansion33(1:3,1:3,3,phase)*(T - TRef)**3 / 3. &
      )
 dLi_dTstar3333 = 0.0_pReal 
  
end subroutine kinematics_thermal_expansion_LiAndItsTangent

end module kinematics_thermal_expansion
