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
!> @brief material subroutine for adiabatic temperature evolution
!> @details to be done
!--------------------------------------------------------------------------------------------------
module thermal_adiabatic
 use prec, only: &
   pReal, &
   pInt

 implicit none
 private

 integer(pInt),                       dimension(:,:),         allocatable, target, public :: &
   thermal_adiabatic_sizePostResult                                                            !< size of each post result output
 character(len=64),                   dimension(:,:),         allocatable, target, public :: &
   thermal_adiabatic_output                                                                    !< name of each post result output
   
 integer(pInt),                       dimension(:),           allocatable, target, public :: &
   thermal_adiabatic_Noutput                                                                   !< number of outputs per instance of this thermal model 

 enum, bind(c) 
   enumerator :: undefined_ID, &
                 temperature_ID
 end enum
 integer(kind(undefined_ID)),         dimension(:,:),         allocatable,          private :: & 
   thermal_adiabatic_outputID                                                                  !< ID of each post result output


 public :: &
   thermal_adiabatic_init, &
   thermal_adiabatic_updateState, &
   thermal_adiabatic_getSourceAndItsTangent, &
   thermal_adiabatic_getSpecificHeat, &
   thermal_adiabatic_getMassDensity, &
   thermal_adiabatic_postResults

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine thermal_adiabatic_init
 use material, only: &
   thermal_type, &
   thermal_typeInstance, &
   homogenization_Noutput, &
   THERMAL_ADIABATIC_label, &
   THERMAL_adiabatic_ID, &
   material_homogenizationAt, & 
   mappingHomogenization, & 
   thermalState, &
   thermalMapping, &
   thermal_initialT, &
   temperature, &
   temperatureRate
 use config, only: &
   config_homogenization

 implicit none
 integer(pInt) :: maxNinstance,section,instance,i
 integer(pInt) :: sizeState
 integer(pInt) :: NofMyHomog   
 character(len=65536),   dimension(0), parameter :: emptyStringArray = [character(len=65536)::]
 character(len=65536), dimension(:), allocatable :: outputs

 write(6,'(/,a)')   ' <<<+-  thermal_'//THERMAL_ADIABATIC_label//' init  -+>>>'
 
 maxNinstance = int(count(thermal_type == THERMAL_adiabatic_ID),pInt)
 if (maxNinstance == 0_pInt) return
 
 allocate(thermal_adiabatic_sizePostResult (maxval(homogenization_Noutput),maxNinstance),source=0_pInt)
 allocate(thermal_adiabatic_output         (maxval(homogenization_Noutput),maxNinstance))
          thermal_adiabatic_output = ''
 allocate(thermal_adiabatic_outputID       (maxval(homogenization_Noutput),maxNinstance),source=undefined_ID)
 allocate(thermal_adiabatic_Noutput        (maxNinstance),                               source=0_pInt) 

 
 initializeInstances: do section = 1_pInt, size(thermal_type)
   if (thermal_type(section) /= THERMAL_adiabatic_ID) cycle
   NofMyHomog=count(material_homogenizationAt==section)
   instance = thermal_typeInstance(section)
   outputs = config_homogenization(section)%getStrings('(output)',defaultVal=emptyStringArray)
   do i=1_pInt, size(outputs)
     select case(outputs(i))
       case('temperature')
             thermal_adiabatic_Noutput(instance) = thermal_adiabatic_Noutput(instance) + 1_pInt
             thermal_adiabatic_outputID(thermal_adiabatic_Noutput(instance),instance) = temperature_ID
             thermal_adiabatic_output(thermal_adiabatic_Noutput(instance),instance) = outputs(i)
             thermal_adiabatic_sizePostResult(thermal_adiabatic_Noutput(instance),instance) = 1_pInt
     end select
   enddo

! allocate state arrays
   sizeState = 1_pInt
   thermalState(section)%sizeState = sizeState
   thermalState(section)%sizePostResults = sum(thermal_adiabatic_sizePostResult(:,instance))
   allocate(thermalState(section)%state0   (sizeState,NofMyHomog), source=thermal_initialT(section))
   allocate(thermalState(section)%subState0(sizeState,NofMyHomog), source=thermal_initialT(section))
   allocate(thermalState(section)%state    (sizeState,NofMyHomog), source=thermal_initialT(section))

   nullify(thermalMapping(section)%p)
   thermalMapping(section)%p => mappingHomogenization(1,:,:)
   deallocate(temperature(section)%p)
   temperature(section)%p => thermalState(section)%state(1,:)
   deallocate(temperatureRate(section)%p)
   allocate  (temperatureRate(section)%p(NofMyHomog), source=0.0_pReal)
     
 enddo initializeInstances
 
end subroutine thermal_adiabatic_init

!--------------------------------------------------------------------------------------------------
!> @brief  calculates adiabatic change in temperature based on local heat generation model  
!--------------------------------------------------------------------------------------------------
function thermal_adiabatic_updateState(subdt, ip, el)
 use numerics, only: &
   err_thermal_tolAbs, &
   err_thermal_tolRel
 use material, only: &
   material_homogenizationAt, &
   mappingHomogenization, &
   thermalState, &
   temperature, &
   temperatureRate, &
   thermalMapping
 
 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal),   intent(in) :: &
   subdt
 logical,                    dimension(2)                             :: &
   thermal_adiabatic_updateState
 integer(pInt) :: &
   homog, &
   offset
 real(pReal) :: &
   T, Tdot, dTdot_dT  

 homog  = material_homogenizationAt(el)
 offset = mappingHomogenization(1,ip,el)
 
 T = thermalState(homog)%subState0(1,offset)
 call thermal_adiabatic_getSourceAndItsTangent(Tdot, dTdot_dT, T, ip, el)
 T = T + subdt*Tdot/(thermal_adiabatic_getSpecificHeat(ip,el)*thermal_adiabatic_getMassDensity(ip,el))
 
 thermal_adiabatic_updateState = [     abs(T - thermalState(homog)%state(1,offset)) &
                                    <= err_thermal_tolAbs &
                                  .or. abs(T - thermalState(homog)%state(1,offset)) &
                                    <= err_thermal_tolRel*abs(thermalState(homog)%state(1,offset)), &
                                  .true.]

 temperature    (homog)%p(thermalMapping(homog)%p(ip,el)) = T  
 temperatureRate(homog)%p(thermalMapping(homog)%p(ip,el)) = &
   (thermalState(homog)%state(1,offset) - thermalState(homog)%subState0(1,offset))/(subdt+tiny(0.0_pReal))
 
end function thermal_adiabatic_updateState

!--------------------------------------------------------------------------------------------------
!> @brief returns heat generation rate
!--------------------------------------------------------------------------------------------------
subroutine thermal_adiabatic_getSourceAndItsTangent(Tdot, dTdot_dT, T, ip, el)
 use material, only: &
   homogenization_Ngrains, &
   material_homogenizationAt, &
   mappingHomogenization, &
   phaseAt, &
   phasememberAt, &
   thermal_typeInstance, &
   phase_Nsources, &
   phase_source, &
   SOURCE_thermal_dissipation_ID, &
   SOURCE_thermal_externalheat_ID
 use source_thermal_dissipation, only: &
   source_thermal_dissipation_getRateAndItsTangent
 use source_thermal_externalheat, only: &
   source_thermal_externalheat_getRateAndItsTangent
 use crystallite, only: &
   crystallite_S, &
   crystallite_Lp  

 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal), intent(in) :: &
   T
 real(pReal), intent(out) :: &
   Tdot, dTdot_dT
 real(pReal) :: &
   my_Tdot, my_dTdot_dT
 integer(pInt) :: &
   phase, &
   homog, &
   instance, &
   grain, &
   source, &
   constituent
   
 homog  = material_homogenizationAt(el)
 instance = thermal_typeInstance(homog)
  
 Tdot = 0.0_pReal
 dTdot_dT = 0.0_pReal
 do grain = 1, homogenization_Ngrains(homog)
   phase = phaseAt(grain,ip,el)
   constituent = phasememberAt(grain,ip,el)
   do source = 1, phase_Nsources(phase)
     select case(phase_source(source,phase))                                                   
       case (SOURCE_thermal_dissipation_ID)
        call source_thermal_dissipation_getRateAndItsTangent(my_Tdot, my_dTdot_dT, &
                                                             crystallite_S(1:3,1:3,grain,ip,el), &
                                                             crystallite_Lp(1:3,1:3,grain,ip,el), &
                                                             phase)

       case (SOURCE_thermal_externalheat_ID)
        call source_thermal_externalheat_getRateAndItsTangent(my_Tdot, my_dTdot_dT, &
                                                              phase, constituent)

       case default
        my_Tdot = 0.0_pReal
        my_dTdot_dT = 0.0_pReal
     end select
     Tdot = Tdot + my_Tdot
     dTdot_dT = dTdot_dT + my_dTdot_dT
   enddo  
 enddo
 
 Tdot = Tdot/real(homogenization_Ngrains(homog),pReal)
 dTdot_dT = dTdot_dT/real(homogenization_Ngrains(homog),pReal)
 
end subroutine thermal_adiabatic_getSourceAndItsTangent
 
!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized specific heat capacity
!--------------------------------------------------------------------------------------------------
function thermal_adiabatic_getSpecificHeat(ip,el)
 use lattice, only: &
   lattice_specificHeat
 use material, only: &
   homogenization_Ngrains, &
   material_phase
 use mesh, only: &
   mesh_element

 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal) :: &
   thermal_adiabatic_getSpecificHeat
 integer(pInt) :: &
   grain
  
 thermal_adiabatic_getSpecificHeat = 0.0_pReal
 
  
 do grain = 1, homogenization_Ngrains(mesh_element(3,el))
   thermal_adiabatic_getSpecificHeat = thermal_adiabatic_getSpecificHeat + &
    lattice_specificHeat(material_phase(grain,ip,el))
 enddo

 thermal_adiabatic_getSpecificHeat = &
   thermal_adiabatic_getSpecificHeat/real(homogenization_Ngrains(mesh_element(3,el)),pReal)
 
end function thermal_adiabatic_getSpecificHeat
 
 
!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized mass density
!--------------------------------------------------------------------------------------------------
function thermal_adiabatic_getMassDensity(ip,el)
 use lattice, only: &
   lattice_massDensity
 use material, only: &
   homogenization_Ngrains, &
   material_phase
 use mesh, only: &
   mesh_element
   
 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal) :: &
   thermal_adiabatic_getMassDensity
 integer(pInt) :: &
   grain
  
 thermal_adiabatic_getMassDensity = 0.0_pReal

  
 do grain = 1, homogenization_Ngrains(mesh_element(3,el))
   thermal_adiabatic_getMassDensity = thermal_adiabatic_getMassDensity + &
    lattice_massDensity(material_phase(grain,ip,el))
 enddo

 thermal_adiabatic_getMassDensity = &
   thermal_adiabatic_getMassDensity/real(homogenization_Ngrains(mesh_element(3,el)),pReal)
 
end function thermal_adiabatic_getMassDensity


!--------------------------------------------------------------------------------------------------
!> @brief return array of thermal results
!--------------------------------------------------------------------------------------------------
function thermal_adiabatic_postResults(homog,instance,of) result(postResults)
 use material, only: &
   temperature

 implicit none
 integer(pInt), intent(in) :: &
   homog, &
   instance, &
   of

 real(pReal), dimension(sum(thermal_adiabatic_sizePostResult(:,instance))) :: &
   postResults

 integer(pInt) :: &
   o, c

 c = 0_pInt

 do o = 1_pInt,thermal_adiabatic_Noutput(instance)
    select case(thermal_adiabatic_outputID(o,instance))
 
      case (temperature_ID)
        postResults(c+1_pInt) = temperature(homog)%p(of)
        c = c + 1
    end select
 enddo
 
end function thermal_adiabatic_postResults

end module thermal_adiabatic
