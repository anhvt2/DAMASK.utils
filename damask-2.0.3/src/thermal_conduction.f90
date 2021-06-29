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
!> @brief material subroutine for temperature evolution from heat conduction
!> @details to be done
!--------------------------------------------------------------------------------------------------
module thermal_conduction
 use prec, only: &
   pReal, &
   pInt

 implicit none
 private

 integer(pInt),                       dimension(:,:),         allocatable, target, public :: &
   thermal_conduction_sizePostResult                                                            !< size of each post result output
 character(len=64),                   dimension(:,:),         allocatable, target, public :: &
   thermal_conduction_output                                                                    !< name of each post result output
   
 integer(pInt),                       dimension(:),           allocatable, target, public :: &
   thermal_conduction_Noutput                                                                   !< number of outputs per instance of this damage 

 enum, bind(c) 
   enumerator :: undefined_ID, &
                 temperature_ID
 end enum
 integer(kind(undefined_ID)),         dimension(:,:),         allocatable,          private :: & 
   thermal_conduction_outputID                                                                  !< ID of each post result output


 public :: &
   thermal_conduction_init, &
   thermal_conduction_getSourceAndItsTangent, &
   thermal_conduction_getConductivity33, &
   thermal_conduction_getSpecificHeat, &
   thermal_conduction_getMassDensity, &
   thermal_conduction_putTemperatureAndItsRate, &
   thermal_conduction_postResults

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_init
 use material, only: &
   thermal_type, &
   thermal_typeInstance, &
   homogenization_Noutput, &
   THERMAL_conduction_label, &
   THERMAL_conduction_ID, &
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

 write(6,'(/,a)')   ' <<<+-  thermal_'//THERMAL_CONDUCTION_label//' init  -+>>>'
 
 maxNinstance = count(thermal_type == THERMAL_conduction_ID)
 if (maxNinstance == 0_pInt) return
 
 allocate(thermal_conduction_sizePostResult (maxval(homogenization_Noutput),maxNinstance),source=0_pInt)
 allocate(thermal_conduction_output         (maxval(homogenization_Noutput),maxNinstance))
          thermal_conduction_output = ''
 allocate(thermal_conduction_outputID       (maxval(homogenization_Noutput),maxNinstance),source=undefined_ID)
 allocate(thermal_conduction_Noutput        (maxNinstance),                               source=0_pInt) 

 
 initializeInstances: do section = 1_pInt, size(thermal_type)
   if (thermal_type(section) /= THERMAL_conduction_ID) cycle
   NofMyHomog=count(material_homogenizationAt==section)
   instance = thermal_typeInstance(section)
   outputs = config_homogenization(section)%getStrings('(output)',defaultVal=emptyStringArray)
   do i=1_pInt, size(outputs)
     select case(outputs(i))
       case('temperature')
             thermal_conduction_Noutput(instance) = thermal_conduction_Noutput(instance) + 1_pInt
             thermal_conduction_outputID(thermal_conduction_Noutput(instance),instance) = temperature_ID
             thermal_conduction_output(thermal_conduction_Noutput(instance),instance) = outputs(i)
             thermal_conduction_sizePostResult(thermal_conduction_Noutput(instance),instance) = 1_pInt
     end select
   enddo


! allocate state arrays
   sizeState = 0_pInt
   thermalState(section)%sizeState = sizeState
   thermalState(section)%sizePostResults = sum(thermal_conduction_sizePostResult(:,instance))
   allocate(thermalState(section)%state0   (sizeState,NofMyHomog))
   allocate(thermalState(section)%subState0(sizeState,NofMyHomog))
   allocate(thermalState(section)%state    (sizeState,NofMyHomog))

   nullify(thermalMapping(section)%p)
   thermalMapping(section)%p => mappingHomogenization(1,:,:)
   deallocate(temperature    (section)%p)
   allocate  (temperature    (section)%p(NofMyHomog), source=thermal_initialT(section))
   deallocate(temperatureRate(section)%p)
   allocate  (temperatureRate(section)%p(NofMyHomog), source=0.0_pReal)
     
 enddo initializeInstances
 
end subroutine thermal_conduction_init

!--------------------------------------------------------------------------------------------------
!> @brief returns heat generation rate
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_getSourceAndItsTangent(Tdot, dTdot_dT, T, ip, el)
 use material, only: &
   material_homogenizationAt, &
   homogenization_Ngrains, &
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
   offset, &
   instance, &
   grain, &
   source, &
   constituent
   
 homog  = material_homogenizationAt(el)
 offset = mappingHomogenization(1,ip,el)
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
 
end subroutine thermal_conduction_getSourceAndItsTangent
 

!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized thermal conductivity in reference configuration
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getConductivity33(ip,el)
 use lattice, only: &
   lattice_thermalConductivity33
 use material, only: &
   homogenization_Ngrains, &
   material_phase
 use mesh, only: &
   mesh_element
 use crystallite, only: &
   crystallite_push33ToRef

 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal), dimension(3,3) :: &
   thermal_conduction_getConductivity33
 integer(pInt) :: &
   grain
   
  
 thermal_conduction_getConductivity33 = 0.0_pReal
 do grain = 1, homogenization_Ngrains(mesh_element(3,el))
   thermal_conduction_getConductivity33 = thermal_conduction_getConductivity33 + &
    crystallite_push33ToRef(grain,ip,el,lattice_thermalConductivity33(:,:,material_phase(grain,ip,el)))
 enddo

 thermal_conduction_getConductivity33 = &
   thermal_conduction_getConductivity33/real(homogenization_Ngrains(mesh_element(3,el)),pReal)
 
end function thermal_conduction_getConductivity33


!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized specific heat capacity
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getSpecificHeat(ip,el)
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
   thermal_conduction_getSpecificHeat
 integer(pInt) :: &
   grain
  
 thermal_conduction_getSpecificHeat = 0.0_pReal
 
  
 do grain = 1, homogenization_Ngrains(mesh_element(3,el))
   thermal_conduction_getSpecificHeat = thermal_conduction_getSpecificHeat + &
    lattice_specificHeat(material_phase(grain,ip,el))
 enddo

 thermal_conduction_getSpecificHeat = &
   thermal_conduction_getSpecificHeat/real(homogenization_Ngrains(mesh_element(3,el)),pReal)
 
end function thermal_conduction_getSpecificHeat
 
!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized mass density
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getMassDensity(ip,el)
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
   thermal_conduction_getMassDensity
 integer(pInt) :: &
   grain
  
 thermal_conduction_getMassDensity = 0.0_pReal
 
  
 do grain = 1, homogenization_Ngrains(mesh_element(3,el))
   thermal_conduction_getMassDensity = thermal_conduction_getMassDensity &
                                     + lattice_massDensity(material_phase(grain,ip,el))
 enddo

 thermal_conduction_getMassDensity = &
   thermal_conduction_getMassDensity/real(homogenization_Ngrains(mesh_element(3,el)),pReal)
 
end function thermal_conduction_getMassDensity


!--------------------------------------------------------------------------------------------------
!> @brief updates thermal state with solution from heat conduction PDE
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_putTemperatureAndItsRate(T,Tdot,ip,el)
 use material, only: &
   material_homogenizationAt, &
   temperature, &
   temperatureRate, &
   thermalMapping

 implicit none
 integer(pInt), intent(in) :: &
   ip, &                                                                                            !< integration point number
   el                                                                                               !< element number
 real(pReal),   intent(in) :: &
   T, &
   Tdot
 integer(pInt) :: &
   homog, &
   offset  
 
 homog  = material_homogenizationAt(el)
 offset = thermalMapping(homog)%p(ip,el)
 temperature    (homog)%p(offset) = T
 temperatureRate(homog)%p(offset) = Tdot

end subroutine thermal_conduction_putTemperatureAndItsRate
 
 
!--------------------------------------------------------------------------------------------------
!> @brief return array of thermal results
!--------------------------------------------------------------------------------------------------
function thermal_conduction_postResults(homog,instance,of) result(postResults)
 use material, only: &
   temperature

 implicit none
 integer(pInt),              intent(in) :: &
   homog, &
   instance, &
   of

 real(pReal), dimension(sum(thermal_conduction_sizePostResult(:,instance))) :: &
   postResults

 integer(pInt) :: &
   o, c

 c = 0_pInt

 do o = 1_pInt,thermal_conduction_Noutput(instance)
    select case(thermal_conduction_outputID(o,instance))
 
      case (temperature_ID)
        postResults(c+1_pInt) = temperature(homog)%p(of)
        c = c + 1
    end select
 enddo
 
end function thermal_conduction_postResults

end module thermal_conduction
