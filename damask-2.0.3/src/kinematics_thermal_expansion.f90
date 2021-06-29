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
!> @brief material subroutine incorporating kinematics resulting from thermal expansion
!> @details to be done
!--------------------------------------------------------------------------------------------------
module kinematics_thermal_expansion
  use prec, only: &
    pReal, &
    pInt

  implicit none
  private
 
  type, private :: tParameters
    real(pReal), allocatable, dimension(:,:,:) :: &
      expansion
  end type tParameters
 
  type(tParameters), dimension(:), allocatable :: param
  
  public :: &
    kinematics_thermal_expansion_init, &
    kinematics_thermal_expansion_initialStrain, &
    kinematics_thermal_expansion_LiAndItsTangent

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine kinematics_thermal_expansion_init()
  use debug, only: &
    debug_level,&
    debug_constitutive,&
    debug_levelBasic
  use material, only: &
    phase_kinematics, &
    KINEMATICS_thermal_expansion_label, &
    KINEMATICS_thermal_expansion_ID
  use config, only: &
    config_phase
 
  implicit none
  integer(pInt) :: &
    Ninstance, &
    p, i
  real(pReal),  dimension(:), allocatable :: &
    temp
  
  write(6,'(/,a)')   ' <<<+-  kinematics_'//KINEMATICS_thermal_expansion_LABEL//' init  -+>>>'
 
  Ninstance = int(count(phase_kinematics == KINEMATICS_thermal_expansion_ID),pInt)
  
  if (iand(debug_level(debug_constitutive),debug_levelBasic) /= 0_pInt) &
    write(6,'(a16,1x,i5,/)') '# instances:',Ninstance
  
  allocate(param(Ninstance))
  
  do p = 1_pInt, size(phase_kinematics)
    if (all(phase_kinematics(:,p) /= KINEMATICS_thermal_expansion_ID)) cycle
    
    ! ToDo: Here we need to decide how to extend the concept of instances to
    ! kinetics and sources. I would suggest that the same mechanism exists at maximum once per phase
    
    ! read up to three parameters (constant, linear, quadratic with T)
    temp = config_phase(p)%getFloats('thermal_expansion11')                   
    !lattice_thermalExpansion33(1,1,1:size(temp),p) = temp
    temp = config_phase(p)%getFloats('thermal_expansion22', &
                                     defaultVal=[(0.0_pReal, i=1,size(temp))],requiredSize=size(temp)) 
    !lattice_thermalExpansion33(2,2,1:size(temp),p) = temp
    temp = config_phase(p)%getFloats('thermal_expansion33', &
                                     defaultVal=[(0.0_pReal, i=1,size(temp))],requiredSize=size(temp))
  enddo

end subroutine kinematics_thermal_expansion_init


!--------------------------------------------------------------------------------------------------
!> @brief  report initial thermal strain based on current temperature deviation from reference
!--------------------------------------------------------------------------------------------------
pure function kinematics_thermal_expansion_initialStrain(homog,phase,offset)
  use material, only: &
    temperature
  use lattice, only: &
    lattice_thermalExpansion33, &
    lattice_referenceTemperature
  
  implicit none
  integer(pInt), intent(in) :: &
    phase, &
    homog, offset
  real(pReal), dimension(3,3) :: &
    kinematics_thermal_expansion_initialStrain                                                       !< initial thermal strain (should be small strain, though)
 
  
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
subroutine kinematics_thermal_expansion_LiAndItsTangent(Li, dLi_dTstar, ipc, ip, el)
  use material, only: &
    material_phase, &
   material_homogenizationAt, &
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
    dLi_dTstar                                                                                       !< derivative of Li with respect to Tstar (4th-order tensor defined to be zero)
  integer(pInt) :: &
    phase, &
    homog, offset
  real(pReal) :: &
    T, TRef, TDot  
    
  phase = material_phase(ipc,ip,el)
  homog = material_homogenizationAt(el)
  offset = thermalMapping(homog)%p(ip,el)
  T = temperature(homog)%p(offset)
  TDot = temperatureRate(homog)%p(offset)
  TRef = lattice_referenceTemperature(phase)
  
  Li = TDot * ( &
                lattice_thermalExpansion33(1:3,1:3,1,phase)*(T - TRef)**0 &                           ! constant  coefficient
              + lattice_thermalExpansion33(1:3,1:3,2,phase)*(T - TRef)**1 &                           ! linear    coefficient
              + lattice_thermalExpansion33(1:3,1:3,3,phase)*(T - TRef)**2 &                           ! quadratic coefficient
              ) / &
       (1.0_pReal &
             + lattice_thermalExpansion33(1:3,1:3,1,phase)*(T - TRef)**1 / 1. &
             + lattice_thermalExpansion33(1:3,1:3,2,phase)*(T - TRef)**2 / 2. &
             + lattice_thermalExpansion33(1:3,1:3,3,phase)*(T - TRef)**3 / 3. &
       )
  dLi_dTstar = 0.0_pReal 
  
end subroutine kinematics_thermal_expansion_LiAndItsTangent

end module kinematics_thermal_expansion
