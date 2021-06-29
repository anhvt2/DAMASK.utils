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
!> @brief material subroutine incorporating kinematics resulting from thermal expansion
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_thermal) kinematics_thermal_expansion

  integer, dimension(:), allocatable :: kinematics_thermal_expansion_instance

  type :: tParameters
    real(pReal) :: &
      T_ref
    real(pReal), dimension(3,3,3) :: &
      A = 0.0_pReal
  end type tParameters

  type(tParameters), dimension(:), allocatable :: param


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function kinematics_thermal_expansion_init(kinematics_length) result(myKinematics)

  integer, intent(in)                  :: kinematics_length  
  logical, dimension(:,:), allocatable :: myKinematics

  integer :: Ninstances,p,i,k
  real(pReal), dimension(:), allocatable :: temp
  class(tNode), pointer :: &
    phases, &
    phase, &
    kinematics, &
    kinematic_type 
 
  print'(/,a)', ' <<<+-  kinematics_thermal_expansion init  -+>>>'

  myKinematics = kinematics_active('thermal_expansion',kinematics_length)
  Ninstances = count(myKinematics)
  print'(a,i2)', ' # instances: ',Ninstances; flush(IO_STDOUT)
  if(Ninstances == 0) return

  phases => config_material%get('phase')
  allocate(param(Ninstances))
  allocate(kinematics_thermal_expansion_instance(phases%length), source=0)

  do p = 1, phases%length
    if(any(myKinematics(:,p))) kinematics_thermal_expansion_instance(p) = count(myKinematics(:,1:p))
    phase => phases%get(p) 
    if(count(myKinematics(:,p)) == 0) cycle
    kinematics => phase%get('kinematics')
    do k = 1, kinematics%length
      if(myKinematics(k,p)) then
        associate(prm  => param(kinematics_thermal_expansion_instance(p)))
        kinematic_type => kinematics%get(k) 

        prm%T_ref = kinematic_type%get_asFloat('T_ref', defaultVal=0.0_pReal)

        ! read up to three parameters (constant, linear, quadratic with T)
        temp = kinematic_type%get_asFloats('A_11')
        prm%A(1,1,1:size(temp)) = temp
        temp = kinematic_type%get_asFloats('A_22',defaultVal=[(0.0_pReal, i=1,size(temp))],requiredSize=size(temp))
        prm%A(2,2,1:size(temp)) = temp
        temp = kinematic_type%get_asFloats('A_33',defaultVal=[(0.0_pReal, i=1,size(temp))],requiredSize=size(temp))
        prm%A(3,3,1:size(temp)) = temp
        do i=1, size(prm%A,3)
          prm%A(1:3,1:3,i) = lattice_applyLatticeSymmetry33(prm%A(1:3,1:3,i),&
                                                   phase%get_asString('lattice'))
        enddo

        end associate
      endif
    enddo
  enddo


end function kinematics_thermal_expansion_init


!--------------------------------------------------------------------------------------------------
!> @brief  report initial thermal strain based on current temperature deviation from reference
!--------------------------------------------------------------------------------------------------
pure module function kinematics_thermal_expansion_initialStrain(homog,phase,offset) result(initialStrain)

 integer, intent(in) :: &
   phase, &
   homog, &
   offset

 real(pReal), dimension(3,3) :: &
   initialStrain                                                                                    !< initial thermal strain (should be small strain, though)

 associate(prm => param(kinematics_thermal_expansion_instance(phase)))
 initialStrain = &
   (temperature(homog)%p(offset) - prm%T_ref)**1 / 1. * prm%A(1:3,1:3,1) + &                        ! constant  coefficient
   (temperature(homog)%p(offset) - prm%T_ref)**2 / 2. * prm%A(1:3,1:3,2) + &                        ! linear    coefficient
   (temperature(homog)%p(offset) - prm%T_ref)**3 / 3. * prm%A(1:3,1:3,3)                            ! quadratic coefficient
 end associate

end function kinematics_thermal_expansion_initialStrain


!--------------------------------------------------------------------------------------------------
!> @brief constitutive equation for calculating the velocity gradient
!--------------------------------------------------------------------------------------------------
module subroutine kinematics_thermal_expansion_LiAndItsTangent(Li, dLi_dTstar, ipc, ip, el)

  integer, intent(in) :: &
    ipc, &                                                                                          !< grain number
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal),   intent(out), dimension(3,3) :: &
    Li                                                                                              !< thermal velocity gradient
  real(pReal),   intent(out), dimension(3,3,3,3) :: &
    dLi_dTstar                                                                                      !< derivative of Li with respect to Tstar (4th-order tensor defined to be zero)

  integer :: &
    phase, &
    homog
  real(pReal) :: &
    T, TDot

  phase = material_phaseAt(ipc,el)
  homog = material_homogenizationAt(el)
  T = temperature(homog)%p(thermalMapping(homog)%p(ip,el))
  TDot = temperatureRate(homog)%p(thermalMapping(homog)%p(ip,el))

  associate(prm => param(kinematics_thermal_expansion_instance(phase)))
  Li = TDot * ( &
                prm%A(1:3,1:3,1)*(T - prm%T_ref)**0 &                                               ! constant  coefficient
              + prm%A(1:3,1:3,2)*(T - prm%T_ref)**1 &                                               ! linear    coefficient
              + prm%A(1:3,1:3,3)*(T - prm%T_ref)**2 &                                               ! quadratic coefficient
              ) / &
       (1.0_pReal &
             + prm%A(1:3,1:3,1)*(T - prm%T_ref)**1 / 1. &
             + prm%A(1:3,1:3,2)*(T - prm%T_ref)**2 / 2. &
             + prm%A(1:3,1:3,3)*(T - prm%T_ref)**3 / 3. &
       )
  end associate
  dLi_dTstar = 0.0_pReal

end subroutine kinematics_thermal_expansion_LiAndItsTangent

end submodule kinematics_thermal_expansion
