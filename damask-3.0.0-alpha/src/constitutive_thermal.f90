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
!----------------------------------------------------------------------------------------------------
!> @brief internal microstructure state for all thermal sources and kinematics constitutive models
!----------------------------------------------------------------------------------------------------
submodule(constitutive) constitutive_thermal

  interface

  module function source_thermal_dissipation_init(source_length) result(mySources)
    integer, intent(in) :: source_length
    logical, dimension(:,:), allocatable :: mySources
  end function source_thermal_dissipation_init
 
  module function source_thermal_externalheat_init(source_length) result(mySources)
    integer, intent(in) :: source_length
    logical, dimension(:,:), allocatable :: mySources
  end function source_thermal_externalheat_init

  module function kinematics_thermal_expansion_init(kinematics_length) result(myKinematics)
    integer, intent(in) :: kinematics_length
    logical, dimension(:,:), allocatable :: myKinematics
  end function kinematics_thermal_expansion_init


  module subroutine source_thermal_dissipation_getRateAndItsTangent(TDot, dTDot_dT, Tstar, Lp, phase)
    integer, intent(in) :: &
      phase                                                                                         !< phase ID of element
    real(pReal),  intent(in), dimension(3,3) :: &
      Tstar                                                                                         !< 2nd Piola Kirchhoff stress tensor for a given element
    real(pReal),  intent(in), dimension(3,3) :: &
      Lp                                                                                            !< plastic velocuty gradient for a given element
    real(pReal),  intent(out) :: &
      TDot, &
      dTDot_dT
  end subroutine source_thermal_dissipation_getRateAndItsTangent

  module subroutine source_thermal_externalheat_getRateAndItsTangent(TDot, dTDot_dT, phase, of)
    integer, intent(in) :: &
      phase, &
      of
    real(pReal),  intent(out) :: &
      TDot, &
      dTDot_dT
  end subroutine source_thermal_externalheat_getRateAndItsTangent

 end interface

contains

!----------------------------------------------------------------------------------------------
!< @brief initializes thermal sources and kinematics mechanism
!----------------------------------------------------------------------------------------------
module subroutine thermal_init

! initialize source mechanisms
  if(maxval(phase_Nsources) /= 0) then
    where(source_thermal_dissipation_init (maxval(phase_Nsources))) phase_source = SOURCE_thermal_dissipation_ID
    where(source_thermal_externalheat_init(maxval(phase_Nsources))) phase_source = SOURCE_thermal_externalheat_ID
  endif 
 
!--------------------------------------------------------------------------------------------------
!initialize kinematic mechanisms
  if(maxval(phase_Nkinematics) /= 0) where(kinematics_thermal_expansion_init(maxval(phase_Nkinematics))) &
                                           phase_kinematics = KINEMATICS_thermal_expansion_ID

end subroutine thermal_init


!----------------------------------------------------------------------------------------------
!< @brief calculates thermal dissipation rate
!----------------------------------------------------------------------------------------------
module subroutine constitutive_thermal_getRateAndItsTangents(TDot, dTDot_dT, T, S, Lp, ip, el)
  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal), intent(in) :: &
    T
  real(pReal),  intent(in), dimension(:,:,:,:,:) :: &
    S, &                                                                                            !< current 2nd Piola Kirchhoff stress
    Lp                                                                                              !< plastic velocity gradient
  real(pReal), intent(inout) :: &
    TDot, &
    dTDot_dT

  real(pReal) :: &
    my_Tdot, &
    my_dTdot_dT
  integer :: &
    phase, &
    homog, &
    instance, &
    grain, &
    source, &
    constituent

   homog  = material_homogenizationAt(el)
   instance = thermal_typeInstance(homog)

  do grain = 1, homogenization_Ngrains(homog)
     phase = material_phaseAt(grain,el)
     constituent = material_phasememberAt(grain,ip,el)
     do source = 1, phase_Nsources(phase)
       select case(phase_source(source,phase))
         case (SOURCE_thermal_dissipation_ID)
          call source_thermal_dissipation_getRateAndItsTangent(my_Tdot, my_dTdot_dT, &
                                                               S(1:3,1:3,grain,ip,el), &
                                                                Lp(1:3,1:3,grain,ip,el), &
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

end subroutine constitutive_thermal_getRateAndItsTangents


end submodule constitutive_thermal
