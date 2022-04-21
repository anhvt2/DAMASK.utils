! Copyright 2011-2022 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Affero General Public License for more details.
! 
! You should have received a copy of the GNU Affero General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine incorporating kinematics resulting from thermal expansion
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule(phase:eigen) thermalexpansion

  integer, dimension(:), allocatable :: kinematics_thermal_expansion_instance

  type :: tParameters
    type(tPolynomial) :: &
      A_11, &
      A_33
  end type tParameters

  type(tParameters), dimension(:), allocatable :: param

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function thermalexpansion_init(kinematics_length) result(myKinematics)

  integer, intent(in)                  :: kinematics_length
  logical, dimension(:,:), allocatable :: myKinematics

  integer :: Ninstances, p, k
  class(tNode), pointer :: &
    phases, &
    phase, &
    mech, &
    kinematics, &
    myConfig

  print'(/,1x,a)', '<<<+-  phase:mechanical:eigen:thermalexpansion init  -+>>>'

  myKinematics = kinematics_active('thermalexpansion',kinematics_length)
  Ninstances = count(myKinematics)
  print'(/,a,i2)', ' # phases: ',Ninstances; flush(IO_STDOUT)
  if (Ninstances == 0) return

  phases => config_material%get('phase')
  allocate(param(Ninstances))
  allocate(kinematics_thermal_expansion_instance(phases%length), source=0)

  do p = 1, phases%length
    if (any(myKinematics(:,p))) kinematics_thermal_expansion_instance(p) = count(myKinematics(:,1:p))
    phase => phases%get(p)
    if (count(myKinematics(:,p)) == 0) cycle
    mech => phase%get('mechanical')
    kinematics => mech%get('eigen')
    do k = 1, kinematics%length
      if (myKinematics(k,p)) then
        associate(prm  => param(kinematics_thermal_expansion_instance(p)))

          myConfig => kinematics%get(k)

          prm%A_11 = polynomial(myConfig%asDict(),'A_11','T')
          if (any(phase_lattice(p) == ['hP','tI'])) &
            prm%A_33 = polynomial(myConfig%asDict(),'A_33','T')

        end associate
      end if
    end do
  end do

end function thermalexpansion_init


!--------------------------------------------------------------------------------------------------
!> @brief constitutive equation for calculating the velocity gradient
!--------------------------------------------------------------------------------------------------
module subroutine thermalexpansion_LiAndItsTangent(Li, dLi_dTstar, ph,me)

  integer, intent(in) :: ph, me
  real(pReal),   intent(out), dimension(3,3) :: &
    Li                                                                                              !< thermal velocity gradient
  real(pReal),   intent(out), dimension(3,3,3,3) :: &
    dLi_dTstar                                                                                      !< derivative of Li with respect to Tstar (4th-order tensor defined to be zero)

  real(pReal) :: T, dot_T
  real(pReal), dimension(3,3) :: A


  T     = thermal_T(ph,me)
  dot_T = thermal_dot_T(ph,me)

  associate(prm => param(kinematics_thermal_expansion_instance(ph)))

    A = 0.0_pReal
    A(1,1) = prm%A_11%at(T)
    if (any(phase_lattice(ph) == ['hP','tI'])) A(3,3) = prm%A_33%at(T)
    A = lattice_symmetrize_33(A,phase_lattice(ph))
    Li = dot_T * A

  end associate
  dLi_dTstar = 0.0_pReal

end subroutine thermalexpansion_LiAndItsTangent

end submodule thermalexpansion
