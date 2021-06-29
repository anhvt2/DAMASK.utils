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
!> @brief material subroutine for locally evolving damage field
!--------------------------------------------------------------------------------------------------
module damage_local
  use prec
  use IO
  use material
  use config
  use YAML_types
  use constitutive
  use results

  implicit none
  private

  type :: tParameters
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
  end type tParameters

  type, private :: tNumerics
    real(pReal) :: &
      residualStiffness                                                                             !< non-zero residual damage
  end type tNumerics

  type(tparameters),             dimension(:),   allocatable :: &
    param

  type(tNumerics), private :: num  

  public :: &
    damage_local_init, &
    damage_local_updateState, &
    damage_local_results

contains

!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine damage_local_init

  integer :: Ninstance,NofMyHomog,h
  class(tNode), pointer :: &
    num_generic, &
    material_homogenization, &
    homog, &
    homogDamage

  write(6,'(/,a)') ' <<<+-  damage_local init  -+>>>'; flush(6)

!----------------------------------------------------------------------------------------------
! read numerics parameter and do sanity check
  num_generic => numerics_root%get('generic',defaultVal=emptyDict)
  num%residualStiffness = num_generic%get_asFloat('residualStiffness', defaultVal=1.0e-6_pReal)
  if (num%residualStiffness < 0.0_pReal)   call IO_error(301,ext_msg='residualStiffness')

  Ninstance = count(damage_type == DAMAGE_local_ID)
  allocate(param(Ninstance))

  material_homogenization => material_root%get('homogenization')
  do h = 1, material_homogenization%length
    if (damage_type(h) /= DAMAGE_LOCAL_ID) cycle
    homog => material_homogenization%get(h)
    homogDamage => homog%get('damage')
    associate(prm => param(damage_typeInstance(h)))

#if defined (__GFORTRAN__)
    prm%output = output_asStrings(homogDamage)
#else
    prm%output = homogDamage%get_asStrings('output',defaultVal=emptyStringArray)
#endif

    NofMyHomog = count(material_homogenizationAt == h)
    damageState(h)%sizeState = 1
    allocate(damageState(h)%state0   (1,NofMyHomog), source=damage_initialPhi(h))
    allocate(damageState(h)%subState0(1,NofMyHomog), source=damage_initialPhi(h))
    allocate(damageState(h)%state    (1,NofMyHomog), source=damage_initialPhi(h))

    nullify(damageMapping(h)%p)
    damageMapping(h)%p => material_homogenizationMemberAt
    deallocate(damage(h)%p)
    damage(h)%p => damageState(h)%state(1,:)

    end associate
  enddo

end subroutine damage_local_init


!--------------------------------------------------------------------------------------------------
!> @brief  calculates local change in damage field
!--------------------------------------------------------------------------------------------------
function damage_local_updateState(subdt, ip, el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal),   intent(in) :: &
    subdt
  logical,    dimension(2)  :: &
    damage_local_updateState
  integer :: &
    homog, &
    offset
  real(pReal) :: &
    phi, phiDot, dPhiDot_dPhi

  homog  = material_homogenizationAt(el)
  offset = material_homogenizationMemberAt(ip,el)
  phi = damageState(homog)%subState0(1,offset)
  call damage_local_getSourceAndItsTangent(phiDot, dPhiDot_dPhi, phi, ip, el)
  phi = max(num%residualStiffness,min(1.0_pReal,phi + subdt*phiDot))

  damage_local_updateState = [     abs(phi - damageState(homog)%state(1,offset)) &
                                <= 1.0e-2_pReal &
                              .or. abs(phi - damageState(homog)%state(1,offset)) &
                                <= 1.0e-6_pReal*abs(damageState(homog)%state(1,offset)), &
                              .true.]

  damageState(homog)%state(1,offset) = phi

end function damage_local_updateState


!--------------------------------------------------------------------------------------------------
!> @brief  calculates homogenized local damage driving forces
!--------------------------------------------------------------------------------------------------
subroutine damage_local_getSourceAndItsTangent(phiDot, dPhiDot_dPhi, phi, ip, el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal),   intent(in) :: &
    phi
  real(pReal) :: &
    phiDot, dPhiDot_dPhi

  phiDot = 0.0_pReal
  dPhiDot_dPhi = 0.0_pReal
 
  call constitutive_damage_getRateAndItsTangents(phiDot, dPhiDot_dPhi, phi, ip, el)

  phiDot = phiDot/real(homogenization_Ngrains(material_homogenizationAt(el)),pReal)
  dPhiDot_dPhi = dPhiDot_dPhi/real(homogenization_Ngrains(material_homogenizationAt(el)),pReal)

end subroutine damage_local_getSourceAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief writes results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine damage_local_results(homog,group)

  integer,          intent(in) :: homog
  character(len=*), intent(in) :: group

  integer :: o

  associate(prm => param(damage_typeInstance(homog)))
  outputsLoop: do o = 1,size(prm%output)
    select case(prm%output(o))
      case ('phi')
        call results_writeDataset(group,damage(homog)%p,prm%output(o),&
                                  'damage indicator','-')
    end select
  enddo outputsLoop
  end associate

end subroutine damage_local_results


end module damage_local
