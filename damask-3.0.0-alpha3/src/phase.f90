! Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
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
!> @brief elasticity, plasticity, damage & thermal internal microstructure state
!--------------------------------------------------------------------------------------------------
module phase
  use prec
  use math
  use rotations
  use IO
  use config
  use material
  use results
  use lattice
  use discretization
  use parallelization
  use HDF5_utilities

  implicit none
  private

  type(Rotation), dimension(:,:,:), allocatable :: &
    material_orientation0                                                                           !< initial orientation of each grain,IP,element

  type(rotation),            dimension(:,:,:),        allocatable :: &
    crystallite_orientation                                                                         !< current orientation

  type :: tTensorContainer
    real(pReal), dimension(:,:,:), allocatable :: data
  end type

  type :: tNumerics
    integer :: &
      iJacoLpresiduum, &                                                                            !< frequency of Jacobian update of residuum in Lp
      nState, &                                                                                     !< state loop limit
      nStress                                                                                       !< stress loop limit
    real(pReal) :: &
      subStepMinCryst, &                                                                            !< minimum (relative) size of sub-step allowed during cutback
      subStepSizeCryst, &                                                                           !< size of first substep when cutback
      subStepSizeLp, &                                                                              !< size of first substep when cutback in Lp calculation
      subStepSizeLi, &                                                                              !< size of first substep when cutback in Li calculation
      stepIncreaseCryst, &                                                                          !< increase of next substep size when previous substep converged
      rtol_crystalliteState, &                                                                      !< relative tolerance in state loop
      rtol_crystalliteStress, &                                                                     !< relative tolerance in stress loop
      atol_crystalliteStress                                                                        !< absolute tolerance in stress loop
  end type tNumerics

  type(tNumerics) :: num                                                                            ! numerics parameters. Better name?

  type :: tDebugOptions
    logical :: &
      basic, &
      extensive, &
      selective
    integer :: &
      element, &
      ip, &
      grain
  end type tDebugOptions

  integer, dimension(:), allocatable, public :: &                                                   !< ToDo: should be protected (bug in Intel compiler)
    phase_elasticityInstance, &
    phase_NstiffnessDegradations

  logical, dimension(:), allocatable, public :: &                                                   ! ToDo: should be protected (bug in Intel Compiler)
    phase_localPlasticity                                                                           !< flags phases with local constitutive law

  type(tPlasticState), allocatable, dimension(:), public :: &
    plasticState
  type(tState),  allocatable, dimension(:), public :: &
    damageState


  integer, public, protected :: &
    phase_plasticity_maxSizeDotState, &
    phase_source_maxSizeDotState

  interface

! == cleaned:begin =================================================================================
    module subroutine mechanical_init(materials,phases)
      class(tNode), pointer :: materials,phases
    end subroutine mechanical_init

    module subroutine damage_init
    end subroutine damage_init

    module subroutine thermal_init(phases)
      class(tNode), pointer :: phases
    end subroutine thermal_init


    module subroutine mechanical_results(group,ph)
      character(len=*), intent(in) :: group
      integer,          intent(in) :: ph
    end subroutine mechanical_results

    module subroutine damage_results(group,ph)
      character(len=*), intent(in) :: group
      integer,          intent(in) :: ph
    end subroutine damage_results

    module subroutine mechanical_forward()
    end subroutine mechanical_forward

    module subroutine damage_forward()
    end subroutine damage_forward

    module subroutine thermal_forward()
    end subroutine thermal_forward


    module subroutine mechanical_restore(ce,includeL)
      integer, intent(in) :: ce
      logical, intent(in) :: includeL
    end subroutine mechanical_restore


    module function phase_mechanical_dPdF(dt,co,ce) result(dPdF)
      real(pReal), intent(in) :: dt
      integer, intent(in) :: &
        co, &                                                                                       !< counter in constituent loop
        ce
      real(pReal), dimension(3,3,3,3) :: dPdF
    end function phase_mechanical_dPdF

    module subroutine mechanical_restartWrite(groupHandle,ph)
      integer(HID_T), intent(in) :: groupHandle
      integer, intent(in) :: ph
    end subroutine mechanical_restartWrite

    module subroutine mechanical_restartRead(groupHandle,ph)
      integer(HID_T), intent(in) :: groupHandle
      integer, intent(in) :: ph
    end subroutine mechanical_restartRead


    module function mechanical_S(ph,en) result(S)
      integer, intent(in) :: ph,en
      real(pReal), dimension(3,3) :: S
    end function mechanical_S

    module function mechanical_L_p(ph,en) result(L_p)
      integer, intent(in) :: ph,en
      real(pReal), dimension(3,3) :: L_p
    end function mechanical_L_p

    module function mechanical_F_e(ph,en) result(F_e)
      integer, intent(in) :: ph,en
      real(pReal), dimension(3,3) :: F_e
    end function mechanical_F_e


    module function phase_F(co,ce) result(F)
      integer, intent(in) :: co, ce
      real(pReal), dimension(3,3) :: F
    end function phase_F

    module function phase_P(co,ce) result(P)
      integer, intent(in) :: co, ce
      real(pReal), dimension(3,3) :: P
    end function phase_P

    module function thermal_T(ph,me) result(T)
      integer, intent(in) :: ph,me
      real(pReal) :: T
    end function thermal_T

    module function thermal_dot_T(ph,me) result(dot_T)
      integer, intent(in) :: ph,me
      real(pReal) :: dot_T
    end function thermal_dot_T

    module function damage_phi(ph,me) result(phi)
      integer, intent(in) :: ph,me
      real(pReal) :: phi
    end function damage_phi


    module subroutine phase_set_F(F,co,ce)
      real(pReal), dimension(3,3), intent(in) :: F
      integer, intent(in) :: co, ce
    end subroutine phase_set_F

    module subroutine phase_thermal_setField(T,dot_T, co,ce)
      real(pReal), intent(in) :: T, dot_T
      integer, intent(in) :: co, ce
    end subroutine phase_thermal_setField

    module subroutine phase_set_phi(phi,co,ce)
      real(pReal), intent(in) :: phi
      integer, intent(in) :: co, ce
    end subroutine phase_set_phi


    module function phase_mu_phi(co,ce) result(mu)
      integer, intent(in) :: co, ce
      real(pReal) :: mu
    end function phase_mu_phi

    module function phase_K_phi(co,ce) result(K)
      integer, intent(in) :: co, ce
      real(pReal), dimension(3,3) :: K
    end function phase_K_phi


    module function phase_mu_T(co,ce) result(mu)
      integer, intent(in) :: co, ce
      real(pReal) :: mu
    end function phase_mu_T

    module function phase_K_T(co,ce) result(K)
      integer, intent(in) :: co, ce
      real(pReal), dimension(3,3) :: K
    end function phase_K_T

! == cleaned:end ===================================================================================

    module function thermal_stress(Delta_t,ph,me) result(converged_)

      real(pReal), intent(in) :: Delta_t
      integer, intent(in) :: ph, me
      logical :: converged_

    end function thermal_stress

    module function integrateDamageState(dt,co,ip,el) result(broken)
      real(pReal), intent(in) :: dt
      integer, intent(in) :: &
        el, &                                                                                            !< element index in element loop
        ip, &                                                                                            !< integration point index in ip loop
        co                                                                                               !< grain index in grain loop
      logical :: broken
    end function integrateDamageState

    module function crystallite_stress(dt,co,ip,el) result(converged_)
      real(pReal), intent(in) :: dt
      integer, intent(in) :: co, ip, el
      logical :: converged_
    end function crystallite_stress

    module function phase_homogenizedC(ph,en) result(C)
      integer, intent(in) :: ph, en
      real(pReal), dimension(6,6) :: C
    end function phase_homogenizedC


    module function phase_f_phi(phi,co,ce) result(f)
      integer, intent(in) :: ce,co
      real(pReal), intent(in) :: &
        phi                                                                                         !< damage parameter
      real(pReal) :: &
        f
    end function phase_f_phi

    module function phase_f_T(ph,me) result(f)
      integer, intent(in) :: ph, me
      real(pReal) :: f
    end function phase_f_T

    module subroutine plastic_nonlocal_updateCompatibility(orientation,ph,i,e)
      integer, intent(in) :: &
        ph, &
        i, &
        e
      type(rotation), dimension(1,discretization_nIPs,discretization_Nelems), intent(in) :: &
        orientation                                                                                 !< crystal orientation
    end subroutine plastic_nonlocal_updateCompatibility

    module subroutine plastic_dependentState(co,ip,el)
      integer, intent(in) :: &
        co, &                                                                                       !< component-ID of integration point
        ip, &                                                                                       !< integration point
        el                                                                                          !< element
    end subroutine plastic_dependentState


    module subroutine damage_anisobrittle_LiAndItsTangent(Ld, dLd_dTstar, S, ph,me)
      integer, intent(in) :: ph, me
      real(pReal),   intent(in),  dimension(3,3) :: &
        S
      real(pReal),   intent(out), dimension(3,3) :: &
        Ld                                                                                          !< damage velocity gradient
      real(pReal),   intent(out), dimension(3,3,3,3) :: &
        dLd_dTstar                                                                                  !< derivative of Ld with respect to Tstar (4th-order tensor)
    end subroutine damage_anisobrittle_LiAndItsTangent

    module subroutine damage_isoductile_LiAndItsTangent(Ld, dLd_dTstar, S, ph,me)
      integer, intent(in) :: ph, me
      real(pReal),   intent(in),  dimension(3,3) :: &
        S
      real(pReal),   intent(out), dimension(3,3) :: &
        Ld                                                                                          !< damage velocity gradient
      real(pReal),   intent(out), dimension(3,3,3,3) :: &
        dLd_dTstar                                                                                  !< derivative of Ld with respect to Tstar (4th-order tensor)
    end subroutine damage_isoductile_LiAndItsTangent

  end interface



  type(tDebugOptions) :: debugConstitutive
#if __INTEL_COMPILER >= 1900
  public :: &
    prec, &
    math, &
    rotations, &
    IO, &
    config, &
    material, &
    results, &
    lattice, &
    discretization, &
    HDF5_utilities
#endif

  public :: &
    phase_init, &
    phase_homogenizedC, &
    phase_f_phi, &
    phase_f_T, &
    phase_K_phi, &
    phase_K_T, &
    phase_mu_phi, &
    phase_mu_T, &
    phase_results, &
    phase_allocateState, &
    phase_forward, &
    phase_restore, &
    plastic_nonlocal_updateCompatibility, &
    converged, &
    crystallite_init, &
    crystallite_stress, &
    thermal_stress, &
    phase_mechanical_dPdF, &
    crystallite_orientations, &
    crystallite_push33ToRef, &
    phase_restartWrite, &
    phase_restartRead, &
    integrateDamageState, &
    phase_thermal_setField, &
    phase_set_phi, &
    phase_P, &
    phase_set_F, &
    phase_F

contains

!--------------------------------------------------------------------------------------------------
!> @brief Initialize constitutive models for individual physics
!--------------------------------------------------------------------------------------------------
subroutine phase_init

  integer :: &
    ph
  class (tNode), pointer :: &
    debug_constitutive, &
    materials, &
    phases


  print'(/,a)', ' <<<+-  phase init  -+>>>'; flush(IO_STDOUT)

  debug_constitutive => config_debug%get('phase', defaultVal=emptyList)
  debugConstitutive%basic     = debug_constitutive%contains('basic')
  debugConstitutive%extensive = debug_constitutive%contains('extensive')
  debugConstitutive%selective = debug_constitutive%contains('selective')
  debugConstitutive%element   = config_debug%get_asInt('element',         defaultVal = 1)
  debugConstitutive%ip        = config_debug%get_asInt('integrationpoint',defaultVal = 1)
  debugConstitutive%grain     = config_debug%get_asInt('constituent',     defaultVal = 1)


  materials => config_material%get('material')
  phases    => config_material%get('phase')

  call mechanical_init(materials,phases)
  call damage_init
  call thermal_init(phases)


  phase_source_maxSizeDotState = 0
  PhaseLoop2:do ph = 1,phases%length
!--------------------------------------------------------------------------------------------------
! partition and initialize state
    plasticState(ph)%state = plasticState(ph)%state0
    if(damageState(ph)%sizeState > 0) &
      damageState(ph)%state  = damageState(ph)%state0
  enddo PhaseLoop2

  phase_source_maxSizeDotState     = maxval(damageState%sizeDotState)
  phase_plasticity_maxSizeDotState = maxval(plasticState%sizeDotState)

end subroutine phase_init


!--------------------------------------------------------------------------------------------------
!> @brief Allocate the components of the state structure for a given phase
!--------------------------------------------------------------------------------------------------
subroutine phase_allocateState(state, &
                               NEntries,sizeState,sizeDotState,sizeDeltaState)

  class(tState), intent(out) :: &
    state
  integer, intent(in) :: &
    NEntries, &
    sizeState, &
    sizeDotState, &
    sizeDeltaState


  state%sizeState        = sizeState
  state%sizeDotState     = sizeDotState
  state%sizeDeltaState   = sizeDeltaState
  state%offsetDeltaState = sizeState-sizeDeltaState                                                 ! deltaState occupies latter part of state by definition

  allocate(state%atol             (sizeState),          source=0.0_pReal)
  allocate(state%state0           (sizeState,NEntries), source=0.0_pReal)
  allocate(state%state            (sizeState,NEntries), source=0.0_pReal)

  allocate(state%dotState      (sizeDotState,NEntries), source=0.0_pReal)

  allocate(state%deltaState  (sizeDeltaState,NEntries), source=0.0_pReal)


end subroutine phase_allocateState


!--------------------------------------------------------------------------------------------------
!> @brief Restore data after homog cutback.
!--------------------------------------------------------------------------------------------------
subroutine phase_restore(ce,includeL)

  logical, intent(in) :: includeL
  integer, intent(in) :: ce

  integer :: &
    co


  do co = 1,homogenization_Nconstituents(material_homogenizationID(ce))
    if (damageState(material_phaseID(co,ce))%sizeState > 0) &
    damageState(material_phaseID(co,ce))%state( :,material_phaseEntry(co,ce)) = &
      damageState(material_phaseID(co,ce))%state0(:,material_phaseEntry(co,ce))
  enddo

  call mechanical_restore(ce,includeL)

end subroutine phase_restore


!--------------------------------------------------------------------------------------------------
!> @brief Forward data after successful increment.
!--------------------------------------------------------------------------------------------------
subroutine phase_forward()

  call mechanical_forward()
  call damage_forward()
  call thermal_forward()

end subroutine phase_forward


!--------------------------------------------------------------------------------------------------
!> @brief writes constitutive results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine phase_results()

  integer :: ph
  character(len=:), allocatable :: group


  call results_closeGroup(results_addGroup('/current/phase/'))

  do ph = 1, size(material_name_phase)

    group = '/current/phase/'//trim(material_name_phase(ph))//'/'
    call results_closeGroup(results_addGroup(group))

    call mechanical_results(group,ph)
    call damage_results(group,ph)

  enddo

end subroutine phase_results


!--------------------------------------------------------------------------------------------------
!> @brief allocates and initialize per grain variables
!--------------------------------------------------------------------------------------------------
subroutine crystallite_init()

  integer :: &
    ph, &
    co, &                                                                                           !< counter in integration point component loop
    ip, &                                                                                           !< counter in integration point loop
    el, &                                                                                           !< counter in element loop
    cMax, &                                                                                         !< maximum number of  integration point components
    iMax, &                                                                                         !< maximum number of integration points
    eMax                                                                                            !< maximum number of elements

  class(tNode), pointer :: &
    num_crystallite, &
    phases


  print'(/,a)', ' <<<+-  crystallite init  -+>>>'

  cMax = homogenization_maxNconstituents
  iMax = discretization_nIPs
  eMax = discretization_Nelems

  allocate(crystallite_orientation(cMax,iMax,eMax))

  num_crystallite => config_numerics%get('crystallite',defaultVal=emptyDict)

  num%subStepMinCryst        = num_crystallite%get_asFloat ('subStepMin',       defaultVal=1.0e-3_pReal)
  num%subStepSizeCryst       = num_crystallite%get_asFloat ('subStepSize',      defaultVal=0.25_pReal)
  num%stepIncreaseCryst      = num_crystallite%get_asFloat ('stepIncrease',     defaultVal=1.5_pReal)
  num%subStepSizeLp          = num_crystallite%get_asFloat ('subStepSizeLp',    defaultVal=0.5_pReal)
  num%subStepSizeLi          = num_crystallite%get_asFloat ('subStepSizeLi',    defaultVal=0.5_pReal)
  num%rtol_crystalliteState  = num_crystallite%get_asFloat ('rtol_State',       defaultVal=1.0e-6_pReal)
  num%rtol_crystalliteStress = num_crystallite%get_asFloat ('rtol_Stress',      defaultVal=1.0e-6_pReal)
  num%atol_crystalliteStress = num_crystallite%get_asFloat ('atol_Stress',      defaultVal=1.0e-8_pReal)
  num%iJacoLpresiduum        = num_crystallite%get_asInt   ('iJacoLpresiduum',  defaultVal=1)
  num%nState                 = num_crystallite%get_asInt   ('nState',           defaultVal=20)
  num%nStress                = num_crystallite%get_asInt   ('nStress',          defaultVal=40)

  if(num%subStepMinCryst   <= 0.0_pReal)      call IO_error(301,ext_msg='subStepMinCryst')
  if(num%subStepSizeCryst  <= 0.0_pReal)      call IO_error(301,ext_msg='subStepSizeCryst')
  if(num%stepIncreaseCryst <= 0.0_pReal)      call IO_error(301,ext_msg='stepIncreaseCryst')

  if(num%subStepSizeLp <= 0.0_pReal)          call IO_error(301,ext_msg='subStepSizeLp')
  if(num%subStepSizeLi <= 0.0_pReal)          call IO_error(301,ext_msg='subStepSizeLi')

  if(num%rtol_crystalliteState  <= 0.0_pReal) call IO_error(301,ext_msg='rtol_crystalliteState')
  if(num%rtol_crystalliteStress <= 0.0_pReal) call IO_error(301,ext_msg='rtol_crystalliteStress')
  if(num%atol_crystalliteStress <= 0.0_pReal) call IO_error(301,ext_msg='atol_crystalliteStress')

  if(num%iJacoLpresiduum < 1)                 call IO_error(301,ext_msg='iJacoLpresiduum')

  if(num%nState < 1)                          call IO_error(301,ext_msg='nState')
  if(num%nStress< 1)                          call IO_error(301,ext_msg='nStress')


  phases => config_material%get('phase')

  do ph = 1, phases%length
    if (damageState(ph)%sizeState > 0) &
      allocate(damageState(ph)%subState0,source=damageState(ph)%state0)                 ! ToDo: hack
  enddo

  print'(a42,1x,i10)', '    # of elements:                       ', eMax
  print'(a42,1x,i10)', '    # of integration points/element:     ', iMax
  print'(a42,1x,i10)', 'max # of constituents/integration point: ', cMax
  flush(IO_STDOUT)


  !$OMP PARALLEL DO
  do el = 1, size(material_phaseMemberAt,3)
    do ip = 1, size(material_phaseMemberAt,2)
      do co = 1,homogenization_Nconstituents(material_homogenizationAt(el))
        call crystallite_orientations(co,ip,el)
        call plastic_dependentState(co,ip,el)                                          ! update dependent state variables to be consistent with basic states
     enddo
    enddo
  enddo
  !$OMP END PARALLEL DO


end subroutine crystallite_init


!--------------------------------------------------------------------------------------------------
!> @brief calculates orientations
!--------------------------------------------------------------------------------------------------
subroutine crystallite_orientations(co,ip,el)

  integer, intent(in) :: &
    co, &                                                                                            !< counter in integration point component loop
    ip, &                                                                                            !< counter in integration point loop
    el                                                                                               !< counter in element loop


  call crystallite_orientation(co,ip,el)%fromMatrix(transpose(math_rotationalPart(&
    mechanical_F_e(material_phaseAt(co,el),material_phaseMemberAt(co,ip,el)))))

  if (plasticState(material_phaseAt(1,el))%nonlocal) &
    call plastic_nonlocal_updateCompatibility(crystallite_orientation, &
                                              material_phaseAt(1,el),ip,el)


end subroutine crystallite_orientations


!--------------------------------------------------------------------------------------------------
!> @brief Map 2nd order tensor to reference config
!--------------------------------------------------------------------------------------------------
function crystallite_push33ToRef(co,ce, tensor33)

  real(pReal), dimension(3,3), intent(in) :: tensor33
  integer, intent(in):: &
    co, &
    ce
  real(pReal), dimension(3,3) :: crystallite_push33ToRef

  real(pReal), dimension(3,3)             :: T
  integer :: ph, en

  ph = material_phaseID(co,ce)
  en = material_phaseEntry(co,ce)
  T = matmul(material_orientation0(co,ph,en)%asMatrix(),transpose(math_inv33(phase_F(co,ce)))) ! ToDo: initial orientation correct?

  crystallite_push33ToRef = matmul(transpose(T),matmul(tensor33,T))

end function crystallite_push33ToRef


!--------------------------------------------------------------------------------------------------
!> @brief determines whether a point is converged
!--------------------------------------------------------------------------------------------------
logical pure function converged(residuum,state,atol)

  real(pReal), intent(in), dimension(:) ::&
    residuum, state, atol
  real(pReal) :: &
    rTol

  rTol = num%rTol_crystalliteState

  converged = all(abs(residuum) <= max(atol, rtol*abs(state)))

end function converged


!--------------------------------------------------------------------------------------------------
!> @brief Write restart data to file.
!--------------------------------------------------------------------------------------------------
subroutine phase_restartWrite(fileHandle)

  integer(HID_T), intent(in) :: fileHandle

  integer(HID_T), dimension(2) :: groupHandle
  integer :: ph


  groupHandle(1) = HDF5_addGroup(fileHandle,'phase')

  do ph = 1, size(material_name_phase)

    groupHandle(2) = HDF5_addGroup(groupHandle(1),material_name_phase(ph))

    call mechanical_restartWrite(groupHandle(2),ph)

    call HDF5_closeGroup(groupHandle(2))

  enddo

  call HDF5_closeGroup(groupHandle(1))

end subroutine phase_restartWrite


!--------------------------------------------------------------------------------------------------
!> @brief Read restart data from file.
!--------------------------------------------------------------------------------------------------
subroutine phase_restartRead(fileHandle)

  integer(HID_T), intent(in) :: fileHandle

  integer(HID_T), dimension(2) :: groupHandle
  integer :: ph


  groupHandle(1) = HDF5_openGroup(fileHandle,'phase')

  do ph = 1, size(material_name_phase)

    groupHandle(2) = HDF5_openGroup(groupHandle(1),material_name_phase(ph))

    call mechanical_restartRead(groupHandle(2),ph)

    call HDF5_closeGroup(groupHandle(2))

  enddo

  call HDF5_closeGroup(groupHandle(1))

end subroutine phase_restartRead


end module phase
