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
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Christoph Kords, Max-Planck-Institut für Eisenforschung GmbH
!> @author Chen Zhang, Michigan State University
!> @brief crystallite state integration functions and reporting of results
!--------------------------------------------------------------------------------------------------

module crystallite
  use prec
  use IO
  use HDF5_utilities
  use DAMASK_interface
  use config
  use rotations
  use math
  use FEsolving
  use material
  use constitutive
  use discretization
  use lattice
  use results

  implicit none
  private

  real(pReal),               dimension(:,:,:),        allocatable, public :: &
    crystallite_dt                                                                                  !< requested time increment of each grain
  real(pReal),               dimension(:,:,:),        allocatable :: &
    crystallite_subdt, &                                                                            !< substepped time increment of each grain
    crystallite_subFrac, &                                                                          !< already calculated fraction of increment
    crystallite_subStep                                                                             !< size of next integration step
  type(rotation),            dimension(:,:,:),        allocatable :: &
    crystallite_orientation                                                                         !< current orientation
  real(pReal),               dimension(:,:,:,:,:),    allocatable, public, protected :: &
    crystallite_Fe, &                                                                               !< current "elastic" def grad (end of converged time step)
    crystallite_P, &                                                                                !< 1st Piola-Kirchhoff stress per grain
    crystallite_S0, &                                                                               !< 2nd Piola-Kirchhoff stress vector at start of FE inc
    crystallite_Fp0, &                                                                              !< plastic def grad at start of FE inc
    crystallite_Fi0, &                                                                              !< intermediate def grad at start of FE inc
    crystallite_F0, &                                                                               !< def grad at start of FE inc
    crystallite_Lp0, &                                                                              !< plastic velocitiy grad at start of FE inc
    crystallite_Li0                                                                                 !< intermediate velocitiy grad at start of FE inc
  real(pReal),               dimension(:,:,:,:,:),    allocatable, public :: &
    crystallite_S, &                                                                                !< current 2nd Piola-Kirchhoff stress vector (end of converged time step)
    crystallite_partionedS0, &                                                                      !< 2nd Piola-Kirchhoff stress vector at start of homog inc
    crystallite_Fp, &                                                                               !< current plastic def grad (end of converged time step)
    crystallite_partionedFp0,&                                                                      !< plastic def grad at start of homog inc
    crystallite_Fi, &                                                                               !< current intermediate def grad (end of converged time step)
    crystallite_partionedFi0,&                                                                      !< intermediate def grad at start of homog inc
    crystallite_partionedF,  &                                                                      !< def grad to be reached at end of homog inc
    crystallite_partionedF0, &                                                                      !< def grad at start of homog inc
    crystallite_Lp, &                                                                               !< current plastic velocitiy grad (end of converged time step)
    crystallite_partionedLp0, &                                                                     !< plastic velocity grad at start of homog inc
    crystallite_Li, &                                                                               !< current intermediate velocitiy grad (end of converged time step)
    crystallite_partionedLi0                                                                        !< intermediate velocity grad at start of homog inc
  real(pReal),                dimension(:,:,:,:,:),    allocatable :: &
    crystallite_subFp0,&                                                                            !< plastic def grad at start of crystallite inc
    crystallite_subFi0,&                                                                            !< intermediate def grad at start of crystallite inc
    crystallite_subF,  &                                                                            !< def grad to be reached at end of crystallite inc
    crystallite_subF0, &                                                                            !< def grad at start of crystallite inc
    crystallite_subLp0,&                                                                            !< plastic velocity grad at start of crystallite inc
    crystallite_subLi0                                                                              !< intermediate velocity grad at start of crystallite inc
  real(pReal),                dimension(:,:,:,:,:,:,:), allocatable, public, protected :: &
    crystallite_dPdF                                                                                !< current individual dPdF per grain (end of converged time step)
  logical,                    dimension(:,:,:),         allocatable, public :: &
    crystallite_requested                                                                           !< used by upper level (homogenization) to request crystallite calculation
  logical,                    dimension(:,:,:),         allocatable :: &
    crystallite_converged                                                                           !< convergence flag

  type :: tOutput                                                                                   !< new requested output (per phase)
    character(len=pStringLen), allocatable, dimension(:) :: &
      label
  end type tOutput
  type(tOutput), allocatable, dimension(:) :: output_constituent

  type :: tNumerics
    integer :: &
      iJacoLpresiduum, &                                                                            !< frequency of Jacobian update of residuum in Lp
      nState, &                                                                                     !< state loop limit
      nStress                                                                                       !< stress loop limit
    character(len=:), allocatable :: &
      integrator                                                                                    !< integration scheme
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

  type(tDebugOptions) :: debugCrystallite

  procedure(integrateStateFPI), pointer :: integrateState

  public :: &
    crystallite_init, &
    crystallite_stress, &
    crystallite_stressTangent, &
    crystallite_orientations, &
    crystallite_push33ToRef, &
    crystallite_results, &
    crystallite_restartWrite, &
    crystallite_restartRead, &
    crystallite_forward

contains


!--------------------------------------------------------------------------------------------------
!> @brief allocates and initialize per grain variables
!--------------------------------------------------------------------------------------------------
subroutine crystallite_init

  logical, dimension(discretization_nIP,discretization_nElem) :: devNull
  integer :: &
    c, &                                                                                            !< counter in integration point component loop
    i, &                                                                                            !< counter in integration point loop
    e, &                                                                                            !< counter in element loop
    cMax, &                                                                                         !< maximum number of  integration point components
    iMax, &                                                                                         !< maximum number of integration points
    eMax, &                                                                                         !< maximum number of elements
    myNcomponents                                                                                   !< number of components at current IP

  class(tNode), pointer :: &
    num_crystallite, &
    debug_crystallite, &                                                                            ! pointer to debug options for crystallite
    phases, &
    phase, &
    generic_param

  write(6,'(/,a)')   ' <<<+-  crystallite init  -+>>>'

  debug_crystallite => debug_root%get('crystallite', defaultVal=emptyList)
  debugCrystallite%basic     = debug_crystallite%contains('basic')
  debugCrystallite%extensive = debug_crystallite%contains('extensive')
  debugCrystallite%selective = debug_crystallite%contains('selective')
  debugCrystallite%element   = debug_root%get_asInt('element', defaultVal=1)
  debugCrystallite%ip        = debug_root%get_asInt('integrationpoint', defaultVal=1)
  debugCrystallite%grain     = debug_root%get_asInt('grain', defaultVal=1)

  cMax = homogenization_maxNgrains
  iMax = discretization_nIP
  eMax = discretization_nElem

  allocate(crystallite_partionedF(3,3,cMax,iMax,eMax),source=0.0_pReal)

  allocate(crystallite_S0, &
           crystallite_F0, crystallite_Fi0,crystallite_Fp0, &
                           crystallite_Li0,crystallite_Lp0, &
           crystallite_partionedS0, &
           crystallite_partionedF0,crystallite_partionedFp0,crystallite_partionedFi0, &
                                   crystallite_partionedLp0,crystallite_partionedLi0, &
           crystallite_S,crystallite_P, &
           crystallite_Fe,crystallite_Fi,crystallite_Fp, &
                          crystallite_Li,crystallite_Lp, &
           crystallite_subF,crystallite_subF0, &
           crystallite_subFp0,crystallite_subFi0, &
           crystallite_subLi0,crystallite_subLp0, &
           source = crystallite_partionedF)

  allocate(crystallite_dPdF(3,3,3,3,cMax,iMax,eMax),source=0.0_pReal)

  allocate(crystallite_dt(cMax,iMax,eMax),source=0.0_pReal)
  allocate(crystallite_subdt,crystallite_subFrac,crystallite_subStep, &
           source = crystallite_dt)

  allocate(crystallite_orientation(cMax,iMax,eMax))

  allocate(crystallite_requested(cMax,iMax,eMax),             source=.false.)
  allocate(crystallite_converged(cMax,iMax,eMax),             source=.true.)

  num_crystallite => numerics_root%get('crystallite',defaultVal=emptyDict)

  num%subStepMinCryst        = num_crystallite%get_asFloat ('subStepMin',       defaultVal=1.0e-3_pReal)
  num%subStepSizeCryst       = num_crystallite%get_asFloat ('subStepSize',      defaultVal=0.25_pReal)
  num%stepIncreaseCryst      = num_crystallite%get_asFloat ('stepIncrease',     defaultVal=1.5_pReal)
  num%subStepSizeLp          = num_crystallite%get_asFloat ('subStepSizeLp',    defaultVal=0.5_pReal)
  num%subStepSizeLi          = num_crystallite%get_asFloat ('subStepSizeLi',    defaultVal=0.5_pReal)
  num%rtol_crystalliteState  = num_crystallite%get_asFloat ('rtol_State',       defaultVal=1.0e-6_pReal)
  num%rtol_crystalliteStress = num_crystallite%get_asFloat ('rtol_Stress',      defaultVal=1.0e-6_pReal)
  num%atol_crystalliteStress = num_crystallite%get_asFloat ('atol_Stress',      defaultVal=1.0e-8_pReal)
  num%iJacoLpresiduum        = num_crystallite%get_asInt   ('iJacoLpresiduum',  defaultVal=1)
  num%integrator             = num_crystallite%get_asString('integrator',       defaultVal='FPI')
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


  select case(num%integrator)
    case('FPI')
      integrateState => integrateStateFPI
    case('Euler')
      integrateState => integrateStateEuler
    case('AdaptiveEuler')
      integrateState => integrateStateAdaptiveEuler
    case('RK4')
      integrateState => integrateStateRK4
    case('RKCK45')
      integrateState => integrateStateRKCK45
    case default
     call IO_error(301,ext_msg='integrator')
  end select

  phases => material_root%get('phase')

  allocate(output_constituent(phases%length))
  do c = 1, phases%length
    phase => phases%get(c)
    generic_param  => phase%get('generic',defaultVal = emptyDict) 
#if defined(__GFORTRAN__)
    output_constituent(c)%label  = output_asStrings(generic_param)
#else
    output_constituent(c)%label  = generic_param%get_asStrings('output',defaultVal=emptyStringArray)
#endif
  enddo


!--------------------------------------------------------------------------------------------------
! initialize
 !$OMP PARALLEL DO PRIVATE(myNcomponents,i,c)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    myNcomponents = homogenization_Ngrains(material_homogenizationAt(e))
    do i = FEsolving_execIP(1), FEsolving_execIP(2); do c = 1, myNcomponents
      crystallite_Fp0(1:3,1:3,c,i,e) = material_orientation0(c,i,e)%asMatrix()                      ! plastic def gradient reflects init orientation
      crystallite_Fp0(1:3,1:3,c,i,e) = crystallite_Fp0(1:3,1:3,c,i,e) &
                                     / math_det33(crystallite_Fp0(1:3,1:3,c,i,e))**(1.0_pReal/3.0_pReal)
      crystallite_Fi0(1:3,1:3,c,i,e) = constitutive_initialFi(c,i,e)
      crystallite_F0(1:3,1:3,c,i,e)  = math_I3
      crystallite_Fe(1:3,1:3,c,i,e)  = math_inv33(matmul(crystallite_Fi0(1:3,1:3,c,i,e), &
                                                         crystallite_Fp0(1:3,1:3,c,i,e)))           ! assuming that euler angles are given in internal strain free configuration
      crystallite_Fp(1:3,1:3,c,i,e)  = crystallite_Fp0(1:3,1:3,c,i,e)
      crystallite_Fi(1:3,1:3,c,i,e)  = crystallite_Fi0(1:3,1:3,c,i,e)
      crystallite_requested(c,i,e) = .true.
    enddo; enddo
  enddo
  !$OMP END PARALLEL DO


  crystallite_partionedFp0 = crystallite_Fp0
  crystallite_partionedFi0 = crystallite_Fi0
  crystallite_partionedF0  = crystallite_F0
  crystallite_partionedF   = crystallite_F0

  call crystallite_orientations()

  !$OMP PARALLEL DO
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do c = 1,homogenization_Ngrains(material_homogenizationAt(e))
        call constitutive_dependentState(crystallite_partionedF0(1:3,1:3,c,i,e), &
                                         crystallite_partionedFp0(1:3,1:3,c,i,e), &
                                         c,i,e)                                                     ! update dependent state variables to be consistent with basic states
     enddo
    enddo
  enddo
  !$OMP END PARALLEL DO

  devNull = crystallite_stress()
  call crystallite_stressTangent

#ifdef DEBUG
  if (debugCrystallite%basic) then
    write(6,'(a42,1x,i10)') '    # of elements:                       ', eMax
    write(6,'(a42,1x,i10)') '    # of integration points/element:     ', iMax
    write(6,'(a42,1x,i10)') 'max # of constituents/integration point: ', cMax
    flush(6)
  endif

#endif

end subroutine crystallite_init


!--------------------------------------------------------------------------------------------------
!> @brief calculate stress (P)
!--------------------------------------------------------------------------------------------------
function crystallite_stress()

  logical, dimension(discretization_nIP,discretization_nElem) :: crystallite_stress
  real(pReal) :: &
    formerSubStep
  integer :: &
    NiterationCrystallite, &                                                                        ! number of iterations in crystallite loop
    c, &                                                                                            !< counter in integration point component loop
    i, &                                                                                            !< counter in integration point loop
    e, &                                                                                            !< counter in element loop
    startIP, endIP, &
    s
  logical, dimension(homogenization_maxNgrains,discretization_nIP,discretization_nElem) :: todo     !ToDo: need to set some values to false for different Ngrains
    todo = .false.

#ifdef DEBUG
  if (debugCrystallite%selective &
      .and. FEsolving_execElem(1) <= debugCrystallite%element &
      .and.                          debugCrystallite%element <= FEsolving_execElem(2)) then
      write(6,'(/,a,i8,1x,i2,1x,i3)')    '<< CRYST stress >> boundary and initial values at el ip ipc ', &
        debugCrystallite%element,debugCrystallite%ip, debugCrystallite%grain
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> F  ', &
                                          transpose(crystallite_partionedF(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> F0 ', &
                                          transpose(crystallite_partionedF0(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> Fp0', &
                                          transpose(crystallite_partionedFp0(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> Fi0', &
                                          transpose(crystallite_partionedFi0(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> Lp0', &
                                          transpose(crystallite_partionedLp0(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
    write(6,'(a,/,3(12x,3(f14.9,1x)/))') '<< CRYST stress >> Li0', &
                                          transpose(crystallite_partionedLi0(1:3,1:3,debugCrystallite%grain, &
                                                      debugCrystallite%ip,debugCrystallite%element))
  endif
#endif

!--------------------------------------------------------------------------------------------------
! initialize to starting condition
  crystallite_subStep = 0.0_pReal
  !$OMP PARALLEL DO
  elementLooping1: do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2); do c = 1,homogenization_Ngrains(material_homogenizationAt(e))
      homogenizationRequestsCalculation: if (crystallite_requested(c,i,e)) then
        plasticState    (material_phaseAt(c,e))%subState0(      :,material_phaseMemberAt(c,i,e)) = &
        plasticState    (material_phaseAt(c,e))%partionedState0(:,material_phaseMemberAt(c,i,e))

        do s = 1, phase_Nsources(material_phaseAt(c,e))
          sourceState(material_phaseAt(c,e))%p(s)%subState0(      :,material_phaseMemberAt(c,i,e)) = &
          sourceState(material_phaseAt(c,e))%p(s)%partionedState0(:,material_phaseMemberAt(c,i,e))
        enddo
        crystallite_subFp0(1:3,1:3,c,i,e) = crystallite_partionedFp0(1:3,1:3,c,i,e)
        crystallite_subLp0(1:3,1:3,c,i,e) = crystallite_partionedLp0(1:3,1:3,c,i,e)
        crystallite_subFi0(1:3,1:3,c,i,e) = crystallite_partionedFi0(1:3,1:3,c,i,e)
        crystallite_subLi0(1:3,1:3,c,i,e) = crystallite_partionedLi0(1:3,1:3,c,i,e)
        crystallite_subF0(1:3,1:3,c,i,e)  = crystallite_partionedF0(1:3,1:3,c,i,e)
        crystallite_subFrac(c,i,e) = 0.0_pReal
        crystallite_subStep(c,i,e) = 1.0_pReal/num%subStepSizeCryst
        todo(c,i,e) = .true.
        crystallite_converged(c,i,e) = .false.                                                      ! pretend failed step of 1/subStepSizeCryst
      endif homogenizationRequestsCalculation
    enddo; enddo
  enddo elementLooping1
  !$OMP END PARALLEL DO

  singleRun: if (FEsolving_execELem(1) == FEsolving_execElem(2) .and. &
                 FEsolving_execIP  (1) == FEsolving_execIP  (2)) then
    startIP = FEsolving_execIP(1)
    endIP   = startIP
  else singleRun
    startIP = 1
    endIP   = discretization_nIP
  endif singleRun

  NiterationCrystallite = 0
  cutbackLooping: do while (any(todo(:,startIP:endIP,FEsolving_execELem(1):FEsolving_execElem(2))))
    NiterationCrystallite = NiterationCrystallite + 1

#ifdef DEBUG
    if (debugCrystallite%extensive) &
      write(6,'(a,i6)') '<< CRYST stress >> crystallite iteration ',NiterationCrystallite
#endif
    !$OMP PARALLEL DO PRIVATE(formerSubStep)
    elementLooping3: do e = FEsolving_execElem(1),FEsolving_execElem(2)
      do i = FEsolving_execIP(1),FEsolving_execIP(2)
        do c = 1,homogenization_Ngrains(material_homogenizationAt(e))
!--------------------------------------------------------------------------------------------------
!  wind forward
          if (crystallite_converged(c,i,e)) then
            formerSubStep = crystallite_subStep(c,i,e)
            crystallite_subFrac(c,i,e) = crystallite_subFrac(c,i,e) + crystallite_subStep(c,i,e)
            crystallite_subStep(c,i,e) = min(1.0_pReal - crystallite_subFrac(c,i,e), &
                                             num%stepIncreaseCryst * crystallite_subStep(c,i,e))

            todo(c,i,e) = crystallite_subStep(c,i,e) > 0.0_pReal                        ! still time left to integrate on?
            if (todo(c,i,e)) then
              crystallite_subF0 (1:3,1:3,c,i,e) = crystallite_subF(1:3,1:3,c,i,e)
              crystallite_subLp0(1:3,1:3,c,i,e) = crystallite_Lp  (1:3,1:3,c,i,e)
              crystallite_subLi0(1:3,1:3,c,i,e) = crystallite_Li  (1:3,1:3,c,i,e)
              crystallite_subFp0(1:3,1:3,c,i,e) = crystallite_Fp  (1:3,1:3,c,i,e)
              crystallite_subFi0(1:3,1:3,c,i,e) = crystallite_Fi  (1:3,1:3,c,i,e)
              !if abbrevation, make c and p private in omp
              plasticState(    material_phaseAt(c,e))%subState0(:,material_phaseMemberAt(c,i,e)) &
                = plasticState(material_phaseAt(c,e))%state(    :,material_phaseMemberAt(c,i,e))
              do s = 1, phase_Nsources(material_phaseAt(c,e))
                sourceState(    material_phaseAt(c,e))%p(s)%subState0(:,material_phaseMemberAt(c,i,e)) &
                  = sourceState(material_phaseAt(c,e))%p(s)%state(    :,material_phaseMemberAt(c,i,e))
              enddo
            endif

!--------------------------------------------------------------------------------------------------
!  cut back (reduced time and restore)
          else
            crystallite_subStep(c,i,e)       = num%subStepSizeCryst * crystallite_subStep(c,i,e)
            crystallite_Fp   (1:3,1:3,c,i,e) =            crystallite_subFp0(1:3,1:3,c,i,e)
            crystallite_Fi   (1:3,1:3,c,i,e) =            crystallite_subFi0(1:3,1:3,c,i,e)
            crystallite_S    (1:3,1:3,c,i,e) =            crystallite_S0    (1:3,1:3,c,i,e)
            if (crystallite_subStep(c,i,e) < 1.0_pReal) then                                        ! actual (not initial) cutback
              crystallite_Lp (1:3,1:3,c,i,e) =            crystallite_subLp0(1:3,1:3,c,i,e)
              crystallite_Li (1:3,1:3,c,i,e) =            crystallite_subLi0(1:3,1:3,c,i,e)
            endif
            plasticState    (material_phaseAt(c,e))%state(    :,material_phaseMemberAt(c,i,e)) &
              = plasticState(material_phaseAt(c,e))%subState0(:,material_phaseMemberAt(c,i,e))
            do s = 1, phase_Nsources(material_phaseAt(c,e))
              sourceState(    material_phaseAt(c,e))%p(s)%state(    :,material_phaseMemberAt(c,i,e)) &
                = sourceState(material_phaseAt(c,e))%p(s)%subState0(:,material_phaseMemberAt(c,i,e))
            enddo

                                                                                                    ! cant restore dotState here, since not yet calculated in first cutback after initialization
            todo(c,i,e) = crystallite_subStep(c,i,e) > num%subStepMinCryst                          ! still on track or already done (beyond repair)
          endif

!--------------------------------------------------------------------------------------------------
!  prepare for integration
          if (todo(c,i,e)) then
            crystallite_subF(1:3,1:3,c,i,e) = crystallite_subF0(1:3,1:3,c,i,e) &
                                            + crystallite_subStep(c,i,e) *( crystallite_partionedF (1:3,1:3,c,i,e) &
                                                                           -crystallite_partionedF0(1:3,1:3,c,i,e))
            crystallite_Fe(1:3,1:3,c,i,e) = matmul(matmul(crystallite_subF(1:3,1:3,c,i,e), &
                                                          math_inv33(crystallite_Fp(1:3,1:3,c,i,e))), &
                                                   math_inv33(crystallite_Fi(1:3,1:3,c,i,e)))
            crystallite_subdt(c,i,e) = crystallite_subStep(c,i,e) * crystallite_dt(c,i,e)
            crystallite_converged(c,i,e) = .false.
          endif

        enddo
      enddo
    enddo elementLooping3
    !$OMP END PARALLEL DO

!--------------------------------------------------------------------------------------------------
!  integrate --- requires fully defined state array (basic + dependent state)
    if (any(todo)) call integrateState(todo)                                                              ! TODO: unroll into proper elementloop to avoid N^2 for single point evaluation
    where(.not. crystallite_converged .and. crystallite_subStep > num%subStepMinCryst) &            ! do not try non-converged but fully cutbacked any further
      todo = .true.                                                                                 ! TODO: again unroll this into proper elementloop to avoid N^2 for single point evaluation


  enddo cutbackLooping

! return whether converged or not
  crystallite_stress = .false.
  elementLooping5: do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      crystallite_stress(i,e) = all(crystallite_converged(:,i,e))
    enddo
  enddo elementLooping5

end function crystallite_stress


!--------------------------------------------------------------------------------------------------
!> @brief calculate tangent (dPdF)
!--------------------------------------------------------------------------------------------------
subroutine crystallite_stressTangent

  integer :: &
    c, &                                                                                            !< counter in integration point component loop
    i, &                                                                                            !< counter in integration point loop
    e, &                                                                                            !< counter in element loop
    o, &
    p

  real(pReal), dimension(3,3)     ::   devNull, &
                                       invSubFp0,invSubFi0,invFp,invFi, &
                                       temp_33_1, temp_33_2, temp_33_3, temp_33_4
  real(pReal), dimension(3,3,3,3) ::   dSdFe, &
                                       dSdF, &
                                       dSdFi, &
                                       dLidS, &                                                     ! tangent in lattice configuration
                                       dLidFi, &
                                       dLpdS, &
                                       dLpdFi, &
                                       dFidS, &
                                       dFpinvdF, &
                                       rhs_3333, &
                                       lhs_3333, &
                                       temp_3333
  real(pReal), dimension(9,9)::        temp_99
  logical :: error

  !$OMP PARALLEL DO PRIVATE(dSdF,dSdFe,dSdFi,dLpdS,dLpdFi,dFpinvdF,dLidS,dLidFi,dFidS,o,p, &
  !$OMP                     invSubFp0,invSubFi0,invFp,invFi, &
  !$OMP                     rhs_3333,lhs_3333,temp_99,temp_33_1,temp_33_2,temp_33_3,temp_33_4,temp_3333,error)
  elementLooping: do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do c = 1,homogenization_Ngrains(material_homogenizationAt(e))

        call constitutive_SandItsTangents(devNull,dSdFe,dSdFi, &
                                         crystallite_Fe(1:3,1:3,c,i,e), &
                                         crystallite_Fi(1:3,1:3,c,i,e),c,i,e)
        call constitutive_LiAndItsTangents(devNull,dLidS,dLidFi, &
                                           crystallite_S (1:3,1:3,c,i,e), &
                                           crystallite_Fi(1:3,1:3,c,i,e), &
                                           c,i,e)

        invFp = math_inv33(crystallite_Fp(1:3,1:3,c,i,e))
        invFi = math_inv33(crystallite_Fi(1:3,1:3,c,i,e))
        invSubFp0 = math_inv33(crystallite_subFp0(1:3,1:3,c,i,e))
        invSubFi0 = math_inv33(crystallite_subFi0(1:3,1:3,c,i,e))

        if (sum(abs(dLidS)) < tol_math_check) then
          dFidS = 0.0_pReal
        else
          lhs_3333 = 0.0_pReal; rhs_3333 = 0.0_pReal
          do o=1,3; do p=1,3
            lhs_3333(1:3,1:3,o,p) = lhs_3333(1:3,1:3,o,p) &
                                  + crystallite_subdt(c,i,e)*matmul(invSubFi0,dLidFi(1:3,1:3,o,p))
            lhs_3333(1:3,o,1:3,p) = lhs_3333(1:3,o,1:3,p) &
                                  + invFi*invFi(p,o)
            rhs_3333(1:3,1:3,o,p) = rhs_3333(1:3,1:3,o,p) &
                                  - crystallite_subdt(c,i,e)*matmul(invSubFi0,dLidS(1:3,1:3,o,p))
          enddo; enddo
          call math_invert(temp_99,error,math_3333to99(lhs_3333))
          if (error) then
            call IO_warning(warning_ID=600,el=e,ip=i,g=c, &
                            ext_msg='inversion error in analytic tangent calculation')
            dFidS = 0.0_pReal
          else
            dFidS = math_mul3333xx3333(math_99to3333(temp_99),rhs_3333)
          endif
          dLidS = math_mul3333xx3333(dLidFi,dFidS) + dLidS
        endif

        call constitutive_LpAndItsTangents(devNull,dLpdS,dLpdFi, &
                                           crystallite_S (1:3,1:3,c,i,e), &
                                           crystallite_Fi(1:3,1:3,c,i,e),c,i,e)                     ! call constitutive law to calculate Lp tangent in lattice configuration
        dLpdS = math_mul3333xx3333(dLpdFi,dFidS) + dLpdS

!--------------------------------------------------------------------------------------------------
! calculate dSdF
        temp_33_1 = transpose(matmul(invFp,invFi))
        temp_33_2 = matmul(crystallite_subF(1:3,1:3,c,i,e),invSubFp0)
        temp_33_3 = matmul(matmul(crystallite_subF(1:3,1:3,c,i,e),invFp), invSubFi0)

        do o=1,3; do p=1,3
          rhs_3333(p,o,1:3,1:3)  = matmul(dSdFe(p,o,1:3,1:3),temp_33_1)
          temp_3333(1:3,1:3,p,o) = matmul(matmul(temp_33_2,dLpdS(1:3,1:3,p,o)), invFi) &
                                 + matmul(temp_33_3,dLidS(1:3,1:3,p,o))
        enddo; enddo
        lhs_3333 = crystallite_subdt(c,i,e)*math_mul3333xx3333(dSdFe,temp_3333) &
                 + math_mul3333xx3333(dSdFi,dFidS)

        call math_invert(temp_99,error,math_identity2nd(9)+math_3333to99(lhs_3333))
        if (error) then
          call IO_warning(warning_ID=600,el=e,ip=i,g=c, &
                          ext_msg='inversion error in analytic tangent calculation')
          dSdF = rhs_3333
        else
          dSdF = math_mul3333xx3333(math_99to3333(temp_99),rhs_3333)
        endif

!--------------------------------------------------------------------------------------------------
! calculate dFpinvdF
        temp_3333 = math_mul3333xx3333(dLpdS,dSdF)
        do o=1,3; do p=1,3
          dFpinvdF(1:3,1:3,p,o) = -crystallite_subdt(c,i,e) &
                                * matmul(invSubFp0, matmul(temp_3333(1:3,1:3,p,o),invFi))
        enddo; enddo

!--------------------------------------------------------------------------------------------------
! assemble dPdF
        temp_33_1 = matmul(crystallite_S(1:3,1:3,c,i,e),transpose(invFp))
        temp_33_2 = matmul(invFp,temp_33_1)
        temp_33_3 = matmul(crystallite_subF(1:3,1:3,c,i,e),invFp)
        temp_33_4 = matmul(temp_33_3,crystallite_S(1:3,1:3,c,i,e))

        crystallite_dPdF(1:3,1:3,1:3,1:3,c,i,e) = 0.0_pReal
        do p=1,3
          crystallite_dPdF(p,1:3,p,1:3,c,i,e) = transpose(temp_33_2)
        enddo
        do o=1,3; do p=1,3
          crystallite_dPdF(1:3,1:3,p,o,c,i,e) = crystallite_dPdF(1:3,1:3,p,o,c,i,e) &
                                              + matmul(matmul(crystallite_subF(1:3,1:3,c,i,e), &
                                                       dFpinvdF(1:3,1:3,p,o)),temp_33_1) &
                                              + matmul(matmul(temp_33_3,dSdF(1:3,1:3,p,o)), &
                                                       transpose(invFp)) &
                                              + matmul(temp_33_4,transpose(dFpinvdF(1:3,1:3,p,o)))
        enddo; enddo

    enddo; enddo
  enddo elementLooping
  !$OMP END PARALLEL DO

end subroutine crystallite_stressTangent


!--------------------------------------------------------------------------------------------------
!> @brief calculates orientations
!--------------------------------------------------------------------------------------------------
subroutine crystallite_orientations

  integer &
    c, &                                                                                            !< counter in integration point component loop
    i, &                                                                                            !< counter in integration point loop
    e                                                                                               !< counter in element loop

  !$OMP PARALLEL DO
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do c = 1,homogenization_Ngrains(material_homogenizationAt(e))
        call crystallite_orientation(c,i,e)%fromMatrix(transpose(math_rotationalPart(crystallite_Fe(1:3,1:3,c,i,e))))
  enddo; enddo; enddo
  !$OMP END PARALLEL DO

  nonlocalPresent: if (any(plasticState%nonlocal)) then
    !$OMP PARALLEL DO
    do e = FEsolving_execElem(1),FEsolving_execElem(2)
      if (plasticState(material_phaseAt(1,e))%nonlocal) then
        do i = FEsolving_execIP(1),FEsolving_execIP(2)
          call plastic_nonlocal_updateCompatibility(crystallite_orientation, &
                                                    phase_plasticityInstance(material_phaseAt(1,e)),i,e)
        enddo
      endif
    enddo
    !$OMP END PARALLEL DO
  endif nonlocalPresent

end subroutine crystallite_orientations


!--------------------------------------------------------------------------------------------------
!> @brief Map 2nd order tensor to reference config
!--------------------------------------------------------------------------------------------------
function crystallite_push33ToRef(ipc,ip,el, tensor33)

  real(pReal), dimension(3,3) :: crystallite_push33ToRef
  real(pReal), dimension(3,3), intent(in) :: tensor33
  real(pReal), dimension(3,3)             :: T
  integer, intent(in):: &
    el, &
    ip, &
    ipc

  T = matmul(material_orientation0(ipc,ip,el)%asMatrix(), &                                         ! ToDo: initial orientation correct?
             transpose(math_inv33(crystallite_subF(1:3,1:3,ipc,ip,el))))
  crystallite_push33ToRef = matmul(transpose(T),matmul(tensor33,T))

end function crystallite_push33ToRef


!--------------------------------------------------------------------------------------------------
!> @brief writes crystallite results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine crystallite_results

  integer :: p,o
  real(pReal),    allocatable, dimension(:,:,:) :: selected_tensors
  type(rotation), allocatable, dimension(:)     :: selected_rotations
  character(len=pStringLen)                     :: group,structureLabel

  do p=1,size(material_name_phase)
    group = trim('current/constituent')//'/'//trim(material_name_phase(p))//'/generic'

    call results_closeGroup(results_addGroup(group))

    do o = 1, size(output_constituent(p)%label)
      select case (output_constituent(p)%label(o))
        case('F')
          selected_tensors = select_tensors(crystallite_partionedF,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'deformation gradient','1')
        case('Fe')
          selected_tensors = select_tensors(crystallite_Fe,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'elastic deformation gradient','1')
        case('Fp')
          selected_tensors = select_tensors(crystallite_Fp,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'plastic deformation gradient','1')
        case('Fi')
          selected_tensors = select_tensors(crystallite_Fi,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'inelastic deformation gradient','1')
        case('Lp')
          selected_tensors = select_tensors(crystallite_Lp,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'plastic velocity gradient','1/s')
        case('Li')
          selected_tensors = select_tensors(crystallite_Li,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'inelastic velocity gradient','1/s')
        case('P')
          selected_tensors = select_tensors(crystallite_P,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'First Piola-Kirchoff stress','Pa')
        case('S')
          selected_tensors = select_tensors(crystallite_S,p)
          call results_writeDataset(group,selected_tensors,output_constituent(p)%label(o),&
                                   'Second Piola-Kirchoff stress','Pa')
        case('O')
          select case(lattice_structure(p))
            case(lattice_ISO_ID)
              structureLabel = 'iso'
            case(lattice_FCC_ID)
              structureLabel = 'fcc'
            case(lattice_BCC_ID)
              structureLabel = 'bcc'
            case(lattice_BCT_ID)
              structureLabel = 'bct'
            case(lattice_HEX_ID)
              structureLabel = 'hex'
            case(lattice_ORT_ID)
              structureLabel = 'ort'
          end select
          selected_rotations = select_rotations(crystallite_orientation,p)
          call results_writeDataset(group,selected_rotations,output_constituent(p)%label(o),&
                                   'crystal orientation as quaternion',structureLabel)
      end select
    enddo
  enddo

  contains

  !------------------------------------------------------------------------------------------------
  !> @brief select tensors for output
  !------------------------------------------------------------------------------------------------
  function select_tensors(dataset,instance)

    integer, intent(in) :: instance
    real(pReal), dimension(:,:,:,:,:), intent(in) :: dataset
    real(pReal), allocatable, dimension(:,:,:) :: select_tensors
    integer :: e,i,c,j

    allocate(select_tensors(3,3,count(material_phaseAt==instance)*discretization_nIP))

    j=0
    do e = 1, size(material_phaseAt,2)
      do i = 1, discretization_nIP
        do c = 1, size(material_phaseAt,1)                                                          !ToDo: this needs to be changed for varying Ngrains
          if (material_phaseAt(c,e) == instance) then
            j = j + 1
            select_tensors(1:3,1:3,j) = dataset(1:3,1:3,c,i,e)
          endif
        enddo
      enddo
    enddo

  end function select_tensors


!--------------------------------------------------------------------------------------------------
!> @brief select rotations for output
!--------------------------------------------------------------------------------------------------
  function select_rotations(dataset,instance)

    integer, intent(in) :: instance
    type(rotation), dimension(:,:,:), intent(in) :: dataset
    type(rotation), allocatable, dimension(:) :: select_rotations
    integer :: e,i,c,j

    allocate(select_rotations(count(material_phaseAt==instance)*homogenization_maxNgrains*discretization_nIP))

    j=0
    do e = 1, size(material_phaseAt,2)
      do i = 1, discretization_nIP
        do c = 1, size(material_phaseAt,1)                                                          !ToDo: this needs to be changed for varying Ngrains
           if (material_phaseAt(c,e) == instance) then
             j = j + 1
             select_rotations(j) = dataset(c,i,e)
           endif
        enddo
      enddo
   enddo

 end function select_rotations

end subroutine crystallite_results


!--------------------------------------------------------------------------------------------------
!> @brief calculation of stress (P) with time integration based on a residuum in Lp and
!> intermediate acceleration of the Newton-Raphson correction
!--------------------------------------------------------------------------------------------------
function integrateStress(ipc,ip,el,timeFraction) result(broken)

  integer, intent(in)::         el, &                                                               ! element index
                                      ip, &                                                         ! integration point index
                                      ipc                                                           ! grain index
  real(pReal), optional, intent(in) :: timeFraction                                                 ! fraction of timestep

  real(pReal), dimension(3,3)::       F, &                                                          ! deformation gradient at end of timestep
                                      Fp_new, &                                                     ! plastic deformation gradient at end of timestep
                                      invFp_new, &                                                  ! inverse of Fp_new
                                      invFp_current, &                                              ! inverse of Fp_current
                                      Lpguess, &                                                    ! current guess for plastic velocity gradient
                                      Lpguess_old, &                                                ! known last good guess for plastic velocity gradient
                                      Lp_constitutive, &                                            ! plastic velocity gradient resulting from constitutive law
                                      residuumLp, &                                                 ! current residuum of plastic velocity gradient
                                      residuumLp_old, &                                             ! last residuum of plastic velocity gradient
                                      deltaLp, &                                                    ! direction of next guess
                                      Fi_new, &                                                     ! gradient of intermediate deformation stages
                                      invFi_new, &
                                      invFi_current, &                                              ! inverse of Fi_current
                                      Liguess, &                                                    ! current guess for intermediate velocity gradient
                                      Liguess_old, &                                                ! known last good guess for intermediate velocity gradient
                                      Li_constitutive, &                                            ! intermediate velocity gradient resulting from constitutive law
                                      residuumLi, &                                                 ! current residuum of intermediate velocity gradient
                                      residuumLi_old, &                                             ! last residuum of intermediate velocity gradient
                                      deltaLi, &                                                    ! direction of next guess
                                      Fe, &                                                         ! elastic deformation gradient
                                      S, &                                                          ! 2nd Piola-Kirchhoff Stress in plastic (lattice) configuration
                                      A, &
                                      B, &
                                      temp_33
  real(pReal), dimension(9) ::        temp_9                                                        ! needed for matrix inversion by LAPACK
  integer,     dimension(9) ::        devNull_9                                                     ! needed for matrix inversion by LAPACK
  real(pReal), dimension(9,9) ::      dRLp_dLp, &                                                   ! partial derivative of residuum (Jacobian for Newton-Raphson scheme)
                                      dRLi_dLi                                                      ! partial derivative of residuumI (Jacobian for Newton-Raphson scheme)
  real(pReal), dimension(3,3,3,3)::   dS_dFe, &                                                     ! partial derivative of 2nd Piola-Kirchhoff stress
                                      dS_dFi, &
                                      dFe_dLp, &                                                    ! partial derivative of elastic deformation gradient
                                      dFe_dLi, &
                                      dFi_dLi, &
                                      dLp_dFi, &
                                      dLi_dFi, &
                                      dLp_dS, &
                                      dLi_dS
  real(pReal)                         steplengthLp, &
                                      steplengthLi, &
                                      dt, &                                                         ! time increment
                                      atol_Lp, &
                                      atol_Li, &
                                      devNull
  integer                             NiterationStressLp, &                                         ! number of stress integrations
                                      NiterationStressLi, &                                         ! number of inner stress integrations
                                      ierr, &                                                       ! error indicator for LAPACK
                                      o, &
                                      p, &
                                      jacoCounterLp, &
                                      jacoCounterLi                                                 ! counters to check for Jacobian update
  logical :: error,broken

  broken = .true.

  if (present(timeFraction)) then
    dt = crystallite_subdt(ipc,ip,el) * timeFraction
    F  = crystallite_subF0(1:3,1:3,ipc,ip,el) &
       + (crystallite_subF(1:3,1:3,ipc,ip,el) - crystallite_subF0(1:3,1:3,ipc,ip,el)) * timeFraction
  else
    dt = crystallite_subdt(ipc,ip,el)
    F  = crystallite_subF(1:3,1:3,ipc,ip,el)
  endif

  call constitutive_dependentState(crystallite_partionedF(1:3,1:3,ipc,ip,el), &
                                   crystallite_Fp(1:3,1:3,ipc,ip,el),ipc,ip,el)

  Lpguess = crystallite_Lp(1:3,1:3,ipc,ip,el)                                                       ! take as first guess
  Liguess = crystallite_Li(1:3,1:3,ipc,ip,el)                                                       ! take as first guess

  call math_invert33(invFp_current,devNull,error,crystallite_subFp0(1:3,1:3,ipc,ip,el))
  if (error) return ! error
  call math_invert33(invFi_current,devNull,error,crystallite_subFi0(1:3,1:3,ipc,ip,el))
  if (error) return ! error

  A = matmul(F,invFp_current)                                                                       ! intermediate tensor needed later to calculate dFe_dLp

  jacoCounterLi  = 0
  steplengthLi   = 1.0_pReal
  residuumLi_old = 0.0_pReal
  Liguess_old    = Liguess

  NiterationStressLi = 0
  LiLoop: do
    NiterationStressLi = NiterationStressLi + 1
    if (NiterationStressLi>num%nStress) return ! error

    invFi_new = matmul(invFi_current,math_I3 - dt*Liguess)
    Fi_new    = math_inv33(invFi_new)

    jacoCounterLp  = 0
    steplengthLp   = 1.0_pReal
    residuumLp_old = 0.0_pReal
    Lpguess_old    = Lpguess

    NiterationStressLp = 0
    LpLoop: do
      NiterationStressLp = NiterationStressLp + 1
      if (NiterationStressLp>num%nStress) return ! error

      B  = math_I3 - dt*Lpguess
      Fe = matmul(matmul(A,B), invFi_new)
      call constitutive_SandItsTangents(S, dS_dFe, dS_dFi, &
                                        Fe, Fi_new, ipc, ip, el)

      call constitutive_LpAndItsTangents(Lp_constitutive, dLp_dS, dLp_dFi, &
                                         S, Fi_new, ipc, ip, el)

      !* update current residuum and check for convergence of loop
      atol_Lp = max(num%rtol_crystalliteStress * max(norm2(Lpguess),norm2(Lp_constitutive)), &      ! absolute tolerance from largest acceptable relative error
                    num%atol_crystalliteStress)                                                     ! minimum lower cutoff
      residuumLp = Lpguess - Lp_constitutive

      if (any(IEEE_is_NaN(residuumLp))) then
        return ! error
      elseif (norm2(residuumLp) < atol_Lp) then                                                     ! converged if below absolute tolerance
        exit LpLoop
      elseif (NiterationStressLp == 1 .or. norm2(residuumLp) < norm2(residuumLp_old)) then          ! not converged, but improved norm of residuum (always proceed in first iteration)...
        residuumLp_old = residuumLp                                                                 ! ...remember old values and...
        Lpguess_old    = Lpguess
        steplengthLp   = 1.0_pReal                                                                  ! ...proceed with normal step length (calculate new search direction)
      else                                                                                          ! not converged and residuum not improved...
        steplengthLp = num%subStepSizeLp * steplengthLp                                             ! ...try with smaller step length in same direction
        Lpguess      = Lpguess_old &
                     + deltaLp * stepLengthLp
        cycle LpLoop
      endif

      calculateJacobiLi: if (mod(jacoCounterLp, num%iJacoLpresiduum) == 0) then
        jacoCounterLp = jacoCounterLp + 1

        do o=1,3; do p=1,3
          dFe_dLp(o,1:3,p,1:3) = - dt * A(o,p)*transpose(invFi_new)                                 ! dFe_dLp(i,j,k,l) = -dt * A(i,k) invFi(l,j)
        enddo; enddo
        dRLp_dLp = math_identity2nd(9) &
                 - math_3333to99(math_mul3333xx3333(math_mul3333xx3333(dLp_dS,dS_dFe),dFe_dLp))
        temp_9 = math_33to9(residuumLp)
        call dgesv(9,1,dRLp_dLp,9,devNull_9,temp_9,9,ierr)                                          ! solve dRLp/dLp * delta Lp = -res for delta Lp
        if (ierr /= 0) return ! error
        deltaLp = - math_9to33(temp_9)
      endif calculateJacobiLi

      Lpguess = Lpguess &
              + deltaLp * steplengthLp
    enddo LpLoop

    call constitutive_LiAndItsTangents(Li_constitutive, dLi_dS, dLi_dFi, &
                                       S, Fi_new, ipc, ip, el)

    !* update current residuum and check for convergence of loop
    atol_Li = max(num%rtol_crystalliteStress * max(norm2(Liguess),norm2(Li_constitutive)), &        ! absolute tolerance from largest acceptable relative error
                  num%atol_crystalliteStress)                                                       ! minimum lower cutoff
    residuumLi = Liguess - Li_constitutive
    if (any(IEEE_is_NaN(residuumLi))) then
      return ! error
    elseif (norm2(residuumLi) < atol_Li) then                                                       ! converged if below absolute tolerance
      exit LiLoop
    elseif (NiterationStressLi == 1 .or. norm2(residuumLi) < norm2(residuumLi_old)) then            ! not converged, but improved norm of residuum (always proceed in first iteration)...
      residuumLi_old = residuumLi                                                                   ! ...remember old values and...
      Liguess_old    = Liguess
      steplengthLi   = 1.0_pReal                                                                    ! ...proceed with normal step length (calculate new search direction)
    else                                                                                            ! not converged and residuum not improved...
      steplengthLi = num%subStepSizeLi * steplengthLi                                               ! ...try with smaller step length in same direction
      Liguess      = Liguess_old &
                   + deltaLi * steplengthLi
      cycle LiLoop
    endif

    calculateJacobiLp: if (mod(jacoCounterLi, num%iJacoLpresiduum) == 0) then
      jacoCounterLi = jacoCounterLi + 1

      temp_33 = matmul(matmul(A,B),invFi_current)
      do o=1,3; do p=1,3
        dFe_dLi(1:3,o,1:3,p) = -dt*math_I3(o,p)*temp_33                                             ! dFe_dLp(i,j,k,l) = -dt * A(i,k) invFi(l,j)
        dFi_dLi(1:3,o,1:3,p) = -dt*math_I3(o,p)*invFi_current
      enddo; enddo
      do o=1,3; do p=1,3
        dFi_dLi(1:3,1:3,o,p) = matmul(matmul(Fi_new,dFi_dLi(1:3,1:3,o,p)),Fi_new)
      enddo; enddo
      dRLi_dLi  = math_identity2nd(9) &
                - math_3333to99(math_mul3333xx3333(dLi_dS,  math_mul3333xx3333(dS_dFe, dFe_dLi) &
                                                          + math_mul3333xx3333(dS_dFi, dFi_dLi)))  &
                - math_3333to99(math_mul3333xx3333(dLi_dFi, dFi_dLi))
      temp_9 = math_33to9(residuumLi)
      call dgesv(9,1,dRLi_dLi,9,devNull_9,temp_9,9,ierr)                                            ! solve dRLi/dLp * delta Li = -res for delta Li
      if (ierr /= 0) return ! error
      deltaLi = - math_9to33(temp_9)
    endif calculateJacobiLp

    Liguess = Liguess &
            + deltaLi * steplengthLi
  enddo LiLoop

  invFp_new = matmul(invFp_current,B)
  call math_invert33(Fp_new,devNull,error,invFp_new)
  if (error) return ! error

  crystallite_P    (1:3,1:3,ipc,ip,el) = matmul(matmul(F,invFp_new),matmul(S,transpose(invFp_new)))
  crystallite_S    (1:3,1:3,ipc,ip,el) = S
  crystallite_Lp   (1:3,1:3,ipc,ip,el) = Lpguess
  crystallite_Li   (1:3,1:3,ipc,ip,el) = Liguess
  crystallite_Fp   (1:3,1:3,ipc,ip,el) = Fp_new / math_det33(Fp_new)**(1.0_pReal/3.0_pReal)         ! regularize
  crystallite_Fi   (1:3,1:3,ipc,ip,el) = Fi_new
  crystallite_Fe   (1:3,1:3,ipc,ip,el) = matmul(matmul(F,invFp_new),invFi_new)
  broken = .false.

end function integrateStress


!--------------------------------------------------------------------------------------------------
!> @brief integrate stress, state with adaptive 1st order explicit Euler method
!> using Fixed Point Iteration to adapt the stepsize
!--------------------------------------------------------------------------------------------------
subroutine integrateStateFPI(todo)

  logical, dimension(:,:,:), intent(in) :: todo
  integer :: &
    NiterationState, &                                                                              !< number of iterations in state loop
    e, &                                                                                            !< element index in element loop
    i, &                                                                                            !< integration point index in ip loop
    g, &                                                                                            !< grain index in grain loop
    p, &
    c, &
    s, &
    size_pl
  integer, dimension(maxval(phase_Nsources)) :: &
    size_so
  real(pReal) :: &
    zeta
  real(pReal), dimension(max(constitutive_plasticity_maxSizeDotState,constitutive_source_maxSizeDotState)) :: &
    r                                                                                               ! state residuum
  real(pReal), dimension(constitutive_plasticity_maxSizeDotState,2) :: &
    plastic_dotState
  real(pReal), dimension(constitutive_source_maxSizeDotState,2,maxval(phase_Nsources)) :: source_dotState
  logical :: &
    nonlocalBroken, broken

  nonlocalBroken = .false.
  !$OMP PARALLEL DO PRIVATE(size_pl,size_so,r,zeta,p,c,plastic_dotState,source_dotState,broken)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do g = 1,homogenization_Ngrains(material_homogenizationAt(e))
        p = material_phaseAt(g,e)
        if(todo(g,i,e) .and. .not. (nonlocalBroken .and. plasticState(p)%nonlocal)) then

          c = material_phaseMemberAt(g,i,e)

          broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                            crystallite_partionedF0, &
                                            crystallite_Fi(1:3,1:3,g,i,e), &
                                            crystallite_partionedFp0, &
                                            crystallite_subdt(g,i,e), g,i,e,p,c)
          if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
          if(broken) cycle

          size_pl = plasticState(p)%sizeDotState
          plasticState(p)%state(1:size_pl,c) = plasticState(p)%subState0(1:size_pl,c) &
                                             + plasticState(p)%dotState (1:size_pl,c) &
                                             * crystallite_subdt(g,i,e)
          plastic_dotState(1:size_pl,2) = 0.0_pReal
          do s = 1, phase_Nsources(p)
            size_so(s) = sourceState(p)%p(s)%sizeDotState
            sourceState(p)%p(s)%state(1:size_so(s),c) = sourceState(p)%p(s)%subState0(1:size_so(s),c) &
                                                      + sourceState(p)%p(s)%dotState (1:size_so(s),c) &
                                                      * crystallite_subdt(g,i,e)
            source_dotState(1:size_so(s),2,s) = 0.0_pReal
          enddo

          iteration: do NiterationState = 1, num%nState

            if(nIterationState > 1) plastic_dotState(1:size_pl,2) = plastic_dotState(1:size_pl,1)
            plastic_dotState(1:size_pl,1) = plasticState(p)%dotState(:,c)
            do s = 1, phase_Nsources(p)
              if(nIterationState > 1) source_dotState(1:size_so(s),2,s) = source_dotState(1:size_so(s),1,s)
              source_dotState(1:size_so(s),1,s) = sourceState(p)%p(s)%dotState(:,c)
            enddo

            broken = integrateStress(g,i,e)
            if(broken) exit iteration

            broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                                  crystallite_partionedF0, &
                                                  crystallite_Fi(1:3,1:3,g,i,e), &
                                                  crystallite_partionedFp0, &
                                                  crystallite_subdt(g,i,e), g,i,e,p,c)
            if(broken) exit iteration

            zeta = damper(plasticState(p)%dotState(:,c),plastic_dotState(1:size_pl,1),&
                                                        plastic_dotState(1:size_pl,2))
            plasticState(p)%dotState(:,c) = plasticState(p)%dotState(:,c) * zeta &
                                          + plastic_dotState(1:size_pl,1) * (1.0_pReal - zeta)
            r(1:size_pl) = plasticState(p)%state    (1:size_pl,c) &
                         - plasticState(p)%subState0(1:size_pl,c)  &
                         - plasticState(p)%dotState (1:size_pl,c) * crystallite_subdt(g,i,e)
            plasticState(p)%state(1:size_pl,c) = plasticState(p)%state(1:size_pl,c) &
                                               - r(1:size_pl)
            crystallite_converged(g,i,e) = converged(r(1:size_pl), &
                                                     plasticState(p)%state(1:size_pl,c), &
                                                     plasticState(p)%atol(1:size_pl))
            do s = 1, phase_Nsources(p)
              zeta = damper(sourceState(p)%p(s)%dotState(:,c), &
                            source_dotState(1:size_so(s),1,s),&
                            source_dotState(1:size_so(s),2,s))
              sourceState(p)%p(s)%dotState(:,c) = sourceState(p)%p(s)%dotState(:,c) * zeta &
                                                + source_dotState(1:size_so(s),1,s)* (1.0_pReal - zeta)
              r(1:size_so(s)) = sourceState(p)%p(s)%state    (1:size_so(s),c)  &
                              - sourceState(p)%p(s)%subState0(1:size_so(s),c)  &
                              - sourceState(p)%p(s)%dotState (1:size_so(s),c) * crystallite_subdt(g,i,e)
              sourceState(p)%p(s)%state(1:size_so(s),c) = sourceState(p)%p(s)%state(1:size_so(s),c) &
                                                        - r(1:size_so(s))
              crystallite_converged(g,i,e) = &
              crystallite_converged(g,i,e) .and. converged(r(1:size_so(s)), &
                                                           sourceState(p)%p(s)%state(1:size_so(s),c), &
                                                           sourceState(p)%p(s)%atol(1:size_so(s)))
            enddo

            if(crystallite_converged(g,i,e)) then
              broken = constitutive_deltaState(crystallite_S(1:3,1:3,g,i,e), &
                                               crystallite_Fe(1:3,1:3,g,i,e), &
                                               crystallite_Fi(1:3,1:3,g,i,e),g,i,e,p,c)
              exit iteration
            endif

          enddo iteration
          if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
        endif
  enddo; enddo; enddo
  !$OMP END PARALLEL DO

  if (nonlocalBroken) call nonlocalConvergenceCheck

  contains

  !--------------------------------------------------------------------------------------------------
  !> @brief calculate the damping for correction of state and dot state
  !--------------------------------------------------------------------------------------------------
  real(pReal) pure function damper(current,previous,previous2)

  real(pReal), dimension(:), intent(in) ::&
    current, previous, previous2

  real(pReal) :: dot_prod12, dot_prod22

  dot_prod12 = dot_product(current  - previous,  previous - previous2)
  dot_prod22 = dot_product(previous - previous2, previous - previous2)
  if ((dot_product(current,previous) < 0.0_pReal .or. dot_prod12 < 0.0_pReal) .and. dot_prod22 > 0.0_pReal) then
    damper = 0.75_pReal + 0.25_pReal * tanh(2.0_pReal + 4.0_pReal * dot_prod12 / dot_prod22)
  else
    damper = 1.0_pReal
  endif

  end function damper

end subroutine integrateStateFPI


!--------------------------------------------------------------------------------------------------
!> @brief integrate state with 1st order explicit Euler method
!--------------------------------------------------------------------------------------------------
subroutine integrateStateEuler(todo)

  logical, dimension(:,:,:), intent(in) :: todo

  integer :: &
    e, &                                                                                            !< element index in element loop
    i, &                                                                                            !< integration point index in ip loop
    g, &                                                                                            !< grain index in grain loop
    p, &
    c, &
    s, &
    sizeDotState
  logical :: &
    nonlocalBroken, broken

  nonlocalBroken = .false.
  !$OMP PARALLEL DO PRIVATE (sizeDotState,p,c,broken)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do g = 1,homogenization_Ngrains(material_homogenizationAt(e))
        p = material_phaseAt(g,e)
        if(todo(g,i,e) .and. .not. (nonlocalBroken .and. plasticState(p)%nonlocal)) then

          c = material_phaseMemberAt(g,i,e)

          broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                            crystallite_partionedF0, &
                                            crystallite_Fi(1:3,1:3,g,i,e), &
                                            crystallite_partionedFp0, &
                                            crystallite_subdt(g,i,e), g,i,e,p,c)
          if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
          if(broken) cycle

          sizeDotState = plasticState(p)%sizeDotState
          plasticState(p)%state(1:sizeDotState,c) = plasticState(p)%subState0(1:sizeDotState,c) &
                                                  + plasticState(p)%dotState (1:sizeDotState,c) &
                                                    * crystallite_subdt(g,i,e)
          do s = 1, phase_Nsources(p)
            sizeDotState = sourceState(p)%p(s)%sizeDotState
            sourceState(p)%p(s)%state(1:sizeDotState,c) = sourceState(p)%p(s)%subState0(1:sizeDotState,c) &
                                                        + sourceState(p)%p(s)%dotState (1:sizeDotState,c) &
                                                          * crystallite_subdt(g,i,e)
          enddo

          broken = constitutive_deltaState(crystallite_S(1:3,1:3,g,i,e), &
                                           crystallite_Fe(1:3,1:3,g,i,e), &
                                          crystallite_Fi(1:3,1:3,g,i,e),g,i,e,p,c)
          if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
          if(broken) cycle

          broken = integrateStress(g,i,e)
          if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
          crystallite_converged(g,i,e) = .not. broken
        endif
  enddo; enddo; enddo
  !$OMP END PARALLEL DO

  if (nonlocalBroken) call nonlocalConvergenceCheck

end subroutine integrateStateEuler


!--------------------------------------------------------------------------------------------------
!> @brief integrate stress, state with 1st order Euler method with adaptive step size
!--------------------------------------------------------------------------------------------------
subroutine integrateStateAdaptiveEuler(todo)

  logical, dimension(:,:,:), intent(in) :: todo

  integer :: &
    e, &                                                                                             ! element index in element loop
    i, &                                                                                             ! integration point index in ip loop
    g, &                                                                                             ! grain index in grain loop
    p, &
    c, &
    s, &
    sizeDotState
  logical :: &
    nonlocalBroken, broken

  real(pReal), dimension(constitutive_plasticity_maxSizeDotState) :: residuum_plastic
  real(pReal), dimension(constitutive_source_maxSizeDotState,maxval(phase_Nsources)) :: residuum_source

  nonlocalBroken = .false.
  !$OMP PARALLEL DO PRIVATE(sizeDotState,p,c,residuum_plastic,residuum_source,broken)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do g = 1,homogenization_Ngrains(material_homogenizationAt(e))
        broken = .false.
        p = material_phaseAt(g,e)
        if(todo(g,i,e) .and. .not. (nonlocalBroken .and. plasticState(p)%nonlocal)) then

          c = material_phaseMemberAt(g,i,e)

          broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                                crystallite_partionedF0, &
                                                crystallite_Fi(1:3,1:3,g,i,e), &
                                                crystallite_partionedFp0, &
                                                crystallite_subdt(g,i,e), g,i,e,p,c)
          if(broken) cycle

          sizeDotState = plasticState(p)%sizeDotState

          residuum_plastic(1:sizeDotState) = - plasticState(p)%dotstate(1:sizeDotState,c) * 0.5_pReal * crystallite_subdt(g,i,e)
          plasticState(p)%state(1:sizeDotState,c) = plasticState(p)%subState0(1:sizeDotState,c) &
                                                  + plasticState(p)%dotstate(1:sizeDotState,c) * crystallite_subdt(g,i,e)
          do s = 1, phase_Nsources(p)
            sizeDotState = sourceState(p)%p(s)%sizeDotState

            residuum_source(1:sizeDotState,s)  = - sourceState(p)%p(s)%dotstate(1:sizeDotState,c) &
                                               * 0.5_pReal * crystallite_subdt(g,i,e)
            sourceState(p)%p(s)%state(1:sizeDotState,c) = sourceState(p)%p(s)%subState0(1:sizeDotState,c) &
                                                        + sourceState(p)%p(s)%dotstate(1:sizeDotState,c) * crystallite_subdt(g,i,e)
          enddo

          broken = constitutive_deltaState(crystallite_S(1:3,1:3,g,i,e), &
                                           crystallite_Fe(1:3,1:3,g,i,e), &
                                           crystallite_Fi(1:3,1:3,g,i,e),g,i,e,p,c)
          if(broken) cycle

          broken = integrateStress(g,i,e)
          if(broken) cycle

          broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                                crystallite_partionedF0, &
                                                crystallite_Fi(1:3,1:3,g,i,e), &
                                                crystallite_partionedFp0, &
                                                crystallite_subdt(g,i,e), g,i,e,p,c)
          if(broken) cycle


          sizeDotState = plasticState(p)%sizeDotState
          crystallite_converged(g,i,e) = converged(residuum_plastic(1:sizeDotState) &
                                                   + 0.5_pReal * plasticState(p)%dotState(:,c) * crystallite_subdt(g,i,e), &
                                                   plasticState(p)%state(1:sizeDotState,c), &
                                                   plasticState(p)%atol(1:sizeDotState))

          do s = 1, phase_Nsources(p)
            sizeDotState = sourceState(p)%p(s)%sizeDotState
            crystallite_converged(g,i,e) = &
            crystallite_converged(g,i,e) .and. converged(residuum_source(1:sizeDotState,s) &
                                                         + 0.5_pReal*sourceState(p)%p(s)%dotState(:,c)*crystallite_subdt(g,i,e), &
                                                         sourceState(p)%p(s)%state(1:sizeDotState,c), &
                                                         sourceState(p)%p(s)%atol(1:sizeDotState))
           enddo

        endif
        if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
  enddo; enddo; enddo
  !$OMP END PARALLEL DO

  if (nonlocalBroken) call nonlocalConvergenceCheck

end subroutine integrateStateAdaptiveEuler


!---------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with the classic Runge Kutta method
!---------------------------------------------------------------------------------------------------
subroutine integrateStateRK4(todo)

  logical, dimension(:,:,:), intent(in) :: todo

  real(pReal), dimension(3,3), parameter :: &
    A = reshape([&
      0.5_pReal, 0.0_pReal, 0.0_pReal, &
      0.0_pReal, 0.5_pReal, 0.0_pReal, &
      0.0_pReal, 0.0_pReal, 1.0_pReal],&
      shape(A))
  real(pReal), dimension(3), parameter :: &
    C = [0.5_pReal, 0.5_pReal, 1.0_pReal]
  real(pReal), dimension(4), parameter :: &
    B = [1.0_pReal/6.0_pReal, 1.0_pReal/3.0_pReal, 1.0_pReal/3.0_pReal, 1.0_pReal/6.0_pReal]

  call integrateStateRK(todo,A,B,C)

end subroutine integrateStateRK4


!---------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with the Cash-Carp method
!---------------------------------------------------------------------------------------------------
subroutine integrateStateRKCK45(todo)

  logical, dimension(:,:,:), intent(in) :: todo

  real(pReal), dimension(5,5), parameter :: &
    A = reshape([&
      1._pReal/5._pReal,       .0_pReal,             .0_pReal,               .0_pReal,                  .0_pReal, &
      3._pReal/40._pReal,      9._pReal/40._pReal,   .0_pReal,               .0_pReal,                  .0_pReal, &
      3_pReal/10._pReal,       -9._pReal/10._pReal,  6._pReal/5._pReal,      .0_pReal,                  .0_pReal, &
      -11._pReal/54._pReal,    5._pReal/2._pReal,    -70.0_pReal/27.0_pReal, 35.0_pReal/27.0_pReal,     .0_pReal, &
      1631._pReal/55296._pReal,175._pReal/512._pReal,575._pReal/13824._pReal,44275._pReal/110592._pReal,253._pReal/4096._pReal],&
      shape(A))
  real(pReal), dimension(5), parameter :: &
    C = [0.2_pReal, 0.3_pReal, 0.6_pReal, 1.0_pReal, 0.875_pReal]
  real(pReal), dimension(6), parameter :: &
    B = &
      [37.0_pReal/378.0_pReal, .0_pReal, 250.0_pReal/621.0_pReal, &
      125.0_pReal/594.0_pReal, .0_pReal, 512.0_pReal/1771.0_pReal], &
    DB = B - &
      [2825.0_pReal/27648.0_pReal,    .0_pReal,                18575.0_pReal/48384.0_pReal,&
      13525.0_pReal/55296.0_pReal, 277.0_pReal/14336.0_pReal,  1._pReal/4._pReal]

  call integrateStateRK(todo,A,B,C,DB)

end subroutine integrateStateRKCK45


!--------------------------------------------------------------------------------------------------
!> @brief Integrate state (including stress integration) with an explicit Runge-Kutta method or an
!! embedded explicit Runge-Kutta method
!--------------------------------------------------------------------------------------------------
subroutine integrateStateRK(todo,A,B,CC,DB)

  logical, dimension(:,:,:), intent(in) :: todo

  real(pReal), dimension(:,:), intent(in) :: A
  real(pReal), dimension(:),   intent(in) :: B, CC
  real(pReal), dimension(:),   intent(in), optional :: DB

  integer :: &
    e, &                                                                                            ! element index in element loop
    i, &                                                                                            ! integration point index in ip loop
    g, &                                                                                            ! grain index in grain loop
    stage, &                                                                                        ! stage index in integration stage loop
    n, &
    p, &
    c, &
    s, &
    sizeDotState
  logical :: &
    nonlocalBroken, broken
  real(pReal), dimension(constitutive_source_maxSizeDotState,size(B),maxval(phase_Nsources)) :: source_RKdotState
  real(pReal), dimension(constitutive_plasticity_maxSizeDotState,size(B))                    :: plastic_RKdotState

  nonlocalBroken = .false.
  !$OMP PARALLEL DO PRIVATE(sizeDotState,p,c,plastic_RKdotState,source_RKdotState,broken)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      do g = 1,homogenization_Ngrains(material_homogenizationAt(e))
        broken = .false.
        p = material_phaseAt(g,e)
        if(todo(g,i,e) .and. .not. (nonlocalBroken .and. plasticState(p)%nonlocal)) then

          c = material_phaseMemberAt(g,i,e)

          broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                                crystallite_partionedF0, &
                                                crystallite_Fi(1:3,1:3,g,i,e), &
                                                crystallite_partionedFp0, &
                                                crystallite_subdt(g,i,e), g,i,e,p,c)
          if(broken) cycle

          do stage = 1,size(A,1)
            sizeDotState = plasticState(p)%sizeDotState
            plastic_RKdotState(1:sizeDotState,stage) = plasticState(p)%dotState(:,c)
            plasticState(p)%dotState(:,c) = A(1,stage) * plastic_RKdotState(1:sizeDotState,1)
            do s = 1, phase_Nsources(p)
              sizeDotState = sourceState(p)%p(s)%sizeDotState
              source_RKdotState(1:sizeDotState,stage,s) = sourceState(p)%p(s)%dotState(:,c)
              sourceState(p)%p(s)%dotState(:,c) = A(1,stage) * source_RKdotState(1:sizeDotState,1,s)
            enddo

            do n = 2, stage
              sizeDotState = plasticState(p)%sizeDotState
              plasticState(p)%dotState(:,c) = plasticState(p)%dotState(:,c) &
                                            + A(n,stage) * plastic_RKdotState(1:sizeDotState,n)
              do s = 1, phase_Nsources(p)
                sizeDotState = sourceState(p)%p(s)%sizeDotState
                sourceState(p)%p(s)%dotState(:,c) = sourceState(p)%p(s)%dotState(:,c) &
                                                  + A(n,stage) * source_RKdotState(1:sizeDotState,n,s)
              enddo
            enddo

            sizeDotState = plasticState(p)%sizeDotState
            plasticState(p)%state(1:sizeDotState,c) = plasticState(p)%subState0(1:sizeDotState,c) &
                                                    + plasticState(p)%dotState (1:sizeDotState,c) &
                                                      * crystallite_subdt(g,i,e)
            do s = 1, phase_Nsources(p)
              sizeDotState = sourceState(p)%p(s)%sizeDotState
              sourceState(p)%p(s)%state(1:sizeDotState,c) = sourceState(p)%p(s)%subState0(1:sizeDotState,c) &
                                                          + sourceState(p)%p(s)%dotState (1:sizeDotState,c) &
                                                            * crystallite_subdt(g,i,e)
            enddo

            broken = integrateStress(g,i,e,CC(stage))
            if(broken) exit

            broken = constitutive_collectDotState(crystallite_S(1:3,1:3,g,i,e), &
                                                  crystallite_partionedF0, &
                                                  crystallite_Fi(1:3,1:3,g,i,e), &
                                                  crystallite_partionedFp0, &
                                                  crystallite_subdt(g,i,e)*CC(stage), g,i,e,p,c)
            if(broken) exit

          enddo
          if(broken) cycle

          sizeDotState = plasticState(p)%sizeDotState

          plastic_RKdotState(1:sizeDotState,size(B)) = plasticState (p)%dotState(:,c)
          plasticState(p)%dotState(:,c) = matmul(plastic_RKdotState(1:sizeDotState,1:size(B)),B)
          plasticState(p)%state(1:sizeDotState,c) = plasticState(p)%subState0(1:sizeDotState,c) &
                                                  + plasticState(p)%dotState (1:sizeDotState,c) &
                                                    * crystallite_subdt(g,i,e)
          if(present(DB)) &
            broken = .not. converged( matmul(plastic_RKdotState(1:sizeDotState,1:size(DB)),DB) &
                                                     * crystallite_subdt(g,i,e), &
                                                plasticState(p)%state(1:sizeDotState,c), &
                                                plasticState(p)%atol(1:sizeDotState))

          do s = 1, phase_Nsources(p)
            sizeDotState = sourceState(p)%p(s)%sizeDotState

            source_RKdotState(1:sizeDotState,size(B),s) = sourceState(p)%p(s)%dotState(:,c)
            sourceState(p)%p(s)%dotState(:,c)  = matmul(source_RKdotState(1:sizeDotState,1:size(B),s),B)
            sourceState(p)%p(s)%state(1:sizeDotState,c) = sourceState(p)%p(s)%subState0(1:sizeDotState,c) &
                                                        + sourceState(p)%p(s)%dotState (1:sizeDotState,c) &
                                                          * crystallite_subdt(g,i,e)
            if(present(DB)) &
              broken = broken .or. .not. converged(matmul(source_RKdotState(1:sizeDotState,1:size(DB),s),DB) &
                                                         * crystallite_subdt(g,i,e), &
                                                   sourceState(p)%p(s)%state(1:sizeDotState,c), &
                                                   sourceState(p)%p(s)%atol(1:sizeDotState))
          enddo
          if(broken) cycle

          broken = constitutive_deltaState(crystallite_S(1:3,1:3,g,i,e), &
                                           crystallite_Fe(1:3,1:3,g,i,e), &
                                           crystallite_Fi(1:3,1:3,g,i,e),g,i,e,p,c)
          if(broken) cycle

          broken = integrateStress(g,i,e)
          crystallite_converged(g,i,e) = .not. broken

        endif
        if(broken .and. plasticState(p)%nonlocal) nonlocalBroken = .true.
  enddo; enddo; enddo
  !$OMP END PARALLEL DO

  if(nonlocalBroken) call nonlocalConvergenceCheck

end subroutine integrateStateRK


!--------------------------------------------------------------------------------------------------
!> @brief sets convergence flag for nonlocal calculations
!> @details one non-converged nonlocal sets all other nonlocals to non-converged to trigger cut back
!--------------------------------------------------------------------------------------------------
subroutine nonlocalConvergenceCheck

  integer :: e,i,p

  !$OMP PARALLEL DO PRIVATE(p)
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    p = material_phaseAt(1,e)
    do i = FEsolving_execIP(1),FEsolving_execIP(2)
      if(plasticState(p)%nonlocal) crystallite_converged(1,i,e) = .false.
    enddo
  enddo
  !$OMP END PARALLEL DO

end subroutine nonlocalConvergenceCheck


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
!> @brief Write current  restart information (Field and constitutive data) to file.
! ToDo: Merge data into one file for MPI, move state to constitutive and homogenization, respectively
!--------------------------------------------------------------------------------------------------
subroutine crystallite_restartWrite

  integer :: i
  integer(HID_T) :: fileHandle, groupHandle
  character(len=pStringLen) :: fileName, datasetName

  write(6,'(a)') ' writing field and constitutive data required for restart to file';flush(6)

  write(fileName,'(a,i0,a)') trim(getSolverJobName())//'_',worldrank,'.hdf5'
  fileHandle = HDF5_openFile(fileName,'a')

  call HDF5_write(fileHandle,crystallite_partionedF,'F')
  call HDF5_write(fileHandle,crystallite_Fp,        'Fp')
  call HDF5_write(fileHandle,crystallite_Fi,        'Fi')
  call HDF5_write(fileHandle,crystallite_Lp,        'Lp')
  call HDF5_write(fileHandle,crystallite_Li,        'Li')
  call HDF5_write(fileHandle,crystallite_S,         'S')

  groupHandle = HDF5_addGroup(fileHandle,'constituent')
  do i = 1,size(material_name_phase)
    write(datasetName,'(i0,a)') i,'_omega_plastic'
    call HDF5_write(groupHandle,plasticState(i)%state,datasetName)
  enddo
  call HDF5_closeGroup(groupHandle)

  groupHandle = HDF5_addGroup(fileHandle,'materialpoint')
  do i = 1, material_Nhomogenization
    write(datasetName,'(i0,a)') i,'_omega_homogenization'
    call HDF5_write(groupHandle,homogState(i)%state,datasetName)
  enddo
  call HDF5_closeGroup(groupHandle)

  call HDF5_closeFile(fileHandle)

end subroutine crystallite_restartWrite


!--------------------------------------------------------------------------------------------------
!> @brief Read data for restart
! ToDo: Merge data into one file for MPI, move state to constitutive and homogenization, respectively
!--------------------------------------------------------------------------------------------------
subroutine crystallite_restartRead

  integer :: i
  integer(HID_T) :: fileHandle, groupHandle
  character(len=pStringLen) :: fileName, datasetName

  write(6,'(/,a,i0,a)') ' reading restart information of increment from file'

  write(fileName,'(a,i0,a)') trim(getSolverJobName())//'_',worldrank,'.hdf5'
  fileHandle = HDF5_openFile(fileName)

  call HDF5_read(fileHandle,crystallite_F0, 'F')
  call HDF5_read(fileHandle,crystallite_Fp0,'Fp')
  call HDF5_read(fileHandle,crystallite_Fi0,'Fi')
  call HDF5_read(fileHandle,crystallite_Lp0,'Lp')
  call HDF5_read(fileHandle,crystallite_Li0,'Li')
  call HDF5_read(fileHandle,crystallite_S0, 'S')

  groupHandle = HDF5_openGroup(fileHandle,'constituent')
  do i = 1,size(material_name_phase)
    write(datasetName,'(i0,a)') i,'_omega_plastic'
    call HDF5_read(groupHandle,plasticState(i)%state0,datasetName)
  enddo
  call HDF5_closeGroup(groupHandle)

  groupHandle = HDF5_openGroup(fileHandle,'materialpoint')
  do i = 1, material_Nhomogenization
    write(datasetName,'(i0,a)') i,'_omega_homogenization'
    call HDF5_read(groupHandle,homogState(i)%state0,datasetName)
  enddo
  call HDF5_closeGroup(groupHandle)

  call HDF5_closeFile(fileHandle)

end subroutine crystallite_restartRead


!--------------------------------------------------------------------------------------------------
!> @brief Forward data after successful increment.
! ToDo: Any guessing for the current states possible?
!--------------------------------------------------------------------------------------------------
subroutine crystallite_forward

  integer :: i, j

  crystallite_F0  = crystallite_partionedF
  crystallite_Fp0 = crystallite_Fp
  crystallite_Lp0 = crystallite_Lp
  crystallite_Fi0 = crystallite_Fi
  crystallite_Li0 = crystallite_Li
  crystallite_S0  = crystallite_S

  do i = 1, size(plasticState)
    plasticState(i)%state0 = plasticState(i)%state
  enddo
  do i = 1, size(sourceState)
    do j = 1,phase_Nsources(i)
      sourceState(i)%p(j)%state0 = sourceState(i)%p(j)%state
  enddo; enddo
  do i = 1, material_Nhomogenization
    homogState  (i)%state0 = homogState  (i)%state
    thermalState(i)%state0 = thermalState(i)%state
    damageState (i)%state0 = damageState (i)%state
  enddo

end subroutine crystallite_forward

end module crystallite
