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
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Driver controlling inner and outer load case looping of the various spectral solvers
!> @details doing cutbacking, forwarding in case of restart, reporting statistics, writing
!> results
!--------------------------------------------------------------------------------------------------
program DAMASK_grid
#include <petsc/finclude/petscsys.h>
  use PETScsys
  use prec
  use DAMASK_interface
  use IO
  use config
  use math
  use CPFEM2
  use material
  use spectral_utilities
  use grid_mech_spectral_basic
  use grid_mech_spectral_polarisation
  use grid_mech_FEM
  use grid_damage_spectral
  use grid_thermal_spectral
  use results
  
  implicit none

!--------------------------------------------------------------------------------------------------
! variables related to information from load case and geom file
  real(pReal), dimension(9) :: temp_valueVector = 0.0_pReal                                         !< temporarily from loadcase file when reading in tensors (initialize to 0.0)
  logical,     dimension(9) :: temp_maskVector  = .false.                                           !< temporarily from loadcase file when reading in tensors
  integer, allocatable, dimension(:) :: chunkPos
  integer :: &
    N_t   = 0, &                                                                                    !< # of time indicators found in load case file
    N_n   = 0, &                                                                                    !< # of increment specifiers found in load case file
    N_def = 0                                                                                       !< # of rate of deformation specifiers found in load case file
  character(len=:), allocatable :: &
    line

!--------------------------------------------------------------------------------------------------
! loop variables, convergence etc.
  real(pReal), dimension(3,3), parameter :: &
    ones  = 1.0_pReal, &
    zeros = 0.0_pReal
  integer, parameter :: &
    subStepFactor = 2                                                                               !< for each substep, divide the last time increment by 2.0
  real(pReal) :: &
    time = 0.0_pReal, &                                                                             !< elapsed time
    time0 = 0.0_pReal, &                                                                            !< begin of interval
    timeinc = 1.0_pReal, &                                                                          !< current time interval
    timeIncOld = 0.0_pReal, &                                                                       !< previous time interval
    remainingLoadCaseTime = 0.0_pReal                                                               !< remaining time of current load case
  logical :: &
    guess, &                                                                                        !< guess along former trajectory
    stagIterate, &
    cutBack = .false.
  integer :: &
    i, j, k, l, field, &
    errorID = 0, &
    cutBackLevel = 0, &                                                                             !< cut back level \f$ t = \frac{t_{inc}}{2^l} \f$
    stepFraction = 0                                                                                !< fraction of current time interval
  integer :: &
    currentLoadcase = 0, &                                                                          !< current load case
    inc, &                                                                                          !< current increment in current load case
    totalIncsCounter = 0, &                                                                         !< total # of increments
    statUnit = 0, &                                                                                 !< file unit for statistics output
    stagIter, &
    nActiveFields = 0
  character(len=pStringLen), dimension(:), allocatable :: fileContent
  character(len=pStringLen) :: &
    incInfo, &
    loadcase_string
  integer :: &
    maxCutBack, &                                                                                   !< max number of cut backs
    stagItMax                                                                                       !< max number of field level staggered iterations
  type(tLoadCase), allocatable, dimension(:) :: loadCases                                           !< array of all load cases
  type(tLoadCase) :: newLoadCase
  type(tSolutionState), allocatable, dimension(:) :: solres
  procedure(grid_mech_spectral_basic_init), pointer :: &
    mech_init
  procedure(grid_mech_spectral_basic_forward), pointer :: &
    mech_forward
  procedure(grid_mech_spectral_basic_solution), pointer :: &
    mech_solution
  procedure(grid_mech_spectral_basic_updateCoords), pointer :: &
    mech_updateCoords
  procedure(grid_mech_spectral_basic_restartWrite), pointer :: &
    mech_restartWrite
  
  external :: &
    quit
  class (tNode), pointer :: &
    num_grid, &
    debug_grid                                                                                      ! pointer to grid debug options

!--------------------------------------------------------------------------------------------------
! init DAMASK (all modules)
 
  call CPFEM_initAll 
  write(6,'(/,a)')   ' <<<+-  DAMASK_spectral init  -+>>>'; flush(6)
  
  write(6,'(/,a)') ' Shanthraj et al., Handbook of Mechanics of Materials, 2019'
  write(6,'(a)')   ' https://doi.org/10.1007/978-981-10-6855-3_80'

!--------------------------------------------------------------------------------------------------
! initialize field solver information
  nActiveFields = 1
  if (any(thermal_type  == THERMAL_conduction_ID  )) nActiveFields = nActiveFields + 1
  if (any(damage_type   == DAMAGE_nonlocal_ID     )) nActiveFields = nActiveFields + 1
  allocate(solres(nActiveFields))
  allocate(newLoadCase%ID(nActiveFields))

!-------------------------------------------------------------------------------------------------
! reading field paramters from numerics file and do sanity checks
  num_grid => numerics_root%get('grid', defaultVal=emptyDict)
  stagItMax  = num_grid%get_asInt('maxStaggeredIter',defaultVal=10)
  maxCutBack = num_grid%get_asInt('maxCutBack',defaultVal=3)

  if (stagItMax < 0)    call IO_error(301,ext_msg='maxStaggeredIter')
  if (maxCutBack < 0)   call IO_error(301,ext_msg='maxCutBack')

!--------------------------------------------------------------------------------------------------
! assign mechanics solver depending on selected type
 
  debug_grid => debug_root%get('grid',defaultVal=emptyList)
  select case (trim(num_grid%get_asString('solver', defaultVal = 'Basic'))) 
    case ('Basic')
      mech_init         => grid_mech_spectral_basic_init
      mech_forward      => grid_mech_spectral_basic_forward
      mech_solution     => grid_mech_spectral_basic_solution
      mech_updateCoords => grid_mech_spectral_basic_updateCoords
      mech_restartWrite => grid_mech_spectral_basic_restartWrite
  
    case ('Polarisation')
     if(debug_grid%contains('basic')) &
        call IO_warning(42, ext_msg='debug Divergence')
      mech_init         => grid_mech_spectral_polarisation_init
      mech_forward      => grid_mech_spectral_polarisation_forward
      mech_solution     => grid_mech_spectral_polarisation_solution
      mech_updateCoords => grid_mech_spectral_polarisation_updateCoords
      mech_restartWrite => grid_mech_spectral_polarisation_restartWrite
  
    case ('FEM')
     if(debug_grid%contains('basic')) &
        call IO_warning(42, ext_msg='debug Divergence')
      mech_init         => grid_mech_FEM_init
      mech_forward      => grid_mech_FEM_forward
      mech_solution     => grid_mech_FEM_solution
      mech_updateCoords => grid_mech_FEM_updateCoords
      mech_restartWrite => grid_mech_FEM_restartWrite
  
    case default
      call IO_error(error_ID = 891, ext_msg = trim(num_grid%get_asString('solver')))
  
  end select
  

!--------------------------------------------------------------------------------------------------
! reading information from load case file and to sanity checks 
  fileContent = IO_readlines(trim(loadCaseFile))
  if(size(fileContent) == 0) call IO_error(307,ext_msg='No load case specified')
  
  allocate (loadCases(0))                                                                           ! array of load cases
  do currentLoadCase = 1, size(fileContent)
    line = fileContent(currentLoadCase)
    if (IO_isBlank(line)) cycle
    chunkPos = IO_stringPos(line)
  
    do i = 1, chunkPos(1)                                                                           ! reading compulsory parameters for loadcase
      select case (IO_lc(IO_stringValue(line,chunkPos,i)))
        case('l','fdot','dotf','f')
          N_def = N_def + 1
        case('t','time','delta')
          N_t = N_t + 1
        case('n','incs','increments','logincs','logincrements')
          N_n = N_n + 1
      end select
    enddo
    if ((N_def /= N_n) .or. (N_n /= N_t) .or. N_n < 1) &                                            ! sanity check
      call IO_error(error_ID=837,el=currentLoadCase,ext_msg = trim(loadCaseFile))                   ! error message for incomplete loadcase
  
    newLoadCase%stress%myType='stress'
    field = 1
    newLoadCase%ID(field) = FIELD_MECH_ID                                                           ! mechanical active by default
    thermalActive: if (any(thermal_type  == THERMAL_conduction_ID)) then
      field = field + 1
      newLoadCase%ID(field) = FIELD_THERMAL_ID
    endif thermalActive
    damageActive: if (any(damage_type   == DAMAGE_nonlocal_ID)) then
      field = field + 1
      newLoadCase%ID(field) = FIELD_DAMAGE_ID
    endif damageActive
  
    call newLoadCase%rot%fromEulers(real([0.0,0.0,0.0],pReal))
    readIn: do i = 1, chunkPos(1)
      select case (IO_lc(IO_stringValue(line,chunkPos,i)))
        case('fdot','dotf','l','f')                                                                 ! assign values for the deformation BC matrix
          temp_valueVector = 0.0_pReal
          if (IO_lc(IO_stringValue(line,chunkPos,i)) == 'fdot'.or. &                                ! in case of Fdot, set type to fdot
              IO_lc(IO_stringValue(line,chunkPos,i)) == 'dotf') then
            newLoadCase%deformation%myType = 'fdot'
          else if (IO_lc(IO_stringValue(line,chunkPos,i)) == 'f') then
            newLoadCase%deformation%myType = 'f'
          else
            newLoadCase%deformation%myType = 'l'
          endif
          do j = 1, 9
            temp_maskVector(j) = IO_stringValue(line,chunkPos,i+j) /= '*'                           ! true if not a *
            if (temp_maskVector(j)) temp_valueVector(j) = IO_floatValue(line,chunkPos,i+j)          ! read value where applicable
          enddo
          newLoadCase%deformation%maskLogical = transpose(reshape(temp_maskVector,[ 3,3]))          ! logical mask in 3x3 notation
          newLoadCase%deformation%maskFloat = merge(ones,zeros,newLoadCase%deformation%maskLogical) ! float (1.0/0.0) mask in 3x3 notation
          newLoadCase%deformation%values    = math_9to33(temp_valueVector)                          ! values in 3x3 notation
        case('p','stress', 's')
          temp_valueVector = 0.0_pReal
          do j = 1, 9
            temp_maskVector(j) = IO_stringValue(line,chunkPos,i+j) /= '*'                           ! true if not an asterisk
            if (temp_maskVector(j)) temp_valueVector(j) = IO_floatValue(line,chunkPos,i+j)          ! read value where applicable
          enddo
          newLoadCase%stress%maskLogical = transpose(reshape(temp_maskVector,[ 3,3]))
          newLoadCase%stress%maskFloat   = merge(ones,zeros,newLoadCase%stress%maskLogical)
          newLoadCase%stress%values      = math_9to33(temp_valueVector)
        case('t','time','delta')                                                                    ! increment time
          newLoadCase%time = IO_floatValue(line,chunkPos,i+1)
        case('n','incs','increments')                                                               ! number of increments
          newLoadCase%incs = IO_intValue(line,chunkPos,i+1)
        case('logincs','logincrements')                                                             ! number of increments (switch to log time scaling)
          newLoadCase%incs = IO_intValue(line,chunkPos,i+1)
          newLoadCase%logscale = 1
        case('freq','frequency','outputfreq')                                                       ! frequency of result writings
          newLoadCase%outputfrequency = IO_intValue(line,chunkPos,i+1)
        case('r','restart','restartwrite')                                                          ! frequency of writing restart information
          newLoadCase%restartfrequency = IO_intValue(line,chunkPos,i+1)
        case('guessreset','dropguessing')
          newLoadCase%followFormerTrajectory = .false.                                              ! do not continue to predict deformation along former trajectory
        case('euler')                                                                               ! rotation of load case given in euler angles
          temp_valueVector = 0.0_pReal
          l = 1                                                                                     ! assuming values given in degrees
          k = 1                                                                                     ! assuming keyword indicating degree/radians present
          select case (IO_lc(IO_stringValue(line,chunkPos,i+1)))
            case('deg','degree')
            case('rad','radian')                                                                    ! don't convert from degree to radian
              l = 0
            case default
              k = 0
          end select
          do j = 1, 3
            temp_valueVector(j) = IO_floatValue(line,chunkPos,i+k+j)
          enddo
          call newLoadCase%rot%fromEulers(temp_valueVector(1:3),degrees=(l==1))
        case('rotation','rot')                                                                      ! assign values for the rotation  matrix
          temp_valueVector = 0.0_pReal
          do j = 1, 9
            temp_valueVector(j) = IO_floatValue(line,chunkPos,i+j)
          enddo
          call newLoadCase%rot%fromMatrix(math_9to33(temp_valueVector))
      end select
    enddo readIn
  
    newLoadCase%followFormerTrajectory = merge(.true.,.false.,currentLoadCase > 1)                  ! by default, guess from previous load case
  
    reportAndCheck: if (worldrank == 0) then
      write (loadcase_string, '(i0)' ) currentLoadCase
      write(6,'(/,1x,a,i0)') 'load case: ', currentLoadCase
      if (.not. newLoadCase%followFormerTrajectory) &
        write(6,'(2x,a)') 'drop guessing along trajectory'
      if (newLoadCase%deformation%myType == 'l') then
        do j = 1, 3
          if (any(newLoadCase%deformation%maskLogical(j,1:3) .eqv. .true.) .and. &
              any(newLoadCase%deformation%maskLogical(j,1:3) .eqv. .false.)) errorID = 832          ! each row should be either fully or not at all defined
        enddo
        write(6,'(2x,a)') 'velocity gradient:'
      else if (newLoadCase%deformation%myType == 'f') then
        write(6,'(2x,a)') 'deformation gradient at end of load case:'
      else
        write(6,'(2x,a)') 'deformation gradient rate:'
      endif
      do i = 1, 3; do j = 1, 3
        if(newLoadCase%deformation%maskLogical(i,j)) then
          write(6,'(2x,f12.7)',advance='no') newLoadCase%deformation%values(i,j)
        else
          write(6,'(2x,12a)',advance='no') '     *      '
          endif
        enddo; write(6,'(/)',advance='no')
      enddo
      if (any(newLoadCase%stress%maskLogical .eqv. &
              newLoadCase%deformation%maskLogical)) errorID = 831                                   ! exclusive or masking only
      if (any(newLoadCase%stress%maskLogical .and. transpose(newLoadCase%stress%maskLogical) &
          .and. (math_I3<1))) errorID = 838                                                         ! no rotation is allowed by stress BC
      write(6,'(2x,a)') 'stress / GPa:'
      do i = 1, 3; do j = 1, 3
        if(newLoadCase%stress%maskLogical(i,j)) then
          write(6,'(2x,f12.7)',advance='no') newLoadCase%stress%values(i,j)*1e-9_pReal
        else
          write(6,'(2x,12a)',advance='no') '     *      '
        endif
        enddo; write(6,'(/)',advance='no')
      enddo
      if (any(abs(matmul(newLoadCase%rot%asMatrix(), &
                         transpose(newLoadCase%rot%asMatrix()))-math_I3) > &
                 reshape(spread(tol_math_check,1,9),[ 3,3]))) errorID = 846                         ! given rotation matrix contains strain
      if (any(dNeq(newLoadCase%rot%asMatrix(), math_I3))) &
        write(6,'(2x,a,/,3(3(3x,f12.7,1x)/))',advance='no') 'rotation of loadframe:',&
                 transpose(newLoadCase%rot%asMatrix())
      if (newLoadCase%time < 0.0_pReal) errorID = 834                                               ! negative time increment
      write(6,'(2x,a,f0.3)') 'time: ', newLoadCase%time
      if (newLoadCase%incs < 1)    errorID = 835                                                    ! non-positive incs count
      write(6,'(2x,a,i0)')  'increments: ', newLoadCase%incs
      if (newLoadCase%outputfrequency < 1)  errorID = 836                                           ! non-positive result frequency
      write(6,'(2x,a,i0)')  'output frequency: ', newLoadCase%outputfrequency
      if (newLoadCase%restartfrequency < 1)  errorID = 839                                          ! non-positive restart frequency
      if (newLoadCase%restartfrequency < huge(0)) &
        write(6,'(2x,a,i0)')  'restart frequency: ', newLoadCase%restartfrequency
      if (errorID > 0) call IO_error(error_ID = errorID, ext_msg = loadcase_string)                 ! exit with error message
    endif reportAndCheck
    loadCases = [loadCases,newLoadCase]                                                             ! load case is ok, append it
  enddo

!--------------------------------------------------------------------------------------------------
! doing initialization depending on active solvers
  call spectral_Utilities_init
  do field = 1, nActiveFields
    select case (loadCases(1)%ID(field))
      case(FIELD_MECH_ID)
        call mech_init
      
      case(FIELD_THERMAL_ID)
        call grid_thermal_spectral_init
  
      case(FIELD_DAMAGE_ID)
        call grid_damage_spectral_init
  
    end select
  enddo

!--------------------------------------------------------------------------------------------------
! write header of output file
  if (worldrank == 0) then
    writeHeader: if (interface_restartInc < 1) then
      open(newunit=statUnit,file=trim(getSolverJobName())//'.sta',form='FORMATTED',status='REPLACE')
      write(statUnit,'(a)') 'Increment Time CutbackLevel Converged IterationsNeeded'                ! statistics file
      if (debug_grid%contains('basic')) &
        write(6,'(/,a)') ' header of statistics file written out'
      flush(6)
    else writeHeader
      open(newunit=statUnit,file=trim(getSolverJobName())//&
                                  '.sta',form='FORMATTED', position='APPEND', status='OLD')
    endif writeHeader
  endif
  
  writeUndeformed: if (interface_restartInc < 1) then
    write(6,'(1/,a)') ' ... writing initial configuration to file ........................'
    call CPFEM_results(0,0.0_pReal)
  endif writeUndeformed
  
  loadCaseLooping: do currentLoadCase = 1, size(loadCases)
    time0 = time                                                                                    ! load case start time
    guess = loadCases(currentLoadCase)%followFormerTrajectory                                       ! change of load case? homogeneous guess for the first inc
  
    incLooping: do inc = 1, loadCases(currentLoadCase)%incs
      totalIncsCounter = totalIncsCounter + 1

!--------------------------------------------------------------------------------------------------
! forwarding time
      timeIncOld = timeinc                                                                          ! last timeinc that brought former inc to an end
      if (loadCases(currentLoadCase)%logscale == 0) then                                            ! linear scale
        timeinc = loadCases(currentLoadCase)%time/real(loadCases(currentLoadCase)%incs,pReal)
      else
        if (currentLoadCase == 1) then                                                              ! 1st load case of logarithmic scale
          if (inc == 1) then                                                                        ! 1st inc of 1st load case of logarithmic scale
            timeinc = loadCases(1)%time*(2.0_pReal**real(    1-loadCases(1)%incs ,pReal))           ! assume 1st inc is equal to 2nd
          else                                                                                      ! not-1st inc of 1st load case of logarithmic scale
            timeinc = loadCases(1)%time*(2.0_pReal**real(inc-1-loadCases(1)%incs ,pReal))
          endif
        else                                                                                        ! not-1st load case of logarithmic scale
          timeinc = time0 * &
               ( (1.0_pReal + loadCases(currentLoadCase)%time/time0 )**(real( inc         ,pReal)/&
                                                     real(loadCases(currentLoadCase)%incs ,pReal))&
                -(1.0_pReal + loadCases(currentLoadCase)%time/time0 )**(real( inc-1  ,pReal)/&
                                                     real(loadCases(currentLoadCase)%incs ,pReal)))
        endif
      endif
      timeinc = timeinc * real(subStepFactor,pReal)**real(-cutBackLevel,pReal)                      ! depending on cut back level, decrease time step
  
      skipping: if (totalIncsCounter <= interface_restartInc) then                                  ! not yet at restart inc?
        time = time + timeinc                                                                       ! just advance time, skip already performed calculation
        guess = .true.                                                                              ! QUESTION:why forced guessing instead of inheriting loadcase preference
      else skipping
        stepFraction = 0                                                                            ! fraction scaled by stepFactor**cutLevel
  
        subStepLooping: do while (stepFraction < subStepFactor**cutBackLevel)
          remainingLoadCaseTime = loadCases(currentLoadCase)%time+time0 - time
          time = time + timeinc                                                                     ! forward target time
          stepFraction = stepFraction + 1                                                           ! count step

!--------------------------------------------------------------------------------------------------
! report begin of new step
          write(6,'(/,a)') ' ###########################################################################'
          write(6,'(1x,a,es12.5,6(a,i0))') &
                  'Time', time, &
                  's: Increment ', inc,'/',loadCases(currentLoadCase)%incs,&
                  '-', stepFraction,'/',subStepFactor**cutBackLevel,&
                  ' of load case ', currentLoadCase,'/',size(loadCases)
          write(incInfo,'(4(a,i0))') &
                  'Increment ',totalIncsCounter,'/',sum(loadCases%incs),&
                  '-', stepFraction,'/',subStepFactor**cutBackLevel
          flush(6)

!--------------------------------------------------------------------------------------------------
! forward fields
          do field = 1, nActiveFields
            select case(loadCases(currentLoadCase)%ID(field))
              case(FIELD_MECH_ID)
                call mech_forward (&
                        cutBack,guess,timeinc,timeIncOld,remainingLoadCaseTime, &
                        deformation_BC     = loadCases(currentLoadCase)%deformation, &
                        stress_BC          = loadCases(currentLoadCase)%stress, &
                        rotation_BC        = loadCases(currentLoadCase)%rot)
  
              case(FIELD_THERMAL_ID); call grid_thermal_spectral_forward(cutBack)
              case(FIELD_DAMAGE_ID);  call grid_damage_spectral_forward(cutBack)
            end select
          enddo
          if(.not. cutBack) call CPFEM_forward

!--------------------------------------------------------------------------------------------------
! solve fields
          stagIter = 0
          stagIterate = .true.
          do while (stagIterate)
            do field = 1, nActiveFields
              select case(loadCases(currentLoadCase)%ID(field))
                case(FIELD_MECH_ID)
                  solres(field) = mech_solution (&
                                         incInfo,timeinc,timeIncOld, &
                                         stress_BC   = loadCases(currentLoadCase)%stress, &
                                         rotation_BC = loadCases(currentLoadCase)%rot)
  
                case(FIELD_THERMAL_ID)
                  solres(field) = grid_thermal_spectral_solution(timeinc,timeIncOld)
  
                case(FIELD_DAMAGE_ID)
                  solres(field) = grid_damage_spectral_solution(timeinc,timeIncOld)
  
              end select
  
              if (.not. solres(field)%converged) exit                                               ! no solution found
  
            enddo
            stagIter = stagIter + 1
            stagIterate =            stagIter < stagItMax &
                         .and.       all(solres(:)%converged) &
                         .and. .not. all(solres(:)%stagConverged)                                   ! stationary with respect to staggered iteration
          enddo

!--------------------------------------------------------------------------------------------------
! check solution for either advance or retry

          if ( (all(solres(:)%converged .and. solres(:)%stagConverged)) &                           ! converged
               .and. .not. solres(1)%termIll) then                                                  ! and acceptable solution found
            call mech_updateCoords
            timeIncOld = timeinc
            cutBack = .false.
            guess = .true.                                                                          ! start guessing after first converged (sub)inc
            if (worldrank == 0) then
              write(statUnit,*) totalIncsCounter, time, cutBackLevel, &
                                solres%converged, solres%iterationsNeeded
              flush(statUnit)
            endif
          elseif (cutBackLevel < maxCutBack) then                                                   ! further cutbacking tolerated?
            cutBack = .true.
            stepFraction = (stepFraction - 1) * subStepFactor                                       ! adjust to new denominator
            cutBackLevel = cutBackLevel + 1
            time    = time - timeinc                                                                ! rewind time
            timeinc = timeinc/real(subStepFactor,pReal)                                             ! cut timestep
            write(6,'(/,a)') ' cutting back '
          else                                                                                      ! no more options to continue
            call IO_warning(850)
            if (worldrank == 0) close(statUnit)
            call quit(0)                                                                            ! quit
          endif
  
        enddo subStepLooping
  
        cutBackLevel = max(0, cutBackLevel - 1)                                                     ! try half number of subincs next inc
  
        if (all(solres(:)%converged)) then
          write(6,'(/,a,i0,a)') ' increment ', totalIncsCounter, ' converged'
        else
          write(6,'(/,a,i0,a)') ' increment ', totalIncsCounter, ' NOT converged'
        endif; flush(6)
  
        if (mod(inc,loadCases(currentLoadCase)%outputFrequency) == 0) then                          ! at output frequency
          write(6,'(1/,a)') ' ... writing results to file ......................................'
          flush(6)
          call CPFEM_results(totalIncsCounter,time)
        endif
        if (mod(inc,loadCases(currentLoadCase)%restartFrequency) == 0) then
          call mech_restartWrite
          call CPFEM_restartWrite
        endif
      endif skipping
  
     enddo incLooping
  
  enddo loadCaseLooping
 
 
!--------------------------------------------------------------------------------------------------
! report summary of whole calculation
  write(6,'(/,a)') ' ###########################################################################'
  if (worldrank == 0) close(statUnit)
  
  call quit(0)                                                                                      ! no complains ;)
  
end program DAMASK_grid
