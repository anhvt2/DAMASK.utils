!--------------------------------------------------------------------------------------------------
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Grid solver for mechanics: Spectral Polarisation
!--------------------------------------------------------------------------------------------------
module grid_mechanical_spectral_polarization
#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/petscdmda.h>
  use PETScDMDA
  use PETScSNES
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  use MPI_f08
#endif

  use prec
  use parallelization
  use CLI
  use misc
  use IO
  use HDF5
  use HDF5_utilities
  use math
  use rotations
  use spectral_utilities
  use grid_mech_utilities
  use config
  use homogenization
  use discretization_grid
  use constants

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  implicit none(type,external)
#else
  implicit none
#endif
  private

  type(tSolutionParams) :: params

  type :: tNumerics
    logical :: update_gamma                                                                         !< update gamma operator with current stiffness
    integer :: &
      itmin, &                                                                                      !< minimum number of iterations
      itmax                                                                                         !< maximum number of iterations
    real(pREAL) :: &
      eps_div_atol, &                                                                               !< absolute tolerance for equilibrium
      eps_div_rtol, &                                                                               !< relative tolerance for equilibrium
      eps_curl_atol, &                                                                              !< absolute tolerance for compatibility
      eps_curl_rtol, &                                                                              !< relative tolerance for compatibility
      eps_stress_atol, &                                                                            !< absolute tolerance for fullfillment of stress BC
      eps_stress_rtol                                                                               !< relative tolerance for fullfillment of stress BC
    real(pREAL) :: &
      alpha, &                                                                                      !< polarization scheme parameter 0.0 < alpha < 2.0. alpha = 1.0 ==> AL scheme, alpha = 2.0 ==> accelerated scheme
      beta                                                                                          !< polarization scheme parameter 0.0 < beta < 2.0. beta = 1.0 ==> AL scheme, beta = 2.0 ==> accelerated scheme
  end type tNumerics

  type(tNumerics) :: num                                                                            ! numerics parameters. Better name?

!--------------------------------------------------------------------------------------------------
! PETSc data
  DM   :: DM_mech
  SNES :: SNES_mech
  Vec  :: FandF_tau_PETSc

!--------------------------------------------------------------------------------------------------
! common pointwise data
  real(pREAL), dimension(:,:,:,:,:), allocatable :: &
    F_lastInc, &                                                                                    !< field of previous compatible deformation gradients
    F_tau_lastInc, &                                                                                !< field of previous incompatible deformation gradient
    Fdot, &                                                                                         !< field of assumed rate of compatible deformation gradient
    F_tauDot                                                                                        !< field of assumed rate of incopatible deformation gradient

!--------------------------------------------------------------------------------------------------
! stress, stiffness and compliance average etc.
  real(pREAL), dimension(3,3) :: &
    F_aimDot = 0.0_pREAL, &                                                                         !< assumed rate of average deformation gradient
    F_aim = math_I3, &                                                                              !< current prescribed deformation gradient
    F_aim_lastInc = math_I3, &                                                                      !< previous average deformation gradient
    F_av = 0.0_pREAL, &                                                                             !< average incompatible def grad field
    P_av = 0.0_pREAL, &                                                                             !< average 1st Piola--Kirchhoff stress
    P_aim = 0.0_pREAL
  character(len=:), allocatable :: incInfo                                                          !< time and increment information
  real(pREAL), dimension(3,3,3,3) :: &
    C_volAvg = 0.0_pREAL, &                                                                         !< current volume average stiffness
    C_volAvgLastInc = 0.0_pREAL, &                                                                  !< previous volume average stiffness
    C_minMaxAvg = 0.0_pREAL, &                                                                      !< current (min+max)/2 stiffness
    C_minMaxAvgLastInc = 0.0_pREAL, &                                                               !< previous (min+max)/2 stiffness
    C_minMaxAvgRestart = 0.0_pREAL, &                                                              !< (min+max)/2 stiffnes (restart)
    S = 0.0_pREAL, &                                                                                !< current compliance (filled up with zeros)
    C_scale = 0.0_pREAL, &
    S_scale = 0.0_pREAL

  real(pREAL) :: &
    err_BC, &                                                                                       !< deviation from stress BC
    err_curl, &                                                                                     !< RMS of curl of F
    err_div                                                                                         !< RMS of div of P

  integer :: totalIter = 0                                                                          !< total iteration in current increment
  integer(kind(STATUS_OK)) :: status

  public :: &
    grid_mechanical_spectral_polarization_init, &
    grid_mechanical_spectral_polarization_solution, &
    grid_mechanical_spectral_polarization_forward, &
    grid_mechanical_spectral_polarization_updateCoords, &
    grid_mechanical_spectral_polarization_restartWrite

contains

!--------------------------------------------------------------------------------------------------
!> @brief Allocate all necessary fields and fill them with data, potentially from restart info.
!--------------------------------------------------------------------------------------------------
subroutine grid_mechanical_spectral_polarization_init(num_grid)

  type(tDict), pointer, intent(in) :: num_grid

  real(pREAL), dimension(3,3,cells(1),cells(2),cells3) :: P
  PetscErrorCode :: err_PETSc
  integer(MPI_INTEGER_KIND) :: err_MPI
  real(pREAL), pointer, dimension(:,:,:,:) :: &
    FandF_tau, &                                                                                    ! overall pointer to solution data
    F, &                                                                                            ! specific (sub)pointer
    F_tau                                                                                           ! specific (sub)pointer
  integer(MPI_INTEGER_KIND), dimension(0:worldsize-1) :: cells3_global
  real(pREAL), dimension(3,3,product(cells(1:2))*cells3) :: temp33n
  integer(HID_T) :: fileHandle, groupHandle
  type(tDict), pointer :: &
    num_grid_mech
  character(len=:), allocatable :: &
    extmsg, &
    petsc_options


  print '(/,1x,a)', '<<<+-  grid_mechanical_spectral_polarization init  -+>>>'; flush(IO_STDOUT)

  print '(/,1x,a)', 'P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015'
  print '(  1x,a)', 'https://doi.org/10.1016/j.ijplas.2014.02.006'

!-------------------------------------------------------------------------------------------------
! read numerical parameters and do sanity checks
  num_grid_mech => num_grid%get_dict('mechanical',defaultVal=emptyDict)

  num%itmin           = num_grid_mech%get_asInt('N_iter_min',defaultVal=1)
  num%itmax           = num_grid_mech%get_asInt('N_iter_max',defaultVal=100)
  num%update_gamma    = num_grid_mech%get_asBool('update_gamma',defaultVal=.false.)
  num%eps_div_atol    = num_grid_mech%get_asReal('eps_abs_div(P)', defaultVal=1.0e-4_pREAL)
  num%eps_div_rtol    = num_grid_mech%get_asReal('eps_rel_div(P)', defaultVal=5.0e-4_pREAL)
  num%eps_curl_atol   = num_grid_mech%get_asReal('eps_abs_curl(F)',defaultVal=1.0e-10_pREAL)
  num%eps_curl_rtol   = num_grid_mech%get_asReal('eps_rel_curl(F)',defaultVal=5.0e-4_pREAL)
  num%eps_stress_atol = num_grid_mech%get_asReal('eps_abs_P',      defaultVal=1.0e3_pREAL)
  num%eps_stress_rtol = num_grid_mech%get_asReal('eps_rel_P',      defaultVal=1.0e-3_pREAL)
  num%alpha           = num_grid_mech%get_asReal('alpha',          defaultVal=1.0_pREAL)
  num%beta            = num_grid_mech%get_asReal('beta',           defaultVal=1.0_pREAL)

  extmsg = ''
  if (num%eps_div_atol <= 0.0_pREAL)                      extmsg = trim(extmsg)//' eps_abs_div(P)'
  if (num%eps_div_rtol <= 0.0_pREAL)                      extmsg = trim(extmsg)//' eps_rel_div(P)'
  if (num%eps_curl_atol <= 0.0_pREAL)                     extmsg = trim(extmsg)//' eps_abs_curl(F)'
  if (num%eps_curl_rtol <= 0.0_pREAL)                     extmsg = trim(extmsg)//' eps_rel_curl(F)'
  if (num%eps_stress_atol <= 0.0_pREAL)                   extmsg = trim(extmsg)//' eps_abs_P'
  if (num%eps_stress_rtol <= 0.0_pREAL)                   extmsg = trim(extmsg)//' eps_rel_P'
  if (num%alpha <= 0.0_pREAL .or. num%alpha >  2.0_pREAL) extmsg = trim(extmsg)//' alpha'
  if (num%beta < 0.0_pREAL .or. num%beta > 2.0_pREAL)     extmsg = trim(extmsg)//' beta'
  if (num%itmax < 1)                                      extmsg = trim(extmsg)//' N_iter_max'
  if (num%itmin > num%itmax .or. num%itmin < 1)           extmsg = trim(extmsg)//' N_iter_min'

  if (extmsg /= '') call IO_error(301,ext_msg=trim(extmsg))

!--------------------------------------------------------------------------------------------------
! set default and user defined options for PETSc
  petsc_options = misc_prefixOptions('-snes_type ngmres '//num_grid_mech%get_asStr('PETSc_options',defaultVal=''), &
                                     'mechanical_')
  call PetscOptionsInsertString(PETSC_NULL_OPTIONS,petsc_options,err_PETSc)
  CHKERRQ(err_PETSc)

!--------------------------------------------------------------------------------------------------
! allocate global fields
  allocate(F_lastInc    (3,3,cells(1),cells(2),cells3),source = 0.0_pREAL)
  allocate(Fdot         (3,3,cells(1),cells(2),cells3),source = 0.0_pREAL)
  allocate(F_tau_lastInc(3,3,cells(1),cells(2),cells3),source = 0.0_pREAL)
  allocate(F_tauDot     (3,3,cells(1),cells(2),cells3),source = 0.0_pREAL)

!--------------------------------------------------------------------------------------------------
! initialize solver specific parts of PETSc
  call SNESCreate(PETSC_COMM_WORLD,SNES_mech,err_PETSc)
  CHKERRQ(err_PETSc)
  call SNESSetOptionsPrefix(SNES_mech,'mechanical_',err_PETSc)
  CHKERRQ(err_PETSc)
  call MPI_Allgather(int(cells3,pPetscInt),1_MPI_INTEGER_KIND,MPI_INTEGER,&
                     cells3_global,1_MPI_INTEGER_KIND,MPI_INTEGER,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)
  call DMDACreate3d(PETSC_COMM_WORLD, &
         DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &                                    ! cut off stencil at boundary
         DMDA_STENCIL_BOX, &                                                                        ! Moore (26) neighborhood around central point
         int(cells(1),pPETSCINT),int(cells(2),pPETSCINT),int(cells(3),pPETSCINT), &                 ! global cells
         1_pPETSCINT, 1_pPETSCINT, int(worldsize,pPETSCINT), &
         18_pPETSCINT, 0_pPETSCINT, &                                                               ! #dof (2xtensor), ghost boundary width (domain overlap)
         [int(cells(1),pPETSCINT)],[int(cells(2),pPETSCINT)],int(cells3_global,pPETSCINT), &        ! local cells
         DM_mech,err_PETSc)                                                                         ! handle, error
  CHKERRQ(err_PETSc)
  call DMsetFromOptions(DM_mech,err_PETSc)
  CHKERRQ(err_PETSc)
  call DMsetUp(DM_mech,err_PETSc)
  CHKERRQ(err_PETSc)
  call DMcreateGlobalVector(DM_mech,FandF_tau_PETSc,err_PETSc)                                      ! global solution vector (cells x 18, i.e. every def grad tensor)
  CHKERRQ(err_PETSc)
  call DMDASNESsetFunctionLocal(DM_mech,INSERT_VALUES,formResidual,PETSC_NULL_SNES,err_PETSc)       ! residual vector of same shape as solution vector
  CHKERRQ(err_PETSc)
  call SNESsetConvergenceTest(SNES_mech,converged,PETSC_NULL_SNES,PETSC_NULL_FUNCTION,err_PETSc)    ! specify custom convergence check function "converged"
  CHKERRQ(err_PETSc)
  call SNESSetDM(SNES_mech,DM_mech,err_PETSc)
  CHKERRQ(err_PETSc)
  call SNESsetFromOptions(SNES_mech,err_PETSc)                                                      ! pull it all together with additional CLI arguments
  CHKERRQ(err_PETSc)

!--------------------------------------------------------------------------------------------------
! init fields
  call DMDAVecGetArrayF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)                              ! places pointer on PETSc data
  CHKERRQ(err_PETSc)
  F     => FandF_tau(0: 8,:,:,:)
  F_tau => FandF_tau(9:17,:,:,:)

  restartRead: if (CLI_restartInc > 0) then
    print '(/,1x,a,1x,i0)', 'loading restart data of increment', CLI_restartInc

    fileHandle  = HDF5_openFile(getSolverJobName()//'_restart.hdf5','r')
    groupHandle = HDF5_openGroup(fileHandle,'solver')

    call HDF5_read(P_aim,groupHandle,'P_aim',.false.)
    call MPI_Bcast(P_aim,9_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(F_aim,groupHandle,'F_aim',.false.)
    call MPI_Bcast(F_aim,9_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(F_aim_lastInc,groupHandle,'F_aim_lastInc',.false.)
    call MPI_Bcast(F_aim_lastInc,9_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(F_aimDot,groupHandle,'F_aimDot',.false.)
    call MPI_Bcast(F_aimDot,9_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(temp33n,groupHandle,'F')
    F = reshape(temp33n,[9,cells(1),cells(2),cells3])
    call HDF5_read(temp33n,groupHandle,'F_lastInc')
    F_lastInc = reshape(temp33n,[3,3,cells(1),cells(2),cells3])
    call HDF5_read(temp33n,groupHandle,'F_tau')
    F_tau = reshape(temp33n,[9,cells(1),cells(2),cells3])
    call HDF5_read(temp33n,groupHandle,'F_tau_lastInc')
    F_tau_lastInc = reshape(temp33n,[3,3,cells(1),cells(2),cells3])

  elseif (CLI_restartInc == 0) then restartRead
    F_lastInc = spread(spread(spread(math_I3,3,cells(1)),4,cells(2)),5,cells3)                      ! initialize to identity
    F = reshape(F_lastInc,[9,cells(1),cells(2),cells3])
    F_tau = 2.0_pREAL*F
    F_tau_lastInc = 2.0_pREAL*F_lastInc
  end if restartRead

  call utilities_updateCoords(reshape(F,shape(F_lastInc)))
  call utilities_constitutiveResponse(status,P,P_av,C_volAvg,C_minMaxAvg, &                         ! stress field, stress avg, global average of stiffness and (min+max)/2
                                      reshape(F,shape(F_lastInc)), &                                ! target F
                                      0.0_pREAL)                                                    ! time increment
  call DMDAVecRestoreArrayF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)                          ! deassociate pointer
  CHKERRQ(err_PETSc)

  restartRead2: if (CLI_restartInc > 0) then
    print '(1x,a,1x,i0)', 'loading additional restart data of increment', CLI_restartInc
    call HDF5_read(C_volAvg,groupHandle,'C_volAvg',.false.)
    call MPI_Bcast(C_volAvg,81_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(C_volAvgLastInc,groupHandle,'C_volAvgLastInc',.false.)
    call MPI_Bcast(C_volAvgLastInc,81_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)
    call HDF5_read(C_minMaxAvg,groupHandle,'C_minMaxAvg',.false.)
    call MPI_Bcast(C_minMaxAvg,81_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
    call parallelization_chkerr(err_MPI)

    call HDF5_closeGroup(groupHandle)
    call HDF5_closeFile(fileHandle)

  end if restartRead2

  call utilities_updateGamma(C_minMaxAvg)
  C_minMaxAvgRestart = C_minMaxAvg
  C_scale = C_minMaxAvg
  S_scale = math_invSym3333(C_minMaxAvg)

end subroutine grid_mechanical_spectral_polarization_init


!--------------------------------------------------------------------------------------------------
!> @brief solution for the Polarisation scheme with internal iterations
!--------------------------------------------------------------------------------------------------
function grid_mechanical_spectral_polarization_solution(incInfoIn) result(solution)

!--------------------------------------------------------------------------------------------------
! input data for solution
  character(len=*), intent(in) :: &
    incInfoIn
  type(tSolutionState) :: &
    solution
!--------------------------------------------------------------------------------------------------
! PETSc Data
  PetscErrorCode :: err_PETSc
  SNESConvergedReason :: reason

  incInfo = incInfoIn

!--------------------------------------------------------------------------------------------------
! update stiffness (and gamma operator)
  S = utilities_maskedCompliance(params%rotation_BC,params%stress_mask,C_volAvg)
  if (num%update_gamma) then
    call utilities_updateGamma(C_minMaxAvg)
    C_scale = C_minMaxAvg
    S_scale = math_invSym3333(C_minMaxAvg)
  end if

  call SNESSolve(SNES_mech,PETSC_NULL_VEC,FandF_tau_PETSc,err_PETSc)
  CHKERRQ(err_PETSc)
  call SNESGetConvergedReason(SNES_mech,reason,err_PETSc)
  CHKERRQ(err_PETSc)

  solution%converged = reason > 0
  solution%iterationsNeeded = totalIter
  P_aim = merge(P_av,P_aim,params%stress_mask)

end function grid_mechanical_spectral_polarization_solution


!--------------------------------------------------------------------------------------------------
!> @brief forwarding routine
!> @details find new boundary conditions and best F estimate for end of current timestep
!--------------------------------------------------------------------------------------------------
subroutine grid_mechanical_spectral_polarization_forward(cutBack,guess,Delta_t,Delta_t_old,t_remaining,&
                                                   deformation_BC,stress_BC,rotation_BC)

  logical,                  intent(in) :: &
    cutBack, &
    guess
  real(pREAL),              intent(in) :: &
    Delta_t_old, &
    Delta_t, &
    t_remaining                                                                                     !< remaining time of current load case
  type(tBoundaryCondition), intent(in) :: &
    stress_BC, &
    deformation_BC
  type(tRotation),           intent(in) :: &
    rotation_BC
  PetscErrorCode :: err_PETSc
  real(pREAL), pointer, dimension(:,:,:,:) :: FandF_tau, F, F_tau
  integer :: i, j, k
  real(pREAL), dimension(3,3) :: F_lambda33


  call DMDAVecGetArrayF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)
  F     => FandF_tau(0: 8,:,:,:)
  F_tau => FandF_tau(9:17,:,:,:)

  if (cutBack) then
    C_volAvg    = C_volAvgLastInc
    C_minMaxAvg = C_minMaxAvgLastInc
  else
    C_volAvgLastInc    = C_volAvg
    C_minMaxAvgLastInc = C_minMaxAvg

    F_aimDot = merge(merge(.0_pREAL,(F_aim-F_aim_lastInc)/Delta_t_old,stress_BC%mask),.0_pREAL,guess)  ! estimate deformation rate for prescribed stress components
    F_aim_lastInc = F_aim

    !-----------------------------------------------------------------------------------------------
    ! calculate rate for aim
    if     (deformation_BC%myType=='L') then                                                        ! calculate F_aimDot from given L and current F
      F_aimDot = F_aimDot &
               + matmul(merge(.0_pREAL,deformation_BC%values,deformation_BC%mask),F_aim_lastInc)
    elseif (deformation_BC%myType=='dot_F') then                                                    ! F_aimDot is prescribed
      F_aimDot = F_aimDot &
               + merge(.0_pREAL,deformation_BC%values,deformation_BC%mask)
    elseif (deformation_BC%myType=='F') then                                                        ! aim at end of load case is prescribed
      F_aimDot = F_aimDot &
               + merge(.0_pREAL,(deformation_BC%values - F_aim_lastInc)/t_remaining,deformation_BC%mask)
    end if

    Fdot     = utilities_calculateRate(guess, &
                                       F_lastInc,reshape(F,[3,3,cells(1),cells(2),cells3]),Delta_t_old, &
                                       rotation_BC%rotate(F_aimDot,active=.true.))
    F_tauDot = utilities_calculateRate(guess, &
                                       F_tau_lastInc,reshape(F_tau,[3,3,cells(1),cells(2),cells3]), Delta_t_old, &
                                       rotation_BC%rotate(F_aimDot,active=.true.))
    F_lastInc     = reshape(F,    [3,3,cells(1),cells(2),cells3])
    F_tau_lastInc = reshape(F_tau,[3,3,cells(1),cells(2),cells3])

  end if

!--------------------------------------------------------------------------------------------------
! update average and local deformation gradients
  F_aim = F_aim_lastInc + F_aimDot * Delta_t
  if (stress_BC%myType=='P')     P_aim = P_aim &
                                      + merge(.0_pREAL,(stress_BC%values - P_aim)/t_remaining,stress_BC%mask)*Delta_t
  if (stress_BC%myType=='dot_P') P_aim = P_aim &
                                      + merge(.0_pREAL,stress_BC%values,stress_BC%mask)*Delta_t

  F = reshape(utilities_forwardTensorField(Delta_t,F_lastInc,Fdot, &                                ! estimate of F at end of time+Delta_t that matches rotated F_aim on average
                                           rotation_BC%rotate(F_aim,active=.true.)),&
              [9,cells(1),cells(2),cells3])
  if (guess) then
     F_tau = reshape(Utilities_forwardTensorField(Delta_t,F_tau_lastInc,F_taudot), &
                     [9,cells(1),cells(2),cells3])                                                  ! does not have any average value as boundary condition
   else
    do k = 1, cells3; do j = 1, cells(2); do i = 1, cells(1)
       F_lambda33 = reshape(F_tau(1:9,i,j,k)-F(1:9,i,j,k),[3,3])
       F_lambda33 = math_I3 &
                  + math_mul3333xx33(S_scale,0.5_pREAL*matmul(F_lambda33, &
                    math_mul3333xx33(C_scale,matmul(transpose(F_lambda33),F_lambda33)-math_I3)))
       F_tau(1:9,i,j,k) = reshape(F_lambda33,[9])+F(1:9,i,j,k)
    end do; end do; end do
  end if

  call DMDAVecRestoreArrayF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)

!--------------------------------------------------------------------------------------------------
! set module wide available data
  params%stress_mask = stress_BC%mask
  params%rotation_BC = rotation_BC
  params%Delta_t     = Delta_t

end subroutine grid_mechanical_spectral_polarization_forward


!--------------------------------------------------------------------------------------------------
!> @brief Update coordinates.
!--------------------------------------------------------------------------------------------------
subroutine grid_mechanical_spectral_polarization_updateCoords()

  PetscErrorCode :: err_PETSc
  real(pREAL), dimension(:,:,:,:), pointer :: FandF_tau

  call DMDAVecGetArrayReadF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)
  call utilities_updateCoords(reshape(FandF_tau(0:8,:,:,:),[3,3,size(FandF_tau,2),size(FandF_tau,3),size(FandF_tau,4)]))
  call DMDAVecRestoreArrayReadF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)

end subroutine grid_mechanical_spectral_polarization_updateCoords


!--------------------------------------------------------------------------------------------------
!> @brief Write current solver and constitutive data for restart to file.
!--------------------------------------------------------------------------------------------------
subroutine grid_mechanical_spectral_polarization_restartWrite()

  PetscErrorCode :: err_PETSc
  integer(HID_T) :: fileHandle, groupHandle
  real(pREAL), dimension(:,:,:,:), pointer :: FandF_tau, F, F_tau

  call DMDAVecGetArrayReadF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)
  F     => FandF_tau(0: 8,:,:,:)
  F_tau => FandF_tau(9:17,:,:,:)

  if (num%update_gamma) C_minMaxAvgRestart = C_minMaxAvg

  print '(1x,a)', 'saving solver data required for restart'; flush(IO_STDOUT)

  fileHandle  = HDF5_openFile(getSolverJobName()//'_restart.hdf5','w')
  groupHandle = HDF5_addGroup(fileHandle,'solver')
  call HDF5_write(reshape(F,[3,3,product(cells(1:2))*cells3]),groupHandle,'F')
  call HDF5_write(reshape(F_lastInc,[3,3,product(cells(1:2))*cells3]),groupHandle,'F_lastInc')
  call HDF5_write(reshape(F_tau,[3,3,product(cells(1:2))*cells3]),groupHandle,'F_tau')
  call HDF5_write(reshape(F_tau_lastInc,[3,3,product(cells(1:2))*cells3]),groupHandle,'F_tau_lastInc')
  call HDF5_closeGroup(groupHandle)
  call HDF5_closeFile(fileHandle)

  if (worldrank == 0) then
    fileHandle  = HDF5_openFile(getSolverJobName()//'_restart.hdf5','a',.false.)
    groupHandle = HDF5_openGroup(fileHandle,'solver')
    call HDF5_write(F_aim,groupHandle,'P_aim',.false.)
    call HDF5_write(F_aim,groupHandle,'F_aim',.false.)
    call HDF5_write(F_aim_lastInc,groupHandle,'F_aim_lastInc',.false.)
    call HDF5_write(F_aimDot,groupHandle,'F_aimDot',.false.)
    call HDF5_write(C_volAvg,groupHandle,'C_volAvg',.false.)
    call HDF5_write(C_volAvgLastInc,groupHandle,'C_volAvgLastInc',.false.)
    call HDF5_write(C_minMaxAvgRestart,groupHandle,'C_minMaxAvg',.false.)
    call HDF5_closeGroup(groupHandle)
    call HDF5_closeFile(fileHandle)
  end if

  call DMDAVecRestoreArrayReadF90(DM_mech,FandF_tau_PETSc,FandF_tau,err_PETSc)
  CHKERRQ(err_PETSc)

end subroutine grid_mechanical_spectral_polarization_restartWrite


!--------------------------------------------------------------------------------------------------
!> @brief convergence check
!--------------------------------------------------------------------------------------------------
subroutine converged(snes_local,PETScIter,devNull1,devNull2,devNull3,reason,dummy,err_PETSc)

  SNES :: snes_local
  PetscInt,  intent(in) :: PETScIter
  PetscReal, intent(in) :: &
    devNull1, &
    devNull2, &
    devNull3
  SNESConvergedReason :: reason
  PetscObject :: dummy
  PetscErrorCode :: err_PETSc
  real(pREAL) :: &
    curlTol, &
    divTol, &
    BCTol

  curlTol = max(maxval(abs(F_aim-math_I3))*num%eps_curl_rtol, num%eps_curl_atol)
  divTol = max(maxval(abs(P_av))*num%eps_div_rtol, num%eps_div_atol)
  BCTol = max(maxval(abs(P_av))*num%eps_stress_rtol, num%eps_stress_atol)

  if (totalIter >= num%itmin .and. all([err_div/divTol, err_curl/curlTol, err_BC/BCTol] < 1.0_pREAL) &
       .and. status == STATUS_OK) then
    reason = 1
  elseif (totalIter >= num%itmax) then
    reason = -1
  else
    reason = 0
  end if

  print '(/,1x,a)', '... reporting .............................................................'
  print '(/,1x,a,f12.2,a,es8.2,a,es9.2,a)', 'error divergence = ', &
            err_div/divTol,  ' (',err_div, ' / m, tol = ',divTol,')'
  print   '(1x,a,f12.2,a,es8.2,a,es9.2,a)', 'error curl       = ', &
            err_curl/curlTol,' (',err_curl,' -,   tol = ',curlTol,')'
  print   '(1x,a,f12.2,a,es8.2,a,es9.2,a)', 'error mech BC    = ', &
            err_BC/BCTol,    ' (',err_BC,  ' Pa,  tol = ',BCTol,')'
  print '(/,1x,a)', '==========================================================================='
  flush(IO_STDOUT)
  err_PETSc = 0

end subroutine converged


!--------------------------------------------------------------------------------------------------
!> @brief Construct the residual vector.
!--------------------------------------------------------------------------------------------------
subroutine formResidual(residual_subdomain, FandF_tau, &
                        r, dummy,err_PETSc)

  DMDALocalInfo, dimension(DMDA_LOCAL_INFO_SIZE) :: residual_subdomain                              !< DMDA info (needs to be named "in" for macros like XRANGE to work)
  real(pREAL), dimension(3,3,2,cells(1),cells(2),cells3), target, intent(in) :: &
    FandF_tau                                                                                       !< deformation gradient field
  real(pREAL), dimension(3,3,2,cells(1),cells(2),cells3), target, intent(out) :: &
    r                                                                                               !< residuum field
  PetscObject :: dummy
  PetscErrorCode :: err_PETSc

  real(pREAL), pointer, dimension(:,:,:,:,:) :: &
    F, &
    F_tau, &
    r_F, &
    r_F_tau
  PetscInt :: &
    PETScIter, &
    nfuncs
  integer(MPI_INTEGER_KIND) :: err_MPI
  integer :: &
    i, j, k, ce


  F       => FandF_tau(1:3,1:3,1,1:cells(1),1:cells(2),1:cells3)
  F_tau   => FandF_tau(1:3,1:3,2,1:cells(1),1:cells(2),1:cells3)
  r_F     => r(1:3,1:3,1,1:cells(1),1:cells(2),1:cells3)
  r_F_tau => r(1:3,1:3,2,1:cells(1),1:cells(2),1:cells3)

  F_av = sum(sum(sum(F,dim=5),dim=4),dim=3) * wgt
  call MPI_Allreduce(MPI_IN_PLACE,F_av,9_MPI_INTEGER_KIND,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)

  call SNESGetNumberFunctionEvals(SNES_mech,nfuncs,err_PETSc)
  CHKERRQ(err_PETSc)
  call SNESGetIterationNumber(SNES_mech,PETScIter,err_PETSc)
  CHKERRQ(err_PETSc)

  if (nfuncs == 0 .and. PETScIter == 0) totalIter = -1                                              ! new increment

  newIteration: if (totalIter <= PETScIter) then
    totalIter = totalIter + 1
    print '(1x,a,3(a,i0))', trim(incInfo), ' @ Iteration ', num%itmin, '≤',totalIter, '≤', num%itmax
    if (any(dNeq(params%rotation_BC%asQuaternion(), real([1.0, 0.0, 0.0, 0.0],pREAL)))) &
      print '(/,1x,a,/,2(3(f12.7,1x)/),3(f12.7,1x))', &
      'deformation gradient aim (lab) =', transpose(params%rotation_BC%rotate(F_aim,active=.true.))
    print '(/,1x,a,/,2(3(f12.7,1x)/),3(f12.7,1x))', &
      'deformation gradient aim       =', transpose(F_aim)
    flush(IO_STDOUT)
  end if newIteration

  do k = 1, cells3; do j = 1, cells(2); do i = 1, cells(1)
    r_F_tau(1:3,1:3,i,j,k) = &
      num%beta*math_mul3333xx33(C_scale,F(1:3,1:3,i,j,k) - math_I3) -&
      num%alpha*matmul(F(1:3,1:3,i,j,k), &
                         math_mul3333xx33(C_scale,F_tau(1:3,1:3,i,j,k) - F(1:3,1:3,i,j,k) - math_I3))
  end do; end do; end do
  r_F_tau = num%beta*F &
          - utilities_GammaConvolution(r_F_tau,params%rotation_BC%rotate(num%beta*F_aim,active=.true.))

  err_curl = utilities_curlRMS(F)

#ifdef __GFORTRAN__
  call utilities_constitutiveResponse(status,r_F, &
#else
  associate (P => r_F)
    call utilities_constitutiveResponse(status, P, &
#endif
                                        P_av,C_volAvg,C_minMaxAvg, &
                                        F - r_F_tau/num%beta,params%Delta_t,params%rotation_BC)
    call MPI_Allreduce(MPI_IN_PLACE,status,1_MPI_INTEGER_KIND,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err_MPI)
#ifdef __GFORTRAN__
    err_div = utilities_divergenceRMS(r_F)
#else
    err_div = utilities_divergenceRMS(P)
#endif
    ce = 0
    do k = 1, cells3; do j = 1, cells(2); do i = 1, cells(1)
      ce = ce + 1
      r_F(1:3,1:3,i,j,k) = &
        math_mul3333xx33(math_invSym3333(homogenization_dPdF(1:3,1:3,1:3,1:3,ce) + C_scale), &
#ifdef __GFORTRAN__
                         r_F(1:3,1:3,i,j,k) - matmul(F(1:3,1:3,i,j,k), &
#else
                         P(1:3,1:3,i,j,k) - matmul(F(1:3,1:3,i,j,k), &
#endif
                         math_mul3333xx33(C_scale,F_tau(1:3,1:3,i,j,k) - F(1:3,1:3,i,j,k) - math_I3))) &
                         + r_F_tau(1:3,1:3,i,j,k)
    end do; end do; end do
#ifndef __GFORTRAN__
  end associate
#endif

  F_aim = F_aim - math_mul3333xx33(S, P_av - P_aim)                                                 ! S = 0.0 for no bc
  err_BC = maxval(abs(merge(math_mul3333xx33(C_scale,F_aim-params%rotation_BC%rotate(F_av)), &
                            P_av-P_aim, &
                            params%stress_mask)))


end subroutine formResidual

end module grid_mechanical_spectral_polarization
