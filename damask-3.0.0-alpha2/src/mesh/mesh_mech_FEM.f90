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
!> @brief FEM PETSc solver
!--------------------------------------------------------------------------------------------------
module mesh_mech_FEM
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petsc.h>

  use PETScsnes
  use PETScDM
  use PETScDMplex
  use PETScDT

  use prec
  use FEM_utilities
  use discretization_mesh
  use DAMASK_interface
  use config
  use IO
  use FEM_quadrature
  use homogenization
  use math

  implicit none
  private

!--------------------------------------------------------------------------------------------------
! derived types
  type tSolutionParams
    type(tFieldBC)  :: fieldBC
    real(pReal)     :: timeinc
    real(pReal)     :: timeincOld
  end type tSolutionParams

  type(tSolutionParams)  :: params

  type, private :: tNumerics
    integer :: &
      integrationOrder, &                                                                           !< order of quadrature rule required
      itmax
    logical :: &
      BBarStabilisation
    real(pReal) :: &
      eps_struct_atol, &                                                                            !< absolute tolerance for mechanical equilibrium
      eps_struct_rtol                                                                               !< relative tolerance for mechanical equilibrium
  end type tNumerics   

  type(tNumerics), private :: num 
!--------------------------------------------------------------------------------------------------
! PETSc data
  SNES                           :: mech_snes
  Vec                            :: solution, solution_rate, solution_local
  PetscInt                       :: dimPlex, cellDof, nQuadrature, nBasis
  PetscReal, allocatable, target :: qPoints(:), qWeights(:)
  MatNullSpace                   :: matnull

!--------------------------------------------------------------------------------------------------
! stress, stiffness and compliance average etc.
  character(len=pStringLen) :: incInfo
  real(pReal), dimension(3,3) :: &
    P_av = 0.0_pReal
  logical :: ForwardData
  real(pReal), parameter :: eps = 1.0e-18_pReal

  public :: &
    FEM_mech_init, &
    FEM_mech_solution, &
    FEM_mech_forward

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields and fills them with data
!--------------------------------------------------------------------------------------------------
subroutine FEM_mech_init(fieldBC)

  type(tFieldBC),             intent(in) :: fieldBC

  DM                                     :: mech_mesh
  PetscFE                                :: mechFE
  PetscQuadrature                        :: mechQuad, functional
  PetscDS                                :: mechDS
  PetscDualSpace                         :: mechDualSpace
  DMLabel, dimension(:),pointer          :: nolabel=>  NULL()
  DMLabel                                :: BCLabel

  PetscInt,  dimension(:),       pointer :: pNumComp, pNumDof, pBcField, pBcPoint
  PetscInt                               :: numBC, bcSize, nc, &
                                            field, faceSet, topologDim, nNodalPoints, &
                                            cellStart, cellEnd, cell, basis

  IS                                     :: bcPoint
  IS,        dimension(:),       pointer :: pBcComps, pBcPoints
  PetscSection                           :: section

  PetscReal,      dimension(:),  pointer :: qPointsP, qWeightsP, &
                                            nodalPointsP, nodalWeightsP,pV0, pCellJ, pInvcellJ
  PetscReal                              :: detJ
  PetscReal,         allocatable, target :: cellJMat(:,:)

  PetscScalar,                   pointer :: px_scal(:)
  PetscScalar,       allocatable, target ::  x_scal(:)

  character(len=*), parameter            :: prefix = 'mechFE_'
  PetscErrorCode                         :: ierr

  class(tNode), pointer :: &
    num_mesh
   
  print'(/,a)', ' <<<+-  FEM_mech init  -+>>>'; flush(IO_STDOUT)

!-----------------------------------------------------------------------------
! read numerical parametes and do sanity checks
  num_mesh => config_numerics%get('mesh',defaultVal=emptyDict)
  num%integrationOrder  = num_mesh%get_asInt('integrationorder',defaultVal = 2)
  num%itmax             = num_mesh%get_asInt('itmax',defaultVal=250)
  num%BBarStabilisation = num_mesh%get_asBool('bbarstabilisation',defaultVal = .false.)
  num%eps_struct_atol   = num_mesh%get_asFloat('eps_struct_atol', defaultVal = 1.0e-10_pReal)
  num%eps_struct_rtol   = num_mesh%get_asFloat('eps_struct_rtol', defaultVal = 1.0e-4_pReal)
  
  if (num%itmax <= 1)                       call IO_error(301,ext_msg='itmax')
  if (num%eps_struct_rtol <= 0.0_pReal)     call IO_error(301,ext_msg='eps_struct_rtol')
  if (num%eps_struct_atol <= 0.0_pReal)     call IO_error(301,ext_msg='eps_struct_atol')

!--------------------------------------------------------------------------------------------------
! Setup FEM mech mesh
  call DMClone(geomMesh,mech_mesh,ierr); CHKERRQ(ierr)
  call DMGetDimension(mech_mesh,dimPlex,ierr); CHKERRQ(ierr)

!--------------------------------------------------------------------------------------------------
! Setup FEM mech discretization
  qPoints  = FEM_quadrature_points( dimPlex,num%integrationOrder)%p
  qWeights = FEM_quadrature_weights(dimPlex,num%integrationOrder)%p
  nQuadrature = FEM_nQuadrature(    dimPlex,num%integrationOrder)
  qPointsP  => qPoints
  qWeightsP => qWeights
  call PetscQuadratureCreate(PETSC_COMM_SELF,mechQuad,ierr); CHKERRQ(ierr)
  CHKERRQ(ierr)
  nc = dimPlex
  call PetscQuadratureSetData(mechQuad,dimPlex,nc,nQuadrature,qPointsP,qWeightsP,ierr)
  CHKERRQ(ierr)
  call PetscFECreateDefault(PETSC_COMM_SELF,dimPlex,nc,PETSC_TRUE,prefix, &
                            num%integrationOrder,mechFE,ierr); CHKERRQ(ierr)
  call PetscFESetQuadrature(mechFE,mechQuad,ierr); CHKERRQ(ierr)
  call PetscFEGetDimension(mechFE,nBasis,ierr); CHKERRQ(ierr)
  nBasis = nBasis/nc
#if (PETSC_VERSION_MINOR > 10)
  call DMAddField(mech_mesh,PETSC_NULL_DMLABEL,mechFE,ierr); CHKERRQ(ierr)
  call DMCreateDS(mech_mesh,ierr); CHKERRQ(ierr)
#endif
  call DMGetDS(mech_mesh,mechDS,ierr); CHKERRQ(ierr)
#if (PETSC_VERSION_MINOR < 11)
  call PetscDSAddDiscretization(mechDS,mechFE,ierr); CHKERRQ(ierr)
#endif
  call PetscDSGetTotalDimension(mechDS,cellDof,ierr); CHKERRQ(ierr)
  call PetscFEDestroy(mechFE,ierr); CHKERRQ(ierr)
  call PetscQuadratureDestroy(mechQuad,ierr); CHKERRQ(ierr)

!--------------------------------------------------------------------------------------------------
! Setup FEM mech boundary conditions
  call DMGetLabel(mech_mesh,'Face Sets',BCLabel,ierr); CHKERRQ(ierr)
  call DMPlexLabelComplete(mech_mesh,BCLabel,ierr); CHKERRQ(ierr)
#if (PETSC_VERSION_MINOR < 12)
  call DMGetSection(mech_mesh,section,ierr); CHKERRQ(ierr)
#else
  call DMGetLocalSection(mech_mesh,section,ierr); CHKERRQ(ierr)
#endif
  allocate(pnumComp(1), source=dimPlex)
  allocate(pnumDof(0:dimPlex), source = 0)
  do topologDim = 0, dimPlex
    call DMPlexGetDepthStratum(mech_mesh,topologDim,cellStart,cellEnd,ierr)
    CHKERRQ(ierr)
    call PetscSectionGetDof(section,cellStart,pnumDof(topologDim),ierr)
    CHKERRQ(ierr)
  enddo
  numBC = 0
  do field = 1, dimPlex; do faceSet = 1, mesh_Nboundaries
    if (fieldBC%componentBC(field)%Mask(faceSet)) numBC = numBC + 1
  enddo; enddo
  allocate(pbcField(numBC), source=0)
  allocate(pbcComps(numBC))
  allocate(pbcPoints(numBC))
  numBC = 0
  do field = 1, dimPlex; do faceSet = 1, mesh_Nboundaries
    if (fieldBC%componentBC(field)%Mask(faceSet)) then
      numBC = numBC + 1
      call ISCreateGeneral(PETSC_COMM_WORLD,1,[field-1],PETSC_COPY_VALUES,pbcComps(numBC),ierr)
      CHKERRQ(ierr)
      call DMGetStratumSize(mech_mesh,'Face Sets',mesh_boundaries(faceSet),bcSize,ierr)
      CHKERRQ(ierr)
      if (bcSize > 0) then
        call DMGetStratumIS(mech_mesh,'Face Sets',mesh_boundaries(faceSet),bcPoint,ierr)
        CHKERRQ(ierr)
        call ISGetIndicesF90(bcPoint,pBcPoint,ierr); CHKERRQ(ierr)
        call ISCreateGeneral(PETSC_COMM_WORLD,bcSize,pBcPoint,PETSC_COPY_VALUES,pbcPoints(numBC),ierr)
        CHKERRQ(ierr)
        call ISRestoreIndicesF90(bcPoint,pBcPoint,ierr); CHKERRQ(ierr)
        call ISDestroy(bcPoint,ierr); CHKERRQ(ierr)
      else
        call ISCreateGeneral(PETSC_COMM_WORLD,0,[0],PETSC_COPY_VALUES,pbcPoints(numBC),ierr)
        CHKERRQ(ierr)
      endif
    endif
  enddo; enddo
#if (PETSC_VERSION_MINOR < 11)
  call DMPlexCreateSection(mech_mesh,dimPlex,1,pNumComp,pNumDof, &
                           numBC,pBcField,pBcComps,pBcPoints,PETSC_NULL_IS,section,ierr)
#else
  call DMPlexCreateSection(mech_mesh,nolabel,pNumComp,pNumDof, &
                           numBC,pBcField,pBcComps,pBcPoints,PETSC_NULL_IS,section,ierr)

#endif
  CHKERRQ(ierr)
  call DMSetSection(mech_mesh,section,ierr); CHKERRQ(ierr)
  do faceSet = 1, numBC
    call ISDestroy(pbcPoints(faceSet),ierr); CHKERRQ(ierr)
  enddo

!--------------------------------------------------------------------------------------------------
! initialize solver specific parts of PETSc
  call SNESCreate(PETSC_COMM_WORLD,mech_snes,ierr);CHKERRQ(ierr)
  call SNESSetOptionsPrefix(mech_snes,'mech_',ierr);CHKERRQ(ierr)
  call SNESSetDM(mech_snes,mech_mesh,ierr); CHKERRQ(ierr)                                           !< set the mesh for non-linear solver
  call DMCreateGlobalVector(mech_mesh,solution        ,ierr); CHKERRQ(ierr)                         !< locally owned displacement Dofs
  call DMCreateGlobalVector(mech_mesh,solution_rate   ,ierr); CHKERRQ(ierr)                         !< locally owned velocity Dofs to guess solution at next load step
  call DMCreateLocalVector (mech_mesh,solution_local  ,ierr); CHKERRQ(ierr)                         !< locally owned velocity Dofs to guess solution at next load step
  call DMSNESSetFunctionLocal(mech_mesh,FEM_mech_formResidual,PETSC_NULL_VEC,ierr)                  !< function to evaluate residual forces
  CHKERRQ(ierr)
  call DMSNESSetJacobianLocal(mech_mesh,FEM_mech_formJacobian,PETSC_NULL_VEC,ierr)                  !< function to evaluate stiffness matrix
  CHKERRQ(ierr)
  call SNESSetMaxLinearSolveFailures(mech_snes, huge(1), ierr); CHKERRQ(ierr)                       !< ignore linear solve failures
  call SNESSetConvergenceTest(mech_snes,FEM_mech_converged,PETSC_NULL_VEC,PETSC_NULL_FUNCTION,ierr)
  CHKERRQ(ierr)
  call SNESSetTolerances(mech_snes,1.0,0.0,0.0,num%itmax,num%itmax,ierr)
  CHKERRQ(ierr)
  call SNESSetFromOptions(mech_snes,ierr); CHKERRQ(ierr)

!--------------------------------------------------------------------------------------------------
! init fields
  call VecSet(solution        ,0.0,ierr); CHKERRQ(ierr)
  call VecSet(solution_rate   ,0.0,ierr); CHKERRQ(ierr)
  allocate(x_scal(cellDof))
  allocate(nodalWeightsP(1))
  allocate(nodalPointsP(dimPlex))
  allocate(pv0(dimPlex))
  allocate(pcellJ(dimPlex**2))
  allocate(pinvcellJ(dimPlex**2))
  allocate(cellJMat(dimPlex,dimPlex))
  call PetscDSGetDiscretization(mechDS,0,mechFE,ierr)
  CHKERRQ(ierr)
  call PetscFEGetDualSpace(mechFE,mechDualSpace,ierr); CHKERRQ(ierr)
  call DMPlexGetHeightStratum(mech_mesh,0,cellStart,cellEnd,ierr)
  CHKERRQ(ierr)
  do cell = cellStart, cellEnd-1                                                                    !< loop over all elements
    x_scal = 0.0_pReal
    call  DMPlexComputeCellGeometryAffineFEM(mech_mesh,cell,pV0,pCellJ,pInvcellJ,detJ,ierr)
    CHKERRQ(ierr)
    cellJMat = reshape(pCellJ,shape=[dimPlex,dimPlex])
    do basis = 0, nBasis*dimPlex-1, dimPlex
      call PetscDualSpaceGetFunctional(mechDualSpace,basis,functional,ierr)
      CHKERRQ(ierr)
      call PetscQuadratureGetData(functional,dimPlex,nc,nNodalPoints,nodalPointsP,nodalWeightsP,ierr)
      CHKERRQ(ierr)
      x_scal(basis+1:basis+dimPlex) = pV0 + matmul(transpose(cellJMat),nodalPointsP + 1.0_pReal)
    enddo
    px_scal => x_scal
#if (PETSC_VERSION_MINOR < 11)
    call DMPlexVecSetClosure(mech_mesh,section,solution_local,cell,px_scal,INSERT_ALL_VALUES,ierr)
#else
    call DMPlexVecSetClosure(mech_mesh,section,solution_local,cell,px_scal,5,ierr)                  ! PETSc: cbee0a90b60958e5c50c89b1e41f4451dfa6008c
#endif
    CHKERRQ(ierr)
  enddo

end subroutine FEM_mech_init


!--------------------------------------------------------------------------------------------------
!> @brief solution for the FEM load step
!--------------------------------------------------------------------------------------------------
type(tSolutionState) function FEM_mech_solution( &
             incInfoIn,timeinc,timeinc_old,fieldBC)

!--------------------------------------------------------------------------------------------------
! input data for solution
  real(pReal), intent(in) :: &
    timeinc, &                                                                                      !< increment in time for current solution
    timeinc_old                                                                                     !< increment in time of last increment
  type(tFieldBC),      intent(in) :: &
    fieldBC
  character(len=*), intent(in) :: &
    incInfoIn

  PetscErrorCode :: ierr
  SNESConvergedReason :: reason

  incInfo = incInfoIn
  FEM_mech_solution%converged =.false.
!--------------------------------------------------------------------------------------------------
! set module wide availabe data
  params%timeinc = timeinc
  params%timeincOld = timeinc_old
  params%fieldBC = fieldBC

  call SNESSolve(mech_snes,PETSC_NULL_VEC,solution,ierr); CHKERRQ(ierr)                             ! solve mech_snes based on solution guess (result in solution)
  call SNESGetConvergedReason(mech_snes,reason,ierr); CHKERRQ(ierr)                                 ! solution converged?
  terminallyIll = .false.

  if (reason < 1) then                                                                              ! 0: still iterating (will not occur), negative -> convergence error
    FEM_mech_solution%converged = .false.
    FEM_mech_solution%iterationsNeeded = num%itmax
  else                                                                                              ! >= 1 proper convergence (or terminally ill)
    FEM_mech_solution%converged = .true.
    call SNESGetIterationNumber(mech_snes,FEM_mech_solution%iterationsNeeded,ierr)
    CHKERRQ(ierr)
  endif

  print'(/,a)', ' ==========================================================================='
  flush(IO_STDOUT)

end function FEM_mech_solution


!--------------------------------------------------------------------------------------------------
!> @brief forms the FEM residual vector
!--------------------------------------------------------------------------------------------------
subroutine FEM_mech_formResidual(dm_local,xx_local,f_local,dummy,ierr)

  DM                                 :: dm_local
  PetscObject,intent(in)             :: dummy
  PetscErrorCode                     :: ierr

  PetscDS                            :: prob
  Vec                                :: x_local, f_local, xx_local
  PetscSection                       :: section
  PetscScalar, dimension(:), pointer :: x_scal, pf_scal
  PetscScalar,                target :: f_scal(cellDof)
  PetscReal                          :: detJ, IcellJMat(dimPlex,dimPlex)
  PetscReal,  pointer,dimension(:)   :: pV0, pCellJ, pInvcellJ, basisField, basisFieldDer
  PetscInt                           :: cellStart, cellEnd, cell, field, face, &
                                        qPt, basis, comp, cidx, &
                                        numFields
  PetscReal                          :: detFAvg
  PetscReal                          :: BMat(dimPlex*dimPlex,cellDof)

  PetscInt                           :: bcSize
  IS                                 :: bcPoints


  allocate(pV0(dimPlex))
  allocate(pcellJ(dimPlex**2))
  allocate(pinvcellJ(dimPlex**2))
  allocate(x_scal(cellDof))

#if (PETSC_VERSION_MINOR < 12)
  call DMGetSection(dm_local,section,ierr); CHKERRQ(ierr)
#else
  call DMGetLocalSection(dm_local,section,ierr); CHKERRQ(ierr)
#endif
  call DMGetDS(dm_local,prob,ierr); CHKERRQ(ierr)
  call PetscDSGetTabulation(prob,0,basisField,basisFieldDer,ierr)
  CHKERRQ(ierr)
  call DMPlexGetHeightStratum(dm_local,0,cellStart,cellEnd,ierr); CHKERRQ(ierr)
  call DMGetLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)
  call VecWAXPY(x_local,1.0,xx_local,solution_local,ierr); CHKERRQ(ierr)
  do field = 1, dimPlex; do face = 1, mesh_Nboundaries
    if (params%fieldBC%componentBC(field)%Mask(face)) then
      call DMGetStratumSize(dm_local,'Face Sets',mesh_boundaries(face),bcSize,ierr)
      if (bcSize > 0) then
        call DMGetStratumIS(dm_local,'Face Sets',mesh_boundaries(face),bcPoints,ierr)
        CHKERRQ(ierr)
        call utilities_projectBCValues(x_local,section,0,field-1,bcPoints, &
                                       0.0_pReal,params%fieldBC%componentBC(field)%Value(face),params%timeinc)
        call ISDestroy(bcPoints,ierr); CHKERRQ(ierr)
      endif
    endif
  enddo; enddo

!--------------------------------------------------------------------------------------------------
! evaluate field derivatives
  do cell = cellStart, cellEnd-1                                                                    !< loop over all elements
    
    call PetscSectionGetNumFields(section,numFields,ierr)
    CHKERRQ(ierr)
    call DMPlexVecGetClosure(dm_local,section,x_local,cell,x_scal,ierr)                             !< get Dofs belonging to element
    CHKERRQ(ierr)
    call  DMPlexComputeCellGeometryAffineFEM(dm_local,cell,pV0,pCellJ,pInvcellJ,detJ,ierr)
    CHKERRQ(ierr)
    IcellJMat = reshape(pInvcellJ,shape=[dimPlex,dimPlex])
    do qPt = 0, nQuadrature-1
      BMat = 0.0
      do basis = 0, nBasis-1
        do comp = 0, dimPlex-1
          cidx = basis*dimPlex+comp
          BMat(comp*dimPlex+1:(comp+1)*dimPlex,basis*dimPlex+comp+1) = &
            matmul(IcellJMat,basisFieldDer((((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp  )*dimPlex+1: &
                                           (((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp+1)*dimPlex))
        enddo
      enddo
      homogenization_F(1:dimPlex,1:dimPlex,qPt+1,cell+1) = &
        reshape(matmul(BMat,x_scal),shape=[dimPlex,dimPlex], order=[2,1])
    enddo
    if (num%BBarStabilisation) then
      detFAvg = math_det33(sum(homogenization_F(1:3,1:3,1:nQuadrature,cell+1),dim=3)/real(nQuadrature))
      do qPt = 1, nQuadrature
        homogenization_F(1:dimPlex,1:dimPlex,qPt,cell+1) = &
          homogenization_F(1:dimPlex,1:dimPlex,qPt,cell+1)* &
          (detFAvg/math_det33(homogenization_F(1:3,1:3,qPt,cell+1)))**(1.0/real(dimPlex))

      enddo
    endif
    call DMPlexVecRestoreClosure(dm_local,section,x_local,cell,x_scal,ierr)
    CHKERRQ(ierr)
  enddo

!--------------------------------------------------------------------------------------------------
! evaluate constitutive response
  call Utilities_constitutiveResponse(params%timeinc,P_av,ForwardData)
  call MPI_Allreduce(MPI_IN_PLACE,terminallyIll,1,MPI_LOGICAL,MPI_LOR,PETSC_COMM_WORLD,ierr)
  ForwardData = .false.

!--------------------------------------------------------------------------------------------------
! integrating residual
  do cell = cellStart, cellEnd-1                                                                    !< loop over all elements
    call DMPlexVecGetClosure(dm_local,section,x_local,cell,x_scal,ierr)                             !< get Dofs belonging to element
    CHKERRQ(ierr)
    call  DMPlexComputeCellGeometryAffineFEM(dm_local,cell,pV0,pCellJ,pInvcellJ,detJ,ierr)
    CHKERRQ(ierr)
    IcellJMat = reshape(pInvcellJ,shape=[dimPlex,dimPlex])
    f_scal = 0.0
    do qPt = 0, nQuadrature-1
      BMat = 0.0
      do basis = 0, nBasis-1
        do comp = 0, dimPlex-1
          cidx = basis*dimPlex+comp
          BMat(comp*dimPlex+1:(comp+1)*dimPlex,basis*dimPlex+comp+1) = &
            matmul(IcellJMat,basisFieldDer((((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp  )*dimPlex+1: &
                                           (((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp+1)*dimPlex))
        enddo
      enddo
      f_scal = f_scal + &
               matmul(transpose(BMat), &
                      reshape(transpose(homogenization_P(1:dimPlex,1:dimPlex,qPt+1,cell+1)), &
                              shape=[dimPlex*dimPlex]))*qWeights(qPt+1)
    enddo
    f_scal = f_scal*abs(detJ)
    pf_scal => f_scal
    call DMPlexVecSetClosure(dm_local,section,f_local,cell,pf_scal,ADD_VALUES,ierr)
    CHKERRQ(ierr)
    call DMPlexVecRestoreClosure(dm_local,section,x_local,cell,x_scal,ierr)
    CHKERRQ(ierr)
  enddo
  call DMRestoreLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)

end subroutine FEM_mech_formResidual


!--------------------------------------------------------------------------------------------------
!> @brief forms the FEM stiffness matrix
!--------------------------------------------------------------------------------------------------
subroutine FEM_mech_formJacobian(dm_local,xx_local,Jac_pre,Jac,dummy,ierr)


  DM                      :: dm_local
  Mat                     :: Jac_pre, Jac
  PetscObject, intent(in) :: dummy
  PetscErrorCode          :: ierr

  PetscDS                              :: prob
  Vec                                  :: x_local, xx_local

  PetscSection                         :: section, gSection

  PetscReal, dimension(1,         cellDof)  :: MatB
  PetscReal, dimension(dimPlex**2,cellDof)  :: BMat, BMatAvg, MatA
  PetscReal,   dimension(3,3)          :: F, FAvg, FInv
  PetscReal                            :: detJ
  PetscReal,   dimension(:),   pointer :: basisField, basisFieldDer, &
                                          pV0, pCellJ, pInvcellJ

  PetscScalar, dimension(:),   pointer :: pK_e, x_scal

  PetscScalar,dimension(cellDOF,cellDOF),  target :: K_e
  PetscScalar,dimension(cellDOF,cellDOF)  :: K_eA  , &
                                          K_eB

  PetscInt                             :: cellStart, cellEnd, cell, field, face, &
                                          qPt, basis, comp, cidx,bcSize

  IS                                   :: bcPoints


  allocate(pV0(dimPlex))
  allocate(pcellJ(dimPlex**2))
  allocate(pinvcellJ(dimPlex**2))

  call MatSetOption(Jac,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE,ierr); CHKERRQ(ierr)
  call MatSetOption(Jac,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE,ierr); CHKERRQ(ierr)
  call MatZeroEntries(Jac,ierr); CHKERRQ(ierr)
  call DMGetDS(dm_local,prob,ierr); CHKERRQ(ierr)
  call PetscDSGetTabulation(prob,0,basisField,basisFieldDer,ierr)
#if (PETSC_VERSION_MINOR < 12)
  call DMGetSection(dm_local,section,ierr); CHKERRQ(ierr)
#else
  call DMGetLocalSection(dm_local,section,ierr); CHKERRQ(ierr)
#endif
  call DMGetGlobalSection(dm_local,gSection,ierr); CHKERRQ(ierr)

  call DMGetLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)
  call VecWAXPY(x_local,1.0_pReal,xx_local,solution_local,ierr); CHKERRQ(ierr)
  do field = 1, dimPlex; do face = 1, mesh_Nboundaries
    if (params%fieldBC%componentBC(field)%Mask(face)) then
      call DMGetStratumSize(dm_local,'Face Sets',mesh_boundaries(face),bcSize,ierr)
      if (bcSize > 0) then
        call DMGetStratumIS(dm_local,'Face Sets',mesh_boundaries(face),bcPoints,ierr)
        CHKERRQ(ierr)
        call utilities_projectBCValues(x_local,section,0,field-1,bcPoints, &
                                       0.0_pReal,params%fieldBC%componentBC(field)%Value(face),params%timeinc)
        call ISDestroy(bcPoints,ierr); CHKERRQ(ierr)
      endif
    endif
  enddo; enddo
  call DMPlexGetHeightStratum(dm_local,0,cellStart,cellEnd,ierr); CHKERRQ(ierr)
  do cell = cellStart, cellEnd-1                                                                    !< loop over all elements
    call DMPlexVecGetClosure(dm_local,section,x_local,cell,x_scal,ierr)                             !< get Dofs belonging to element
    CHKERRQ(ierr)
    call  DMPlexComputeCellGeometryAffineFEM(dm_local,cell,pV0,pCellJ,pInvcellJ,detJ,ierr)
    CHKERRQ(ierr)
    K_eA = 0.0
    K_eB = 0.0
    MatB = 0.0
    FAvg = 0.0
    BMatAvg = 0.0
    do qPt = 0, nQuadrature-1
      BMat = 0.0
      do basis = 0, nBasis-1
        do comp = 0, dimPlex-1
          cidx = basis*dimPlex+comp
          BMat(comp*dimPlex+1:(comp+1)*dimPlex,basis*dimPlex+comp+1) = &
            matmul( reshape(pInvcellJ, shape = [dimPlex,dimPlex]),&
                             basisFieldDer((((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp  )*dimPlex+1: &
                                           (((qPt*nBasis + basis)*dimPlex + comp)*dimPlex+comp+1)*dimPlex))
        enddo
      enddo
      MatA = matmul(reshape(reshape(homogenization_dPdF(1:dimPlex,1:dimPlex,1:dimPlex,1:dimPlex,qPt+1,cell+1), &
                                    shape=[dimPlex,dimPlex,dimPlex,dimPlex], order=[2,1,4,3]), &
                            shape=[dimPlex*dimPlex,dimPlex*dimPlex]),BMat)*qWeights(qPt+1)
      if (num%BBarStabilisation) then
        F(1:dimPlex,1:dimPlex) = reshape(matmul(BMat,x_scal),shape=[dimPlex,dimPlex])
        FInv = math_inv33(F)
        K_eA = K_eA + matmul(transpose(BMat),MatA)*math_det33(FInv)**(1.0/real(dimPlex))
        K_eB = K_eB - &
               matmul(transpose(matmul(reshape(homogenization_F(1:dimPlex,1:dimPlex,qPt+1,cell+1), &
                                               shape=[dimPlex*dimPlex,1]), &
                                       matmul(reshape(FInv(1:dimPlex,1:dimPlex), &
                                                      shape=[1,dimPlex*dimPlex],order=[2,1]),BMat))),MatA)
        MatB = MatB + &
               matmul(reshape(homogenization_F(1:dimPlex,1:dimPlex,qPt+1,cell+1),shape=[1,dimPlex*dimPlex]),MatA)
        FAvg = FAvg + F
        BMatAvg = BMatAvg + BMat
      else
        K_eA = K_eA + matmul(transpose(BMat),MatA)
      endif
    enddo
    if (num%BBarStabilisation) then
      FInv = math_inv33(FAvg)
      K_e = K_eA*math_det33(FAvg/real(nQuadrature))**(1.0/real(dimPlex)) + &
            (matmul(matmul(transpose(BMatAvg), &
                           reshape(FInv(1:dimPlex,1:dimPlex),shape=[dimPlex*dimPlex,1],order=[2,1])),MatB) + &
             K_eB)/real(dimPlex)
    else
      K_e = K_eA
    endif
    K_e = (K_e + eps*math_eye(cellDof)) * abs(detJ)
#ifndef __INTEL_COMPILER
    pK_e(1:cellDOF**2) => K_e
#else
    ! https://software.intel.com/en-us/forums/intel-fortran-compiler/topic/782230 (bug)
    allocate(pK_e(cellDOF**2),source = reshape(K_e,[cellDOF**2]))
#endif
    call DMPlexMatSetClosure(dm_local,section,gSection,Jac,cell,pK_e,ADD_VALUES,ierr)
    CHKERRQ(ierr)
    call DMPlexVecRestoreClosure(dm_local,section,x_local,cell,x_scal,ierr)
    CHKERRQ(ierr)
  enddo
  call MatAssemblyBegin(Jac,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)
  call MatAssemblyEnd(Jac,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)
  call MatAssemblyBegin(Jac_pre,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)
  call MatAssemblyEnd(Jac_pre,MAT_FINAL_ASSEMBLY,ierr); CHKERRQ(ierr)
  call DMRestoreLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)

!--------------------------------------------------------------------------------------------------
! apply boundary conditions
#if (PETSC_VERSION_MINOR < 14)
  call DMPlexCreateRigidBody(dm_local,matnull,ierr); CHKERRQ(ierr)
#else
  call DMPlexCreateRigidBody(dm_local,0,matnull,ierr); CHKERRQ(ierr)
#endif
  call MatSetNullSpace(Jac,matnull,ierr); CHKERRQ(ierr)
  call MatSetNearNullSpace(Jac,matnull,ierr); CHKERRQ(ierr)
  call MatNullSpaceDestroy(matnull,ierr); CHKERRQ(ierr)

end subroutine FEM_mech_formJacobian


!--------------------------------------------------------------------------------------------------
!> @brief forwarding routine
!--------------------------------------------------------------------------------------------------
subroutine FEM_mech_forward(guess,timeinc,timeinc_old,fieldBC)

  type(tFieldBC), intent(in) :: &
    fieldBC
  real(pReal),    intent(in) :: &
    timeinc_old, &
    timeinc
  logical,        intent(in) :: &
    guess

  PetscInt       :: field, face, bcSize
  DM             :: dm_local
  Vec            :: x_local
  PetscSection   :: section
  IS             :: bcPoints
  PetscErrorCode :: ierr

!--------------------------------------------------------------------------------------------------
! forward last inc
  if (guess .and. .not. cutBack) then
    ForwardData = .True.
    homogenization_F0 = homogenization_F
    call SNESGetDM(mech_snes,dm_local,ierr); CHKERRQ(ierr)                                          !< retrieve mesh info from mech_snes into dm_local
    call DMGetSection(dm_local,section,ierr); CHKERRQ(ierr)
    call DMGetLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)
    call VecSet(x_local,0.0_pReal,ierr); CHKERRQ(ierr)
    call DMGlobalToLocalBegin(dm_local,solution,INSERT_VALUES,x_local,ierr)                         !< retrieve my partition of global solution vector
    CHKERRQ(ierr)
    call DMGlobalToLocalEnd(dm_local,solution,INSERT_VALUES,x_local,ierr)
    CHKERRQ(ierr)
    call VecAXPY(solution_local,1.0,x_local,ierr); CHKERRQ(ierr)
    do field = 1, dimPlex; do face = 1, mesh_Nboundaries
      if (fieldBC%componentBC(field)%Mask(face)) then
        call DMGetStratumSize(dm_local,'Face Sets',mesh_boundaries(face),bcSize,ierr)
        if (bcSize > 0) then
          call DMGetStratumIS(dm_local,'Face Sets',mesh_boundaries(face),bcPoints,ierr)
          CHKERRQ(ierr)
          call utilities_projectBCValues(solution_local,section,0,field-1,bcPoints, &
                                         0.0_pReal,fieldBC%componentBC(field)%Value(face),timeinc_old)
          call ISDestroy(bcPoints,ierr); CHKERRQ(ierr)
        endif
      endif
    enddo; enddo
    call DMRestoreLocalVector(dm_local,x_local,ierr); CHKERRQ(ierr)

!--------------------------------------------------------------------------------------------------
! update rate and forward last inc
    call VecCopy(solution,solution_rate,ierr); CHKERRQ(ierr)
    call VecScale(solution_rate,1.0/timeinc_old,ierr); CHKERRQ(ierr)
  endif
  call VecCopy(solution_rate,solution,ierr); CHKERRQ(ierr)
  call VecScale(solution,timeinc,ierr); CHKERRQ(ierr)

end subroutine FEM_mech_forward


!--------------------------------------------------------------------------------------------------
!> @brief reporting
!--------------------------------------------------------------------------------------------------
subroutine FEM_mech_converged(snes_local,PETScIter,xnorm,snorm,fnorm,reason,dummy,ierr)

  SNES :: snes_local
  PetscInt :: PETScIter
  PetscReal :: xnorm,snorm,fnorm,divTol
  SNESConvergedReason :: reason
  PetscObject :: dummy
  PetscErrorCode :: ierr

!--------------------------------------------------------------------------------------------------
! report
  divTol = max(maxval(abs(P_av(1:dimPlex,1:dimPlex)))*num%eps_struct_rtol,num%eps_struct_atol)
  call SNESConvergedDefault(snes_local,PETScIter,xnorm,snorm,fnorm/divTol,reason,dummy,ierr)
  CHKERRQ(ierr)
  if (terminallyIll) reason = SNES_DIVERGED_FUNCTION_DOMAIN
  print'(/,1x,a,a,i0,a,i0,f0.3)', trim(incInfo), &
                  ' @ Iteration ',PETScIter,' mechanical residual norm = ', &
                                                  int(fnorm/divTol),fnorm/divTol-int(fnorm/divTol)
  write(IO_STDOUT,'(/,a,/,3(3(2x,f12.4,1x)/))',advance='no') ' Piola--Kirchhoff stress / MPa =',&
                                                        transpose(P_av)*1.e-6_pReal
  flush(IO_STDOUT)

end subroutine FEM_mech_converged

end module mesh_mech_FEM
