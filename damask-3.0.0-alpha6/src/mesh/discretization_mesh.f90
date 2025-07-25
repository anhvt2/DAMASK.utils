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
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!--------------------------------------------------------------------------------------------------
module discretization_mesh
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscis.h>
#include <petsc/finclude/petscdmda.h>
  use PETScDMplex
  use PETScDMDA
  use PETScIS
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  use MPI_f08
#endif

  use DAMASK_interface
  use parallelization
  use IO
  use config
  use discretization
  use results
  use FEM_quadrature
  use YAML_types
  use prec

  implicit none
  private

  integer, public, protected :: &
    mesh_Nboundaries, &
    mesh_NcpElemsGlobal

  integer, public, protected :: &
    mesh_NcpElems                                                                                   !< total number of CP elements in mesh

!!!! BEGIN DEPRECATED !!!!!
  integer, public, protected :: &
    mesh_maxNips                                                                                    !< max number of IPs in any CP element
!!!! BEGIN DEPRECATED !!!!!

  DM, public :: geomMesh

  PetscInt, dimension(:), allocatable, public, protected :: &
    mesh_boundaries

  real(pReal), dimension(:,:), allocatable :: &
    mesh_ipVolume, &                                                                                !< volume associated with IP (initially!)
    mesh_node0                                                                                      !< node x,y,z coordinates (initially!)

  real(pReal), dimension(:,:,:), allocatable :: &
    mesh_ipCoordinates                                                                              !< IP x,y,z coordinates (after deformation!)

  public :: &
    discretization_mesh_init, &
    mesh_FEM_build_ipVolumes, &
    mesh_FEM_build_ipCoordinates

contains


!--------------------------------------------------------------------------------------------------
!> @brief initializes the mesh by calling all necessary private routines the mesh module
!! Order and routines strongly depend on type of solver
!--------------------------------------------------------------------------------------------------
subroutine discretization_mesh_init(restart)

  logical, intent(in) :: restart

  PetscInt :: dimPlex, &
    mesh_Nnodes, &                                                                                  !< total number of nodes in mesh
    j, &
    debug_element, debug_ip
  PetscSF :: sf
  DM :: globalMesh
  PetscInt :: nFaceSets, Nboundaries, NelemsGlobal, Nelems
  PetscInt, pointer, dimension(:) :: pFaceSets
  IS :: faceSetIS
  PetscErrorCode :: err_PETSc
  integer(MPI_INTEGER_KIND) :: err_MPI
  PetscInt, dimension(:), allocatable :: &
    materialAt
  class(tNode), pointer :: &
    num_mesh
  integer :: p_i, dim                                                                               !< integration order (quadrature rule)
  type(tvec) :: coords_node0
  real(pReal), pointer, dimension(:) :: &
    mesh_node0_temp

  print'(/,1x,a)',   '<<<+-  discretization_mesh init  -+>>>'

!--------------------------------------------------------------------------------
! read numerics parameter
  num_mesh => config_numerics%get('mesh',defaultVal=emptyDict)
  p_i = num_mesh%get_asInt('p_i',defaultVal = 2)

!---------------------------------------------------------------------------------
! read debug parameters
  debug_element = config_debug%get_asInt('element',defaultVal=1)
  debug_ip      = config_debug%get_asInt('integrationpoint',defaultVal=1)

  call DMPlexCreateFromFile(PETSC_COMM_WORLD,interface_geomFile,PETSC_TRUE,globalMesh,err_PETSc)
  CHKERRQ(err_PETSc)
  call DMGetDimension(globalMesh,dimPlex,err_PETSc)
  CHKERRQ(err_PETSc)
  call DMGetStratumSize(globalMesh,'depth',dimPlex,NelemsGlobal,err_PETSc)
  CHKERRQ(err_PETSc)
  mesh_NcpElemsGlobal = int(NelemsGlobal)
  call DMView(globalMesh, PETSC_VIEWER_STDOUT_WORLD,err_PETSc)
  CHKERRQ(err_PETSc)

  ! get number of IDs in face sets (for boundary conditions?)
  call DMGetLabelSize(globalMesh,'Face Sets',Nboundaries,err_PETSc)
  CHKERRQ(err_PETSc)
  mesh_Nboundaries = int(Nboundaries)
  call MPI_Bcast(mesh_Nboundaries,1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI error'
  call MPI_Bcast(mesh_NcpElemsGlobal,1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI error'
  dim = int(dimPlex)
  call MPI_Bcast(dim,1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  dimPlex = int(dim,pPETSCINT)
  if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI error'

  if (worldsize == 1) then
    call DMClone(globalMesh,geomMesh,err_PETSc)
  else
    call DMPlexDistribute(globalMesh,0_pPETSCINT,sf,geomMesh,err_PETSc)
  endif
  CHKERRQ(err_PETSc)

  allocate(mesh_boundaries(mesh_Nboundaries), source = 0_pPETSCINT)
  call DMGetLabelSize(globalMesh,'Face Sets',nFaceSets,err_PETSc)
  CHKERRQ(err_PETSc)
  call DMGetLabelIdIS(globalMesh,'Face Sets',faceSetIS,err_PETSc)
  CHKERRQ(err_PETSc)
  if (nFaceSets > 0) then
    call ISGetIndicesF90(faceSetIS,pFaceSets,err_PETSc)
    CHKERRQ(err_PETSc)
    mesh_boundaries(1:nFaceSets) = pFaceSets
    CHKERRQ(err_PETSc)
    call ISRestoreIndicesF90(faceSetIS,pFaceSets,err_PETSc)
  endif
  call MPI_Bcast(mesh_boundaries,mesh_Nboundaries,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI error'

  call DMDestroy(globalMesh,err_PETSc); CHKERRQ(err_PETSc)

  call DMGetStratumSize(geomMesh,'depth',dimPlex,Nelems,err_PETSc)
  CHKERRQ(err_PETSc)
  mesh_NcpElems = int(Nelems)
  call DMGetStratumSize(geomMesh,'depth',0_pPETSCINT,mesh_Nnodes,err_PETSc)
  CHKERRQ(err_PETSc)

! Get initial nodal coordinates
  call DMGetCoordinatesLocal(geomMesh,coords_node0,err_PETSc)
  CHKERRQ(err_PETSc)
  call VecGetArrayF90(coords_node0, mesh_node0_temp,err_PETSc)
  CHKERRQ(err_PETSc)

  mesh_maxNips = FEM_nQuadrature(dimPlex,p_i)

  call mesh_FEM_build_ipCoordinates(dimPlex,FEM_quadrature_points(dimPlex,p_i)%p)
  call mesh_FEM_build_ipVolumes(dimPlex)

  allocate(materialAt(mesh_NcpElems))
  do j = 1, mesh_NcpElems
    call DMGetLabelValue(geomMesh,'Cell Sets',j-1,materialAt(j),err_PETSc)
    CHKERRQ(err_PETSc)
  enddo
  materialAt = materialAt + 1_pPETSCINT

  if (debug_element < 1 .or. debug_element > mesh_NcpElems) call IO_error(602,ext_msg='element')
  if (debug_ip < 1 .or. debug_ip > mesh_maxNips)            call IO_error(602,ext_msg='IP')

  allocate(mesh_node0(3,mesh_Nnodes),source=0.0_pReal)
  mesh_node0(1:dimPlex,:) = reshape(mesh_node0_temp,[dimPlex,mesh_Nnodes])


  call discretization_init(int(materialAt),&
                           reshape(mesh_ipCoordinates,[3,mesh_maxNips*mesh_NcpElems]), &
                           mesh_node0)

  call writeGeometry(reshape(mesh_ipCoordinates,[3,mesh_maxNips*mesh_NcpElems]),mesh_node0)

end subroutine discretization_mesh_init


!--------------------------------------------------------------------------------------------------
!> @brief Calculates IP volume. Allocates global array 'mesh_ipVolume'
!--------------------------------------------------------------------------------------------------
subroutine mesh_FEM_build_ipVolumes(dimPlex)

  PetscInt,intent(in):: dimPlex
  PetscReal          :: vol
  PetscReal, pointer,dimension(:) :: pCent, pNorm
  PetscInt           :: cellStart, cellEnd, cell
  PetscErrorCode     :: err_PETSc

  allocate(mesh_ipVolume(mesh_maxNips,mesh_NcpElems),source=0.0_pReal)

  call DMPlexGetHeightStratum(geomMesh,0_pPETSCINT,cellStart,cellEnd,err_PETSc)
  CHKERRQ(err_PETSc)
  allocate(pCent(dimPlex))
  allocate(pNorm(dimPlex))
  do cell = cellStart, cellEnd-1
    call  DMPlexComputeCellGeometryFVM(geomMesh,cell,vol,pCent,pNorm,err_PETSc)
    CHKERRQ(err_PETSc)
    mesh_ipVolume(:,cell+1) = vol/real(mesh_maxNips,pReal)
  enddo

end subroutine mesh_FEM_build_ipVolumes


!--------------------------------------------------------------------------------------------------
!> @brief Calculates IP Coordinates. Allocates global array 'mesh_ipCoordinates'
!--------------------------------------------------------------------------------------------------
subroutine mesh_FEM_build_ipCoordinates(dimPlex,qPoints)

  PetscInt,      intent(in) :: dimPlex
  PetscReal,     intent(in) :: qPoints(mesh_maxNips*dimPlex)

  PetscReal,        pointer,dimension(:) :: pV0, pCellJ, pInvcellJ
  PetscReal                 :: detJ
  PetscInt                  :: cellStart, cellEnd, cell, qPt, dirI, dirJ, qOffset
  PetscErrorCode            :: err_PETSc


  allocate(mesh_ipCoordinates(3,mesh_maxNips,mesh_NcpElems),source=0.0_pReal)

  allocate(pV0(dimPlex))
  allocatE(pCellJ(dimPlex**2))
  allocatE(pinvCellJ(dimPlex**2))
  call DMPlexGetHeightStratum(geomMesh,0_pPETSCINT,cellStart,cellEnd,err_PETSc)
  CHKERRQ(err_PETSc)
  do cell = cellStart, cellEnd-1                                                                     !< loop over all elements
    call DMPlexComputeCellGeometryAffineFEM(geomMesh,cell,pV0,pCellJ,pInvcellJ,detJ,err_PETSc)
    CHKERRQ(err_PETSc)
    qOffset = 0
    do qPt = 1, mesh_maxNips
      do dirI = 1, dimPlex
        mesh_ipCoordinates(dirI,qPt,cell+1) = pV0(dirI)
        do dirJ = 1, dimPlex
          mesh_ipCoordinates(dirI,qPt,cell+1) = mesh_ipCoordinates(dirI,qPt,cell+1) + &
                                                pCellJ((dirI-1)*dimPlex+dirJ)*(qPoints(qOffset+dirJ) + 1.0)
        enddo
      enddo
      qOffset = qOffset + dimPlex
    enddo
  enddo

end subroutine mesh_FEM_build_ipCoordinates

!--------------------------------------------------------------------------------------------------
!> @brief Write all information needed for the DADF5 geometry
!--------------------------------------------------------------------------------------------------
subroutine writeGeometry(coordinates_points,coordinates_nodes)

  real(pReal), dimension(:,:), intent(in) :: &
  coordinates_nodes, &
  coordinates_points

  call results_openJobFile
  call results_closeGroup(results_addGroup('geometry'))

  call results_writeDataset(coordinates_nodes,'geometry','x_n', &
        'initial coordinates of the nodes','m')

  call results_writeDataset(coordinates_points,'geometry','x_p', &
        'initial coordinates of the materialpoints (cell centers)','m')

  call results_closeJobFile

  end subroutine writeGeometry

end module discretization_mesh
