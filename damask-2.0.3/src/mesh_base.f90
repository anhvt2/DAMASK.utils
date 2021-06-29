! Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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
!> @author Christoph Koords, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Sets up the mesh for the solvers MSC.Marc,FEM, Abaqus and the spectral solver
!--------------------------------------------------------------------------------------------------
module mesh_base

 use, intrinsic :: iso_c_binding
 use prec, only: &
   pStringLen, &
   pReal, &
   pInt
 use element, only: &
   tElement

  implicit none

!---------------------------------------------------------------------------------------------------
!> Properties of a the whole mesh (consisting of one type of elements)
!---------------------------------------------------------------------------------------------------
 type, public :: tMesh
   type(tElement) :: &
     elem
   real(pReal), dimension(:,:), allocatable, public :: &
     ipVolume, &                                                                                 !< volume associated with each IP (initially!)
     node0, &                                                                                    !< node x,y,z coordinates (initially)
     node                                                                                        !< node x,y,z coordinates (deformed)
   integer(pInt), dimension(:,:), allocatable, public :: &    
     cellnodeParent                                                                               !< cellnode's parent element ID, cellnode's intra-element ID 
   character(pStringLen) :: type = "n/a"
   integer(pInt)         :: &
     Nnodes, &                                                                                   !< total number of nodes in mesh
     Nelems = -1_pInt, &
     elemType, &
     Ncells, &
     nIPneighbors, &
     NcellNodes, &
     maxElemsPerNode
   integer(pInt), dimension(:), allocatable, public :: &
     homogenizationAt, &
     microstructureAt
   integer(pInt), dimension(:,:), allocatable, public :: &
     connectivity
   contains
   procedure, pass(self) :: tMesh_base_init
   procedure :: setNelems =>  tMesh_base_setNelems                                                  ! not needed once we compute the cells from the connectivity
   generic, public :: init => tMesh_base_init
 end type tMesh

contains
subroutine tMesh_base_init(self,meshType,elemType,nodes)
 
 implicit none
 class(tMesh) :: self
 character(len=*), intent(in) :: meshType
 integer(pInt), intent(in) :: elemType
 real(pReal), dimension(:,:), intent(in) :: nodes
 
 write(6,'(/,a)')   ' <<<+-  mesh_base_init  -+>>>'
 
 write(6,*)' mesh type ',meshType
 write(6,*)' # node    ',size(nodes,2)

 self%type = meshType
 call self%elem%init(elemType)
 self%node0 = nodes
 self%nNodes = size(nodes,2)

end subroutine tMesh_base_init


subroutine tMesh_base_setNelems(self,Nelems)
 
  implicit none
  class(tMesh) :: self
  integer(pInt), intent(in) :: Nelems

  self%Nelems = Nelems

end subroutine tMesh_base_setNelems

end module mesh_base
