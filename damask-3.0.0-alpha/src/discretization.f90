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
!> @brief spatial discretization
!> @details serves as an abstraction layer between the different solvers and DAMASK
!--------------------------------------------------------------------------------------------------
module discretization

  use prec
  use results

  implicit none
  private
  
  integer,     public, protected :: &
    discretization_nIP, &
    discretization_nElem
    
  integer,     public, protected, dimension(:),   allocatable :: &
    discretization_homogenizationAt, &
    discretization_microstructureAt   

  real(pReal), public, protected, dimension(:,:), allocatable :: & 
    discretization_IPcoords0, &
    discretization_IPcoords, &
    discretization_NodeCoords0, &
    discretization_NodeCoords
    
  integer :: &
    discretization_sharedNodesBegin

  public :: &
    discretization_init, &
    discretization_results, &
    discretization_setIPcoords, &
    discretization_setNodeCoords

contains
  
!--------------------------------------------------------------------------------------------------
!> @brief stores the relevant information in globally accesible variables
!--------------------------------------------------------------------------------------------------
subroutine discretization_init(homogenizationAt,microstructureAt,&
                               IPcoords0,NodeCoords0,&
                               sharedNodesBegin)

  integer,     dimension(:),   intent(in) :: &
    homogenizationAt, &
    microstructureAt
  real(pReal), dimension(:,:), intent(in) :: &
    IPcoords0, &
    NodeCoords0
  integer, optional,           intent(in) :: &
    sharedNodesBegin                                                                                !< index of first node shared among different processes (MPI)

  write(6,'(/,a)') ' <<<+-  discretization init  -+>>>'; flush(6)

  discretization_nElem = size(microstructureAt,1)
  discretization_nIP   = size(IPcoords0,2)/discretization_nElem

  discretization_homogenizationAt = homogenizationAt
  discretization_microstructureAt = microstructureAt  

  discretization_IPcoords0   = IPcoords0
  discretization_IPcoords    = IPcoords0

  discretization_NodeCoords0 = NodeCoords0
  discretization_NodeCoords  = NodeCoords0
  
  if(present(sharedNodesBegin)) then
    discretization_sharedNodesBegin = sharedNodesBegin
  else
    discretization_sharedNodesBegin = size(discretization_NodeCoords0,2)
  endif
  
end subroutine discretization_init


!--------------------------------------------------------------------------------------------------
!> @brief write the displacements
!--------------------------------------------------------------------------------------------------
subroutine discretization_results

  real(pReal), dimension(:,:), allocatable :: u
  
  call results_closeGroup(results_addGroup('current/geometry'))
  
  u = discretization_NodeCoords (1:3,:discretization_sharedNodesBegin) &
    - discretization_NodeCoords0(1:3,:discretization_sharedNodesBegin)
  call results_writeDataset('current/geometry',u,'u_n','displacements of the nodes','m')
  
  u = discretization_IPcoords &
    - discretization_IPcoords0
  call results_writeDataset('current/geometry',u,'u_p','displacements of the materialpoints','m')

end subroutine discretization_results


!--------------------------------------------------------------------------------------------------
!> @brief stores current IP coordinates
!--------------------------------------------------------------------------------------------------
subroutine discretization_setIPcoords(IPcoords)

  real(pReal), dimension(:,:), intent(in) :: IPcoords
  
  discretization_IPcoords = IPcoords

end subroutine discretization_setIPcoords


!--------------------------------------------------------------------------------------------------
!> @brief stores current IP coordinates
!--------------------------------------------------------------------------------------------------
subroutine discretization_setNodeCoords(NodeCoords)

  real(pReal), dimension(:,:), intent(in) :: NodeCoords
  
  discretization_NodeCoords = NodeCoords

end subroutine discretization_setNodeCoords


end module discretization
