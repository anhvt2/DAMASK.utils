! Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as
! published by the Free Software Foundation, either version 3 of the
! License, or (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Affero General Public License for more details.
! 
! You should have received a copy of the GNU Affero General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Christoph Kords, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Sets up the mesh for the solver MSC.Marc
!--------------------------------------------------------------------------------------------------
module discretization_Marc
  use IO
  use prec
  use math
  use DAMASK_interface
  use IO
  use config
  use element
  use discretization
  use geometry_plastic_nonlocal
  use result

  implicit none(type,external)
  private

  real(pREAL),                         public, protected :: &
    mesh_unitlength                                                                                 !< physical length of one unit in mesh MD: needs systematic_name

  integer,  dimension(:), allocatable, public, protected :: &
    discretization_Marc_FEM2DAMASK_elem, &                                                          !< DAMASK element ID for Marc element ID
    discretization_Marc_FEM2DAMASK_node                                                             !< DAMASK node ID for Marc node ID


  type tCellNodeDefinition
    integer, dimension(:,:), allocatable :: parents
    integer, dimension(:,:), allocatable :: weights
  end type tCellNodeDefinition

  type(tCellNodeDefinition), dimension(:), allocatable :: cellNodeDefinition

  integer, dimension(:,:,:), allocatable :: &
    connectivity_cell                                                                               !< cell connectivity for each element,ip/cell

  public :: &
    discretization_Marc_init, &
    discretization_Marc_updateNodeAndIpCoords, &
    discretization_Marc_FEM2DAMASK_cell

contains

!--------------------------------------------------------------------------------------------------
!> @brief initializes the mesh by calling all necessary private routines the mesh module
!! Order and routines strongly depend on type of solver
!--------------------------------------------------------------------------------------------------
subroutine discretization_Marc_init

  real(pREAL), dimension(:,:),     allocatable :: &
   node0_elem, &                                                                                    !< node x,y,z coordinates (initially!)
   node0_cell
  type(tElement) :: elem

  integer,     dimension(:),       allocatable :: &
    materialAt
  integer:: &
    Nelems                                                                                          !< total number of elements in the mesh

  real(pREAL), dimension(:,:),     allocatable :: &
    IP_reshaped
  integer,     dimension(:,:),     allocatable :: &
    connectivity_elem
  real(pREAL), dimension(:,:,:,:), allocatable :: &
    unscaledNormals

  type(tDict), pointer :: &
    num_solver, &
    num_commercialFEM


  print'(/,a)', ' <<<+-  discretization_Marc init  -+>>>'; flush(6)

  num_solver => config_numerics%get_dict('solver',defaultVal=emptyDict)
  num_commercialFEM => num_solver%get_dict('Marc',defaultVal=emptyDict)
  mesh_unitlength = num_commercialFEM%get_asReal('unit_length',defaultVal=1.0_pREAL)                 ! set physical extent of a length unit in mesh
  if (mesh_unitlength <= 0.0_pREAL) call IO_error(301,'unit_length')

  call inputRead(elem,node0_elem,connectivity_elem,materialAt)
  nElems = size(connectivity_elem,2)

  allocate(cellNodeDefinition(elem%nNodes-1))
  allocate(connectivity_cell(elem%NcellNodesPerCell,elem%nIPs,nElems))

  call buildCells(connectivity_cell,cellNodeDefinition,&
                  elem,connectivity_elem)
  node0_cell  = buildCellNodes(node0_elem)

  IP_reshaped = buildIPcoordinates(node0_cell)

  call discretization_init(materialAt, IP_reshaped, node0_cell)

  call writeGeometry(elem,connectivity_elem,&
                     reshape(connectivity_cell,[elem%NcellNodesPerCell,elem%nIPs*nElems]),&
                     node0_cell,IP_reshaped)

!--------------------------------------------------------------------------------------------------
! geometry information required by the nonlocal CP model
  call geometry_plastic_nonlocal_setIPvolume(IPvolume(elem,node0_cell))
  unscaledNormals = IPareaNormal(elem,nElems,node0_cell)
  call geometry_plastic_nonlocal_setIParea(norm2(unscaledNormals,1))
  call geometry_plastic_nonlocal_setIPareaNormal(unscaledNormals/spread(norm2(unscaledNormals,1),1,3))
  call geometry_plastic_nonlocal_setIPneighborhood(IPneighborhood(elem))
  call geometry_plastic_nonlocal_result()

end subroutine discretization_Marc_init


!--------------------------------------------------------------------------------------------------
!> @brief Calculate and set current nodal and IP positions (including cell nodes)
!--------------------------------------------------------------------------------------------------
subroutine discretization_Marc_updateNodeAndIpCoords(d_n)

  real(pREAL), dimension(:,:), intent(in)  :: d_n

  real(pREAL), dimension(:,:), allocatable :: node_cell


  node_cell = buildCellNodes(discretization_NodeCoords0(1:3,1:maxval(discretization_Marc_FEM2DAMASK_node)) + d_n)

  call discretization_setNodeCoords(node_cell)
  call discretization_setIPcoords(buildIPcoordinates(node_cell))

end subroutine discretization_Marc_updateNodeAndIpCoords


!--------------------------------------------------------------------------------------------------
!> @brief Calculate and set current nodal and IP positions (including cell nodes)
!--------------------------------------------------------------------------------------------------
function discretization_Marc_FEM2DAMASK_cell(IP_FEM,elem_FEM) result(cell)

  integer, intent(in) :: IP_FEM, elem_FEM
  integer :: cell

  real(pREAL), dimension(:,:), allocatable :: node_cell


  cell = (discretization_Marc_FEM2DAMASK_elem(elem_FEM)-1)*discretization_nIPs + IP_FEM


end function discretization_Marc_FEM2DAMASK_cell


!--------------------------------------------------------------------------------------------------
!> @brief Write all information needed for the DADF5 geometry
!--------------------------------------------------------------------------------------------------
subroutine writeGeometry(elem, &
                         connectivity_elem,connectivity_cell_reshaped, &
                         coordinates_nodes,coordinates_points)

  type(tElement),              intent(in) :: &
    elem
  integer, dimension(:,:),     intent(in) :: &
    connectivity_elem, &
    connectivity_cell_reshaped
  real(pREAL), dimension(:,:), intent(in) :: &
    coordinates_nodes, &
    coordinates_points


  call result_openJobFile()
  call result_closeGroup(result_addGroup('geometry'))

  call result_writeDataset(connectivity_elem,'geometry','T_e',&
                           'connectivity of the elements','-')

  call result_writeDataset(connectivity_cell_reshaped,'geometry','T_c', &
                           'connectivity of the cells','-')
  call result_addAttribute('VTK_TYPE',elem%vtkType,'geometry/T_c')

  call result_writeDataset(coordinates_nodes,'geometry','x_n', &
                           'initial coordinates of the nodes','m')

  call result_writeDataset(coordinates_points,'geometry','x_p', &
                           'initial coordinates of the materialpoints (cell centers)','m')

  call result_closeJobFile()

end subroutine writeGeometry


!--------------------------------------------------------------------------------------------------
!> @brief Read mesh from marc input file
!--------------------------------------------------------------------------------------------------
subroutine inputRead(elem,node0_elem,connectivity_elem,materialAt)

  type(tElement), intent(out) :: elem
  real(pREAL), dimension(:,:), allocatable, intent(out) :: &
    node0_elem                                                                                      !< node x,y,z coordinates (initially!)
  integer, dimension(:,:),     allocatable, intent(out) :: &
    connectivity_elem
  integer,     dimension(:),   allocatable, intent(out) :: &
    materialAt

  integer :: &
    fileFormatVersion, &
    hypoelasticTableStyle, &
    initialcondTableStyle, &
    nNodes, &
    nElems
  integer, dimension(:), allocatable :: &
    matNumber                                                                                       !< material numbers for hypoelastic material
  character(len=pSTRLEN), dimension(:), allocatable :: &
    inputFile, &                                                                                    !< file content, separated per lines
    nameElemSet
  integer, dimension(:,:), allocatable :: &
    mapElemSet                                                                                      !< list of elements in elementSet


  call result_openJobFile()
  call result_addSetupFile(IO_read(trim(getSolverJobName())//InputFileExtension), &
                                   trim(getSolverJobName())//InputFileExtension, &
                                   'MSC.Marc input deck')
  call result_closeJobFile()

  inputFile = readlines(trim(getSolverJobName())//InputFileExtension)
  call inputRead_fileFormat(fileFormatVersion, &
                            inputFile)
  call inputRead_tableStyles(initialcondTableStyle,hypoelasticTableStyle, &
                             inputFile)
  if (fileFormatVersion > 12) &
    call inputRead_matNumber(matNumber, &
                             hypoelasticTableStyle,inputFile)
  call inputRead_NnodesAndElements(nNodes,nElems,&
                                   inputFile)


  call inputRead_mapElemSets(nameElemSet,mapElemSet,&
                             inputFile)

  call inputRead_elemType(elem, &
                          nElems,inputFile)

  call inputRead_mapElems(discretization_Marc_FEM2DAMASK_elem,&
                          nElems,elem%nNodes,inputFile)

  call inputRead_mapNodes(discretization_Marc_FEM2DAMASK_node,&
                          nNodes,inputFile)

  call inputRead_elemNodes(node0_elem, &
                           Nnodes,inputFile)

  connectivity_elem = inputRead_connectivityElem(nElems,elem%nNodes,inputFile)

  call inputRead_material(materialAt, &
                          nElems,elem%nNodes,nameElemSet,mapElemSet,&
                          initialcondTableStyle,inputFile)

contains
!--------------------------------------------------------------------------------------------------
!> @brief Read ASCII file and split at EOL.
!--------------------------------------------------------------------------------------------------
function readlines(fileName) result(fileContent)

  character(len=*),       intent(in)                :: fileName
  character(len=pSTRLEN), dimension(:), allocatable :: fileContent                                  !< file content, separated per lines

  character(len=pSTRLEN)                            :: line
  character(len=:),                     allocatable :: rawData
  integer ::  &
    startPos, endPos, &
    N_lines, &                                                                                      !< # lines in file
    l
  logical :: warned


  rawData = IO_read(fileName)

  N_lines = count([(rawData(l:l) == IO_EOL,l=1,len(rawData))])
  allocate(fileContent(N_lines))

!--------------------------------------------------------------------------------------------------
! split raw data at end of line
  warned = .false.
  startPos = 1
  l = 1
  do while (l <= N_lines)
    endPos = startPos + scan(rawData(startPos:),IO_EOL) - 2
    if (endPos - startPos > pSTRLEN-1) then
      line = rawData(startPos:startPos+pSTRLEN-1)
      if (.not. warned) then
        call IO_warning(207,trim(fileName),label1='line',ID1=l)
        warned = .true.
      end if
    else
      line = rawData(startPos:endpos)
    end if
    startPos = endPos + 2                                                                           ! jump to next line start

    fileContent(l) = trim(line)//''
    l = l + 1
  end do

end function readlines

end subroutine inputRead



!--------------------------------------------------------------------------------------------------
!> @brief Figures out version of Marc input file format
!--------------------------------------------------------------------------------------------------
subroutine inputRead_fileFormat(fileFormat,fileContent)

  integer,                        intent(out) :: fileFormat
  character(len=*), dimension(:), intent(in)  :: fileContent                                        !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: l

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 2) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'version') then
      fileFormat = intValue(fileContent(l),chunkPos,2)
      exit
    end if
  end do

end subroutine inputRead_fileFormat


!--------------------------------------------------------------------------------------------------
!> @brief Figures out table styles for initial cond and hypoelastic
!--------------------------------------------------------------------------------------------------
subroutine inputRead_tableStyles(initialcond,hypoelastic,fileContent)

  integer,                        intent(out) :: initialcond, hypoelastic
  character(len=*), dimension(:), intent(in)  :: fileContent                                        !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: l

  initialcond = 0
  hypoelastic = 0

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 6) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'table') then
      initialcond = intValue(fileContent(l),chunkPos,4)
      hypoelastic = intValue(fileContent(l),chunkPos,5)
      exit
    end if
  end do

end subroutine inputRead_tableStyles


!--------------------------------------------------------------------------------------------------
!> @brief Figures out material number of hypoelastic material
!--------------------------------------------------------------------------------------------------
subroutine inputRead_matNumber(matNumber, &
                               tableStyle,fileContent)

  integer, allocatable, dimension(:), intent(out) :: matNumber
  integer,                            intent(in)  :: tableStyle
  character(len=*),     dimension(:), intent(in)  :: fileContent                                    !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: i, j, data_blocks, l


  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'hypoelastic') then
      if (len_trim(fileContent(l+1))/=0) then
        chunkPos = strPos(fileContent(l+1))
        data_blocks = intValue(fileContent(l+1),chunkPos,1)
      else
        data_blocks = 1
      end if
      allocate(matNumber(data_blocks), source = 0)
      do i = 0, data_blocks - 1
        j = i*(2+tableStyle) + 1
        chunkPos = strPos(fileContent(l+1+j))
        matNumber(i+1) = intValue(fileContent(l+1+j),chunkPos,1)
      end do
      exit
    end if
  end do

end subroutine inputRead_matNumber


!--------------------------------------------------------------------------------------------------
!> @brief Count overall number of nodes and elements
!--------------------------------------------------------------------------------------------------
subroutine inputRead_NnodesAndElements(nNodes,nElems,&
                                       fileContent)

  integer,                        intent(out) :: nNodes, nElems
  character(len=*), dimension(:), intent(in)  :: fileContent                                        !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: l

  nNodes = 0
  nElems = 0

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if    (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'sizing') then
      nElems = intValue (fileContent(l),chunkPos,3)
    elseif (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'coordinates') then
      chunkPos = strPos(fileContent(l+1))
      nNodes = intValue (fileContent(l+1),chunkPos,2)
    end if
  end do

end subroutine inputRead_NnodesAndElements


!--------------------------------------------------------------------------------------------------
!> @brief Count overall number of element sets in mesh.
!--------------------------------------------------------------------------------------------------
subroutine inputRead_NelemSets(nElemSets,maxNelemInSet,&
                               fileContent)

  integer,                            intent(out) :: nElemSets, maxNelemInSet
  character(len=*),     dimension(:), intent(in)  :: fileContent                                    !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer                            :: i,l,elemInCurrentSet


  nElemSets     = 0
  maxNelemInSet = 0

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 2) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'define' .and. &
       IO_lc(strValue(fileContent(l),chunkPos,2)) == 'element') then
      nElemSets = nElemSets + 1

      chunkPos = strPos(fileContent(l+1))
      if (containsRange(fileContent(l+1),chunkPos)) then
        elemInCurrentSet = 1 + abs( intValue(fileContent(l+1),chunkPos,3) &
                                   -intValue(fileContent(l+1),chunkPos,1))
      else
        elemInCurrentSet = 0
        i = 0
        do while (.true.)
          i = i + 1
          chunkPos = strPos(fileContent(l+i))
          elemInCurrentSet = elemInCurrentSet + chunkPos(1) - 1                                     ! add line's count when assuming 'c'
          if (IO_lc(strValue(fileContent(l+i),chunkPos,chunkPos(1))) /= 'c') then                ! line finished, read last value
            elemInCurrentSet = elemInCurrentSet + 1                                                 ! data ended
            exit
          end if
        end do
      end if
      maxNelemInSet = max(maxNelemInSet, elemInCurrentSet)
    end if
  end do

end subroutine inputRead_NelemSets


!--------------------------------------------------------------------------------------------------
!> @brief map element sets
!--------------------------------------------------------------------------------------------------
subroutine inputRead_mapElemSets(nameElemSet,mapElemSet,&
                                 fileContent)

  character(len=pSTRLEN), dimension(:),   allocatable, intent(out) :: nameElemSet
  integer,                   dimension(:,:), allocatable, intent(out) :: mapElemSet
  character(len=*),          dimension(:),                intent(in)  :: fileContent                !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: elemSet, NelemSets, maxNelemInSet,l


  call inputRead_NelemSets(NelemSets,maxNelemInSet,fileContent)
  allocate(nameElemSet(NelemSets)); nameElemSet = 'n/a'
  allocate(mapElemSet(1+maxNelemInSet,NelemSets),source=0)
  elemSet = 0

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 2) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'define' .and. &
       IO_lc(strValue(fileContent(l),chunkPos,2)) == 'element') then
       elemSet = elemSet+1
       nameElemSet(elemSet)  = trim(strValue(fileContent(l),chunkPos,4))
       mapElemSet(:,elemSet) = continuousIntValues(fileContent(l+1:),size(mapElemSet,1)-1,nameElemSet,mapElemSet,size(nameElemSet))
    end if
  end do

end subroutine inputRead_mapElemSets


!--------------------------------------------------------------------------------------------------
!> @brief Maps elements from FE ID to internal (consecutive) representation.
!--------------------------------------------------------------------------------------------------
subroutine inputRead_mapElems(FEM2DAMASK, &
                              nElems,nNodesPerElem,fileContent)

  integer, allocatable, dimension(:), intent(out) :: FEM2DAMASK

  integer,                            intent(in)  :: nElems, &                                      !< number of elements
                                                     nNodesPerElem                                  !< number of nodes per element
  character(len=*),     dimension(:), intent(in)  :: fileContent                                    !< file content, separated per lines

  integer, dimension(2,nElems)       :: map_unsorted
  integer, allocatable, dimension(:) :: chunkPos
  integer :: i,j,l,nNodesAlreadyRead


  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'connectivity') then
      j = 0
      do i = 1,nElems
        chunkPos = strPos(fileContent(l+1+i+j))
        map_unsorted(:,i) = [intValue(fileContent(l+1+i+j),chunkPos,1),i]
        nNodesAlreadyRead = chunkPos(1) - 2
        do while(nNodesAlreadyRead < nNodesPerElem)                                                 ! read on if not all nodes in one line
          j = j + 1
          chunkPos = strPos(fileContent(l+1+i+j))
          nNodesAlreadyRead = nNodesAlreadyRead + chunkPos(1)
        end do
      end do
      exit
    end if
  end do

  call math_sort(map_unsorted)
  allocate(FEM2DAMASK(minval(map_unsorted(1,:)):maxval(map_unsorted(1,:))),source=-1)
  do i = 1, nElems
    FEM2DAMASK(map_unsorted(1,i)) = map_unsorted(2,i)
  end do

end subroutine inputRead_mapElems


!--------------------------------------------------------------------------------------------------
!> @brief Maps node from FE ID to internal (consecutive) representation.
!--------------------------------------------------------------------------------------------------
subroutine inputRead_mapNodes(FEM2DAMASK, &
                              nNodes,fileContent)

  integer, allocatable, dimension(:), intent(out) :: FEM2DAMASK

  integer,                            intent(in)  :: nNodes                                         !< number of nodes
  character(len=*),     dimension(:), intent(in)  :: fileContent                                    !< file content, separated per lines

  integer, dimension(2,nNodes)       :: map_unsorted
  integer, allocatable, dimension(:) :: chunkPos
  integer :: i, l


  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'coordinates') then
      chunkPos = [1,1,10]
      do i = 1,nNodes
        map_unsorted(:,i) = [intValue(fileContent(l+1+i),chunkPos,1),i]
      end do
      exit
    end if
  end do

  call math_sort(map_unsorted)
  allocate(FEM2DAMASK(minval(map_unsorted(1,:)):maxval(map_unsorted(1,:))),source=-1)
  do i = 1, nNodes
    FEM2DAMASK(map_unsorted(1,i)) = map_unsorted(2,i)
  end do

end subroutine inputRead_mapNodes


!--------------------------------------------------------------------------------------------------
!> @brief store x,y,z coordinates of all nodes in mesh.
!--------------------------------------------------------------------------------------------------
subroutine inputRead_elemNodes(nodes, &
                               nNode,fileContent)

  real(pREAL), allocatable,  dimension(:,:), intent(out) :: nodes
  integer,                                   intent(in)  :: nNode
  character(len=*),            dimension(:), intent(in)  :: fileContent                             !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: i,j,m,l


  allocate(nodes(3,nNode))

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'coordinates') then
      chunkPos = [4,1,10,11,30,31,50,51,70]
      do i=1,nNode
        m = discretization_Marc_FEM2DAMASK_node(intValue(fileContent(l+1+i),chunkPos,1))
        nodes(1:3,m) = [(mesh_unitlength * realValue(fileContent(l+1+i),chunkPos,j+1),j=1,3)]
      end do
      exit
    end if
  end do

end subroutine inputRead_elemNodes


!--------------------------------------------------------------------------------------------------
!> @brief Gets element type (and checks if the whole mesh comprises of only one type)
!--------------------------------------------------------------------------------------------------
subroutine inputRead_elemType(elem, &
                              nElem,fileContent)

  type(tElement),                 intent(out) :: elem
  integer,                        intent(in)  :: nElem
  character(len=*), dimension(:), intent(in)  :: fileContent                                        !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos
  integer :: i,j,t,t_,l,remainingChunks


  t = -1
  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'connectivity') then
      j = 0
      do i=1,nElem                                                                                  ! read all elements
        chunkPos = strPos(fileContent(l+1+i+j))
        if (t == -1) then
          t = mapElemtype(strValue(fileContent(l+1+i+j),chunkPos,2))
          call elem%init(t)
        else
          t_ = mapElemtype(strValue(fileContent(l+1+i+j),chunkPos,2))
          if (t /= t_) call IO_error(191,strValue(fileContent(l+1+i+j),chunkPos,2),label1='type',ID1=t)
        end if
        remainingChunks = elem%nNodes - (chunkPos(1) - 2)
        do while(remainingChunks > 0)
          j = j + 1
          chunkPos = strPos(fileContent(l+1+i+j))
          remainingChunks = remainingChunks - chunkPos(1)
        end do
      end do
      exit
    end if
  end do

  contains

  !--------------------------------------------------------------------------------------------------
  !> @brief mapping of Marc element types to internal representation
  !--------------------------------------------------------------------------------------------------
  integer function mapElemtype(what)

   character(len=*), intent(in) :: what


   select case (what)
      case (   '6')
        mapElemtype = 1            ! Two-dimensional Plane Strain Triangle
      case ( '125')                ! 155, 128 (need test)
        mapElemtype = 2            ! Two-dimensional Plane Strain triangle (155: cubic shape function, 125/128: second order isoparametric)
      case ( '11')
        mapElemtype = 3            ! Arbitrary Quadrilateral Plane-strain
      case ( '27')
        mapElemtype = 4            ! Plane Strain, Eight-node Distorted Quadrilateral
      case ( '54')
        mapElemtype = 5            ! Plane Strain, Eight-node Distorted Quadrilateral with reduced integration
      case ( '134')
        mapElemtype = 6            ! Three-dimensional Four-node Tetrahedron
      !case ( '157')               ! need test
      !  mapElemtype = 7           ! Three-dimensional, Low-order, Tetrahedron, Herrmann Formulations
      case ( '127')
        mapElemtype = 8            ! Three-dimensional Ten-node Tetrahedron
      case ( '136')
        mapElemtype = 9            ! Three-dimensional Arbitrarily Distorted Pentahedral
      case ( '117')                ! 123 (need test)
        mapElemtype = 10           ! Three-dimensional Arbitrarily Distorted linear hexahedral with reduced integration
      case ( '7')
        mapElemtype = 11           ! Three-dimensional Arbitrarily Distorted Brick
      case ( '57')
        mapElemtype = 12           ! Three-dimensional Arbitrarily Distorted quad hexahedral with reduced integration
      case ( '21')
        mapElemtype = 13           ! Three-dimensional Arbitrarily Distorted quadratic hexahedral
      case default
        call IO_error(190,what)
   end select

  end function mapElemtype


end subroutine inputRead_elemType


!--------------------------------------------------------------------------------------------------
!> @brief Stores node IDs
!--------------------------------------------------------------------------------------------------
function inputRead_connectivityElem(nElem,nNodes,fileContent)

  integer, intent(in) :: &
    nElem, &
    nNodes                                                                                          !< number of nodes per element
  character(len=*), dimension(:), intent(in) :: fileContent                                         !< file content, separated per lines

  integer, dimension(nNodes,nElem) :: &
    inputRead_connectivityElem

  integer, allocatable, dimension(:) :: chunkPos

  integer, dimension(1+nElem) :: contInts
  integer :: i,k,j,t,e,l,nNodesAlreadyRead


  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'connectivity') then
      j = 0
      do i = 1,nElem
        chunkPos = strPos(fileContent(l+1+i+j))
        e = discretization_Marc_FEM2DAMASK_elem(intValue(fileContent(l+1+i+j),chunkPos,1))
        if (e /= 0) then                                                                            ! disregard non CP elems
          do k = 1,chunkPos(1)-2
            inputRead_connectivityElem(k,e) = &
              discretization_Marc_FEM2DAMASK_node(intValue(fileContent(l+1+i+j),chunkPos,k+2))
          end do
          nNodesAlreadyRead = chunkPos(1) - 2
          do while(nNodesAlreadyRead < nNodes)                                                      ! read on if not all nodes in one line
            j = j + 1
            chunkPos = strPos(fileContent(l+1+i+j))
            do k = 1,chunkPos(1)
              inputRead_connectivityElem(nNodesAlreadyRead+k,e) = &
                discretization_Marc_FEM2DAMASK_node(intValue(fileContent(l+1+i+j),chunkPos,k))
            end do
            nNodesAlreadyRead = nNodesAlreadyRead + chunkPos(1)
          end do
        end if
      end do
      exit
    end if
  end do

end function inputRead_connectivityElem


!--------------------------------------------------------------------------------------------------
!> @brief Store material ID
!> @details 0-based ID in file is converted to 1-based ID used in DAMASK
!--------------------------------------------------------------------------------------------------
subroutine inputRead_material(materialAt,&
                              nElem,nNodes,nameElemSet,mapElemSet,initialcondTableStyle,fileContent)

  integer, dimension(:), allocatable, intent(out) :: &
    materialAt
  integer, intent(in) :: &
    nElem, &
    nNodes, &                                                                                       !< number of nodes per element
    initialcondTableStyle
  character(len=*), dimension(:), intent(in) :: nameElemSet
  integer, dimension(:,:),        intent(in) :: mapElemSet                                          !< list of elements in elementSet
  character(len=*), dimension(:), intent(in) :: fileContent                                         !< file content, separated per lines

  integer, allocatable, dimension(:) :: chunkPos

  integer, dimension(1+nElem) :: contInts
  integer :: i,j,t,sv,ID,e,nNodesAlreadyRead,l,k,m


  allocate(materialAt(nElem))

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 2) cycle
    if (IO_lc(strValue(fileContent(l),chunkPos,1)) == 'initial' .and. &
       IO_lc(strValue(fileContent(l),chunkPos,2)) == 'state') then
      k = merge(2,1,initialcondTableStyle == 2)
      chunkPos = strPos(fileContent(l+k))
      sv = intValue(fileContent(l+k),chunkPos,1)                                                 ! # of state variable
      if (sv == 2) then                                                                             ! state var 2 gives material ID
        m = 1
        chunkPos = strPos(fileContent(l+k+m))
        do while (scan(strValue(fileContent(l+k+m),chunkPos,1),'+-',back=.true.)>1)              ! is no Efloat value?
          ID = nint(realValue(fileContent(l+k+m),chunkPos,1))
          if (initialcondTableStyle == 2) m = m + 2
          contInts = continuousIntValues(fileContent(l+k+m+1:),nElem,nameElemSet,mapElemSet,size(nameElemSet)) ! get affected elements
          do i = 1,contInts(1)
            e = discretization_Marc_FEM2DAMASK_elem(contInts(1+i))
            materialAt(e) = ID + 1
          end do
          if (initialcondTableStyle == 0) m = m + 1
        end do
      end if
    end if
  end do

  if (any(materialAt < 1)) call IO_error(180)

end subroutine inputRead_material


!--------------------------------------------------------------------------------------------------
!> @brief Calculates cell node coordinates from element node coordinates
!--------------------------------------------------------------------------------------------------
pure subroutine buildCells(connectivity,definition, &
                           elem,connectivity_elem)

  type(tCellNodeDefinition), dimension(:),    intent(out) :: definition                             ! definition of cell nodes for increasing number of parents
  integer,                   dimension(:,:,:),intent(out) :: connectivity

  type(tElement),                             intent(in)  :: elem                                   ! element definition
  integer,                   dimension(:,:),  intent(in)  :: connectivity_elem                      ! connectivity of the elements

  integer,dimension(:),     allocatable :: candidates_local
  integer,dimension(:,:),   allocatable :: parentsAndWeights,candidates_global

  integer :: e, n, c, p, s,i,m,j,nParentNodes,nCellNode,Nelem,candidateID

  Nelem = size(connectivity_elem,2)

!---------------------------------------------------------------------------------------------------
! initialize global connectivity to negative local connectivity
  connectivity = -spread(elem%cell,3,Nelem)                                                         ! local cell node ID

!---------------------------------------------------------------------------------------------------
! set connectivity of cell nodes that coincide with FE nodes (defined by 1 parent node)
! and renumber local (negative) to global (positive) node ID
  do e = 1, Nelem
    do c = 1, elem%NcellNodes
      realNode: if (count(elem%cellNodeParentNodeWeights(:,c) /= 0) == 1) then
        where(connectivity(:,:,e) == -c) connectivity(:,:,e) = connectivity_elem(c,e)
      end if realNode
    end do
  end do

  nCellNode = maxval(connectivity_elem)

!---------------------------------------------------------------------------------------------------
! set connectivity of cell nodes that are defined by 2,...,nNodes real nodes
  do nParentNodes = 2, elem%nNodes

    ! get IDs of local cell nodes that are defined by the current number of parent nodes
    candidates_local = [integer::]
    do c = 1, elem%NcellNodes
      if (count(elem%cellNodeParentNodeWeights(:,c) /= 0) == nParentNodes) &
        candidates_local = [candidates_local,c]
    end do
    s = size(candidates_local)

    if (allocated(candidates_global)) deallocate(candidates_global)
    allocate(candidates_global(nParentNodes*2+2,s*Nelem))                                           ! stores parent node ID + weight together with element ID and cellnode id (local)
    parentsAndWeights = reshape([(0, i = 1,2*nParentNodes)],[nParentNodes,2])                       ! (re)allocate

    do e = 1, Nelem
      do i = 1, size(candidates_local)
        candidateID = (e-1)*size(candidates_local)+i                                                ! including duplicates, runs to (Nelem*size(candidates_local))
        c = candidates_local(i)                                                                     ! c is local cellnode ID for connectivity
        p = 0
        do j = 1, size(elem%cellNodeParentNodeWeights(:,c))
          if (elem%cellNodeParentNodeWeights(j,c) /= 0) then                                        ! real node 'j' partly defines cell node 'c'
            p = p + 1
            parentsAndWeights(p,1:2) = [connectivity_elem(j,e),elem%cellNodeParentNodeWeights(j,c)]
          end if
        end do
        ! store (and order) real node IDs and their weights together with the element number and local ID
        do p = 1, nParentNodes
          m = maxloc(parentsAndWeights(:,1),1)

          candidates_global(p,                                candidateID) = parentsAndWeights(m,1)
          candidates_global(p+nParentNodes,                   candidateID) = parentsAndWeights(m,2)
          candidates_global(nParentNodes*2+1:nParentNodes*2+2,candidateID) = [e,c]

          parentsAndWeights(m,1) = -huge(parentsAndWeights(m,1))                                    ! out of the competition
        end do
      end do
    end do

    ! sort according to real node IDs + weight (from left to right)
    call math_sort(candidates_global,sortDim=1)                                                     ! sort according to first column

    do p = 2, nParentNodes*2
      n = 1
      do while(n <= size(candidates_local)*Nelem)
        j=0
        do while (n+j<= size(candidates_local)*Nelem)
          if (candidates_global(p-1,n+j)/=candidates_global(p-1,n)) exit
          j = j + 1
        end do
        e = n+j-1
        if (any(candidates_global(p,n:e)/=candidates_global(p,n))) &
          call math_sort(candidates_global(:,n:e),sortDim=p)
        n = e+1
      end do
    end do

    i = uniqueRows(candidates_global(1:2*nParentNodes,:))
    allocate(definition(nParentNodes-1)%parents(i,nParentNodes))
    allocate(definition(nParentNodes-1)%weights(i,nParentNodes))

    i = 1
    n = 1
    do while(n <= size(candidates_local)*Nelem)
      j=0
      parentsAndWeights(:,1) = candidates_global(1:nParentNodes,n+j)
      parentsAndWeights(:,2) = candidates_global(nParentNodes+1:nParentNodes*2,n+j)

      e = candidates_global(nParentNodes*2+1,n+j)
      c = candidates_global(nParentNodes*2+2,n+j)

      do while (n+j<= size(candidates_local)*Nelem)
        if (any(candidates_global(1:2*nParentNodes,n+j)/=candidates_global(1:2*nParentNodes,n))) exit
        where (connectivity(:,:,candidates_global(nParentNodes*2+1,n+j)) == -candidates_global(nParentNodes*2+2,n+j)) ! still locally defined
          connectivity(:,:,candidates_global(nParentNodes*2+1,n+j)) = nCellNode + 1                                   ! get current new cell node id
        end where

        j = j+1
      end do
      nCellNode = nCellNode + 1
      definition(nParentNodes-1)%parents(i,:) = parentsAndWeights(:,1)
      definition(nParentNodes-1)%weights(i,:) = parentsAndWeights(:,2)
      i = i + 1
      n = n+j
    end do

  end do

  contains
  !------------------------------------------------------------------------------------------------
  !> @brief count unique rows (same rows need to be stored consecutively)
  !------------------------------------------------------------------------------------------------
  pure function uniqueRows(A) result(u)

    integer, dimension(:,:), intent(in) :: A                                                        !< array, rows need to be sorted

    integer :: &
      u, &                                                                                          !< # of unique rows
      r, &                                                                                          !< row counter
      d                                                                                             !< duplicate counter

    u = 0
    r = 1
    do while(r <= size(A,2))
      d = 0
      do while (r+d<= size(A,2))
        if (any(A(:,r)/=A(:,r+d))) exit
        d = d+1
      end do
      u = u+1
      r = r+d
    end do

  end function uniqueRows

end subroutine buildCells


!--------------------------------------------------------------------------------------------------
!> @brief Calculates cell node coordinates from element node coordinates
!--------------------------------------------------------------------------------------------------
pure function buildCellNodes(node_elem)

  real(pREAL),               dimension(:,:), intent(in)  :: node_elem                               !< element nodes
  real(pREAL),               dimension(:,:), allocatable :: buildCellNodes                          !< cell node coordinates

  integer :: i, j, k, n


  allocate(buildCellNodes(3,maxval(connectivity_cell)))
  n = size(node_elem,2)
  buildCellNodes(:,1:n) = node_elem                                                                 !< initial nodes coincide with element nodes

  do i = 1, size(cellNodeDefinition)
    do j = 1, size(cellNodeDefinition(i)%parents,1)
      n = n+1
      buildCellNodes(:,n) = 0.0_pREAL
      do k = 1, size(cellNodeDefinition(i)%parents,2)
        buildCellNodes(:,n) = buildCellNodes(:,n) &
                            + buildCellNodes(:,cellNodeDefinition(i)%parents(j,k)) &
                            * real(cellNodeDefinition(i)%weights(j,k),pREAL)
      end do
      buildCellNodes(:,n) = buildCellNodes(:,n)/real(sum(cellNodeDefinition(i)%weights(j,:)),pREAL)
    end do
  end do

end function buildCellNodes


!--------------------------------------------------------------------------------------------------
!> @brief Calculates IP coordinates as center of cell
!--------------------------------------------------------------------------------------------------
pure function buildIPcoordinates(node_cell)

  real(pREAL), dimension(:,:), intent(in)  :: node_cell                                             !< cell node coordinates
  real(pREAL), dimension(:,:), allocatable :: buildIPcoordinates                                    !< cell-center/IP coordinates

  integer, dimension(:,:), allocatable :: connectivity_cell_reshaped
  integer :: i, n, NcellNodesPerCell,Ncells


  NcellNodesPerCell = size(connectivity_cell,1)
  Ncells = size(connectivity_cell,2)*size(connectivity_cell,3)
  connectivity_cell_reshaped = reshape(connectivity_cell,[NcellNodesPerCell,Ncells])

  allocate(buildIPcoordinates(3,Ncells))

  do i = 1, size(connectivity_cell_reshaped,2)
    buildIPcoordinates(:,i) = 0.0_pREAL
    do n = 1, size(connectivity_cell_reshaped,1)
      buildIPcoordinates(:,i) = buildIPcoordinates(:,i) &
                              + node_cell(:,connectivity_cell_reshaped(n,i))
    end do
    buildIPcoordinates(:,i) = buildIPcoordinates(:,i)/real(size(connectivity_cell_reshaped,1),pREAL)
  end do

end function buildIPcoordinates


!---------------------------------------------------------------------------------------------------
!> @brief Calculates IP volume.
!> @details The IP volume is calculated differently depending on the cell type.
!> 2D cells assume an element depth of 1.0
!---------------------------------------------------------------------------------------------------
pure function IPvolume(elem,node)

  type(tElement),                intent(in) :: elem
  real(pREAL), dimension(:,:),   intent(in) :: node

  real(pREAL), dimension(elem%nIPs,size(connectivity_cell,3)) :: IPvolume
  real(pREAL), dimension(3) :: x0,x1,x2,x3,x4,x5,x6,x7

  integer :: e,i


  do e = 1,size(connectivity_cell,3)
    do i = 1,elem%nIPs

      select case (elem%cellType)
        case (1)                                                                                    ! 2D 3node
          IPvolume(i,e) = math_areaTriangle(node(1:3,connectivity_cell(1,i,e)), &
                                            node(1:3,connectivity_cell(2,i,e)), &
                                            node(1:3,connectivity_cell(3,i,e)))

        case (2)                                                                                    ! 2D 4node
          IPvolume(i,e) = math_areaTriangle(node(1:3,connectivity_cell(1,i,e)), &                   ! assume planar shape, division in two triangles suffices
                                            node(1:3,connectivity_cell(2,i,e)), &
                                            node(1:3,connectivity_cell(3,i,e))) &
                        + math_areaTriangle(node(1:3,connectivity_cell(3,i,e)), &
                                            node(1:3,connectivity_cell(4,i,e)), &
                                            node(1:3,connectivity_cell(1,i,e)))
        case (3)                                                                                    ! 3D 4node
          IPvolume(i,e) = math_volTetrahedron(node(1:3,connectivity_cell(1,i,e)), &
                                              node(1:3,connectivity_cell(2,i,e)), &
                                              node(1:3,connectivity_cell(3,i,e)), &
                                              node(1:3,connectivity_cell(4,i,e)))
        case (4)                                                                                    ! 3D 8node
          ! J. Grandy, Efficient Calculation of Volume of Hexahedral Cells
          ! Lawrence Livermore National Laboratory
          ! https://www.osti.gov/servlets/purl/632793
          x0 = node(1:3,connectivity_cell(1,i,e))
          x1 = node(1:3,connectivity_cell(2,i,e))
          x2 = node(1:3,connectivity_cell(4,i,e))
          x3 = node(1:3,connectivity_cell(3,i,e))
          x4 = node(1:3,connectivity_cell(5,i,e))
          x5 = node(1:3,connectivity_cell(6,i,e))
          x6 = node(1:3,connectivity_cell(8,i,e))
          x7 = node(1:3,connectivity_cell(7,i,e))
          IPvolume(i,e) = dot_product((x7-x1)+(x6-x0),math_cross((x7-x2),        (x3-x0))) &
                        + dot_product((x6-x0),        math_cross((x7-x2)+(x5-x0),(x7-x4))) &
                        + dot_product((x7-x1),        math_cross((x5-x0),        (x7-x4)+(x3-x0)))
          IPvolume(i,e) = IPvolume(i,e)/12.0_pREAL
      end select
    end do
  end do

end function IPvolume


!--------------------------------------------------------------------------------------------------
!> @brief calculation of IP interface areas
!--------------------------------------------------------------------------------------------------
pure function IPareaNormal(elem,nElem,node)

  type(tElement),                intent(in) :: elem
  integer,                       intent(in) :: nElem
  real(pREAL), dimension(:,:),   intent(in) :: node

  real(pREAL), dimension(3,elem%nIPneighbors,elem%nIPs,nElem) :: ipAreaNormal

  real(pREAL), dimension (3,size(elem%cellFace,1)) :: nodePos
  integer :: e,i,f,n,m

  m = size(elem%cellFace,1)

  do e = 1,nElem
    do i = 1,elem%nIPs
      do f = 1,elem%nIPneighbors
        nodePos = node(1:3,connectivity_cell(elem%cellface(1:m,f),i,e))

        select case (elem%cellType)
          case (1,2)                                                                                ! 2D 3 or 4 node
            IPareaNormal(1,f,i,e) =   nodePos(2,2) - nodePos(2,1)                                   ! x_normal =  y_connectingVector
            IPareaNormal(2,f,i,e) = -(nodePos(1,2) - nodePos(1,1))                                  ! y_normal = -x_connectingVector
            IPareaNormal(3,f,i,e) = 0.0_pREAL
          case (3)                                                                                  ! 3D 4node
            IPareaNormal(1:3,f,i,e) = math_cross(nodePos(1:3,2) - nodePos(1:3,1), &
                                                 nodePos(1:3,3) - nodePos(1:3,1))
          case (4)                                                                                  ! 3D 8node
            ! Get the normal of the quadrilateral face as the average of four normals of triangular
            ! subfaces. Since the face consists only of two triangles, the sum has to be divided
            ! by two. This procedure tries to compensate for probable non-planar cell surfaces
            IPareaNormal(1:3,f,i,e) = 0.0_pREAL
            do n = 1, m
              IPareaNormal(1:3,f,i,e) = IPareaNormal(1:3,f,i,e) &
                                      + math_cross(nodePos(1:3,mod(n+0,m)+1) - nodePos(1:3,n), &
                                                   nodePos(1:3,mod(n+1,m)+1) - nodePos(1:3,n)) * 0.5_pREAL
            end do
        end select
      end do
    end do
  end do

end function IPareaNormal


!--------------------------------------------------------------------------------------------------
!> @brief IP neighborhood
!--------------------------------------------------------------------------------------------------
function IPneighborhood(elem)

  type(tElement),            intent(in) :: elem                                                     ! definition of the element in use
  integer, dimension(3,size(elem%cellFace,2), &
                     size(connectivity_cell,2),size(connectivity_cell,3)) :: IPneighborhood                   ! neighboring IPs as [element ID, IP ID, face ID]

  integer, dimension(size(elem%cellFace,1)+3,&
                     size(elem%cellFace,2)*size(connectivity_cell,2)*size(connectivity_cell,3)) :: face
  integer, dimension(size(connectivity_cell,1))  :: myConnectivity
  integer, dimension(size(elem%cellFace,1))      :: face_unordered
  integer :: e,i,f,n,c,s

  c = 0
  do e = 1, size(connectivity_cell,3)
    do i = 1, size(connectivity_cell,2)
      myConnectivity = connectivity_cell(:,i,e)
      do f = 1, size(elem%cellFace,2)
        c = c + 1
        face_unordered = myConnectivity(elem%cellFace(:,f))
        do n = 1, size(face_unordered)
          face(n,c) = minval(face_unordered)
          face_unordered(minloc(face_unordered)) = huge(face_unordered)
        end do
        face(n:n+3,c) = [e,i,f]
      end do
  end do; end do

!--------------------------------------------------------------------------------------------------
! sort face definitions
  call math_sort(face,sortDim=1)
  do c=2, size(face,1)-4
    s = 1
    e = 1
    do while (e < size(face,2))
      e = e + 1
      if (any(face(:c,s) /= face(:c,e))) then
        if (e-1/=s) call math_sort(face(:,s:e-1),sortDim=c)
        s = e
      end if
    end do
  end do

  IPneighborhood = 0
  do c=1, size(face,2) - 1
    if (all(face(:n-1,c) == face(:n-1,c+1))) then
      IPneighborhood(:,face(n+2,c+1),face(n+1,c+1),face(n+0,c+1)) = face(n:n+3,c+0)
      IPneighborhood(:,face(n+2,c+0),face(n+1,c+0),face(n+0,c+0)) = face(n:n+3,c+1)
    end if
  end do

end function IPneighborhood

!--------------------------------------------------------------------------------------------------
!> @brief Locate all whitespace-separated chunks in given string and returns array containing
!! number them and the left/right position to be used by IO_xxxVal.
!! Array size is dynamically adjusted to number of chunks found in string
!! IMPORTANT: first element contains number of chunks!
!--------------------------------------------------------------------------------------------------
pure function strPos(str)

  character(len=*),                  intent(in) :: str                                              !< string in which chunk positions are searched for
  integer, dimension(:), allocatable            :: strPos

  integer :: left, right


  allocate(strPos(1), source=0)
  right = 0

  do while (verify(str(right+1:),IO_WHITESPACE)>0)
    left  = right + verify(str(right+1:),IO_WHITESPACE)
    right = left + scan(str(left:),IO_WHITESPACE) - 2
    strPos = [strPos,left,right]
    strPos(1) = strPos(1)+1
    endOfStr: if (right < left) then
      strPos(strPos(1)*2+1) = len_trim(str)
      exit
    end if endOfStr
  end do

end function strPos


!--------------------------------------------------------------------------------------------------
!> @brief Read string value at myChunk from string.
!--------------------------------------------------------------------------------------------------
function strValue(str,chunkPos,myChunk)

  character(len=*),             intent(in) :: str                                                   !< raw input with known start and end of each chunk
  integer,   dimension(:),      intent(in) :: chunkPos                                              !< positions of start and end of each tag/chunk in given string
  integer,                      intent(in) :: myChunk                                               !< position number of desired chunk
  character(len=:), allocatable            :: strValue


  validChunk: if (myChunk > chunkPos(1) .or. myChunk < 1) then
    strValue = ''
    call IO_error(110,'strValue: "'//trim(str)//'"',label1='chunk',ID1=myChunk)
  else validChunk
    strValue = str(chunkPos(myChunk*2):chunkPos(myChunk*2+1))
  end if validChunk

end function strValue


!--------------------------------------------------------------------------------------------------
!> @brief Read integer value at myChunk from string.
!--------------------------------------------------------------------------------------------------
integer function intValue(str,chunkPos,myChunk)

  character(len=*),      intent(in) :: str                                                          !< raw input with known start and end of each chunk
  integer, dimension(:), intent(in) :: chunkPos                                                     !< positions of start and end of each tag/chunk in given string
  integer,               intent(in) :: myChunk                                                      !< position number of desired chunk


  intValue = IO_strAsInt(strValue(str,chunkPos,myChunk))

end function intValue


!--------------------------------------------------------------------------------------------------
!> @brief Read real value at myChunk from string.
!--------------------------------------------------------------------------------------------------
real(pREAL) function realValue(str,chunkPos,myChunk)

  character(len=*),        intent(in) :: str                                                        !< raw input with known start and end of each chunk
  integer,   dimension(:), intent(in) :: chunkPos                                                   !< positions of start and end of each tag/chunk in given string
  integer,                 intent(in) :: myChunk                                                    !< position number of desired chunk


  realValue = IO_strAsReal(strValue(str,chunkPos,myChunk))

end function realValue


!--------------------------------------------------------------------------------------------------
!> @brief return integer list corresponding to items in consecutive lines.
!! First integer in array is counter
!> @details ints concatenated by "c" as last char, range of a "to" b, or named set
!--------------------------------------------------------------------------------------------------
function continuousIntValues(fileContent,maxN,lookupName,lookupMap,lookupMaxN)

  character(len=*), dimension(:),   intent(in) :: fileContent                                       !< file content, separated per lines
  integer,                          intent(in) :: maxN
  integer,                          intent(in) :: lookupMaxN
  integer,          dimension(:,:), intent(in) :: lookupMap
  character(len=*), dimension(:),   intent(in) :: lookupName

  integer,           dimension(1+maxN)         :: continuousIntValues

  integer :: l,i,first,last
  integer, allocatable, dimension(:) :: chunkPos
  logical :: rangeGeneration

  continuousIntValues = 0
  rangeGeneration = .false.

  do l = 1, size(fileContent)
    chunkPos = strPos(fileContent(l))
    if (chunkPos(1) < 1) then                                                                       ! empty line
      exit
    elseif (verify(strValue(fileContent(l),chunkPos,1),'0123456789') > 0) then                   ! a non-int, i.e. set name
      do i = 1, lookupMaxN                                                                          ! loop over known set names
        if (strValue(fileContent(l),chunkPos,1) == lookupName(i)) then                           ! found matching name
          continuousIntValues = lookupMap(:,i)                                                      ! return resp. entity list
          exit
        end if
      end do
      exit
    elseif (containsRange(fileContent(l),chunkPos)) then
      first = intValue(fileContent(l),chunkPos,1)
      last  = intValue(fileContent(l),chunkPos,3)
      do i = first, last, sign(1,last-first)
        continuousIntValues(1) = continuousIntValues(1) + 1
        continuousIntValues(1+continuousIntValues(1)) = i
      end do
      exit
    else
      do i = 1,chunkPos(1)-1                                                                        ! interpret up to second to last value
        continuousIntValues(1) = continuousIntValues(1) + 1
        continuousIntValues(1+continuousIntValues(1)) = intValue(fileContent(l),chunkPos,i)
      end do
      if ( IO_lc(strValue(fileContent(l),chunkPos,chunkPos(1))) /= 'c' ) then                    ! line finished, read last value
        continuousIntValues(1) = continuousIntValues(1) + 1
        continuousIntValues(1+continuousIntValues(1)) = intValue(fileContent(l),chunkPos,chunkPos(1))
        exit
      end if
    end if
  end do

end function continuousIntValues


!--------------------------------------------------------------------------------------------------
!> @brief return whether a line contains a range ('X to Y')
!--------------------------------------------------------------------------------------------------
logical function containsRange(str,chunkPos)

  character(len=*),      intent(in) :: str
  integer, dimension(:), intent(in) :: chunkPos                                                     !< positions of start and end of each tag/chunk in given string


  containsRange = .False.
  if (chunkPos(1) == 3) then
    if (IO_lc(strValue(str,chunkPos,2)) == 'to') containsRange = .True.
  end if

end function containsRange

end module discretization_Marc
