!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Parse geometry file to set up discretization and geometry for nonlocal model
!--------------------------------------------------------------------------------------------------
module discretization_grid
#include <petsc/finclude/petscsys.h>
  use PETScSys
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  use MPI_f08
#endif
  use FFTW3

  use prec
  use parallelization
  use system_routines
  use VTI
  use CLI
  use IO
  use config
  use HDF5
  use HDF5_utilities
  use result
  use discretization
  use geometry_plastic_nonlocal

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  implicit none(type,external)
#else
  implicit none
#endif
  private

  integer,     dimension(3), public, protected :: &
    cells                                                                                           !< (global) cells
  integer,                   public, protected :: &
    cells3, &                                                                                       !< (local) cells in 3rd direction
    cells3Offset                                                                                    !< (local) cells offset in 3rd direction
  real(pREAL), dimension(3), public, protected :: &
    geomSize                                                                                        !< (global) physical size
  real(pREAL),               public, protected :: &
    size3, &                                                                                        !< (local) size in 3rd direction
    size3offset                                                                                     !< (local) size offset in 3rd direction

  public :: &
    discretization_grid_init, &
    discretization_grid_getInitialCondition

contains


!--------------------------------------------------------------------------------------------------
!> @brief Read the geometry file to obtain information on discretization.
!--------------------------------------------------------------------------------------------------
subroutine discretization_grid_init(restart)

  logical, intent(in) :: restart

  real(pREAL), dimension(3) :: &
    mySize, &                                                                                       !< domain size of this process
    origin                                                                                          !< (global) distance to origin
  integer,     dimension(3) :: &
    myGrid                                                                                          !< domain cells of this process

  integer,     dimension(:),   allocatable :: &
    materialAt, materialAt_global

  integer :: &
    j
  integer(MPI_INTEGER_KIND) :: err_MPI
  integer(C_INTPTR_T) :: &
    devNull, cells3_, cells3Offset_
  integer, dimension(worldsize) :: &
    displs, sendcounts
  character(len=:), allocatable :: &
    fileContent, fname
  integer(HID_T) :: handle


  print'(/,1x,a)', '<<<+-  discretization_grid init  -+>>>'; flush(IO_STDOUT)


  if (worldrank == 0) then
    fileContent = IO_read(CLI_geomFile)
    call VTI_readCellsSizeOrigin(cells,geomSize,origin,fileContent)
    materialAt_global = VTI_readDataset_int(fileContent,'material') + 1
    if (any(materialAt_global < 1)) &
      call IO_error(180,ext_msg='material ID < 1')
    if (size(materialAt_global) /= product(cells)) &
      call IO_error(180,ext_msg='mismatch in # of material IDs and cells')
    fname = CLI_geomFile
    if (scan(fname,'/') /= 0) fname = fname(scan(fname,'/',.true.)+1:)
    call result_openJobFile(parallel=.false.)
    call result_addSetupFile(fileContent,fname,'geometry definition (grid solver)')
    call result_closeJobFile()
  else
    allocate(materialAt_global(0))                                                                  ! needed for IntelMPI
  end if


  call MPI_Bcast(cells,3_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD, err_MPI)
  call parallelization_chkerr(err_MPI)
  if (cells(1) < 2) call IO_error(844, ext_msg='cells(1) must be larger than 1')
  call MPI_Bcast(geomSize,3_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD, err_MPI)
  call parallelization_chkerr(err_MPI)
  call MPI_Bcast(origin,3_MPI_INTEGER_KIND,MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD, err_MPI)
  call parallelization_chkerr(err_MPI)

  print'(/,1x,a,i0,a,i0,a,i0)',            'cells:  ', cells(1),    ' × ', cells(2),    ' × ', cells(3)
  print  '(1x,a,es8.2,a,es8.2,a,es8.2,a)', 'size:   ', geomSize(1), ' × ', geomSize(2), ' × ', geomSize(3), ' m³'
  print  '(1x,a,es9.2,a,es9.2,a,es9.2,a)', 'origin: ', origin(1),   ' ',   origin(2),   ' ',   origin(3), ' m'

  if (worldsize>cells(3)) call IO_error(894, ext_msg='number of processes exceeds cells(3)')

  call fftw_mpi_init()
  devNull = fftw_mpi_local_size_3d(int(cells(3),C_INTPTR_T),int(cells(2),C_INTPTR_T),int(cells(1)/2+1,C_INTPTR_T), &
                                   PETSC_COMM_WORLD, &
                                   cells3_, &                                                       ! domain cells size along z
                                   cells3Offset_)                                                   ! domain cells offset along z
  if (cells3_==0_C_INTPTR_T) call IO_error(894, ext_msg='Cannot distribute MPI processes')

  cells3       = int(cells3_)
  cells3Offset = int(cells3Offset_)
  size3       = geomSize(3)*real(cells3,pREAL)      /real(cells(3),pREAL)
  size3Offset = geomSize(3)*real(cells3Offset,pREAL)/real(cells(3),pREAL)
  myGrid = [cells(1:2),cells3]
  mySize = [geomSize(1:2),size3]

  call MPI_Gather(product(cells(1:2))*cells3Offset,1_MPI_INTEGER_KIND,MPI_INTEGER,displs,&
                  1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)
  call MPI_Gather(product(myGrid),               1_MPI_INTEGER_KIND,MPI_INTEGER,sendcounts,&
                  1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)

  allocate(materialAt(product(myGrid)))
  call MPI_Scatterv(materialAt_global,sendcounts,displs,MPI_INTEGER,materialAt,size(materialAt),&
                    MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)

  call discretization_init(materialAt, &
                           IPcoordinates0(myGrid,mySize,cells3Offset), &
                           Nodes0(myGrid,mySize,cells3Offset),&
                           merge((cells(1)+1) * (cells(2)+1) * (cells3+1),&                         ! write top layer...
                                 (cells(1)+1) * (cells(2)+1) *  cells3,&                            ! ...unless not last process
                                 worldrank+1==worldsize))

!--------------------------------------------------------------------------------------------------
! store geometry information for post processing
  if (.not. restart .and. worldrank == 0) then
    call result_openJobFile(parallel=.false.)
    handle = result_addGroup('geometry')
    call HDF5_write(cells,   handle,'cells', .false.)
    call HDF5_write(geomSize,handle,'size',  .false.)
    call HDF5_write(origin,  handle,'origin',.false.)
    call HDF5_addAttribute(handle,'unit','1','cells')
    call HDF5_addAttribute(handle,'unit','m³','size')
    call HDF5_addAttribute(handle,'unit','m','origin')
    call result_addAttribute('cells', cells,   '/geometry') ! legacy for DADF5 1.x
    call result_addAttribute('size',  geomSize,'/geometry') ! legacy for DADF5 1.x
    call result_addAttribute('origin',origin,  '/geometry') ! legacy for DADF5 1.x
    call result_closeGroup(handle)
    call result_closeJobFile()
  end if

!--------------------------------------------------------------------------------------------------
! geometry information required by the nonlocal CP model
  call geometry_plastic_nonlocal_setIPvolume(reshape([(product(mySize/real(myGrid,pREAL)),j=1,product(myGrid))], &
                                                     [1,product(myGrid)]))
  call geometry_plastic_nonlocal_setIParea        (cellSurfaceArea(mySize,myGrid))
  call geometry_plastic_nonlocal_setIPareaNormal  (cellSurfaceNormal(product(myGrid)))
  call geometry_plastic_nonlocal_setIPneighborhood(IPneighborhood(myGrid))

end subroutine discretization_grid_init


!---------------------------------------------------------------------------------------------------
!> @brief Calculate undeformed position of IPs/cell centers (pretend to be an element).
!---------------------------------------------------------------------------------------------------
function IPcoordinates0(cells,geomSize,cells3Offset)

  integer,     dimension(3), intent(in) :: cells                                                    ! cells (for this process!)
  real(pREAL), dimension(3), intent(in) :: geomSize                                                 ! size (for this process!)
  integer,                   intent(in) :: cells3Offset                                             ! cells(3) offset

  real(pREAL), dimension(3,product(cells))  :: ipCoordinates0

  integer :: &
    a,b,c, &
    i


  i = 0
  do c = 1, cells(3); do b = 1, cells(2); do a = 1, cells(1)
    i = i + 1
    IPcoordinates0(1:3,i) = geomSize/real(cells,pREAL) * (real([a,b,cells3Offset+c],pREAL) -0.5_pREAL)
  end do; end do; end do

end function IPcoordinates0


!---------------------------------------------------------------------------------------------------
!> @brief Calculate position of undeformed nodes (pretend to be an element).
!---------------------------------------------------------------------------------------------------
pure function nodes0(cells,geomSize,cells3Offset)

  integer,     dimension(3), intent(in) :: cells                                                    ! cells (for this process!)
  real(pREAL), dimension(3), intent(in) :: geomSize                                                 ! size (for this process!)
  integer,                   intent(in) :: cells3Offset                                             ! cells(3) offset

  real(pREAL), dimension(3,product(cells+1)) :: nodes0

  integer :: &
    a,b,c, &
    n

  n = 0
  do c = 0, cells3; do b = 0, cells(2); do a = 0, cells(1)
    n = n + 1
    nodes0(1:3,n) = geomSize/real(cells,pREAL) * real([a,b,cells3Offset+c],pREAL)
  end do; end do; end do

end function nodes0


!--------------------------------------------------------------------------------------------------
!> @brief Calculate IP interface areas.
!--------------------------------------------------------------------------------------------------
pure function cellSurfaceArea(geomSize,cells)

  real(pREAL), dimension(3), intent(in) :: geomSize                                                 ! size (for this process!)
  integer,     dimension(3), intent(in) :: cells                                                    ! cells (for this process!)

  real(pREAL), dimension(6,1,product(cells)) :: cellSurfaceArea


  cellSurfaceArea(1:2,1,:) = geomSize(2)/real(cells(2),pREAL) * geomSize(3)/real(cells(3),pREAL)
  cellSurfaceArea(3:4,1,:) = geomSize(3)/real(cells(3),pREAL) * geomSize(1)/real(cells(1),pREAL)
  cellSurfaceArea(5:6,1,:) = geomSize(1)/real(cells(1),pREAL) * geomSize(2)/real(cells(2),pREAL)

end function cellSurfaceArea


!--------------------------------------------------------------------------------------------------
!> @brief Calculate IP interface areas normals.
!--------------------------------------------------------------------------------------------------
pure function cellSurfaceNormal(nElems)

  integer, intent(in) :: nElems

  real(pREAL), dimension(3,6,1,nElems) :: cellSurfaceNormal

  cellSurfaceNormal(1:3,1,1,:) = spread([+1.0_pREAL, 0.0_pREAL, 0.0_pREAL],2,nElems)
  cellSurfaceNormal(1:3,2,1,:) = spread([-1.0_pREAL, 0.0_pREAL, 0.0_pREAL],2,nElems)
  cellSurfaceNormal(1:3,3,1,:) = spread([ 0.0_pREAL,+1.0_pREAL, 0.0_pREAL],2,nElems)
  cellSurfaceNormal(1:3,4,1,:) = spread([ 0.0_pREAL,-1.0_pREAL, 0.0_pREAL],2,nElems)
  cellSurfaceNormal(1:3,5,1,:) = spread([ 0.0_pREAL, 0.0_pREAL,+1.0_pREAL],2,nElems)
  cellSurfaceNormal(1:3,6,1,:) = spread([ 0.0_pREAL, 0.0_pREAL,-1.0_pREAL],2,nElems)

end function cellSurfaceNormal


!--------------------------------------------------------------------------------------------------
!> @brief Build IP neighborhood relations.
!--------------------------------------------------------------------------------------------------
pure function IPneighborhood(cells)

  integer, dimension(3), intent(in) :: cells                                                        ! cells (for this process!)

  integer, dimension(3,6,1,product(cells)) :: IPneighborhood                                        !< 6 neighboring IPs as [element ID, IP ID, face ID]

  integer :: &
   x,y,z, &
   e

  e = 0
  do z = 0,cells(3)-1; do y = 0,cells(2)-1; do x = 0,cells(1)-1
    e = e + 1
    ! element ID
    IPneighborhood(1,1,1,e) = z * cells(1) * cells(2) &
                            + y * cells(1) &
                            + modulo(x+1,cells(1)) &
                            + 1
    IPneighborhood(1,2,1,e) = z * cells(1) * cells(2) &
                            + y * cells(1) &
                            + modulo(x-1,cells(1)) &
                            + 1
    IPneighborhood(1,3,1,e) = z * cells(1) * cells(2) &
                            + modulo(y+1,cells(2)) * cells(1) &
                            + x &
                            + 1
    IPneighborhood(1,4,1,e) = z * cells(1) * cells(2) &
                            + modulo(y-1,cells(2)) * cells(1) &
                            + x &
                            + 1
    IPneighborhood(1,5,1,e) = modulo(z+1,cells(3)) * cells(1) * cells(2) &
                            + y * cells(1) &
                            + x &
                            + 1
    IPneighborhood(1,6,1,e) = modulo(z-1,cells(3)) * cells(1) * cells(2) &
                            + y * cells(1) &
                            + x &
                            + 1
    ! IP ID
    IPneighborhood(2,:,1,e) = 1

    ! face ID
    IPneighborhood(3,1,1,e) = 2
    IPneighborhood(3,2,1,e) = 1
    IPneighborhood(3,3,1,e) = 4
    IPneighborhood(3,4,1,e) = 3
    IPneighborhood(3,5,1,e) = 6
    IPneighborhood(3,6,1,e) = 5

  end do; end do; end do

end function IPneighborhood


!--------------------------------------------------------------------------------------------------
!> @brief Read initial condition from VTI file.
!--------------------------------------------------------------------------------------------------
function discretization_grid_getInitialCondition(label) result(ic)

  character(len=*), intent(in) :: label
  real(pREAL), dimension(cells(1),cells(2),cells3) :: ic

  real(pREAL), dimension(:), allocatable :: ic_global, ic_local
  integer(MPI_INTEGER_KIND) :: err_MPI
  integer, dimension(worldsize) :: &
    displs, sendcounts


  if (worldrank == 0) then
    ic_global = VTI_readDataset_real(IO_read(CLI_geomFile),label)
  else
    allocate(ic_global(0))                                                                          ! needed for IntelMPI
  end if

  call MPI_Gather(product(cells(1:2))*cells3Offset, 1_MPI_INTEGER_KIND,MPI_INTEGER,displs,&
                  1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)
  call MPI_Gather(product(cells(1:2))*cells3,      1_MPI_INTEGER_KIND,MPI_INTEGER,sendcounts,&
                  1_MPI_INTEGER_KIND,MPI_INTEGER,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)

  allocate(ic_local(product(cells(1:2))*cells3))
  call MPI_Scatterv(ic_global,sendcounts,displs,MPI_DOUBLE,ic_local,size(ic_local),&
                    MPI_DOUBLE,0_MPI_INTEGER_KIND,MPI_COMM_WORLD,err_MPI)
  call parallelization_chkerr(err_MPI)

  ic = reshape(ic_local,[cells(1),cells(2),cells3])

end function discretization_grid_getInitialCondition

end module discretization_grid
