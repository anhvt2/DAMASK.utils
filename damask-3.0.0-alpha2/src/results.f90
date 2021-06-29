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
!> @author Vitesh Shah, Max-Planck-Institut für Eisenforschung GmbH
!> @author Yi-Chin Yang, Max-Planck-Institut für Eisenforschung GmbH
!> @author Jennifer Nastola, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!--------------------------------------------------------------------------------------------------
module results
  use DAMASK_interface
  use parallelization
  use IO
  use rotations
  use HDF5_utilities
#ifdef PETSc
  use PETSC
#endif

  implicit none
  private

  integer(HID_T) :: resultsFile

  interface results_writeDataset

    module procedure results_writeTensorDataset_real
    module procedure results_writeVectorDataset_real
    module procedure results_writeScalarDataset_real

    module procedure results_writeTensorDataset_int
    module procedure results_writeVectorDataset_int

    module procedure results_writeScalarDataset_rotation

  end interface results_writeDataset

  interface results_addAttribute

    module procedure results_addAttribute_real
    module procedure results_addAttribute_int
    module procedure results_addAttribute_str

    module procedure results_addAttribute_int_array
    module procedure results_addAttribute_real_array

  end interface results_addAttribute

  public :: &
    results_init, &
    results_openJobFile, &
    results_closeJobFile, &
    results_addIncrement, &
    results_finalizeIncrement, &
    results_addGroup, &
    results_openGroup, &
    results_closeGroup, &
    results_writeDataset, &
    results_setLink, &
    results_addAttribute, &
    results_removeLink, &
    results_mapping_constituent, &
    results_mapping_homogenization
contains

subroutine results_init(restart)

  logical, intent(in) :: restart

  character(len=pStringLen) :: commandLine

  print'(/,a)', ' <<<+-  results init  -+>>>'; flush(IO_STDOUT)

  print*, 'Diehl et al., Integrating Materials and Manufacturing Innovation 6(1):83–91, 2017'
  print*, 'https://doi.org/10.1007/s40192-017-0084-5'//IO_EOL

  if(.not. restart) then
    resultsFile = HDF5_openFile(trim(getSolverJobName())//'.hdf5','w',.true.)
    call results_addAttribute('DADF5_version_major',0)
    call results_addAttribute('DADF5_version_minor',10)
    call results_addAttribute('DAMASK_version',DAMASKVERSION)
    call get_command(commandLine)
    call results_addAttribute('Call',trim(commandLine))
    call results_closeGroup(results_addGroup('mapping'))
    call results_closeJobFile
  endif

end subroutine results_init


!--------------------------------------------------------------------------------------------------
!> @brief opens the results file to append data
!--------------------------------------------------------------------------------------------------
subroutine results_openJobFile

  resultsFile = HDF5_openFile(trim(getSolverJobName())//'.hdf5','a',.true.)

end subroutine results_openJobFile


!--------------------------------------------------------------------------------------------------
!> @brief closes the results file
!--------------------------------------------------------------------------------------------------
subroutine results_closeJobFile

  call HDF5_closeFile(resultsFile)

end subroutine results_closeJobFile


!--------------------------------------------------------------------------------------------------
!> @brief creates the group of increment and adds time as attribute to the file
!--------------------------------------------------------------------------------------------------
subroutine results_addIncrement(inc,time)

  integer,       intent(in) :: inc
  real(pReal),   intent(in) :: time
  character(len=pStringLen) :: incChar

  write(incChar,'(i10)') inc
  call results_closeGroup(results_addGroup(trim('inc'//trim(adjustl(incChar)))))
  call results_setLink(trim('inc'//trim(adjustl(incChar))),'current')
  call results_addAttribute('time/s',time,trim('inc'//trim(adjustl(incChar))))
  call results_closeGroup(results_addGroup('current/phase'))
  call results_closeGroup(results_addGroup('current/homogenization'))

end subroutine results_addIncrement


!--------------------------------------------------------------------------------------------------
!> @brief finalize increment
!> @details remove soft link
!--------------------------------------------------------------------------------------------------
subroutine results_finalizeIncrement

  call results_removeLink('current')

end subroutine results_finalizeIncrement


!--------------------------------------------------------------------------------------------------
!> @brief open a group from the results file
!--------------------------------------------------------------------------------------------------
integer(HID_T) function results_openGroup(groupName)

  character(len=*), intent(in) :: groupName

  results_openGroup = HDF5_openGroup(resultsFile,groupName)

end function results_openGroup


!--------------------------------------------------------------------------------------------------
!> @brief adds a new group to the results file
!--------------------------------------------------------------------------------------------------
integer(HID_T) function results_addGroup(groupName)

  character(len=*), intent(in) :: groupName

  results_addGroup = HDF5_addGroup(resultsFile,groupName)

end function results_addGroup


!--------------------------------------------------------------------------------------------------
!> @brief close a group
!--------------------------------------------------------------------------------------------------
subroutine results_closeGroup(group_id)

  integer(HID_T), intent(in) :: group_id

  call HDF5_closeGroup(group_id)

end subroutine results_closeGroup


!--------------------------------------------------------------------------------------------------
!> @brief set link to object in results file
!--------------------------------------------------------------------------------------------------
subroutine results_setLink(path,link)

  character(len=*), intent(in) :: path, link

  call HDF5_setLink(resultsFile,path,link)

end subroutine results_setLink

!--------------------------------------------------------------------------------------------------
!> @brief adds a string attribute to an object in the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addAttribute_str(attrLabel,attrValue,path)

  character(len=*), intent(in)           :: attrLabel, attrValue
  character(len=*), intent(in), optional :: path

  if (present(path)) then
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue, path)
  else
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue)
  endif

end subroutine results_addAttribute_str


!--------------------------------------------------------------------------------------------------
!> @brief adds an integer attribute an object in the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addAttribute_int(attrLabel,attrValue,path)

  character(len=*), intent(in)           :: attrLabel
  integer,          intent(in)           :: attrValue
  character(len=*), intent(in), optional :: path

  if (present(path)) then
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue, path)
  else
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue)
  endif

end subroutine results_addAttribute_int


!--------------------------------------------------------------------------------------------------
!> @brief adds a real attribute an object in the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addAttribute_real(attrLabel,attrValue,path)

  character(len=*), intent(in)           :: attrLabel
  real(pReal),      intent(in)           :: attrValue
  character(len=*), intent(in), optional :: path

  if (present(path)) then
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue, path)
  else
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue)
  endif

end subroutine results_addAttribute_real


!--------------------------------------------------------------------------------------------------
!> @brief adds an integer array attribute an object in the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addAttribute_int_array(attrLabel,attrValue,path)

  character(len=*), intent(in)               :: attrLabel
  integer,          intent(in), dimension(:) :: attrValue
  character(len=*), intent(in), optional     :: path

  if (present(path)) then
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue, path)
  else
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue)
  endif

end subroutine results_addAttribute_int_array


!--------------------------------------------------------------------------------------------------
!> @brief adds a real array attribute an object in the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addAttribute_real_array(attrLabel,attrValue,path)

  character(len=*), intent(in)               :: attrLabel
  real(pReal),      intent(in), dimension(:) :: attrValue
  character(len=*), intent(in), optional     :: path

  if (present(path)) then
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue, path)
  else
    call HDF5_addAttribute(resultsFile,attrLabel, attrValue)
  endif

end subroutine results_addAttribute_real_array


!--------------------------------------------------------------------------------------------------
!> @brief remove link to an object
!--------------------------------------------------------------------------------------------------
subroutine results_removeLink(link)

  character(len=*), intent(in) :: link
  integer                      :: hdferr

  call h5ldelete_f(resultsFile,link, hdferr)
  if (hdferr < 0) call IO_error(1,ext_msg = 'results_removeLink: h5ldelete_soft_f ('//trim(link)//')')

end subroutine results_removeLink


!--------------------------------------------------------------------------------------------------
!> @brief stores a scalar dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeScalarDataset_real(group,dataset,label,description,SIunit)

  character(len=*), intent(in)                  :: label,group,description
  character(len=*), intent(in),    optional     :: SIunit
  real(pReal),      intent(inout), dimension(:) :: dataset

  integer(HID_T) :: groupHandle

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset,label,.true.)
#else
  call HDF5_write(groupHandle,dataset,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(SIunit)) &
    call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeScalarDataset_real

!--------------------------------------------------------------------------------------------------
!> @brief stores a vector dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeVectorDataset_real(group,dataset,label,description,SIunit)

  character(len=*), intent(in)                    :: label,group,description
  character(len=*), intent(in),    optional       :: SIunit
  real(pReal),      intent(inout), dimension(:,:) :: dataset

  integer(HID_T) :: groupHandle

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset,label,.true.)
#else
  call HDF5_write(groupHandle,dataset,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(SIunit)) &
    call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeVectorDataset_real


!--------------------------------------------------------------------------------------------------
!> @brief stores a tensor dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeTensorDataset_real(group,dataset,label,description,SIunit,transposed)

  character(len=*), intent(in)                   :: label,group,description
  character(len=*), intent(in), optional         :: SIunit
  logical,          intent(in), optional         :: transposed
  real(pReal),      intent(in), dimension(:,:,:) :: dataset

  integer :: i
  logical :: transposed_
  integer(HID_T) :: groupHandle
  real(pReal), dimension(:,:,:), allocatable :: dataset_transposed


  if(present(transposed)) then
    transposed_ = transposed
  else
    transposed_ = .true.
  endif

  if(transposed_) then
    if(size(dataset,1) /= size(dataset,2)) call IO_error(0,ext_msg='transpose non-symmetric tensor')
    allocate(dataset_transposed,mold=dataset)
    do i=1,size(dataset_transposed,3)
      dataset_transposed(:,:,i) = transpose(dataset(:,:,i))
    enddo
  else
    allocate(dataset_transposed,source=dataset)
  endif

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset_transposed,label,.true.)
#else
  call HDF5_write(groupHandle,dataset_transposed,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(SIunit)) &
    call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeTensorDataset_real


!--------------------------------------------------------------------------------------------------
!> @brief stores a vector dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeVectorDataset_int(group,dataset,label,description,SIunit)

  character(len=*), intent(in)                :: label,group,description
  character(len=*), intent(in), optional      :: SIunit
  integer,      intent(inout), dimension(:,:) :: dataset

  integer(HID_T) :: groupHandle

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset,label,.true.)
#else
  call HDF5_write(groupHandle,dataset,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(SIunit)) &
    call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeVectorDataset_int


!--------------------------------------------------------------------------------------------------
!> @brief stores a tensor dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeTensorDataset_int(group,dataset,label,description,SIunit)

  character(len=*), intent(in)                  :: label,group,description
  character(len=*), intent(in), optional        :: SIunit
  integer,      intent(inout), dimension(:,:,:) :: dataset

  integer(HID_T) :: groupHandle

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset,label,.true.)
#else
  call HDF5_write(groupHandle,dataset,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(SIunit)) &
    call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeTensorDataset_int


!--------------------------------------------------------------------------------------------------
!> @brief stores a scalar dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeScalarDataset_rotation(group,dataset,label,description,lattice_structure)

  character(len=*), intent(in)                  :: label,group,description
  character(len=*), intent(in), optional        :: lattice_structure
  type(rotation),   intent(inout), dimension(:) :: dataset

  integer(HID_T) :: groupHandle

  groupHandle = results_openGroup(group)

#ifdef PETSc
  call HDF5_write(groupHandle,dataset,label,.true.)
#else
  call HDF5_write(groupHandle,dataset,label,.false.)
#endif

  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Description',description,label)
  if (HDF5_objectExists(groupHandle,label) .and. present(lattice_structure)) &
    call HDF5_addAttribute(groupHandle,'Lattice',lattice_structure,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Creator','DAMASK '//DAMASKVERSION,label)
  if (HDF5_objectExists(groupHandle,label)) &
    call HDF5_addAttribute(groupHandle,'Created',now(),label)
  call HDF5_closeGroup(groupHandle)

end subroutine results_writeScalarDataset_rotation


!--------------------------------------------------------------------------------------------------
!> @brief adds the unique mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine results_mapping_constituent(phaseAt,memberAtLocal,label)

  integer,          dimension(:,:),   intent(in) :: phaseAt                                         !< phase section at (constituent,element)
  integer,                   dimension(:,:,:), intent(in) :: memberAtLocal                          !< phase member at (constituent,IP,element)
  character(len=pStringLen), dimension(:),     intent(in) :: label                                  !< label of each phase section

  integer, dimension(size(memberAtLocal,1),size(memberAtLocal,2),size(memberAtLocal,3)) :: &
    phaseAtMaterialpoint, &
    memberAtGlobal
  integer, dimension(size(label),0:worldsize-1) :: memberOffset                                     !< offset in member counting per process
  integer, dimension(0:worldsize-1)             :: writeSize                                        !< amount of data written per process
  integer(HSIZE_T), dimension(2) :: &
    myShape, &                                                                                      !< shape of the dataset (this process)
    myOffset, &
    totalShape                                                                                      !< shape of the dataset (all processes)

  integer(HID_T) :: &
    loc_id, &                                                                                       !< identifier of group in file
    dtype_id, &                                                                                     !< identifier of compound data type
    name_id, &                                                                                      !< identifier of name (string) in compound data type
    position_id, &                                                                                  !< identifier of position/index (integer) in compound data type
    dset_id, &
    memspace_id, &
    filespace_id, &
    plist_id, &
    dt_id


  integer(SIZE_T) :: type_size_string, type_size_int
  integer         :: hdferr, ierr, i

!---------------------------------------------------------------------------------------------------
! compound type: name of phase section + position/index within results array
  call h5tcopy_f(H5T_NATIVE_CHARACTER, dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tset_size_f(dt_id, int(len(label(1)),SIZE_T), hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tget_size_f(dt_id, type_size_string, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tget_size_f(H5T_NATIVE_INTEGER, type_size_int, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tcreate_f(H5T_COMPOUND_F, type_size_string + type_size_int, dtype_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(dtype_id, "Name", 0_SIZE_T, dt_id,hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(dtype_id, "Position", type_size_string, H5T_NATIVE_INTEGER, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! create memory types for each component of the compound type
  call h5tcreate_f(H5T_COMPOUND_F, type_size_string, name_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(name_id, "Name", 0_SIZE_T, dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tcreate_f(H5T_COMPOUND_F, type_size_int, position_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_NATIVE_INTEGER, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tclose_f(dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! prepare MPI communication (transparent for non-MPI runs)
  call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  memberOffset = 0
  do i=1, size(label)
    memberOffset(i,worldrank) = count(phaseAt == i)*size(memberAtLocal,2)                                ! number of points/instance of this process
  enddo
  writeSize = 0
  writeSize(worldrank) = size(memberAtLocal(1,:,:))                                                      ! total number of points by this process

!--------------------------------------------------------------------------------------------------
! MPI settings and communication
#ifdef PETSc
  call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call MPI_allreduce(MPI_IN_PLACE,writeSize,worldsize,MPI_INT,MPI_SUM,PETSC_COMM_WORLD,ierr)        ! get output at each process
  if(ierr /= 0) error stop 'MPI error'

  call MPI_allreduce(MPI_IN_PLACE,memberOffset,size(memberOffset),MPI_INT,MPI_SUM,PETSC_COMM_WORLD,ierr)! get offset at each process
  if(ierr /= 0) error stop 'MPI error'
#endif

  myShape    = int([size(phaseAt,1),writeSize(worldrank)],  HSIZE_T)
  myOffset   = int([0,sum(writeSize(0:worldrank-1))],       HSIZE_T)
  totalShape = int([size(phaseAt,1),sum(writeSize)],        HSIZE_T)

!--------------------------------------------------------------------------------------------------
! create dataspace in memory (local shape = hyperslab) and in file (global shape)
  call h5screate_simple_f(2,myShape,memspace_id,hdferr,myShape)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5screate_simple_f(2,totalShape,filespace_id,hdferr,totalShape)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5sselect_hyperslab_f(filespace_id, H5S_SELECT_SET_F, myOffset, myShape, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!---------------------------------------------------------------------------------------------------
! expand phaseAt to consider IPs (is not stored per IP)
  do i = 1, size(phaseAtMaterialpoint,2)
    phaseAtMaterialpoint(:,i,:) = phaseAt
  enddo

!---------------------------------------------------------------------------------------------------
! renumber member from my process to all processes
  do i = 1, size(label)
    where(phaseAtMaterialpoint == i) memberAtGlobal = memberAtLocal + sum(memberOffset(i,0:worldrank-1)) -1     ! convert to 0-based
  enddo

!--------------------------------------------------------------------------------------------------
! write the components of the compound type individually
  call h5pset_preserve_f(plist_id, .TRUE., hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  loc_id = results_openGroup('/mapping')
  call h5dcreate_f(loc_id, 'phase', dtype_id, filespace_id, dset_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5dwrite_f(dset_id, name_id, reshape(label(pack(phaseAtMaterialpoint,.true.)),myShape), &
                  myShape, hdferr, file_space_id = filespace_id, mem_space_id = memspace_id, xfer_prp = plist_id)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5dwrite_f(dset_id, position_id, reshape(pack(memberAtGlobal,.true.),myShape), &
                  myShape, hdferr, file_space_id = filespace_id, mem_space_id = memspace_id, xfer_prp = plist_id)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! close all
  call HDF5_closeGroup(loc_id)
  call h5pclose_f(plist_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5sclose_f(filespace_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5sclose_f(memspace_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5dclose_f(dset_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(dtype_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(name_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(position_id, hdferr)

end subroutine results_mapping_constituent


!--------------------------------------------------------------------------------------------------
!> @brief adds the unique mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine results_mapping_homogenization(homogenizationAt,memberAtLocal,label)

  integer,          dimension(:),   intent(in) :: homogenizationAt                                  !< homogenization section at (element)
  integer,                   dimension(:,:), intent(in) :: memberAtLocal                            !< homogenization member at (IP,element)
  character(len=pStringLen), dimension(:),   intent(in) :: label                                    !< label of each homogenization section

  integer, dimension(size(memberAtLocal,1),size(memberAtLocal,2)) :: &
    homogenizationAtMaterialpoint, &
    memberAtGlobal
  integer, dimension(size(label),0:worldsize-1) :: memberOffset                                     !< offset in member counting per process
  integer, dimension(0:worldsize-1)             :: writeSize                                        !< amount of data written per process
  integer(HSIZE_T), dimension(1) :: &
    myShape, &                                                                                      !< shape of the dataset (this process)
    myOffset, &
    totalShape                                                                                      !< shape of the dataset (all processes)

  integer(HID_T) :: &
    loc_id, &                                                                                       !< identifier of group in file
    dtype_id, &                                                                                     !< identifier of compound data type
    name_id, &                                                                                      !< identifier of name (string) in compound data type
    position_id, &                                                                                  !< identifier of position/index (integer) in compound data type
    dset_id, &
    memspace_id, &
    filespace_id, &
    plist_id, &
    dt_id


  integer(SIZE_T) :: type_size_string, type_size_int
  integer         :: hdferr, ierr, i

!---------------------------------------------------------------------------------------------------
! compound type: name of phase section + position/index within results array
  call h5tcopy_f(H5T_NATIVE_CHARACTER, dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tset_size_f(dt_id, int(len(label(1)),SIZE_T), hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tget_size_f(dt_id, type_size_string, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tget_size_f(H5T_NATIVE_INTEGER, type_size_int, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tcreate_f(H5T_COMPOUND_F, type_size_string + type_size_int, dtype_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(dtype_id, "Name", 0_SIZE_T, dt_id,hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(dtype_id, "Position", type_size_string, H5T_NATIVE_INTEGER, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! create memory types for each component of the compound type
  call h5tcreate_f(H5T_COMPOUND_F, type_size_string, name_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(name_id, "Name", 0_SIZE_T, dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tcreate_f(H5T_COMPOUND_F, type_size_int, position_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_NATIVE_INTEGER, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5tclose_f(dt_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! prepare MPI communication (transparent for non-MPI runs)
  call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  memberOffset = 0
  do i=1, size(label)
    memberOffset(i,worldrank) = count(homogenizationAt == i)*size(memberAtLocal,1)                  ! number of points/instance of this process
  enddo
  writeSize = 0
  writeSize(worldrank) = size(memberAtLocal)                                                        ! total number of points by this process

!--------------------------------------------------------------------------------------------------
! MPI settings and communication
#ifdef PETSc
  call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call MPI_allreduce(MPI_IN_PLACE,writeSize,worldsize,MPI_INT,MPI_SUM,PETSC_COMM_WORLD,ierr)        ! get output at each process
  if(ierr /= 0) error stop 'MPI error'

  call MPI_allreduce(MPI_IN_PLACE,memberOffset,size(memberOffset),MPI_INT,MPI_SUM,PETSC_COMM_WORLD,ierr)! get offset at each process
  if(ierr /= 0) error stop 'MPI error'
#endif

  myShape    = int([writeSize(worldrank)],          HSIZE_T)
  myOffset   = int([sum(writeSize(0:worldrank-1))], HSIZE_T)
  totalShape = int([sum(writeSize)],                HSIZE_T)

!--------------------------------------------------------------------------------------------------
! create dataspace in memory (local shape = hyperslab) and in file (global shape)
  call h5screate_simple_f(1,myShape,memspace_id,hdferr,myShape)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5screate_simple_f(1,totalShape,filespace_id,hdferr,totalShape)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5sselect_hyperslab_f(filespace_id, H5S_SELECT_SET_F, myOffset, myShape, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

!---------------------------------------------------------------------------------------------------
! expand phaseAt to consider IPs (is not stored per IP)
  do i = 1, size(homogenizationAtMaterialpoint,1)
    homogenizationAtMaterialpoint(i,:) = homogenizationAt
  enddo

!---------------------------------------------------------------------------------------------------
! renumber member from my process to all processes
  do i = 1, size(label)
    where(homogenizationAtMaterialpoint == i) memberAtGlobal = memberAtLocal + sum(memberOffset(i,0:worldrank-1)) - 1  ! convert to 0-based
  enddo

!--------------------------------------------------------------------------------------------------
! write the components of the compound type individually
  call h5pset_preserve_f(plist_id, .TRUE., hdferr)

  loc_id = results_openGroup('/mapping')
  call h5dcreate_f(loc_id, 'homogenization', dtype_id, filespace_id, dset_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

  call h5dwrite_f(dset_id, name_id, reshape(label(pack(homogenizationAtMaterialpoint,.true.)),myShape), &
                  myShape, hdferr, file_space_id = filespace_id, mem_space_id = memspace_id, xfer_prp = plist_id)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5dwrite_f(dset_id, position_id, reshape(pack(memberAtGlobal,.true.),myShape), &
                  myShape, hdferr, file_space_id = filespace_id, mem_space_id = memspace_id, xfer_prp = plist_id)
  if(hdferr < 0) error stop 'HDF5 error'

!--------------------------------------------------------------------------------------------------
! close all
  call HDF5_closeGroup(loc_id)
  call h5pclose_f(plist_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5sclose_f(filespace_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5sclose_f(memspace_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5dclose_f(dset_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(dtype_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(name_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'
  call h5tclose_f(position_id, hdferr)
  if(hdferr < 0) error stop 'HDF5 error'

end subroutine results_mapping_homogenization


!--------------------------------------------------------------------------------------------------
!> @brief current date and time (including time zone information)
!--------------------------------------------------------------------------------------------------
character(len=24) function now()

  character(len=5)      :: zone
  integer, dimension(8) :: values

  call date_and_time(values=values,zone=zone)
  write(now,'(i4.4,5(a,i2.2),a)') &
    values(1),'-',values(2),'-',values(3),' ',values(5),':',values(6),':',values(7),zone

end function now

end module results
