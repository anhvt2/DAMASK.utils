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
!> @author Vitesh Shah, Max-Planck-Institut für Eisenforschung GmbH
!> @author Yi-Chin Yang, Max-Planck-Institut für Eisenforschung GmbH
!> @author Jennifer Nastola, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!--------------------------------------------------------------------------------------------------
module results
  use prec
  use IO
  use HDF5
  use HDF5_utilities
#ifdef PETSc
  use PETSC
#endif

  implicit none
  private
  integer(HID_T), public, protected :: tempCoordinates, tempResults
  integer(HID_T), private :: resultsFile, currentIncID, plist_id


  public :: &
    results_init, &
    results_openJobFile, &
    results_closeJobFile, &
    results_addIncrement, &
    results_addGroup, &
    results_openGroup, &
    results_writeVectorDataset, &
    results_setLink, &
    results_removeLink
contains

subroutine results_init
  use DAMASK_interface, only: &
    getSolverJobName

  implicit none

  write(6,'(/,a)') ' <<<+-  results init  -+>>>'

  write(6,'(/,a)') ' Diehl et al., Integrating Materials and Manufacturing Innovation 6(1):83–91, 2017'
  write(6,'(a)')   ' https://doi.org/10.1007/s40192-018-0118-7'

  call HDF5_closeFile(HDF5_openFile(trim(getSolverJobName())//'.hdf5','w',.true.))

end subroutine results_init


!--------------------------------------------------------------------------------------------------
!> @brief opens the results file to append data
!--------------------------------------------------------------------------------------------------
subroutine results_openJobFile()
 use DAMASK_interface, only: &
   getSolverJobName

 implicit none
 character(len=pStringLen) :: commandLine

 resultsFile = HDF5_openFile(trim(getSolverJobName())//'.hdf5','a',.true.)
 call HDF5_addAttribute(resultsFile,'DADF5',0.1_pReal)
 call HDF5_addAttribute(resultsFile,'DAMASK',DAMASKVERSION)
 call get_command(commandLine)
 call HDF5_addAttribute(resultsFile,'call',trim(commandLine))
 
end subroutine results_openJobFile


!--------------------------------------------------------------------------------------------------
!> @brief closes the results file
!--------------------------------------------------------------------------------------------------
subroutine results_closeJobFile()
 implicit none

 call HDF5_closeFile(resultsFile)

end subroutine results_closeJobFile


!--------------------------------------------------------------------------------------------------
!> @brief closes the results file
!--------------------------------------------------------------------------------------------------
subroutine results_addIncrement(inc,time)
 
 implicit none
 integer(pInt), intent(in) :: inc
 real(pReal),   intent(in) :: time
 character(len=pStringLen) :: incChar

 write(incChar,*) inc
 call HDF5_closeGroup(results_addGroup(trim('inc'//trim(adjustl(incChar)))))
 call results_setLink(trim('inc'//trim(adjustl(incChar))),'current')
 call HDF5_addAttribute(resultsFile,'time/s',time,trim('inc'//trim(adjustl(incChar))))

end subroutine results_addIncrement

!--------------------------------------------------------------------------------------------------
!> @brief open a group from the results file
!--------------------------------------------------------------------------------------------------
integer(HID_T) function results_openGroup(groupName)

 implicit none
 character(len=*), intent(in) :: groupName
 
 results_openGroup = HDF5_openGroup(resultsFile,groupName)

end function results_openGroup


!--------------------------------------------------------------------------------------------------
!> @brief adds a new group to the results file
!--------------------------------------------------------------------------------------------------
integer(HID_T) function results_addGroup(groupName)

 implicit none
 character(len=*), intent(in) :: groupName
 
 results_addGroup = HDF5_addGroup(resultsFile,groupName)

end function results_addGroup


!--------------------------------------------------------------------------------------------------
!> @brief set link to object in results file
!--------------------------------------------------------------------------------------------------
subroutine results_setLink(path,link)
 use hdf5_utilities, only: &
  HDF5_setLink

 implicit none
 character(len=*), intent(in) :: path, link

 call HDF5_setLink(resultsFile,path,link)

end subroutine results_setLink


!--------------------------------------------------------------------------------------------------
!> @brief remove link to an object
!--------------------------------------------------------------------------------------------------
subroutine results_removeLink(link)
 use hdf5

 implicit none
 character(len=*), intent(in) :: link
 integer                      :: hdferr

 call h5ldelete_f(resultsFile,link, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg = 'results_removeLink: h5ldelete_soft_f ('//trim(link)//')')

end subroutine results_removeLink


!--------------------------------------------------------------------------------------------------
!> @brief stores a vector dataset in a group
!--------------------------------------------------------------------------------------------------
subroutine results_writeVectorDataset(group,dataset,label,SIunit)

 implicit none
 character(len=*), intent(in)                 :: SIunit,label,group
 real(pReal),      intent(inout), dimension(:,:) :: dataset
 integer(HID_T)        :: groupHandle
 
 groupHandle = results_openGroup(group)
 call HDF5_write(groupHandle,dataset,label)
 if (HDF5_objectExists(groupHandle,label)) &
   call HDF5_addAttribute(groupHandle,'Unit',SIunit,label)
 call HDF5_closeGroup(groupHandle)

end subroutine results_writeVectorDataset


!--------------------------------------------------------------------------------------------------
!> @brief adds the unique mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_mappingPhase(mapping,mapping2,Nconstituents,material_phase,phase_name,dataspace_size,mpiOffset,mpiOffset_phase)
 use hdf5

 implicit none
 integer(pInt),    intent(in)                   :: Nconstituents, dataspace_size, mpiOffset
 integer(pInt),    intent(in), dimension(:)     :: mapping, mapping2
 character(len=*), intent(in), dimension(:)     :: phase_name
 integer(pInt),    intent(in), dimension(:)     :: mpiOffset_phase
 integer(pInt),    intent(in), dimension(:,:,:) :: material_phase

 character(len=len(phase_name(1))), dimension(:), allocatable :: namesNA
 character(len=len(phase_name(1)))                            :: a
 character(len=*),                  parameter                 :: n = "NULL"

 integer(pInt)   :: hdferr, NmatPoints, i, j, k
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, name_id, position_id, plist_id, memspace
 integer(HID_T)  :: dt5_id      ! Memory datatype identifier
 integer(SIZE_T) :: typesize, type_sizec, type_sizei, type_size

 integer(HSIZE_T),  dimension(2)                :: counter
 integer(HSSIZE_T), dimension(2)                :: fileOffset
 integer(pInt),     dimension(:,:), allocatable :: arrOffset

 a = n
 allocate(namesNA(0:size(phase_name)),source=[a,phase_name])
 NmatPoints = size(mapping,1)/Nconstituents
 mapping_ID = results_openGroup("current/mapGeometry")

 allocate(arrOffset(Nconstituents,NmatPoints))
 do i=1_pInt, NmatPoints
   do k=1_pInt, Nconstituents
     do j=1_pInt, size(phase_name)
       if(material_phase(k,1,i) == j) &
         arrOffset(k,i) = mpiOffset_phase(j)
     enddo
   enddo
 enddo

!--------------------------------------------------------------------------------------------------
! create dataspace
 call h5screate_simple_f(2, int([Nconstituents,dataspace_size],HSIZE_T), space_id, hdferr, &
                            int([Nconstituents,dataspace_size],HSIZE_T))
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping')

!--------------------------------------------------------------------------------------------------
! compound type
     ! First calculate total size by calculating sizes of each member
     !
     CALL h5tcopy_f(H5T_NATIVE_CHARACTER, dt5_id, hdferr)
     typesize = len(phase_name(1))
     CALL h5tset_size_f(dt5_id, typesize, hdferr)
     CALL h5tget_size_f(dt5_id, type_sizec, hdferr)
     CALL h5tget_size_f(H5T_STD_I32LE,type_sizei, hdferr)
     type_size = type_sizec + type_sizei
 call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping: h5tcreate_f dtype_id')

 call h5tinsert_f(dtype_id, "Name",          0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tinsert_f 0')
 call h5tinsert_f(dtype_id, "Position",  type_sizec, H5T_STD_I32LE, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tinsert_f 2')

!--------------------------------------------------------------------------------------------------
! create Dataset
 call h5dcreate_f(mapping_id, 'constitutive', dtype_id, space_id, dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase')

!--------------------------------------------------------------------------------------------------
! Create memory types (one compound datatype for each member)
 call h5tcreate_f(H5T_COMPOUND_F, int(type_sizec,SIZE_T), name_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tcreate_f instance_id')
 call h5tinsert_f(name_id, "Name",        0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tinsert_f instance_id')

 call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tcreate_f position_id')
 call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_STD_I32LE, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tinsert_f position_id')

!--------------------------------------------------------------------------------------------------
! Define and select hyperslabs
 counter(1)    = Nconstituents                  ! how big i am
 counter(2)    = NmatPoints
 fileOffset(1) = 0                              ! where i start to write my data
 fileOffset(2) = mpiOffset

 call h5screate_simple_f(2, counter, memspace, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5screate_simple_f')
 call h5dget_space_f(dset_id, space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5dget_space_f')
 call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
 ! Create property list for collective dataset write
#ifdef PETSc
 call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5pcreate_f')
 call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
! write data by fields in the datatype. Fields order is not important.
 call h5dwrite_f(dset_id, name_id, reshape(namesNA(mapping),[Nconstituents,NmatPoints]), &
                          int([Nconstituents,  dataspace_size],HSIZE_T), hdferr, &
                          file_space_id = space_id, mem_space_id = memspace, xfer_prp = plist_id)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5dwrite_f position_id')

 call h5dwrite_f(dset_id, position_id, reshape(mapping2-1_pInt,[Nconstituents,NmatPoints])+arrOffset, &
                          int([Nconstituents,  dataspace_size],HSIZE_T), hdferr, &
                          file_space_id = space_id, mem_space_id = memspace, xfer_prp = plist_id)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
! close types, dataspaces
 call h5tclose_f(dtype_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tclose_f dtype_id')
 call h5tclose_f(position_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tclose_f position_id')
 call h5tclose_f(name_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tclose_f name_id  ')
 call h5tclose_f(dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5tclose_f dt5_id')
 call h5dclose_f(dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5dclose_f')
 call h5sclose_f(space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5sclose_f space_id')
 call h5sclose_f(memspace, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5sclose_f memspace')
 call h5pclose_f(plist_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingPhase: h5pclose_f')
 call HDF5_closeGroup(mapping_ID)

end subroutine HDF5_mappingPhase

!--------------------------------------------------------------------------------------------------
!> @brief adds the backward mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_backwardMappingPhase(material_phase,phasememberat,phase_name,dataspace_size,mpiOffset,mpiOffset_phase)
 use hdf5

 implicit none
 integer(pInt),    intent(in), dimension(:,:,:) :: material_phase, phasememberat
 character(len=*), intent(in), dimension(:)     :: phase_name
 integer(pInt),    intent(in), dimension(:)     :: dataspace_size, mpiOffset_phase
 integer(pInt),    intent(in)                   :: mpiOffset

 integer(pInt)   :: hdferr, NmatPoints, Nconstituents, i, j
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, position_id, plist_id, memspace
 integer(SIZE_T) :: type_size

 integer(pInt), dimension(:,:), allocatable :: arr

 integer(HSIZE_T),  dimension(1) :: counter
 integer(HSSIZE_T), dimension(1) :: fileOffset

 character(len=64) :: phaseID

 Nconstituents = size(phasememberat,1)
 NmatPoints = count(material_phase /=0_pInt)/Nconstituents

 allocate(arr(2,NmatPoints*Nconstituents))

 do i=1_pInt, NmatPoints
   do j=Nconstituents-1_pInt, 0_pInt, -1_pInt
     arr(1,Nconstituents*i-j) = i-1_pInt
   enddo
 enddo
 arr(2,:) = pack(material_phase,material_phase/=0_pInt)

 do i=1_pInt, size(phase_name)
   write(phaseID, '(i0)') i
   mapping_ID = results_openGroup('/current/constitutive/'//trim(phaseID)//'_'//phase_name(i))
   NmatPoints = count(material_phase == i)

!--------------------------------------------------------------------------------------------------
  ! create dataspace
   call h5screate_simple_f(1, int([dataspace_size(i)],HSIZE_T), space_id, hdferr, &
                              int([dataspace_size(i)],HSIZE_T))
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping')

!--------------------------------------------------------------------------------------------------
  ! compound type
   call h5tget_size_f(H5T_STD_I32LE, type_size, hdferr)
   call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping: h5tcreate_f dtype_id')

   call h5tinsert_f(dtype_id, "Position",          0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5tinsert_f 0')

!--------------------------------------------------------------------------------------------------
  ! create Dataset
   call h5dcreate_f(mapping_id, 'mapGeometry', dtype_id, space_id, dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase')

!--------------------------------------------------------------------------------------------------
  ! Create memory types (one compound datatype for each member)
   call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5tcreate_f position_id')
   call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5tinsert_f position_id')

!--------------------------------------------------------------------------------------------------
  ! Define and select hyperslabs
   counter = NmatPoints                       ! how big i am
   fileOffset = mpiOffset_phase(i)            ! where i start to write my data

   call h5screate_simple_f(1, counter, memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5screate_simple_f')
   call h5dget_space_f(dset_id, space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5dget_space_f')
   call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
 ! Create property list for collective dataset write
#ifdef PETSc
   call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5pcreate_f')
   call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
  ! write data by fields in the datatype. Fields order is not important.
   call h5dwrite_f(dset_id, position_id, pack(arr(1,:),arr(2,:)==i)+mpiOffset, int([dataspace_size(i)],HSIZE_T),&
                              hdferr, file_space_id = space_id, mem_space_id = memspace, xfer_prp = plist_id)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
  !close types, dataspaces
   call h5tclose_f(dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5tclose_f dtype_id')
   call h5tclose_f(position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5tclose_f position_id')
   call h5dclose_f(dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5dclose_f')
   call h5sclose_f(space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5sclose_f space_id')
   call h5sclose_f(memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5sclose_f memspace')
   call h5pclose_f(plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingPhase: h5pclose_f')
   call HDF5_closeGroup(mapping_ID)

 enddo

end subroutine HDF5_backwardMappingPhase

!--------------------------------------------------------------------------------------------------
!> @brief adds the unique mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_mappingHomog(material_homog,homogmemberat,homogenization_name,dataspace_size,mpiOffset,mpiOffset_homog)
 use hdf5

 implicit none
 integer(pInt),    intent(in), dimension(:,:) :: material_homog, homogmemberat
 character(len=*), intent(in), dimension(:)   :: homogenization_name
 integer(pInt),    intent(in), dimension(:)   :: mpiOffset_homog
 integer(pInt),    intent(in)                 :: dataspace_size, mpiOffset

 integer(pInt)   :: hdferr, NmatPoints, i, j
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, name_id, position_id, plist_id, memspace

 integer(HID_T)  :: dt5_id      ! Memory datatype identifier
 integer(SIZE_T) :: typesize, type_sizec, type_sizei, type_size

 integer(HSIZE_T),  dimension(1)              :: counter
 integer(HSSIZE_T), dimension(1)              :: fileOffset
 integer(pInt),     dimension(:), allocatable :: arrOffset

 NmatPoints = count(material_homog /=0_pInt)
 mapping_ID = results_openGroup("current/mapGeometry")

 allocate(arrOffset(NmatPoints))
 do i=1_pInt, NmatPoints
   do j=1_pInt, size(homogenization_name)
     if(material_homog(1,i) == j) &
       arrOffset(i) = mpiOffset_homog(j)
   enddo
 enddo

!--------------------------------------------------------------------------------------------------
! create dataspace
 call h5screate_simple_f(1, int([dataspace_size],HSIZE_T), space_id, hdferr, &
                            int([dataspace_size],HSIZE_T))
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping')

!--------------------------------------------------------------------------------------------------
! compound type
     ! First calculate total size by calculating sizes of each member
     !
     CALL h5tcopy_f(H5T_NATIVE_CHARACTER, dt5_id, hdferr)
     typesize = len(homogenization_name(1))
     CALL h5tset_size_f(dt5_id, typesize, hdferr)
     CALL h5tget_size_f(dt5_id, type_sizec, hdferr)
     CALL h5tget_size_f(H5T_STD_I32LE,type_sizei, hdferr)
     type_size = type_sizec + type_sizei
 call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping: h5tcreate_f dtype_id')

 call h5tinsert_f(dtype_id, "Name",          0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tinsert_f 0')
 call h5tinsert_f(dtype_id, "Position",  type_sizec, H5T_STD_I32LE, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tinsert_f 2')

!--------------------------------------------------------------------------------------------------
! create Dataset
 call h5dcreate_f(mapping_id, 'homogenization', dtype_id, space_id, dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog')

!--------------------------------------------------------------------------------------------------
! Create memory types (one compound datatype for each member)
 call h5tcreate_f(H5T_COMPOUND_F, int(type_sizec,SIZE_T), name_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tcreate_f instance_id')
 call h5tinsert_f(name_id, "Name",        0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tinsert_f instance_id')

 call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tcreate_f position_id')
 call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_STD_I32LE, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tinsert_f position_id')

!--------------------------------------------------------------------------------------------------
! Define and select hyperslabs
 counter = NmatPoints                    ! how big i am
 fileOffset = mpiOffset                  ! where i start to write my data

 call h5screate_simple_f(1, counter, memspace, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5screate_simple_f')
 call h5dget_space_f(dset_id, space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5dget_space_f')
 call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
! Create property list for collective dataset write
#ifdef PETSc
 call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5pcreate_f')
 call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
! write data by fields in the datatype. Fields order is not important.
 call h5dwrite_f(dset_id, name_id, homogenization_name(pack(material_homog,material_homog/=0_pInt)), &
                          int([dataspace_size],HSIZE_T), hdferr, file_space_id = space_id, &
                          mem_space_id = memspace, xfer_prp = plist_id)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5dwrite_f position_id')

 call h5dwrite_f(dset_id, position_id, pack(homogmemberat-1_pInt,homogmemberat/=0_pInt) + arrOffset, &
                          int([dataspace_size],HSIZE_T), hdferr, file_space_id = space_id, &
                          mem_space_id = memspace, xfer_prp = plist_id)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
!close types, dataspaces
call h5tclose_f(dtype_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tclose_f dtype_id')
call h5tclose_f(position_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tclose_f position_id')
call h5tclose_f(name_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tclose_f name_id  ')
call h5tclose_f(dt5_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5tclose_f dt5_id')
call h5dclose_f(dset_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5dclose_f')
call h5sclose_f(space_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5sclose_f space_id')
call h5sclose_f(memspace, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5sclose_f memspace')
call h5pclose_f(plist_id, hdferr)
if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingHomog: h5pclose_f')
call HDF5_closeGroup(mapping_ID)

end subroutine HDF5_mappingHomog

!--------------------------------------------------------------------------------------------------
!> @brief adds the backward mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_backwardMappingHomog(material_homog,homogmemberat,homogenization_name,dataspace_size,mpiOffset,mpiOffset_homog)
 use hdf5

 implicit none
 integer(pInt),    intent(in), dimension(:,:) :: material_homog, homogmemberat
 character(len=*), intent(in), dimension(:)   :: homogenization_name
 integer(pInt),    intent(in), dimension(:)   :: dataspace_size, mpiOffset_homog
 integer(pInt),    intent(in)                 :: mpiOffset

 integer(pInt)   :: hdferr, NmatPoints, i
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, position_id, plist_id, memspace
 integer(SIZE_T) :: type_size

 integer(pInt),     dimension(:,:), allocatable :: arr

 integer(HSIZE_T),  dimension(1) :: counter
 integer(HSSIZE_T), dimension(1) :: fileOffset

 character(len=64) :: homogID

 NmatPoints = count(material_homog /=0_pInt)
 allocate(arr(2,NmatPoints))

 arr(1,:) = (/(i, i=0_pint,NmatPoints-1_pInt)/)
 arr(2,:) = pack(material_homog,material_homog/=0_pInt)

 do i=1_pInt, size(homogenization_name)
   write(homogID, '(i0)') i
   mapping_ID = results_openGroup('/current/homogenization/'//trim(homogID)//'_'//homogenization_name(i))

!--------------------------------------------------------------------------------------------------
  ! create dataspace
   call h5screate_simple_f(1, int([dataspace_size(i)],HSIZE_T), space_id, hdferr, &
                              int([dataspace_size(i)],HSIZE_T))
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping')

!--------------------------------------------------------------------------------------------------
  ! compound type
   call h5tget_size_f(H5T_STD_I32LE, type_size, hdferr)
   call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping: h5tcreate_f dtype_id')

   call h5tinsert_f(dtype_id, "Position",          0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5tinsert_f 0')

!--------------------------------------------------------------------------------------------------
  ! create Dataset
   call h5dcreate_f(mapping_id, 'mapGeometry', dtype_id, space_id, dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog')

!--------------------------------------------------------------------------------------------------
  ! Create memory types (one compound datatype for each member)
   call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5tcreate_f position_id')
   call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5tinsert_f position_id')

!--------------------------------------------------------------------------------------------------
  ! Define and select hyperslabs
   counter = NmatPoints                           ! how big i am
   fileOffset = mpiOffset_homog(i)                ! where i start to write my data

   call h5screate_simple_f(1, counter, memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5screate_simple_f')
   call h5dget_space_f(dset_id, space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5dget_space_f')
   call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
 ! Create property list for collective dataset write
#ifdef PETSc
   call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5pcreate_f')
   call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
  ! write data by fields in the datatype. Fields order is not important.
   call h5dwrite_f(dset_id, position_id, pack(arr(1,:),arr(2,:)==i)+mpiOffset,int([dataspace_size(i)],HSIZE_T),&
                   hdferr, file_space_id = space_id, mem_space_id = memspace, xfer_prp = plist_id)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
  !close types, dataspaces
   call h5tclose_f(dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5tclose_f dtype_id')
   call h5tclose_f(position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5tclose_f position_id')
   call h5dclose_f(dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5dclose_f')
   call h5sclose_f(space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5sclose_f space_id')
   call h5sclose_f(memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5sclose_f memspace')
   call h5pclose_f(plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingHomog: h5pclose_f')
   call HDF5_closeGroup(mapping_ID)

 enddo

end subroutine HDF5_backwardMappingHomog

!--------------------------------------------------------------------------------------------------
!> @brief adds the unique mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_mappingCrystallite(crystalliteAt,crystmemberAt,crystallite_name,dataspace_size,mpiOffset,mpiOffset_cryst)
 use hdf5

 implicit none
 integer(pInt),    intent(in), dimension(:,:)   :: crystalliteAt
 integer(pInt),    intent(in), dimension(:,:,:) :: crystmemberAt
 character(len=*), intent(in), dimension(:)     :: crystallite_name
 integer(pInt),    intent(in), dimension(:)     :: mpiOffset_cryst
 integer(pInt),    intent(in)                   :: dataspace_size, mpiOffset

 integer         :: hdferr
 integer(pInt)   :: NmatPoints, Nconstituents, i, j
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, name_id, plist_id, memspace

 integer(HID_T), dimension(:), allocatable :: position_id

 integer(HID_T)  :: dt5_id      ! Memory datatype identifier
 integer(SIZE_T) :: typesize, type_sizec, type_sizei, type_size

 integer(HSIZE_T),  dimension(1)              :: counter
 integer(HSSIZE_T), dimension(1)              :: fileOffset
 integer(pInt),     dimension(:), allocatable :: arrOffset

 character(len=64) :: m

 Nconstituents = size(crystmemberAt,1)
 NmatPoints = count(crystalliteAt /=0_pInt)
 mapping_ID = results_openGroup("current/mapGeometry")

 allocate(position_id(Nconstituents))

 allocate(arrOffset(NmatPoints))
 do i=1_pInt, NmatPoints
   do j=1_pInt, size(crystallite_name)
     if(crystalliteAt(1,i) == j) &
       arrOffset(i) = Nconstituents*mpiOffset_cryst(j)
   enddo
 enddo

!--------------------------------------------------------------------------------------------------
! create dataspace
 call h5screate_simple_f(1, int([dataspace_size],HSIZE_T), space_id, hdferr, &
                            int([dataspace_size],HSIZE_T))
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping')

!--------------------------------------------------------------------------------------------------
! compound type
     ! First calculate total size by calculating sizes of each member
     !
     CALL h5tcopy_f(H5T_NATIVE_CHARACTER, dt5_id, hdferr)
     typesize = len(crystallite_name(1))
     CALL h5tset_size_f(dt5_id, typesize, hdferr)
     CALL h5tget_size_f(dt5_id, type_sizec, hdferr)
     CALL h5tget_size_f(H5T_STD_I32LE, type_sizei, hdferr)
     type_size = type_sizec + type_sizei*Nconstituents
 call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeMapping: h5tcreate_f dtype_id')

 call h5tinsert_f(dtype_id, "Name",          0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tinsert_f 0')
 do i=1_pInt, Nconstituents
   write(m, '(i0)') i
   call h5tinsert_f(dtype_id, "Position "//trim(m),  type_sizec+(i-1)*type_sizei, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tinsert_f 2 '//trim(m))
 enddo

!--------------------------------------------------------------------------------------------------
! create Dataset
 call h5dcreate_f(mapping_id, 'crystallite', dtype_id, space_id, dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite')

!--------------------------------------------------------------------------------------------------
! Create memory types (one compound datatype for each member)
 call h5tcreate_f(H5T_COMPOUND_F, int(type_sizec,SIZE_T), name_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tcreate_f instance_id')
 call h5tinsert_f(name_id, "Name",        0_SIZE_T, dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tinsert_f instance_id')

 do i=1_pInt, Nconstituents
   write(m, '(i0)') i
   call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id(i), hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tcreate_f position_id')
   call h5tinsert_f(position_id(i), "Position "//trim(m), 0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tinsert_f position_id')
 enddo

!--------------------------------------------------------------------------------------------------
! Define and select hyperslabs
 counter = NmatPoints                    ! how big i am
 fileOffset = mpiOffset                  ! where i start to write my data

 call h5screate_simple_f(1, counter, memspace, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5screate_simple_f')
 call h5dget_space_f(dset_id, space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5dget_space_f')
 call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
 ! Create property list for collective dataset write
#ifdef PETSc
 call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5pcreate_f')
 call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
! write data by fields in the datatype. Fields order is not important.
 call h5dwrite_f(dset_id, name_id, crystallite_name(pack(crystalliteAt,crystalliteAt/=0_pInt)), &
                          int([dataspace_size],HSIZE_T), hdferr, file_space_id = space_id, &
                          mem_space_id = memspace, xfer_prp = plist_id)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5dwrite_f position_id')

 do i=1_pInt, Nconstituents
   call h5dwrite_f(dset_id, position_id(i), pack(crystmemberAt(i,:,:)-1_pInt,crystmemberAt(i,:,:)/=0_pInt)+arrOffset,&
                            int([dataspace_size],HSIZE_T), hdferr, file_space_id = space_id, &
                            mem_space_id = memspace, xfer_prp = plist_id)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5dwrite_f instance_id')
 enddo

!--------------------------------------------------------------------------------------------------
!close types, dataspaces
 call h5tclose_f(dtype_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tclose_f dtype_id')
 do i=1_pInt, Nconstituents
   call h5tclose_f(position_id(i), hdferr)
 enddo
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tclose_f position_id')
 call h5tclose_f(name_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tclose_f name_id')
 call h5tclose_f(dt5_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5tclose_f dt5_id')
 call h5dclose_f(dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5dclose_f')
 call h5sclose_f(space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5sclose_f space_id')
 call h5sclose_f(memspace, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5sclose_f memspace')
 call h5pclose_f(plist_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCrystallite: h5pclose_f')
 call HDF5_closeGroup(mapping_ID)

end subroutine HDF5_mappingCrystallite


!--------------------------------------------------------------------------------------------------
!> @brief adds the backward mapping from spatial position and constituent ID to results
!--------------------------------------------------------------------------------------------------
subroutine HDF5_backwardMappingCrystallite(crystalliteAt,crystmemberAt,crystallite_name,dataspace_size,mpiOffset,mpiOffset_cryst)
 use hdf5

 implicit none
 integer(pInt),    intent(in), dimension(:,:)   :: crystalliteAt
 integer(pInt),    intent(in), dimension(:,:,:) :: crystmemberAt
 character(len=*), intent(in), dimension(:)     :: crystallite_name
 integer(pInt),    intent(in), dimension(:)     :: dataspace_size, mpiOffset_cryst
 integer(pInt),    intent(in)                   :: mpiOffset

 integer         :: hdferr
 integer(pInt)   :: NmatPoints, Nconstituents, i, j
 integer(HID_T)  :: mapping_id, dtype_id, dset_id, space_id, position_id, plist_id, memspace
 integer(SIZE_T) :: type_size

 integer(pInt), dimension(:,:), allocatable     :: h_arr, arr

 integer(HSIZE_T),  dimension(1) :: counter
 integer(HSSIZE_T), dimension(1) :: fileOffset

 character(len=64) :: crystallID

 Nconstituents = size(crystmemberAt,1)
 NmatPoints = count(crystalliteAt /=0_pInt)

 allocate(h_arr(2,NmatPoints))
 allocate(arr(2,Nconstituents*NmatPoints))

 h_arr(1,:) = (/(i, i=0_pInt,NmatPoints-1_pInt)/)
 h_arr(2,:) = pack(crystalliteAt,crystalliteAt/=0_pInt)

 do i=1_pInt, NmatPoints
   do j=Nconstituents-1_pInt, 0_pInt, -1_pInt
     arr(1,Nconstituents*i-j) = h_arr(1,i)
     arr(2,Nconstituents*i-j) = h_arr(2,i)
   enddo
 enddo

 do i=1_pInt, size(crystallite_name)
   if (crystallite_name(i) == 'none') cycle
   write(crystallID, '(i0)') i
   mapping_ID = results_openGroup('/current/crystallite/'//trim(crystallID)//'_'//crystallite_name(i))
   NmatPoints = count(crystalliteAt == i)

!--------------------------------------------------------------------------------------------------
  ! create dataspace
   call h5screate_simple_f(1, int([Nconstituents*dataspace_size(i)],HSIZE_T), space_id, hdferr, &
                              int([Nconstituents*dataspace_size(i)],HSIZE_T))
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping')

!--------------------------------------------------------------------------------------------------
  ! compound type
   call h5tget_size_f(H5T_STD_I32LE, type_size, hdferr)
   call h5tcreate_f(H5T_COMPOUND_F, type_size, dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='HDF5_writeBackwardMapping: h5tcreate_f dtype_id')

   call h5tinsert_f(dtype_id, "Position",          0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5tinsert_f 0')

!--------------------------------------------------------------------------------------------------
  ! create Dataset
   call h5dcreate_f(mapping_id, 'mapGeometry', dtype_id, space_id, dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite')

!--------------------------------------------------------------------------------------------------
  ! Create memory types
   call h5tcreate_f(H5T_COMPOUND_F, int(pInt,SIZE_T), position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5tcreate_f position_id')
   call h5tinsert_f(position_id, "Position", 0_SIZE_T, H5T_STD_I32LE, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5tinsert_f position_id')

!--------------------------------------------------------------------------------------------------
  ! Define and select hyperslabs
   counter = Nconstituents*NmatPoints                       ! how big i am
   fileOffset = Nconstituents*mpiOffset_cryst(i)            ! where i start to write my data

   call h5screate_simple_f(1, counter, memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5screate_simple_f')
   call h5dget_space_f(dset_id, space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5dget_space_f')
   call h5sselect_hyperslab_f(space_id, H5S_SELECT_SET_F, fileOffset, counter, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5sselect_hyperslab_f')

!--------------------------------------------------------------------------------------------------
 ! Create property list for collective dataset write
#ifdef PETSc
   call h5pcreate_f(H5P_DATASET_XFER_F, plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5pcreate_f')
   call h5pset_dxpl_mpio_f(plist_id, H5FD_MPIO_COLLECTIVE_F, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5pset_dxpl_mpio_f')
#endif

!--------------------------------------------------------------------------------------------------
  ! write data by fields in the datatype. Fields order is not important.
   call h5dwrite_f(dset_id, position_id, pack(arr(1,:),arr(2,:)==i) + mpiOffset,&
                              int([Nconstituents*dataspace_size(i)],HSIZE_T), hdferr, file_space_id = space_id, &
                              mem_space_id = memspace, xfer_prp = plist_id)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
  !close types, dataspaces
   call h5tclose_f(dtype_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5tclose_f dtype_id')
   call h5tclose_f(position_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5tclose_f position_id')
   call h5dclose_f(dset_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5dclose_f')
   call h5sclose_f(space_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5sclose_f space_id')
   call h5sclose_f(memspace, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5sclose_f memspace')
   call h5pclose_f(plist_id, hdferr)
   if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_backwardMappingCrystallite: h5pclose_f')
   call HDF5_closeGroup(mapping_ID)

 enddo

end subroutine HDF5_backwardMappingCrystallite

!--------------------------------------------------------------------------------------------------
!> @brief adds the unique cell to node mapping
!--------------------------------------------------------------------------------------------------
subroutine HDF5_mappingCells(mapping)
 use hdf5

 implicit none
 integer(pInt), intent(in), dimension(:) :: mapping

 integer        :: hdferr, Nnodes
 integer(HID_T) :: mapping_id, dset_id, space_id

 Nnodes=size(mapping)
 mapping_ID = results_openGroup("mapping")

!--------------------------------------------------------------------------------------------------
! create dataspace
 call h5screate_simple_f(1, int([Nnodes],HSIZE_T), space_id, hdferr, &
                            int([Nnodes],HSIZE_T))
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCells: h5screate_simple_f')

!--------------------------------------------------------------------------------------------------
! create Dataset
 call h5dcreate_f(mapping_id, "Cell",H5T_NATIVE_INTEGER, space_id, dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCells')

!--------------------------------------------------------------------------------------------------
! write data by fields in the datatype. Fields order is not important.
 call h5dwrite_f(dset_id, H5T_NATIVE_INTEGER, mapping, int([Nnodes],HSIZE_T), hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingCells: h5dwrite_f instance_id')

!--------------------------------------------------------------------------------------------------
!close types, dataspaces
 call h5dclose_f(dset_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingConstitutive: h5dclose_f')
 call h5sclose_f(space_id, hdferr)
 if (hdferr < 0) call IO_error(1_pInt,ext_msg='IO_mappingConstitutive: h5sclose_f')
 call HDF5_closeGroup(mapping_ID)

end subroutine HDF5_mappingCells


end module results
