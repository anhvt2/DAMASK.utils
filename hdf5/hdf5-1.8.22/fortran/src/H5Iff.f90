!****h* ROBODoc/H5I
!
! NAME
!  MODULE H5I
!
!  FILE
!  fortran/src/H5Iff.f90
!
! PURPOSE
!  This file contains Fortran interfaces for H5I functions.
!
! COPYRIGHT
!  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
!  Copyright by The HDF Group.                                                 *
!  Copyright by the Board of Trustees of the University of Illinois.           *
!  All rights reserved.                                                        *
!                                                                              *
!  This file is part of HDF5.  The full HDF5 copyright notice, including       *
!  terms governing use, modification, and redistribution, is contained in      *
!  the COPYING file, which can be found at the root of the source code         *
!  distribution tree, or in https://www.hdfgroup.org/licenses.                 *
!  If you do not have access to either file, you may request a copy from       *
!  help@hdfgroup.org.                                                          *
!  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
!
! NOTES
!                         *** IMPORTANT ***
!  If you add a new H5I function you must add the function name to the
!  Windows dll file 'hdf5_fortrandll.def.in' in the fortran/src directory.
!  This is needed for Windows based operating systems.
!
!*****

MODULE H5I

  USE H5GLOBAL

CONTAINS

!****s* H5I/h5iget_type_f
!
! NAME
!  h5iget_type_f
!
! PURPOSE
!  Retrieves the type of an object.
!
! INPUTS
!  obj_id   - object identifier
! OUTPUTS
!  type     - type of the object, possible values:
!              H5I_FILE_F
!              H5I_GROUP_F
!              H5I_DATATYPE_F
!              H5I_DATASPACE_F
!              H5I_DATASET_F
!              H5I_ATTR_F
!              H5I_BADID_F
!  hdferr:  - error code
!              Success:  0
!              Failure: -1
!
! AUTHOR
!  Elena Pourmal
!  August 12, 1999
!
! HISTORY
!  Explicit Fortran interfaces were added for
!  called C functions (it is needed for Windows
!  port).  March 5, 2001
!
! SOURCE
  SUBROUTINE h5iget_type_f(obj_id, TYPE, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: obj_id  ! Object identifier
    INTEGER, INTENT(OUT) :: TYPE ! type of an object.
                                 ! possible values are:
                                 !   H5I_FILE_F
                                 !   H5I_GROUP_F
                                 !   H5I_DATATYPE_F
                                 !   H5I_DATASPACE_F
                                 !   H5I_DATASET_F
                                 !   H5I_ATTR_F
                                 !   H5I_BADID_F
    INTEGER, INTENT(OUT) :: hdferr  ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5iget_type_c(obj_id, TYPE)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IGET_TYPE_C':: h5iget_type_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: obj_id
         INTEGER, INTENT(OUT) :: TYPE
       END FUNCTION h5iget_type_c
    END INTERFACE
    hdferr = h5iget_type_c(obj_id, TYPE)
  END SUBROUTINE h5iget_type_f

!****s* H5I/h5iget_name_f
!
! NAME
!  h5iget_name_f
!
! PURPOSE
!  Gets a name of an object specified by its idetifier.
!
! INPUTS
!  obj_id    - attribute identifier
!  buf_size  - size of a buffer to read name in
! OUTPUTS
!  buf 	     - buffer to read name in, name will be truncated if
!              buffer is not big enough
!  name_size - name size
!  hdferr:   - error code
!               Success:  0
!               Failure: -1
!
! AUTHOR
!  Elena Pourmal
!  March 12, 2003
! SOURCE
  SUBROUTINE h5iget_name_f(obj_id, buf, buf_size, name_size, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: obj_id      ! Object identifier
    INTEGER(SIZE_T), INTENT(IN) :: buf_size   ! Buffer size
    CHARACTER(LEN=*), INTENT(OUT) :: buf      ! Buffer to hold object name
    INTEGER(SIZE_T), INTENT(OUT) :: name_size ! Actual name size
    INTEGER, INTENT(OUT) :: hdferr            ! Error code:
                                              !   0 if successful,
                                              !   -1 if fail
!*****
    INTERFACE
       INTEGER FUNCTION h5iget_name_c(obj_id, buf, buf_size, name_size)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IGET_NAME_C'::h5iget_name_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: buf
         INTEGER(HID_T), INTENT(IN) :: obj_id
         CHARACTER(LEN=*), INTENT(OUT) :: buf
         INTEGER(SIZE_T), INTENT(IN) :: buf_size
         INTEGER(SIZE_T), INTENT(OUT) :: name_size
       END FUNCTION h5iget_name_c
    END INTERFACE

    hdferr = h5iget_name_c(obj_id, buf, buf_size, name_size)
  END SUBROUTINE h5iget_name_f

!****s* H5I/h5iinc_ref_f
!
! NAME
!  h5iinc_ref_f
!
! PURPOSE
!  Increments the reference count of an ID
!
! INPUTS
!  obj_id 	 - object identifier
! OUTPUTS
!  ref_count 	 - Current reference count of the ID
!  hdferr:		- error code
!  Success:  0
!  Failure: -1
! AUTHOR
!  Quincey Koziol
!  December  9, 2003
!
! SOURCE
  SUBROUTINE h5iinc_ref_f(obj_id, ref_count, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: obj_id ! Object identifier
    INTEGER, INTENT(OUT) :: ref_count    ! Current reference count of ID
    INTEGER, INTENT(OUT) :: hdferr       ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5iinc_ref_c(obj_id, ref_count)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IINC_REF_C':: h5iinc_ref_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: obj_id
         INTEGER, INTENT(OUT) :: ref_count
       END FUNCTION h5iinc_ref_c
    END INTERFACE
    hdferr = h5iinc_ref_c(obj_id, ref_count)
  END SUBROUTINE h5iinc_ref_f

!****s* H5I/h5idec_ref_f
!
! NAME
!  h5idec_ref_f
!
! PURPOSE
!  Decrements the reference count of an ID
!
! INPUTS
!  obj_id 	 - Object identifier
! OUTPUTS
!  ref_count 	 - Current reference count of the ID
!  hdferr:	 - Error code
!                   Success:  0
!                   Failure: -1
! AUTHOR
!  Quincey Koziol
!  December  9, 2003
!
! SOURCE
  SUBROUTINE h5idec_ref_f(obj_id, ref_count, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: obj_id ! Object identifier
    INTEGER, INTENT(OUT) :: ref_count    ! Current reference count of ID
    INTEGER, INTENT(OUT) :: hdferr       ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5idec_ref_c(obj_id, ref_count)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IDEC_REF_C':: h5idec_ref_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: obj_id
         INTEGER, INTENT(OUT) :: ref_count
       END FUNCTION h5idec_ref_c
    END INTERFACE
    hdferr = h5idec_ref_c(obj_id, ref_count)
  END SUBROUTINE h5idec_ref_f

!****s* H5I/h5iget_ref_f
! NAME
!  h5iget_ref_f
!
! PURPOSE
!  Retrieves the reference count of an ID
!
! INPUTS
!  obj_id 	 - object identifier
!
! OUTPUTS
!  ref_count 	 - Current reference count of the ID
!  hdferr:	   - error code
!  Success:  0
!  Failure: -1
! AUTHOR
!  Quincey Koziol
!  December  9, 2003
!
! SOURCE
  SUBROUTINE h5iget_ref_f(obj_id, ref_count, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: obj_id ! Object identifier
    INTEGER, INTENT(OUT) :: ref_count    ! Current reference count of ID
    INTEGER, INTENT(OUT) :: hdferr       ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5iget_ref_c(obj_id, ref_count)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IGET_REF_C':: h5iget_ref_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: obj_id
         INTEGER, INTENT(OUT) :: ref_count
       END FUNCTION h5iget_ref_c
    END INTERFACE
    hdferr = h5iget_ref_c(obj_id, ref_count)
  END SUBROUTINE h5iget_ref_f
!
!****s* H5I/h5iget_file_id_f
! NAME
!  h5iget_file_id_f
!
! PURPOSE
!  Obtains file identifier from the object identifier
!
! INPUTS
!  obj_id 	 - object identifier
! OUTPUTS
!  file_id 	 - file identifier
!  hdferr:       - error code
!                    Success:  0
!                    Failure: -1
!
! AUTHOR
!  Elena Pourmal
!  August 23, 2004
! SOURCE
  SUBROUTINE h5iget_file_id_f(obj_id, file_id, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN)  :: obj_id   ! Object identifier
    INTEGER(HID_T), INTENT(OUT) :: file_id  ! File identifier
    INTEGER, INTENT(OUT) :: hdferr          ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5iget_file_id_c(obj_id, file_id)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IGET_FILE_ID_C':: h5iget_file_id_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN)  :: obj_id
         INTEGER(HID_T), INTENT(OUT) :: file_id
       END FUNCTION h5iget_file_id_c
    END INTERFACE
    hdferr = h5iget_file_id_c(obj_id, file_id)
  END SUBROUTINE h5iget_file_id_f
!
!****s* H5I/h5iis_valid_f
! NAME
!  h5iget_file_id_f
!
! PURPOSE
!  Check if an ID is valid without producing an error message
!
! INPUTS
!  id		- identifier 
! OUTPUTS       
!  valid        - status of id as a valid identifier
!  hdferr:	- error code		
!		   Success:  0
!		   Failure: -1
!
! AUTHOR
!  M. Scot Breitenfeld
!  April 13, 2009
! SOURCE
  SUBROUTINE h5iis_valid_f(id, valid, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN)  :: id ! Identifier 
    LOGICAL, INTENT(OUT) :: valid     ! Status of id as a valid identifier
    INTEGER, INTENT(OUT) :: hdferr    ! Error code
!*****
    INTEGER  :: c_valid ! 0 = .false, 1 = .true.
    
    INTERFACE
       INTEGER FUNCTION h5iis_valid_c(id, c_valid)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5IIS_VALID_C':: h5iis_valid_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN)  :: id   ! Identifier 
         INTEGER  :: c_valid
       END FUNCTION h5iis_valid_c
    END INTERFACE
    
    hdferr = h5iis_valid_c(id, c_valid)
    
    valid = .FALSE. ! Default
    IF(c_valid.EQ.1) valid = .TRUE.
    
  END SUBROUTINE h5iis_valid_f
END MODULE H5I

