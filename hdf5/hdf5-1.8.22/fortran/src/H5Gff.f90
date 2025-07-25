!****h* ROBODoc/H5G
!
! NAME
!  MODULE H5G
!
! FILE
!  fortran/src/H5Gff.f90
!
! PURPOSE
!  This file contains Fortran interfaces for H5G functions.
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
!  If you add a new H5G function you must add the function name to the
!  Windows dll file 'hdf5_fortrandll.def.in' in the fortran/src directory.
!  This is needed for Windows based operating systems.
!
!*****

MODULE H5G
  USE H5GLOBAL

!  PRIVATE :: h5gcreate1_f
!  PRIVATE :: h5gcreate2_f

!  INTERFACE h5gcreate_f
!  MODULE PROCEDURE h5gcreate1_f
!  MODULE PROCEDURE h5gcreate2_f
!  END INTERFACE

CONTAINS

!****s* H5G/h5gcreate_f
!
! NAME
!  h5gcreate_f
!
! PURPOSE
!  Creates a new group.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - group name at the specified location
! OUTPUTS
!  grp_id 	 - group identifier
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  size_hint 	 - a parameter indicating the number of bytes to
!                  reserve for the names that will appear in the group
!  lcpl_id 	 - Property list for link creation
!  gcpl_id 	 - Property list for group creation
!  gapl_id 	 - Property list for group access
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
!  Added additional optional paramaters in 1.8
!  MSB - February 27, 2008
!
! SOURCE
  SUBROUTINE h5gcreate_f(loc_id, name, grp_id, hdferr, size_hint, lcpl_id, gcpl_id, gapl_id)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of the group
    INTEGER(HID_T), INTENT(OUT) :: grp_id  ! Group identifier
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
    INTEGER(SIZE_T), OPTIONAL, INTENT(IN) :: size_hint
                                           ! Parameter indicating
                                           ! the number of bytes
                                           ! to reserve for the
                                           ! names that will appear
                                           ! in the group. Set to OBJECT_NAMELEN_DEFAULT_F
                                           ! if using any of the optional
                                           ! parameters lcpl_id, gcpl_id, and/or gapl_id when not
                                           ! using keywords in specifying the optional parameters
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: lcpl_id  ! Property list for link creation
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gcpl_id  ! Property list for group creation
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gapl_id  ! Property list for group access
!*****
    INTEGER(HID_T) :: lcpl_id_default
    INTEGER(HID_T) :: gcpl_id_default
    INTEGER(HID_T) :: gapl_id_default

    INTEGER :: namelen ! Length of the name character string
    INTEGER(SIZE_T) :: size_hint_default

    INTERFACE
       INTEGER FUNCTION h5gcreate_c(loc_id, name, namelen, &
            size_hint_default, grp_id, lcpl_id_default, gcpl_id_default, gapl_id_default)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GCREATE_C'::h5gcreate_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         INTEGER(SIZE_T) :: size_hint_default
         INTEGER(HID_T), INTENT(OUT) :: grp_id
         INTEGER(HID_T) :: lcpl_id_default
         INTEGER(HID_T) :: gcpl_id_default
         INTEGER(HID_T) :: gapl_id_default
       END FUNCTION h5gcreate_c
    END INTERFACE

    size_hint_default = OBJECT_NAMELEN_DEFAULT_F
    IF (PRESENT(size_hint)) size_hint_default = size_hint
    lcpl_id_default = H5P_DEFAULT_F
    IF(PRESENT(lcpl_id)) lcpl_id_default = lcpl_id
    gcpl_id_default = H5P_DEFAULT_F
    IF(PRESENT(gcpl_id)) gcpl_id_default = gcpl_id
    gapl_id_default = H5P_DEFAULT_F
    IF(PRESENT(gapl_id)) gapl_id_default = gapl_id

    namelen = LEN(name)

    hdferr = h5gcreate_c(loc_id, name, namelen, size_hint_default, grp_id, &
         lcpl_id_default, gcpl_id_default, gapl_id_default)

  END SUBROUTINE h5gcreate_f

!!$!
!!$!****s* H5G/
!!$!
!!$! NAME
!!$!  h5gcreate2_f
!!$!
!!$! PURPOSE
!!$!	Creates a new group.
!!$!
!!$! INPUTS
!!$!		loc_id		- location identifier
!!$!		name		- group name at the specified location
!!$! OUTPUTS
!!$!		grp_id		- group identifier
!!$!		hdferr:		- error code
!!$!				 	Success:  0
!!$!				 	Failure: -1
!!$! OPTIONAL PARAMETERS
!!$!
!!$!    lcpl_id  - Property list for link creation
!!$!    gcpl_id  - Property list for group creation
!!$!    gapl_id  - Property list for group access
!!$!
!!$! AUTHOR	M. Scot Breitenfeld
!!$!		February 27, 2008
!!$!
!!$! HISTORY
!!$!
!!$! NOTES Needed to switch the first 2 arguments to avoid conflect
!!$!          with h5gcreate1_f
!!$!
!!$
!!$  SUBROUTINE h5gcreate2_f(name, loc_id, grp_id, hdferr, &
!!$        lcpl_id, gcpl_id, gapl_id)
!!$    IMPLICIT NONE
!!$    CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of the group
!!$    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
!!$    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!!$    INTEGER(HID_T), INTENT(OUT) :: grp_id  ! Group identifier
!!$
!!$    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: lcpl_id  ! Property list for link creation
!!$    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gcpl_id  ! Property list for group creation
!!$    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gapl_id  ! Property list for group access
!!$
!!$    INTEGER(HID_T) :: lcpl_id_default
!!$    INTEGER(HID_T) :: gcpl_id_default
!!$    INTEGER(HID_T) :: gapl_id_default
!!$
!!$    INTEGER(SIZE_T) :: OBJECT_NAME
!  LEN_DEFAULT ! Dummy argument to pass to c call
!!$    INTEGER :: namelen ! Length of the name character string
!!$
!!$!  MS FORTRAN needs explicit interface for C functions called here.
!!$!
!!$    INTERFACE
!!$       INTEGER FUNCTION h5gcreate_c(loc_id, name, namelen, &
!!$            OBJECT_NAME
!  LEN_DEFAULT, grp_id, lcpl_id_default, gcpl_id_default, gapl_id_default)
!!$         USE H5GLOBAL
!!$         !DEC$IF DEFINED(HDF5F90_WINDOWS)
!!$         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GCREATE_C'::h5gcreate_c
!!$         !DEC$ENDIF
!!$         !DEC$ATTRIBUTES reference :: name
!!$         INTEGER(HID_T), INTENT(IN) :: loc_id
!!$         CHARACTER(LEN=*), INTENT(IN) :: name
!!$         INTEGER :: namelen
!!$         INTEGER(SIZE_T) :: OBJECT_NAME
!  LEN_DEFAULT
!!$         INTEGER(HID_T) :: lcpl_id_default
!!$         INTEGER(HID_T) :: gcpl_id_default
!!$         INTEGER(HID_T) :: gapl_id_default
!!$         INTEGER(HID_T), INTENT(OUT) :: grp_id
!!$       END FUNCTION h5gcreate_c
!!$    END INTERFACE
!!$
!!$    namelen = LEN(name)
!!$    OBJECT_NAME
!  LEN_DEFAULT = OBJECT_NAME
!  LEN_DEFAULT_F
!!$
!!$    lcpl_id_default = H5P_DEFAULT_F
!!$    IF(PRESENT(lcpl_id)) lcpl_id_default = lcpl_id
!!$    gcpl_id_default = H5P_DEFAULT_F
!!$    IF(PRESENT(gcpl_id)) gcpl_id_default = gcpl_id
!!$    gapl_id_default = H5P_DEFAULT_F
!!$    IF(PRESENT(gapl_id)) gapl_id_default = gapl_id
!!$
!!$
!!$    hdferr = h5gcreate_c(loc_id, name, namelen, OBJECT_NAME
!  LEN_DEFAULT, grp_id, &
!!$         lcpl_id_default, gcpl_id_default, gapl_id_default)
!!$
!!$  END SUBROUTINE h5gcreate2_f

!
!****s* H5G/h5gopen_f
!
! NAME
!  h5gopen_f
!
! PURPOSE
!  Opens an existing group.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the group to open
! OUTPUTS
!  grp_id 	 - group identifier
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  gapl_id 	 - Group access property list identifier
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
!  Added 1.8 (optional) parameter gapl_id
!  February, 2008 M. Scot Breitenfeld
!
! SOURCE
  SUBROUTINE h5gopen_f(loc_id, name, grp_id, hdferr, gapl_id)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of the group
    INTEGER(HID_T), INTENT(OUT) :: grp_id  ! File identifier
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gapl_id  ! Group access property list identifier
!*****
    INTEGER(HID_T) :: gapl_id_default
    INTEGER :: namelen ! Length of the name character string

    INTERFACE
       INTEGER FUNCTION h5gopen_c(loc_id, name, namelen, gapl_id_default, grp_id)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GOPEN_C'::h5gopen_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         INTEGER(HID_T), INTENT(IN) :: gapl_id_default
         INTEGER(HID_T), INTENT(OUT) :: grp_id
       END FUNCTION h5gopen_c
    END INTERFACE

    gapl_id_default = H5P_DEFAULT_F
    IF(PRESENT(gapl_id)) gapl_id_default = gapl_id

    namelen = LEN(name)
    hdferr = h5gopen_c(loc_id, name, namelen, gapl_id_default, grp_id)

  END SUBROUTINE h5gopen_f
!
!****s* H5G/h5gclose_f
!
! NAME
!  h5gclose_f
!
! PURPOSE
!  Closes the specified group.
!
! INPUTS
!  grp_id 	 - group identifier
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gclose_f(grp_id, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: grp_id  ! Group identifier
    INTEGER, INTENT(OUT) :: hdferr        ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5gclose_c(grp_id)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GCLOSE_C'::h5gclose_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: grp_id
       END FUNCTION h5gclose_c
    END INTERFACE

    hdferr = h5gclose_c(grp_id)

  END SUBROUTINE h5gclose_f
!
!****s* H5G/h5gget_obj_info_idx_f
!
! NAME
!  h5gget_obj_info_idx_f
!
! PURPOSE
!  Returns name and type of the group member identified by
!  its index.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the group at the specified location
!  idx 	         - object index (zero-based)
! OUTPUTS
!  obj_name 	 - object name
!  obj_type 	 - object type
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gget_obj_info_idx_f(loc_id, name, idx, &
       obj_name, obj_type, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of the group
    INTEGER, INTENT(IN) :: idx             ! Index of member object
    CHARACTER(LEN=*), INTENT(OUT) :: obj_name   ! Name of the object
    INTEGER, INTENT(OUT) :: obj_type       ! Object type
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTEGER :: namelen ! Length of the name character string
    INTEGER :: obj_namelen ! Length of the obj_name character string

    INTERFACE
       INTEGER FUNCTION h5gget_obj_info_idx_c(loc_id, name, &
            namelen, idx, &
            obj_name, obj_namelen, obj_type)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_OBJ_INFO_IDX_C'::h5gget_obj_info_idx_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         !DEC$ATTRIBUTES reference :: obj_name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         INTEGER, INTENT(IN) :: idx
         CHARACTER(LEN=*), INTENT(OUT) :: obj_name
         INTEGER :: obj_namelen
         INTEGER, INTENT(OUT) :: obj_type
       END FUNCTION h5gget_obj_info_idx_c
    END INTERFACE

    namelen = LEN(name)
    obj_namelen = LEN(obj_name)
    hdferr = h5gget_obj_info_idx_c(loc_id, name, namelen, idx, &
                                           obj_name, obj_namelen, obj_type)
  END SUBROUTINE h5gget_obj_info_idx_f

!
!****s* H5G/h5gn_members_f
!
! NAME
!  h5gn_members_f
!
! PURPOSE
!  Returns the number of group members.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the group at the specified location
! OUTPUTS
!  nmembers 	 - number of group members
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gn_members_f(loc_id, name, nmembers, hdferr)
            IMPLICIT NONE
            INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
            CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of the group
            INTEGER, INTENT(OUT) :: nmembers       ! Number of members in the
                                                   ! group
            INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
            INTEGER :: namelen ! Length of the name character string

            INTERFACE
              INTEGER FUNCTION h5gn_members_c(loc_id, name, namelen, nmembers)
              USE H5GLOBAL
              !DEC$IF DEFINED(HDF5F90_WINDOWS)
              !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GN_MEMBERS_C'::h5gn_members_c
              !DEC$ENDIF
              !DEC$ATTRIBUTES reference :: name
              INTEGER(HID_T), INTENT(IN) :: loc_id
              CHARACTER(LEN=*), INTENT(IN) :: name
              INTEGER :: namelen
              INTEGER, INTENT(OUT) :: nmembers
              END FUNCTION h5gn_members_c
            END INTERFACE

            namelen = LEN(name)
            hdferr = h5gn_members_c(loc_id, name, namelen, nmembers)

          END SUBROUTINE h5gn_members_f
!
!****s* H5G/h5glink_f
!
! NAME
!  h5glink_f
!
! PURPOSE
!  Creates a link of the specified type from new_name
!  to current_name.
!
! INPUTS
!  loc_id 	 - location identifier
!  link_type 	 - link type; possible values are:
!                    H5G_LINK_HARD_F (0)
!                    H5G_LINK_SOFT_F (1)
!  current_name  - name of the existing object if link is a
!                  hard link. Can be anything for the soft link
!  new_name 	 - new name for the object
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5glink_f(loc_id, link_type, current_name, &
       new_name, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    INTEGER, INTENT(IN) :: link_type       ! link type
                                                   ! Possible values are:
                                                   ! H5G_LINK_HARD_F (0) or
                                                   ! H5G_LINK_SOFT_F (1)

    CHARACTER(LEN=*), INTENT(IN) :: current_name
                                                   ! Current name of an object
    CHARACTER(LEN=*), INTENT(IN) :: new_name ! New name of an object
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****

    INTEGER :: current_namelen ! Lenghth of the current_name string
    INTEGER :: new_namelen     ! Lenghth of the new_name string

    INTERFACE
       INTEGER FUNCTION h5glink_c(loc_id, link_type, current_name, &
            current_namelen, new_name, new_namelen)

         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GLINK_C'::h5glink_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: current_name
         !DEC$ATTRIBUTES reference :: new_name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         INTEGER, INTENT(IN) :: link_type
         CHARACTER(LEN=*), INTENT(IN) :: current_name
         INTEGER :: current_namelen
         CHARACTER(LEN=*), INTENT(IN) :: new_name
         INTEGER :: new_namelen
       END FUNCTION h5glink_c
    END INTERFACE

    current_namelen = LEN(current_name)
    new_namelen = LEN(new_name)
    hdferr = h5glink_c(loc_id, link_type, current_name, &
         current_namelen, new_name, new_namelen)
  END SUBROUTINE h5glink_f

!
!****s* H5G/h5glink2_f
!
! NAME
!  h5glink2_f
!
! PURPOSE
!  Creates a link of the specified type from new_name
!  to current_name. current_name and new_name are interpreted
!  releative to current and new location identifiers.
!
! INPUTS
!  cur_loc_id 	 - location identifier
!  cur_name 	 - name of the existing object if link is a
!                  hard link. Can be anything for the soft link.
!  link_type 	 - link type; possible values are:
!                    H5G_LINK_HARD_F (0)
!                    H5G_LINK_SOFT_F (1)
!  new_loc_id 	 - new location identifier
!  new_name 	 - new name for the object
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
!
! AUTHOR
!  Elena Pourmal
!  September 25, 2002
!
! SOURCE
  SUBROUTINE h5glink2_f(cur_loc_id, cur_name, link_type, new_loc_id, &
       new_name, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: cur_loc_id ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: cur_name
                                             ! Current name of an object
    INTEGER, INTENT(IN) :: link_type         ! link type
                                                 ! Possible values are:
                                                 ! H5G_LINK_HARD_F (0) or
                                                 ! H5G_LINK_SOFT_F (1)

    INTEGER(HID_T), INTENT(IN) :: new_loc_id ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: new_name ! New name of an object
    INTEGER, INTENT(OUT) :: hdferr           ! Error code
!*****

    INTEGER :: cur_namelen ! Lenghth of the current_name string
    INTEGER :: new_namelen ! Lenghth of the new_name string

    INTERFACE
       INTEGER FUNCTION h5glink2_c(cur_loc_id, cur_name, cur_namelen, &
            link_type, new_loc_id, &
            new_name, new_namelen)

         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GLINK2_C'::h5glink2_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: cur_name
         !DEC$ATTRIBUTES reference :: new_name
         INTEGER(HID_T), INTENT(IN) :: cur_loc_id
         INTEGER(HID_T), INTENT(IN) :: new_loc_id
         INTEGER, INTENT(IN) :: link_type
         CHARACTER(LEN=*), INTENT(IN) :: cur_name
         CHARACTER(LEN=*), INTENT(IN) :: new_name
         INTEGER :: cur_namelen
         INTEGER :: new_namelen
       END FUNCTION h5glink2_c
    END INTERFACE

    cur_namelen = LEN(cur_name)
    new_namelen = LEN(new_name)
    hdferr = h5glink2_c(cur_loc_id, cur_name, cur_namelen, link_type, &
         new_loc_id, new_name, new_namelen)
  END SUBROUTINE h5glink2_f

!
!****s* H5G/h5gunlink_f
!
! NAME
!  h5gunlink_f
!
! PURPOSE
!  Removes the specified name from the group graph and
!  decrements the link count for the object to which name
!  points
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the object to unlink
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gunlink_f(loc_id, name, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Name of an object
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTEGER :: namelen ! Lenghth of the name character string

    INTERFACE
       INTEGER FUNCTION h5gunlink_c(loc_id, name, namelen)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GUNLINK_C'::h5gunlink_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
       END FUNCTION h5gunlink_c
    END INTERFACE

    namelen = LEN(name)
    hdferr = h5gunlink_c(loc_id, name, namelen)
  END SUBROUTINE h5gunlink_f

!
!****s* H5G/h5gmove_f
!
! NAME
!  h5gmove_f
!
! PURPOSE
!  Renames an object within an HDF5 file.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - object's name at specified location
!  new_name 	 - object's new name
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gmove_f(loc_id, name, new_name, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id     ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name     ! Current name of an object
    CHARACTER(LEN=*), INTENT(IN) :: new_name ! New name of an object
    INTEGER, INTENT(OUT) :: hdferr           ! Error code
!*****
    INTEGER :: namelen         ! Lenghth of the current_name string
    INTEGER :: new_namelen     ! Lenghth of the new_name string
    
    INTERFACE
       INTEGER FUNCTION h5gmove_c(loc_id, name, namelen, new_name, new_namelen)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GMOVE_C'::h5gmove_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         !DEC$ATTRIBUTES reference :: new_name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         CHARACTER(LEN=*), INTENT(IN) :: new_name
         INTEGER :: new_namelen
       END FUNCTION h5gmove_c
    END INTERFACE
    
    namelen = LEN(name)
    new_namelen = LEN(new_name)
    hdferr = h5gmove_c(loc_id, name, namelen, new_name, new_namelen)
  END SUBROUTINE h5gmove_f
!
!****s* H5G/h5gmove2_f
!
! NAME
!  h5gmove2_f
!
! PURPOSE
!  Renames an object within an HDF5 file.
!
! INPUTS
!  src_loc_id 	 - original location identifier
!  src_name 	 - object's name at specified original location
!  dst_loc_id 	 - original location identifier
!  dst_name 	 - object's new name
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
! AUTHOR
!  Elena Pourmal
!  September 25, 2002
!
! SOURCE
  SUBROUTINE h5gmove2_f(src_loc_id, src_name, dst_loc_id, dst_name, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN)   :: src_loc_id  ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: src_name    ! Original name of an object
    INTEGER(HID_T), INTENT(IN)   :: dst_loc_id  ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: dst_name    ! New name of an object
    INTEGER, INTENT(OUT)         :: hdferr      ! Error code
!*****
    INTEGER :: src_namelen         ! Length of the current_name string
    INTEGER :: dst_namelen         ! Lenghth of the new_name string
    
    INTERFACE
       INTEGER FUNCTION h5gmove2_c(src_loc_id, src_name, src_namelen, &
            dst_loc_id, dst_name, dst_namelen)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GMOVE2_C'::h5gmove2_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: src_name
         !DEC$ATTRIBUTES reference :: dst_name
         INTEGER(HID_T), INTENT(IN) :: src_loc_id
         INTEGER(HID_T), INTENT(IN) :: dst_loc_id
         CHARACTER(LEN=*), INTENT(IN) :: src_name
         CHARACTER(LEN=*), INTENT(IN) :: dst_name
         INTEGER :: src_namelen
         INTEGER :: dst_namelen
       END FUNCTION h5gmove2_c
    END INTERFACE
    
    src_namelen = LEN(src_name)
    dst_namelen = LEN(dst_name)
    hdferr = h5gmove2_c(src_loc_id, src_name, src_namelen, dst_loc_id, dst_name, dst_namelen)
  END SUBROUTINE h5gmove2_f
!
!****s* H5G/h5gget_linkval_f
!
! NAME
!  h5gget_linkval_f
!
! PURPOSE
!  Returns the name of the object that the symbolic link
!  points to.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - symbolic link to the object whose name
!                  is to be returned.
!  size 	 - maximum number of characters to be returned
! OUTPUTS
!  buffer 	 - a buffer to hold the name of the object
!  being sought
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gget_linkval_f(loc_id, name, size, buffer, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Current name of an object
    INTEGER(SIZE_T), INTENT(IN) :: size    ! Maximum number of buffer
    CHARACTER(LEN=size), INTENT(OUT) :: buffer
                                           ! Buffer to hold a name of
                                           ! the object symbolic link
                                           ! points to
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTEGER :: namelen ! Lenghth of the current_name string
    
    INTERFACE
       INTEGER FUNCTION h5gget_linkval_c(loc_id, name, namelen, size, buffer)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_LINKVAL_C'::h5gget_linkval_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         !DEC$ATTRIBUTES reference :: buffer
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         INTEGER(SIZE_T), INTENT(IN) :: size
         CHARACTER(LEN=*), INTENT(OUT) :: buffer
       END FUNCTION h5gget_linkval_c
    END INTERFACE
    
    namelen = LEN(name)
    hdferr = h5gget_linkval_c(loc_id, name, namelen, size, buffer)
  END SUBROUTINE h5gget_linkval_f

!
!****s* H5G/h5gset_comment_f
!
! NAME
!  h5gset_comment_f
!
! PURPOSE
!  Sets comment for specified object.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the object
!  comment 	 - comment to set for the object
! OUTPUTS
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gset_comment_f(loc_id, name, comment, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Current name of an object
    CHARACTER(LEN=*), INTENT(IN) :: comment ! New name of an object
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTEGER :: namelen ! Lenghth of the current_name string
    INTEGER :: commentlen     ! Lenghth of the comment string
    
    INTERFACE
       INTEGER FUNCTION h5gset_comment_c(loc_id, name, namelen, &
            comment, commentlen)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GSET_COMMENT_C'::h5gset_comment_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name
         !DEC$ATTRIBUTES reference :: comment
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         CHARACTER(LEN=*), INTENT(IN) :: comment
         INTEGER :: commentlen
       END FUNCTION h5gset_comment_c
    END INTERFACE
    
    namelen = LEN(name)
    commentlen = LEN(comment)
    hdferr = h5gset_comment_c(loc_id, name, namelen, comment, commentlen)
  END SUBROUTINE h5gset_comment_f
!
!****s* H5G/h5gget_comment_f
!
! NAME
!  h5gget_comment_f
!
! PURPOSE
!  Retrieves comment for specified object.
!
! INPUTS
!  loc_id 	 - location identifier
!  name 	 - name of the object at specified location
!  size 	 - size of the buffer required to hold comment
! OUTPUTS
!  buffer 	 - buffer to hold object's comment
!  hdferr 	 - Returns 0 if successful and -1 if fails
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
  SUBROUTINE h5gget_comment_f(loc_id, name, size, buffer, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: name   ! Current name of an object
    INTEGER(SIZE_T), INTENT(IN) :: size    ! Maximum number of buffer
    CHARACTER(LEN=size), INTENT(OUT) :: buffer
                                           ! Buffer to hold a comment
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTEGER :: namelen ! Length of the current_name string

    INTERFACE
       INTEGER FUNCTION h5gget_comment_c(loc_id, name, namelen, size, buffer)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_COMMENT_C'::h5gget_comment_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: name, buffer
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: name
         INTEGER :: namelen
         INTEGER(SIZE_T), INTENT(IN) :: size
         CHARACTER(LEN=*), INTENT(OUT) :: buffer
       END FUNCTION h5gget_comment_c
    END INTERFACE

    namelen = LEN(name)
    hdferr = h5gget_comment_c(loc_id, name, namelen, size, buffer)

  END SUBROUTINE h5gget_comment_f
!
!****s* H5G/H5Gcreate_anon_f
!
! NAME
!  H5Gcreate_anon_f
!
! PURPOSE
!  Creates a new empty group without linking it into the file structure.
!
! INPUTS
!  loc_id 	 - Location identifier
! OUTPUTS
!  grp_id 	 - group identifier
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  gcpl_id 	 - Group creation property list identifier
!  gapl_id 	 - Group access property list identifier
!
! AUTHOR
!  M. Scot Breitenfeld
!  February 15, 2008
!
! SOURCE
  SUBROUTINE h5Gcreate_anon_f(loc_id, grp_id, hdferr, gcpl_id, gapl_id)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
    INTEGER(HID_T), INTENT(OUT) :: grp_id  ! Group identifier
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gcpl_id  ! Property list for group creation
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: gapl_id  ! Property list for group access
!*****
    INTEGER(HID_T) :: gcpl_id_default
    INTEGER(HID_T) :: gapl_id_default

    INTERFACE
       INTEGER FUNCTION h5gcreate_anon_c(loc_id, gcpl_id_default, gapl_id_default, grp_id)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GCREATE_ANON_C'::h5gcreate_anon_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: loc_id   ! File or group identifier
         INTEGER(HID_T), INTENT(IN) :: gcpl_id_default  ! Property list for group creation
         INTEGER(HID_T), INTENT(IN) :: gapl_id_default  ! Property list for group access
         INTEGER(HID_T), INTENT(OUT) :: grp_id  ! Group identifier
       END FUNCTION h5gcreate_anon_c
    END INTERFACE

    gcpl_id_default = H5P_DEFAULT_F
    gapl_id_default = H5P_DEFAULT_F

    IF(PRESENT(gcpl_id)) gcpl_id_default = gcpl_id
    IF(PRESENT(gapl_id)) gapl_id_default = gapl_id

    hdferr = h5gcreate_anon_c(loc_id, gcpl_id_default, gapl_id_default, grp_id)

  END SUBROUTINE h5Gcreate_anon_f
!
!****s* H5G/H5Gget_create_plist_f
!
! NAME
!  H5Gget_create_plist_f
!
! PURPOSE
!  Gets a group creation property list identifier.
!
! INPUTS
!  grp_id 	 - group identifier
! OUTPUTS
!  gcpl_id 	 - Group creation property list identifier
!  hdferr 	 - Returns 0 if successful and -1 if fails
! AUTHOR
!  M. Scot Breitenfeld
!  February 15, 2008
!
! SOURCE
  SUBROUTINE h5gget_create_plist_f(grp_id, gcpl_id, hdferr)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN)  :: grp_id  ! Group identifier
    INTEGER(HID_T), INTENT(OUT) :: gcpl_id ! Property list for group creation
    INTEGER, INTENT(OUT) :: hdferr         ! Error code
!*****
    INTERFACE
       INTEGER FUNCTION h5gget_create_plist_c(grp_id, gcpl_id )
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_CREATE_PLIST_C'::h5gget_create_plist_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN)  :: grp_id
         INTEGER(HID_T), INTENT(OUT) :: gcpl_id
       END FUNCTION h5gget_create_plist_c
    END INTERFACE

    hdferr = h5gget_create_plist_c(grp_id, gcpl_id )

  END SUBROUTINE h5gget_create_plist_f

!
!****s* H5G/h5gget_info_f
!
! NAME
!  h5gget_info_f
!
! PURPOSE
!  Retrieves information about a group
!
! INPUTS
!  group_id 	 - Group identifier
!
! OUTPUTS  
!  storage_type  - Type of storage for links in group
!                    H5G_STORAGE_TYPE_COMPACT: Compact storage
!                    H5G_STORAGE_TYPE_DENSE: Indexed storage
!                    H5G_STORAGE_TYPE_SYMBOL_TABLE: Symbol tables, the original HDF5 structure
!  nlinks 	 - Number of links in group
!  max_corder 	 - Current maximum creation order value for group
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  mounted 	 - Whether group has a file mounted on it
!
! AUTHOR
!  M. Scot Breitenfeld
!  February 15, 2008
!
! NOTES
!  In C the output is defined as a structure: H5G_info_t
!
! HISTORY
!
!  - Added 'mounted' paramater
!    M. Scot Breitenfeld
!    July 16, 2008
!
! SOURCE
  SUBROUTINE h5gget_info_f(group_id, storage_type, nlinks, max_corder, hdferr, mounted)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: group_id ! Group identifier

    INTEGER, INTENT(OUT) :: storage_type  ! Type of storage for links in group:
                                          ! H5G_STORAGE_TYPE_COMPACT_F: Compact storage
                                          ! H5G_STORAGE_TYPE_DENSE_F: Indexed storage
                                          ! H5G_STORAGE_TYPE_SYMBOL_TABLE_F: Symbol tables, the original HDF5 structure
    INTEGER, INTENT(OUT) :: nlinks        ! Number of links in group
    INTEGER, INTENT(OUT) :: max_corder    ! Current maximum creation order value for group
    INTEGER, INTENT(OUT) :: hdferr        ! Error code:
                                          ! 0 on success and -1 on failure
    LOGICAL, INTENT(OUT), OPTIONAL :: mounted  ! Whether group has a file mounted on it
!*****
    INTEGER :: mounted_c

    INTERFACE
       INTEGER FUNCTION h5gget_info_c(group_id, storage_type, nlinks, max_corder, mounted_c)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_INFO_C'::h5gget_info_c
         !DEC$ENDIF
         INTEGER(HID_T), INTENT(IN) :: group_id
         INTEGER, INTENT(OUT) :: storage_type
         INTEGER, INTENT(OUT) :: nlinks
         INTEGER, INTENT(OUT) :: max_corder
         INTEGER :: mounted_c
       END FUNCTION h5gget_info_c
    END INTERFACE

    hdferr = h5gget_info_c(group_id, storage_type, nlinks, max_corder, mounted_c)

    IF(PRESENT(mounted))THEN
       IF(mounted_c.EQ.0) THEN
          mounted = .FALSE.
       ELSE
          mounted = .TRUE.
       ENDIF
    ENDIF

  END SUBROUTINE h5gget_info_f
!
!****s* H5G/h5gget_info_by_idx_f
!
! NAME
!  h5gget_info_by_idx_f
!
! PURPOSE
!  Retrieves information about a group, according to the group’s position within an index.
!
! INPUTS
!  loc_id 	 - File or group identifier
!  group_name 	 - Name of group containing group for which information is to be retrieved
!  index_type 	 - Index type
!  order 	 - Order of the count in the index
!  n 	         - Position in the index of the group for which information is retrieved
!
! OUTPUTS
!  storage_type  - Type of storage for links in group
!                    H5G_STORAGE_TYPE_COMPACT: Compact storage
!                    H5G_STORAGE_TYPE_DENSE: Indexed storage
!                    H5G_STORAGE_TYPE_SYMBOL_TABLE: Symbol tables, the original HDF5 structure
!  nlinks 	 - Number of links in group
!  max_corder 	 - Current maximum creation order value for group
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  lapl_id 	 - Link access property list
!  mounted 	 - Whether group has a file mounted on it
!
! NOTES
!  In C the output is defined as a structure: H5G_info_t
!
! AUTHOR
!  M. Scot Breitenfeld
!  February 18, 2008
!
! HISTORY
!  Added 'mounted' paramater
!  M. Scot Breitenfeld
!  July 16, 2008
!
! SOURCE
  SUBROUTINE h5gget_info_by_idx_f(loc_id, group_name, index_type, order, n, &
       storage_type, nlinks, max_corder, hdferr, lapl_id, mounted)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id       ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: group_name ! Name of group containing group for which information is to be retrieved
    INTEGER, INTENT(IN) :: index_type ! Index type
    INTEGER, INTENT(IN) :: order      ! Order of the count in the index
    INTEGER(HSIZE_T), INTENT(IN) :: n ! Position in the index of the group for which information is retrieved

    INTEGER, INTENT(OUT) :: storage_type ! Type of storage for links in group:
                                         !   H5G_STORAGE_TYPE_COMPACT_F: Compact storage
                                         !   H5G_STORAGE_TYPE_DENSE_F: Indexed storage
                                         !   H5G_STORAGE_TYPE_SYMBOL_TABLE_F: Symbol tables, the original HDF5 structure
    INTEGER, INTENT(OUT) :: nlinks      ! Number of links in group
    INTEGER, INTENT(OUT) :: max_corder  ! Current maximum creation order value for group
    INTEGER, INTENT(OUT) :: hdferr      ! Error code:
                                        ! 0 on success and -1 on failure
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: lapl_id ! Link access property list
    LOGICAL, INTENT(OUT), OPTIONAL :: mounted       ! Whether group has a file mounted on it
!*****
    INTEGER :: mounted_c
    INTEGER(HID_T) :: lapl_id_default
    INTEGER(SIZE_T) :: group_name_len ! length of group name

    INTERFACE
       INTEGER FUNCTION h5gget_info_by_idx_c(loc_id, group_name, group_name_len, index_type, order, n, lapl_id_default, &
            storage_type, nlinks, max_corder, mounted_c)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_INFO_BY_IDX_C'::h5gget_info_by_idx_c
         !DEC$ENDIF
         
         !DEC$ATTRIBUTES reference :: group_name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: group_name
         INTEGER, INTENT(IN) :: index_type
         INTEGER, INTENT(IN) :: order
         INTEGER(HSIZE_T), INTENT(IN) :: n
         INTEGER(HID_T) :: lapl_id_default
         INTEGER, INTENT(OUT) :: storage_type
         INTEGER, INTENT(OUT) :: nlinks
         INTEGER, INTENT(OUT) :: max_corder

         INTEGER(SIZE_T) :: group_name_len
         INTEGER :: mounted_c

       END FUNCTION h5gget_info_by_idx_c
    END INTERFACE

    group_name_len = LEN(group_name)

    lapl_id_default = H5P_DEFAULT_F
    IF(PRESENT(lapl_id)) lapl_id_default = lapl_id

    hdferr = h5gget_info_by_idx_c(loc_id, group_name, group_name_len, &
         index_type, order, n, lapl_id_default, &
         storage_type, nlinks, max_corder, mounted_c)

    IF(PRESENT(mounted))THEN
       IF(mounted_c.EQ.0) THEN
          mounted = .FALSE.
       ELSE
          mounted = .TRUE.
       ENDIF
    ENDIF

  END SUBROUTINE h5gget_info_by_idx_f
!
!****s* H5G/h5gget_info_by_name_f
!
! NAME
!  h5gget_info_by_name_f
!
! PURPOSE
!  Retrieves information about a group.
!
! INPUTS
!  loc_id 	 - File or group identifier
!  group_name 	 - Name of group containing group for which information is to be retrieved
!
! OUTPUTS
!
!  storage_type  - Type of storage for links in group
!                    H5G_STORAGE_TYPE_COMPACT: Compact storage
!                    H5G_STORAGE_TYPE_DENSE: Indexed storage
!                    H5G_STORAGE_TYPE_SYMBOL_TABLE: Symbol tables, the original HDF5 structure
!  nlinks 	 - Number of links in group
!  max_corder 	 - Current maximum creation order value for group
!  hdferr 	 - Returns 0 if successful and -1 if fails
! OPTIONAL PARAMETERS
!  lapl_id 	 - Link access property list
!  mounted 	 - Whether group has a file mounted on it
!
! NOTES
!  In C the output is defined as a structure: H5G_info_t
!
! AUTHOR
!  M. Scot Breitenfeld
!  February 18, 2008
!
! HISTORY
!  Added 'mounted' paramater
!  M. Scot Breitenfeld
!  July 16, 2008
! SOURCE
  SUBROUTINE h5gget_info_by_name_f(loc_id, group_name, &
       storage_type, nlinks, max_corder, hdferr, lapl_id, mounted)
    IMPLICIT NONE
    INTEGER(HID_T), INTENT(IN) :: loc_id       ! File or group identifier
    CHARACTER(LEN=*), INTENT(IN) :: group_name ! Name of group containing group for which information is to be retrieved

    INTEGER, INTENT(OUT) :: storage_type  ! Type of storage for links in group:
                                          ! H5G_STORAGE_TYPE_COMPACT_F: Compact storage
                                          ! H5G_STORAGE_TYPE_DENSE_F: Indexed storage
                                          ! H5G_STORAGE_TYPE_SYMBOL_TABLE_F: Symbol tables, the original HDF5 structure
    INTEGER, INTENT(OUT) :: nlinks        ! Number of links in group
    INTEGER, INTENT(OUT) :: max_corder    ! Current maximum creation order value for group
    INTEGER, INTENT(OUT) :: hdferr        ! Error code:
                                          !   0 on success and -1 on failure
    INTEGER(HID_T), OPTIONAL, INTENT(IN) :: lapl_id ! Link access property list
    LOGICAL, INTENT(OUT), OPTIONAL :: mounted       ! Whether group has a file mounted on it
!*****
    INTEGER :: mounted_c
    INTEGER(HID_T) :: lapl_id_default
    INTEGER(SIZE_T) :: group_name_len ! length of group name

    INTERFACE
       INTEGER FUNCTION h5gget_info_by_name_c(loc_id, group_name, group_name_len, lapl_id_default, &
            storage_type, nlinks, max_corder, mounted_c)
         USE H5GLOBAL
         !DEC$IF DEFINED(HDF5F90_WINDOWS)
         !DEC$ATTRIBUTES C,reference,decorate,alias:'H5GGET_INFO_BY_NAME_C'::h5gget_info_by_name_c
         !DEC$ENDIF
         !DEC$ATTRIBUTES reference :: group_name
         INTEGER(HID_T), INTENT(IN) :: loc_id
         CHARACTER(LEN=*), INTENT(IN) :: group_name
         INTEGER(HID_T), INTENT(IN) :: lapl_id_default
         INTEGER, INTENT(OUT) :: storage_type
         INTEGER, INTENT(OUT) :: nlinks
         INTEGER, INTENT(OUT) :: max_corder

         INTEGER(SIZE_T) :: group_name_len
         INTEGER :: mounted_c

       END FUNCTION h5gget_info_by_name_c
    END INTERFACE

    group_name_len = LEN(group_name)

    lapl_id_default = H5P_DEFAULT_F
    IF(PRESENT(lapl_id)) lapl_id_default = lapl_id

    hdferr = h5gget_info_by_name_c(loc_id, group_name, group_name_len, lapl_id_default, &
         storage_type, nlinks, max_corder, mounted_c)

    IF(PRESENT(mounted))THEN
       IF(mounted_c.EQ.0) THEN
          mounted = .FALSE.
       ELSE
          mounted = .TRUE.
       ENDIF
    ENDIF

  END SUBROUTINE h5gget_info_by_name_f

END MODULE H5G
