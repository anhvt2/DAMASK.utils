#
# Copyright by The HDF Group.
# All rights reserved.
#
# This file is part of HDF5.  The full HDF5 copyright notice, including
# terms governing use, modification, and redistribution, is contained in
# the COPYING file, which can be found at the root of the source code
# distribution tree, or in https://www.hdfgroup.org/licenses.
# If you do not have access to either file, you may request a copy from
# help@hdfgroup.org.
#

#
# This file provides functions for HDF5 specific Fortran support.
#
#-------------------------------------------------------------------------------
enable_language (Fortran)

set (HDF_PREFIX "H5")
include (CheckFortranFunctionExists)

if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  include (CheckFortranSourceRuns)
  include (CheckFortranSourceCompiles)
endif ()

# Read source line beginning at the line matching Input:"START" and ending at the line matching Input:"END"
macro (READ_SOURCE SOURCE_START SOURCE_END RETURN_VAR)
  file (READ "${HDF5_SOURCE_DIR}/m4/aclocal_fc.f90" SOURCE_MASTER)
  string (REGEX MATCH "${SOURCE_START}[\\\t\\\n\\\r[].+]*${SOURCE_END}" SOURCE_CODE ${SOURCE_MASTER})
  set (RETURN_VAR "${SOURCE_CODE}")
endmacro ()

set (RUN_OUTPUT_PATH_DEFAULT ${CMAKE_BINARY_DIR})
if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  if (HDF5_REQUIRED_LIBRARIES)
    set (CMAKE_REQUIRED_LIBRARIES "${HDF5_REQUIRED_LIBRARIES}")
  endif ()
else ()
# The provided CMake Fortran macros don't provide a general compile/run function
# so this one is used.
#-----------------------------------------------------------------------------
macro (FORTRAN_RUN FUNCTION_NAME SOURCE_CODE RUN_RESULT_VAR1 COMPILE_RESULT_VAR1 RETURN_VAR)
    message (STATUS "Detecting Fortran ${FUNCTION_NAME}")
    file (WRITE
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testFortranCompiler1.f90
        "${SOURCE_CODE}"
    )
    TRY_RUN (RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testFortranCompiler1.f90
        LINK_LIBRARIES "${HDF5_REQUIRED_LIBRARIES}"
    )

    if (${COMPILE_RESULT_VAR})
      set(${RETURN_VAR} ${RUN_RESULT_VAR})
      if (${RUN_RESULT_VAR} MATCHES 0)
        message (STATUS "Testing Fortran ${FUNCTION_NAME} - OK")
        file (APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
            "Determining if the Fortran ${FUNCTION_NAME} exists passed\n"
        )
      else ()
        message (STATUS "Testing Fortran ${FUNCTION_NAME} - Fail")
        file (APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
            "Determining if the Fortran ${FUNCTION_NAME} exists failed: ${RUN_RESULT_VAR}\n"
        )
      endif ()
    else ()
        message (STATUS "Compiling Fortran ${FUNCTION_NAME} - Fail")
        file (APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
            "Determining if the Fortran ${FUNCTION_NAME} compiles failed: ${COMPILE_RESULT_VAR}\n"
        )
        set(${RETURN_VAR} ${COMPILE_RESULT_VAR})
    endif ()
endmacro ()
endif ()
#-----------------------------------------------------------------------------
#  Check to see C_LONG_DOUBLE is available

READ_SOURCE("PROGRAM PROG_FC_HAVE_C_LONG_DOUBLE" "END PROGRAM PROG_FC_HAVE_C_LONG_DOUBLE" SOURCE_CODE)
if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  check_fortran_source_compiles (${SOURCE_CODE} FORTRAN_HAVE_C_LONG_DOUBLE SRC_EXT f90)
else ()
  CHECK_FORTRAN_FEATURE(c_long_double "${SOURCE_CODE}" FORTRAN_HAVE_C_LONG_DOUBLE)
endif ()

if (${FORTRAN_HAVE_C_LONG_DOUBLE})
  set (${HDF_PREFIX}_FORTRAN_HAVE_C_LONG_DOUBLE 1)
else ()
  set (${HDF_PREFIX}_FORTRAN_HAVE_C_LONG_DOUBLE 0)
endif ()

# Check to see C_LONG_DOUBLE is different from C_DOUBLE

READ_SOURCE("MODULE type_mod" "END PROGRAM PROG_FC_C_LONG_DOUBLE_EQ_C_DOUBLE" SOURCE_CODE)
if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  check_fortran_source_compiles (${SOURCE_CODE} FORTRAN_C_LONG_DOUBLE_IS_UNIQUE SRC_EXT f90)
else ()
  CHECK_FORTRAN_FEATURE(c_long_double "${SOURCE_CODE}" FORTRAN_C_LONG_DOUBLE_IS_UNIQUE)
endif ()
if (${FORTRAN_C_LONG_DOUBLE_IS_UNIQUE})
  set (${HDF_PREFIX}_FORTRAN_C_LONG_DOUBLE_IS_UNIQUE 1)
else ()
  set (${HDF_PREFIX}_FORTRAN_C_LONG_DOUBLE_IS_UNIQUE 0)
endif ()

## Set the sizeof function for use later in the fortran tests
if (${HDF_PREFIX}_FORTRAN_HAVE_STORAGE_SIZE)
  set (FC_SIZEOF_A "STORAGE_SIZE(a, c_size_t)/STORAGE_SIZE(c_char_'a',c_size_t)")
  set (FC_SIZEOF_B "STORAGE_SIZE(b, c_size_t)/STORAGE_SIZE(c_char_'a',c_size_t)")
  set (FC_SIZEOF_C "STORAGE_SIZE(c, c_size_t)/STORAGE_SIZE(c_char_'a',c_size_t)")
elseif (${HDF_PREFIX}_FORTRAN_HAVE_C_SIZEOF)
  set (FC_SIZEOF_A "SIZEOF(a)")
  set (FC_SIZEOF_B "SIZEOF(b)")
  set (FC_SIZEOF_C "SIZEOF(c)")
else ()
  message (FATAL_ERROR "Fortran compiler requires either intrinsic functions SIZEOF or STORAGE_SIZE")
endif ()

#-----------------------------------------------------------------------------
# Determine the available KINDs for REALs and INTEGERs
#-----------------------------------------------------------------------------

#READ_SOURCE ("PROGRAM FC_AVAIL_KINDS" "END PROGRAM FC_AVAIL_KINDS" SOURCE_CODE)
set (PROG_SRC_CODE
  "
       PROGRAM FC_AVAIL_KINDS
          IMPLICIT NONE
          INTEGER :: ik, jk, k, max_decimal_prec
          INTEGER :: num_rkinds = 1, num_ikinds = 1
          INTEGER, DIMENSION(1:10) :: list_ikinds = -1
          INTEGER, DIMENSION(1:10) :: list_rkinds = -1

          OPEN(8, FILE='pac_fconftest.out', FORM='formatted')

          ! Find integer KINDs
          list_ikinds(num_ikinds)=SELECTED_INT_KIND(1)
          DO ik = 2, 36
            k = SELECTED_INT_KIND(ik)
            IF(k.LT.0) EXIT
            IF(k.GT.list_ikinds(num_ikinds))THEN
               num_ikinds = num_ikinds + 1
               list_ikinds(num_ikinds) = k
            ENDIF
          ENDDO

          DO k = 1, num_ikinds
             WRITE(8,'(I0)', ADVANCE='NO') list_ikinds(k)
             IF(k.NE.num_ikinds)THEN
                WRITE(8,'(A)',ADVANCE='NO') ','
             ELSE
                WRITE(8,'()')
             ENDIF
          ENDDO

          ! Find real KINDs
          list_rkinds(num_rkinds)=SELECTED_REAL_KIND(1)
          max_decimal_prec = 1

          prec: DO ik = 2, 36
             exp: DO jk = 1, 17000
                k = SELECTED_REAL_KIND(ik,jk)
                IF(k.LT.0) EXIT exp
                IF(k.GT.list_rkinds(num_rkinds))THEN
                   num_rkinds = num_rkinds + 1
                   list_rkinds(num_rkinds) = k
                ENDIF
                max_decimal_prec = ik
             ENDDO exp
          ENDDO prec

          DO k = 1, num_rkinds
             WRITE(8,'(I0)', ADVANCE='NO') list_rkinds(k)
             IF(k.NE.num_rkinds)THEN
                WRITE(8,'(A)',ADVANCE='NO') ','
             ELSE
                WRITE(8,'()')
             ENDIF
          ENDDO

         WRITE(8,'(I0)') max_decimal_prec
         WRITE(8,'(I0)') num_ikinds
         WRITE(8,'(I0)') num_rkinds
      END PROGRAM FC_AVAIL_KINDS
  "
)
if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  check_fortran_source_runs (${PROG_SRC_CODE} FC_AVAIL_KINDS_RESULT SRC_EXT f90)
else ()
FORTRAN_RUN ("REAL and INTEGER KINDs"
    "${PROG_SRC_CODE}"
    XX
    YY
    FC_AVAIL_KINDS_RESULT
)
endif ()

# dnl The output from the above program will be:
# dnl    -- LINE 1 --  valid integer kinds (comma seperated list)
# dnl    -- LINE 2 --  valid real kinds (comma seperated list)
# dnl    -- LINE 3 --  max decimal precision for reals
# dnl    -- LINE 4 --  number of valid integer kinds
# dnl    -- LINE 5 --  number of valid real kinds

file (READ "${RUN_OUTPUT_PATH_DEFAULT}/pac_fconftest.out" PROG_OUTPUT)
# Convert the string to a list of strings by replacing the carriage return with a semicolon
string (REGEX REPLACE "\n" ";" PROG_OUTPUT "${PROG_OUTPUT}")

list (GET PROG_OUTPUT 0 pac_validIntKinds)
list (GET PROG_OUTPUT 1 pac_validRealKinds)
list (GET PROG_OUTPUT 2 ${HDF_PREFIX}_PAC_FC_MAX_REAL_PRECISION)

# If the lists are empty then something went wrong.
if (NOT pac_validIntKinds)
    message (FATAL_ERROR "Failed to find available INTEGER KINDs for Fortran")
endif ()
if (NOT pac_validRealKinds)
    message (FATAL_ERROR "Failed to find available REAL KINDs for Fortran")
endif ()
if (NOT ${HDF_PREFIX}_PAC_FC_MAX_REAL_PRECISION)
    message (FATAL_ERROR "No output from Fortran decimal precision program")
endif ()

set (PAC_FC_ALL_INTEGER_KINDS "\{${pac_validIntKinds}\}")
set (PAC_FC_ALL_REAL_KINDS "\{${pac_validRealKinds}\}")

list (GET PROG_OUTPUT 3 NUM_IKIND)
list (GET PROG_OUTPUT 4 NUM_RKIND)

set (PAC_FORTRAN_NUM_INTEGER_KINDS "${NUM_IKIND}")

set (${HDF_PREFIX}_H5CONFIG_F_NUM_IKIND "INTEGER, PARAMETER :: num_ikinds = ${NUM_IKIND}")
set (${HDF_PREFIX}_H5CONFIG_F_IKIND "INTEGER, DIMENSION(1:num_ikinds) :: ikind = (/${pac_validIntKinds}/)")

message (STATUS "....NUMBER OF INTEGER KINDS FOUND ${PAC_FORTRAN_NUM_INTEGER_KINDS}")
message (STATUS "....REAL KINDS FOUND ${PAC_FC_ALL_REAL_KINDS}")
message (STATUS "....INTEGER KINDS FOUND ${PAC_FC_ALL_INTEGER_KINDS}")
message (STATUS "....MAX DECIMAL PRECISION ${${HDF_PREFIX}_PAC_FC_MAX_REAL_PRECISION}")

#-----------------------------------------------------------------------------
# Determine the available KINDs for REALs and INTEGERs
#-----------------------------------------------------------------------------
# **********
# INTEGERS
# **********
string (REGEX REPLACE "," ";" VAR "${pac_validIntKinds}")

foreach (KIND ${VAR})
  set (PROG_SRC_${KIND}
  "
       PROGRAM main
          USE ISO_C_BINDING
          IMPLICIT NONE
          INTEGER (KIND=${KIND}) a
          OPEN(8,FILE='pac_validIntKinds.out',FORM='formatted')
          WRITE(8,'(I0)') ${FC_SIZEOF_A}
          CLOSE(8)
       END
   "
  )
  if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
    check_fortran_source_runs (${PROG_SRC_${KIND}} VALIDINTKINDS_RESULT_${KIND} SRC_EXT f90)
  else ()
    FORTRAN_RUN("INTEGER KIND SIZEOF" ${PROG_SRC_${KIND}} XX YY VALIDINTKINDS_RESULT_${KIND})
  endif ()
  file (READ "${RUN_OUTPUT_PATH_DEFAULT}/pac_validIntKinds.out" PROG_OUTPUT1)
  string (REGEX REPLACE "\n" "" PROG_OUTPUT1 "${PROG_OUTPUT1}")
  set (pack_int_sizeof "${pack_int_sizeof} ${PROG_OUTPUT1},")
endforeach ()

if (pack_int_sizeof STREQUAL "")
   message (FATAL_ERROR "Failed to find available INTEGER KINDs for Fortran")
endif ()

string (STRIP ${pack_int_sizeof} pack_int_sizeof)

#Remove trailing comma
string (REGEX REPLACE ",$" "" pack_int_sizeof "${pack_int_sizeof}")
#Remove spaces
string (REGEX REPLACE " " "" pack_int_sizeof "${pack_int_sizeof}")

set (PAC_FC_ALL_INTEGER_KINDS_SIZEOF "\{${pack_int_sizeof}\}")

message (STATUS "....FOUND SIZEOF for INTEGER KINDs ${PAC_FC_ALL_INTEGER_KINDS_SIZEOF}")
# **********
# REALS
# **********
string (REGEX REPLACE "," ";" VAR "${pac_validRealKinds}")

#find the maximum kind of the real
list (LENGTH VAR LEN_VAR)
math (EXPR _LEN "${LEN_VAR}-1")
list (GET VAR ${_LEN} max_real_fortran_kind)

foreach (KIND ${VAR} )
  set (PROG_SRC2_${KIND}
  "
       PROGRAM main
          USE ISO_C_BINDING
          IMPLICIT NONE
          REAL (KIND=${KIND}) a
          OPEN(8,FILE='pac_validRealKinds.out',FORM='formatted')
          WRITE(8,'(I0)') ${FC_SIZEOF_A}
          CLOSE(8)
       END
  "
  )
  if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
    check_fortran_source_runs (${PROG_SRC2_${KIND}} VALIDREALKINDS_RESULT_${KIND} SRC_EXT f90)
  else ()
    FORTRAN_RUN ("REAL KIND SIZEOF" ${PROG_SRC2_${KIND}} XX YY VALIDREALKINDS_RESULT_${KIND})
  endif ()
  file (READ "${RUN_OUTPUT_PATH_DEFAULT}/pac_validRealKinds.out" PROG_OUTPUT1)
  string (REGEX REPLACE "\n" "" PROG_OUTPUT1 "${PROG_OUTPUT1}")
  set (pack_real_sizeof "${pack_real_sizeof} ${PROG_OUTPUT1},")
endforeach ()

if (pack_real_sizeof STREQUAL "")
   message (FATAL_ERROR "Failed to find available REAL KINDs for Fortran")
endif ()

string(STRIP ${pack_real_sizeof} pack_real_sizeof)

#Remove trailing comma
string (REGEX REPLACE ",$" "" pack_real_sizeof "${pack_real_sizeof}")
#Remove spaces
string (REGEX REPLACE " " "" pack_real_sizeof "${pack_real_sizeof}")

set (${HDF_PREFIX}_H5CONFIG_F_RKIND_SIZEOF "INTEGER, DIMENSION(1:num_rkinds) :: rkind_sizeof = (/${pack_real_sizeof}/)")

message (STATUS "....FOUND SIZEOF for REAL KINDs \{${pack_real_sizeof}\}")

set (PAC_FC_ALL_REAL_KINDS_SIZEOF "\{${pack_real_sizeof}\}")

#find the maximum kind of the real
string (REGEX REPLACE "," ";" VAR "${pack_real_sizeof}")
list (LENGTH VAR LEN_VAR)
math (EXPR _LEN "${LEN_VAR}-1")
list (GET VAR ${_LEN} max_real_fortran_sizeof)

#-----------------------------------------------------------------------------
# Find sizeof of native kinds
#-----------------------------------------------------------------------------
set (PROG_SRC3
  "
       PROGRAM main
          USE ISO_C_BINDING
          IMPLICIT NONE
          INTEGER a
          REAL b
          DOUBLE PRECISION c
          OPEN(8,FILE='pac_sizeof_native_kinds.out',FORM='formatted')
          WRITE(8,*) ${FC_SIZEOF_A}
          WRITE(8,*) kind(a)
          WRITE(8,*) ${FC_SIZEOF_B}
          WRITE(8,*) kind(b)
          WRITE(8,*) ${FC_SIZEOF_C}
          WRITE(8,*) kind(c)
          CLOSE(8)
       END
  "
)
if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  check_fortran_source_runs (${PROG_SRC3} PAC_SIZEOF_NATIVE_KINDS_RESULT SRC_EXT f90)
else ()
  FORTRAN_RUN ("SIZEOF NATIVE KINDs" ${PROG_SRC3} XX YY PAC_SIZEOF_NATIVE_KINDS_RESULT)
endif ()
file (READ "${RUN_OUTPUT_PATH_DEFAULT}/pac_sizeof_native_kinds.out" PROG_OUTPUT)
# dnl The output from the above program will be:
# dnl    -- LINE 1 --  sizeof INTEGER
# dnl    -- LINE 2 --  kind of INTEGER
# dnl    -- LINE 3 --  sizeof REAL
# dnl    -- LINE 4 --  kind of REAL
# dnl    -- LINE 5 --  sizeof DOUBLE PRECISION
# dnl    -- LINE 6 --  kind of DOUBLE PRECISION

# Convert the string to a list of strings by replacing the carriage return with a semicolon
string (REGEX REPLACE "\n" ";" PROG_OUTPUT "${PROG_OUTPUT}")

list (GET PROG_OUTPUT 0 PAC_FORTRAN_NATIVE_INTEGER_SIZEOF)
list (GET PROG_OUTPUT 1 PAC_FORTRAN_NATIVE_INTEGER_KIND)
list (GET PROG_OUTPUT 2 PAC_FORTRAN_NATIVE_REAL_SIZEOF)
list (GET PROG_OUTPUT 3 PAC_FORTRAN_NATIVE_REAL_KIND)
list (GET PROG_OUTPUT 4 PAC_FORTRAN_NATIVE_DOUBLE_SIZEOF)
list (GET PROG_OUTPUT 5 PAC_FORTRAN_NATIVE_DOUBLE_KIND)

if (NOT PAC_FORTRAN_NATIVE_INTEGER_SIZEOF)
   message (FATAL_ERROR "Failed to find SIZEOF NATIVE INTEGER KINDs for Fortran")
endif ()
if (NOT PAC_FORTRAN_NATIVE_REAL_SIZEOF)
   message (FATAL_ERROR "Failed to find SIZEOF NATIVE REAL KINDs for Fortran")
endif ()
if (NOT PAC_FORTRAN_NATIVE_DOUBLE_SIZEOF)
   message (FATAL_ERROR "Failed to find SIZEOF NATIVE DOUBLE KINDs for Fortran")
endif ()
if (NOT PAC_FORTRAN_NATIVE_INTEGER_KIND)
   message (FATAL_ERROR "Failed to find KIND of NATIVE INTEGER for Fortran")
endif ()
if (NOT PAC_FORTRAN_NATIVE_REAL_KIND)
   message (FATAL_ERROR "Failed to find KIND of NATIVE REAL for Fortran")
endif ()
if (NOT PAC_FORTRAN_NATIVE_DOUBLE_KIND)
   message (FATAL_ERROR "Failed to find KIND of NATIVE DOUBLE for Fortran")
endif ()


set (${HDF_PREFIX}_FORTRAN_SIZEOF_LONG_DOUBLE ${${HDF_PREFIX}_SIZEOF_LONG_DOUBLE})

set (${HDF_PREFIX}_H5CONFIG_F_NUM_RKIND "INTEGER, PARAMETER :: num_rkinds = ${NUM_RKIND}")

string (REGEX REPLACE "{" "" OUT_VAR ${PAC_FC_ALL_REAL_KINDS})
string (REGEX REPLACE "}" "" OUT_VAR ${OUT_VAR})
set (${HDF_PREFIX}_H5CONFIG_F_RKIND "INTEGER, DIMENSION(1:num_rkinds) :: rkind = (/${OUT_VAR}/)")

string (REGEX REPLACE "{" "" OUT_VAR ${PAC_FC_ALL_REAL_KINDS_SIZEOF})
string (REGEX REPLACE "}" "" OUT_VAR ${OUT_VAR})
set (${HDF_PREFIX}_H5CONFIG_F_RKIND_SIZEOF "INTEGER, DIMENSION(1:num_rkinds) :: rkind_sizeof = (/${OUT_VAR}/)")

ENABLE_LANGUAGE (C)

if (NOT CMAKE_VERSION VERSION_LESS "3.14.0")
  include (CheckCSourceRuns)
else ()
#-----------------------------------------------------------------------------
# The provided CMake C macros don't provide a general compile/run function
# so this one is used.
#-----------------------------------------------------------------------------
macro (C_RUN FUNCTION_NAME SOURCE_CODE RETURN_VAR)
    message (STATUS "Detecting C ${FUNCTION_NAME}")
    if (HDF5_REQUIRED_LIBRARIES)
      set (CHECK_FUNCTION_EXISTS_ADD_LIBRARIES
          "-DLINK_LIBRARIES:STRING=${HDF5_REQUIRED_LIBRARIES}")
    else ()
      set (CHECK_FUNCTION_EXISTS_ADD_LIBRARIES)
    endif ()
    file (WRITE
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCCompiler1.c
        ${SOURCE_CODE}
    )
    TRY_RUN (RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testCCompiler1.c
        CMAKE_FLAGS "${CHECK_FUNCTION_EXISTS_ADD_LIBRARIES}"
        RUN_OUTPUT_VARIABLE OUTPUT_VAR
    )

    set (${RETURN_VAR} ${OUTPUT_VAR})

    #message (STATUS "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    #message (STATUS "Test COMPILE_RESULT_VAR ${COMPILE_RESULT_VAR} ")
    #message (STATUS "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
    #message (STATUS "Test RUN_RESULT_VAR ${RUN_RESULT_VAR} ")
    #message (STATUS "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")

    if (${COMPILE_RESULT_VAR})
      if (${RUN_RESULT_VAR} MATCHES 1)
        set (${RUN_RESULT_VAR} 1 CACHE INTERNAL "Have C function ${FUNCTION_NAME}")
        message (STATUS "Testing C ${FUNCTION_NAME} - OK")
        file (APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
            "Determining if the C ${FUNCTION_NAME} exists passed with the following output:\n"
            "${OUTPUT_VAR}\n\n"
        )
      else ()
        message (STATUS "Testing C ${FUNCTION_NAME} - Fail")
        set (${RUN_RESULT_VAR} 0 CACHE INTERNAL "Have C function ${FUNCTION_NAME}")
        file (APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
            "Determining if the C ${FUNCTION_NAME} exists failed with the following output:\n"
            "${OUTPUT_VAR}\n\n")
      endif ()
    else ()
        message (FATAL_ERROR "Compilation of C ${FUNCTION_NAME} - Failed")
    endif ()
endmacro ()
endif ()


# Setting definition if there is a 16 byte fortran integer
string (FIND ${PAC_FC_ALL_INTEGER_KINDS_SIZEOF} "16" pos)
if (${pos} EQUAL -1)
  set (${HDF_PREFIX}_HAVE_Fortran_INTEGER_SIZEOF_16 0)
else ()
  set (${HDF_PREFIX}_HAVE_Fortran_INTEGER_SIZEOF_16 1)
endif ()
