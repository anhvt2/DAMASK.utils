cmake_minimum_required (VERSION 3.12)
project (HDF5_EXAMPLES C)

#-----------------------------------------------------------------------------
# Define Sources
#-----------------------------------------------------------------------------
set (examples
    h5_crtdat
    h5_rdwt
    h5_crtatt
    h5_crtgrp
    h5_crtgrpar
    h5_crtgrpd
    h5_cmprss
    h5_extend
    h5_subset
    h5_write
    h5_read
    h5_extend_write
    h5_chunk_read
    h5_compound
    h5_group
    h5_select
    h5_attribute
    h5_mount
    h5_reference
    h5_drivers
    h5_ref2reg
    h5_extlink
    h5_elink_unix2win
    h5_shared_mesg
)

foreach (example ${examples})
  add_executable (${example} ${HDF5_EXAMPLES_SOURCE_DIR}/${example}.c)
  target_include_directories (${example} PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT BUILD_SHARED_LIBS)
    TARGET_C_PROPERTIES (${example} STATIC)
    target_link_libraries (${example} PRIVATE ${HDF5_LIB_TARGET})
  else ()
    TARGET_C_PROPERTIES (${example} SHARED)
    target_link_libraries (${example} PRIVATE ${HDF5_LIBSH_TARGET})
  endif ()
  set_target_properties (${example} PROPERTIES FOLDER examples)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_EXAMPLES_${example}_FORMAT ${example})
  endif ()
endforeach ()

if (H5_HAVE_PARALLEL)
  add_executable (ph5example ${HDF5_EXAMPLES_SOURCE_DIR}/ph5example.c)
  target_include_directories (ph5example PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT BUILD_SHARED_LIBS)
    TARGET_C_PROPERTIES (ph5example STATIC)
    target_link_libraries (ph5example PRIVATE ${HDF5_LIB_TARGET} ${MPI_C_LIBRARIES})
  else ()
    TARGET_C_PROPERTIES (ph5example SHARED)
    target_link_libraries (ph5example PRIVATE ${HDF5_LIBSH_TARGET} ${MPI_C_LIBRARIES})
  endif ()
  set_target_properties (ph5example PROPERTIES FOLDER examples)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_EXAMPLES_ph5example_FORMAT ph5example)
  endif ()
endif ()

if (BUILD_TESTING AND HDF5_TEST_EXAMPLES)
  include (CMakeTests.cmake)
endif ()
