cmake_minimum_required (VERSION 3.12)
project (HDF5_TOOLS_H5JAM C)

# --------------------------------------------------------------------
# Add the h5jam executables
# --------------------------------------------------------------------
if (NOT ONLY_SHARED_LIBS)
  add_executable (h5jam ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/h5jam.c)
  target_include_directories (h5jam PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5jam STATIC)
  target_link_libraries (h5jam PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5jam PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5jam")

  add_executable (h5unjam ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/h5unjam.c)
  target_include_directories (h5unjam PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5unjam STATIC)
  target_link_libraries (h5unjam PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5unjam PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5unjam")

  set (H5_DEP_EXECUTABLES
      h5jam
      h5unjam
  )
endif ()

if (BUILD_SHARED_LIBS)
  add_executable (h5jam-shared ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/h5jam.c)
  target_include_directories (h5jam-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5jam-shared SHARED)
  target_link_libraries (h5jam-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5jam-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5jam-shared")

  add_executable (h5unjam-shared ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/h5unjam.c)
  target_include_directories (h5unjam-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5unjam-shared SHARED)
  target_link_libraries (h5unjam-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5unjam-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5unjam-shared")

  set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES}
      h5jam-shared
      h5unjam-shared
  )
endif ()

#-----------------------------------------------------------------------------
# Add Target to clang-format
#-----------------------------------------------------------------------------
if (HDF5_ENABLE_FORMATTERS)
  if (NOT ONLY_SHARED_LIBS)
    clang_format (HDF5_H5JAM_SRC_FORMAT h5jam)
    clang_format (HDF5_H5UNJAM_SRC_FORMAT h5unjam)
  else ()
    clang_format (HDF5_H5JAM_SRC_FORMAT h5jam-shared)
    clang_format (HDF5_H5UNJAM_SRC_FORMAT h5unjam-shared)
  endif ()
endif ()

if (BUILD_TESTING)
  # --------------------------------------------------------------------
  # Add the h5jam test executables
  # --------------------------------------------------------------------
  if (HDF5_BUILD_GENERATORS AND NOT ONLY_SHARED_LIBS)
    add_executable (h5jamgentest ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/h5jamgentest.c)
    target_include_directories (h5jamgentest PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
    TARGET_C_PROPERTIES (h5jamgentest STATIC)
    target_link_libraries (h5jamgentest PRIVATE ${HDF5_LIB_TARGET})
    set_target_properties (h5jamgentest PROPERTIES FOLDER generator/tools)

    #-----------------------------------------------------------------------------
    # Add Target to clang-format
    #-----------------------------------------------------------------------------
    if (HDF5_ENABLE_FORMATTERS)
      clang_format (HDF5_TOOLS_TEST_H5JAM_FORMAT h5jamgentest)
    endif ()

    #add_test (NAME h5jamgentest COMMAND $<TARGET_FILE:h5jamgentest>)
  endif ()

  add_executable (getub ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/getub.c)
  target_include_directories (getub PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT ONLY_SHARED_LIBS)
    TARGET_C_PROPERTIES (getub STATIC)
    target_link_libraries (getub PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  else ()
    TARGET_C_PROPERTIES (getub SHARED)
    target_link_libraries (getub PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  endif ()
  set_target_properties (getub PROPERTIES FOLDER tools)

  add_executable (tellub ${HDF5_TOOLS_H5JAM_SOURCE_DIR}/tellub.c)
  target_include_directories (tellub PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT ONLY_SHARED_LIBS)
    TARGET_C_PROPERTIES (tellub STATIC)
    target_link_libraries (tellub PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  else ()
    TARGET_C_PROPERTIES (tellub SHARED)
    target_link_libraries (tellub PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  endif ()
  set_target_properties (tellub PROPERTIES FOLDER tools)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_TOOLS_TEST_H5JAM_getub_FORMAT getub)
    clang_format (HDF5_TOOLS_TEST_H5JAM_tellub_FORMAT tellub)
  endif ()

  if (HDF5_TEST_TOOLS AND HDF5_TEST_SERIAL)
    include (CMakeTests.cmake)
  endif ()

endif ()

##############################################################################
##############################################################################
###           I N S T A L L A T I O N                                      ###
##############################################################################
##############################################################################

#-----------------------------------------------------------------------------
# Rules for Installation of tools using make Install target
#-----------------------------------------------------------------------------
if (HDF5_EXPORTED_TARGETS)
  foreach (exec ${H5_DEP_EXECUTABLES})
    INSTALL_PROGRAM_PDB (${exec} ${HDF5_INSTALL_BIN_DIR} toolsapplications)
  endforeach ()

  install (
      TARGETS
          ${H5_DEP_EXECUTABLES}
      EXPORT
          ${HDF5_EXPORTED_TARGETS}
      RUNTIME DESTINATION ${HDF5_INSTALL_BIN_DIR} COMPONENT toolsapplications
  )
endif ()
