cmake_minimum_required (VERSION 3.12)
project (HDF5_TOOLS_H5IMPORT C)

# --------------------------------------------------------------------
# Add the h5import executables
# --------------------------------------------------------------------
if (NOT ONLY_SHARED_LIBS)
  add_executable (h5import ${HDF5_TOOLS_H5IMPORT_SOURCE_DIR}/h5import.c ${HDF5_TOOLS_H5IMPORT_SOURCE_DIR}/h5import.h)
  target_include_directories (h5import PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5import STATIC)
  target_link_libraries (h5import PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  #set_target_properties (h5import PROPERTIES COMPILE_DEFINITIONS H5DEBUGIMPORT)
  set_target_properties (h5import PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5import")

  set (H5_DEP_EXECUTABLES h5import)
endif ()

if (BUILD_SHARED_LIBS)
  add_executable (h5import-shared ${HDF5_TOOLS_H5IMPORT_SOURCE_DIR}/h5import.c ${HDF5_TOOLS_H5IMPORT_SOURCE_DIR}/h5import.h)
  target_include_directories (h5import-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5import-shared SHARED)
  target_link_libraries (h5import-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  #set_target_properties (h5import-shared PROPERTIES COMPILE_DEFINITIONS H5DEBUGIMPORT)
  set_target_properties (h5import-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5import-shared")

  set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES} h5import-shared)
endif ()

#-----------------------------------------------------------------------------
# Add Target to clang-format
#-----------------------------------------------------------------------------
if (HDF5_ENABLE_FORMATTERS)
  if (NOT ONLY_SHARED_LIBS)
    clang_format (HDF5_H5IMPORT_SRC_FORMAT h5import)
  else ()
    clang_format (HDF5_H5IMPORT_SRC_FORMAT h5import-shared)
  endif ()
endif ()

if (BUILD_TESTING)
  # --------------------------------------------------------------------
  # Add the h5import executables
  # --------------------------------------------------------------------
  add_executable (h5importtest ${HDF5_TOOLS_H5IMPORT_SOURCE_DIR}/h5importtest.c)
  target_include_directories(h5importtest PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT ONLY_SHARED_LIBS)
    TARGET_C_PROPERTIES (h5importtest STATIC)
    target_link_libraries (h5importtest PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  else ()
    TARGET_C_PROPERTIES (h5importtest SHARED)
    target_link_libraries (h5importtest PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  endif ()
  set_target_properties (h5importtest PROPERTIES FOLDER tools)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_TOOLS_TEST_H5IMPORT_FORMAT h5importtest)
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
