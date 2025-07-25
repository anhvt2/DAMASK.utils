cmake_minimum_required (VERSION 3.12)
project (HDF5_TOOLS_MISC C)

# --------------------------------------------------------------------
# Add the misc executables
# --------------------------------------------------------------------
#-- Misc Executables
if (NOT ONLY_SHARED_LIBS)
  add_executable (h5debug ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5debug.c)
  target_include_directories (h5debug PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5debug PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  TARGET_C_PROPERTIES (h5debug STATIC)
  target_link_libraries (h5debug PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5debug PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5debug")

  add_executable (h5repart ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5repart.c)
  target_include_directories (h5repart PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5repart PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  TARGET_C_PROPERTIES (h5repart STATIC)
  target_link_libraries (h5repart PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5repart PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5repart")

  add_executable (h5mkgrp ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5mkgrp.c)
  target_include_directories (h5mkgrp PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5mkgrp PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  TARGET_C_PROPERTIES (h5mkgrp STATIC)
  target_link_libraries (h5mkgrp PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5mkgrp PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5mkgrp")

  set (H5_DEP_EXECUTABLES
      h5debug
      h5repart
      h5mkgrp
  )
endif ()
if (BUILD_SHARED_LIBS)
  add_executable (h5debug-shared ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5debug.c)
  target_include_directories (h5debug-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5debug-shared SHARED)
  target_compile_options(h5debug-shared PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  target_link_libraries (h5debug-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5debug-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5debug-shared")

  add_executable (h5repart-shared ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5repart.c)
  target_include_directories (h5repart-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5repart-shared PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  TARGET_C_PROPERTIES (h5repart-shared SHARED)
  target_link_libraries (h5repart-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5repart-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5repart-shared")

  add_executable (h5mkgrp-shared ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5mkgrp.c)
  target_include_directories (h5mkgrp-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5mkgrp-shared PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  TARGET_C_PROPERTIES (h5mkgrp-shared SHARED)
  target_link_libraries (h5mkgrp-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5mkgrp-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5mkgrp-shared")

  set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES}
      h5debug-shared
      h5repart-shared
      h5mkgrp-shared
  )
endif ()

#-----------------------------------------------------------------------------
# Add Target to clang-format
#-----------------------------------------------------------------------------
if (HDF5_ENABLE_FORMATTERS)
  if (NOT ONLY_SHARED_LIBS)
    clang_format (HDF5_H5DEBUG_SRC_FORMAT h5debug)
    clang_format (HDF5_H5REPART_SRC_FORMAT h5repart)
    clang_format (HDF5_H5MKGRP_SRC_FORMAT h5mkgrp)
  else ()
    clang_format (HDF5_H5DEBUG_SRC_FORMAT h5debug-shared)
    clang_format (HDF5_H5REPART_SRC_FORMAT h5repart-shared)
    clang_format (HDF5_H5MKGRP_SRC_FORMAT h5mkgrp-shared)
  endif ()
endif ()

if (BUILD_TESTING)
  # --------------------------------------------------------------------
  # Add the misc test executables
  # --------------------------------------------------------------------
  if (HDF5_BUILD_GENERATORS AND NOT ONLY_SHARED_LIBS)
    add_executable (h5repart_gentest ${HDF5_TOOLS_MISC_SOURCE_DIR}/h5repart_gentest.c)
    target_include_directories(h5repart_gentest PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
    TARGET_C_PROPERTIES (h5repart_gentest STATIC)
    target_link_libraries (h5repart_gentest PRIVATE ${HDF5_LIB_TARGET} ${HDF5_TOOLS_LIB_TARGET})
    set_target_properties (h5repart_gentest PROPERTIES FOLDER generator/tools)
    #add_test (NAME h5repart_gentest COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:h5repart_gentest>)

    #-----------------------------------------------------------------------------
    # Add Target to clang-format
    #-----------------------------------------------------------------------------
    if (HDF5_ENABLE_FORMATTERS)
      clang_format (HDF5_TOOLS_TEST_MISC_h5repart_gentest_FORMAT h5repart_gentest)
    endif ()
  endif ()

  add_executable (h5repart_test ${HDF5_TOOLS_MISC_SOURCE_DIR}/repart_test.c)
  target_include_directories (h5repart_test PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  if (NOT ONLY_SHARED_LIBS)
    TARGET_C_PROPERTIES (h5repart_test STATIC)
    target_link_libraries (h5repart_test PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  else ()
    TARGET_C_PROPERTIES (h5repart_test SHARED)
    target_link_libraries (h5repart_test PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  endif ()
  set_target_properties (h5repart_test PROPERTIES FOLDER tools)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_TOOLS_TEST_MISC_h5repart_test_FORMAT h5repart_test)
  endif ()

  if (HDF5_TEST_SERIAL)
    include (CMakeTestsRepart.cmake)
    include (CMakeTestsMkgrp.cmake)
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
