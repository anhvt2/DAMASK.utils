cmake_minimum_required (VERSION 3.12)
project (HDF5_TOOLS_H5DIFF C)

# --------------------------------------------------------------------
# Add the h5diff executables
# --------------------------------------------------------------------
if (NOT ONLY_SHARED_LIBS)
  add_executable (h5diff
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.c
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_main.c
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.h
  )
  target_include_directories (h5diff PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5diff PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  #target_compile_definitions (h5diff PRIVATE H5_TOOLS_DEBUG)
  TARGET_C_PROPERTIES (h5diff STATIC)
  target_link_libraries (h5diff PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET})
  set_target_properties (h5diff PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5diff")

  set (H5_DEP_EXECUTABLES h5diff)
endif ()
if (BUILD_SHARED_LIBS)
  add_executable (h5diff-shared
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.c
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_main.c
      ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.h
  )
  target_include_directories (h5diff-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  target_compile_options(h5diff-shared PRIVATE "${HDF5_CMAKE_C_FLAGS}")
  #target_compile_definitions (h5diff-shared PRIVATE H5_TOOLS_DEBUG)
  TARGET_C_PROPERTIES (h5diff-shared SHARED)
  target_link_libraries (h5diff-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
  set_target_properties (h5diff-shared PROPERTIES FOLDER tools)
  set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};h5diff-shared")

  set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES} h5diff-shared)
endif ()

#-----------------------------------------------------------------------------
# Add Target to clang-format
#-----------------------------------------------------------------------------
if (HDF5_ENABLE_FORMATTERS)
  if (NOT ONLY_SHARED_LIBS)
    clang_format (HDF5_H5DIFF_SRC_FORMAT h5diff)
  else ()
    clang_format (HDF5_H5DIFF_SRC_FORMAT h5diff-shared)
  endif ()
endif ()

if (H5_HAVE_PARALLEL)
  if (NOT ONLY_SHARED_LIBS)
    add_executable (ph5diff
        ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.c
        ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/ph5diff_main.c
    )
    target_include_directories (ph5diff PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
    target_compile_options(ph5diff PRIVATE "${HDF5_CMAKE_C_FLAGS}")
    TARGET_C_PROPERTIES (ph5diff STATIC)
    target_link_libraries (ph5diff PRIVATE ${HDF5_TOOLS_LIB_TARGET} ${HDF5_LIB_TARGET} "$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_LIBRARIES}>")
    set_target_properties (ph5diff PROPERTIES FOLDER tools)
    set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};ph5diff")
    set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES} ph5diff)
  endif ()
  if (BUILD_SHARED_LIBS)
    add_executable (ph5diff-shared
        ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diff_common.c
        ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/ph5diff_main.c
    )
    target_include_directories (ph5diff-shared PRIVATE "${HDF5_TOOLS_DIR}/lib;${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
    target_compile_options(ph5diff-shared PRIVATE "${HDF5_CMAKE_C_FLAGS}")
    TARGET_C_PROPERTIES (ph5diff-shared SHARED)
    target_link_libraries (ph5diff-shared PRIVATE ${HDF5_TOOLS_LIBSH_TARGET} ${HDF5_LIBSH_TARGET} "$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_LIBRARIES}>")
    set_target_properties (ph5diff-shared PROPERTIES FOLDER tools)
    set_global_variable (HDF5_UTILS_TO_EXPORT "${HDF5_UTILS_TO_EXPORT};ph5diff-shared")
    set (H5_DEP_EXECUTABLES ${H5_DEP_EXECUTABLES} ph5diff-shared)
  endif ()
endif ()

if (HDF5_BUILD_GENERATORS AND NOT ONLY_SHARED_LIBS)
  add_executable (h5diffgentest ${HDF5_TOOLS_H5DIFF_SOURCE_DIR}/h5diffgentest.c)
  target_include_directories (h5diffgentest PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (h5diffgentest STATIC)
  target_link_libraries (h5diffgentest PRIVATE ${HDF5_LIB_TARGET})
  set_target_properties (h5diffgentest PROPERTIES FOLDER generator/tools)

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_TOOLS_TEST_H5DIFF_FORMAT h5diffgentest)
  endif ()

  #add_test (NAME h5diffgentest COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:h5diffgentest>)
endif ()

#-----------------------------------------------------------------------------
# If plugin library tests can be tested
#-----------------------------------------------------------------------------
if (BUILD_SHARED_LIBS)
  set (H5DIFF_TOOL_PLUGIN_LIB_CORENAME         "dynlibdiff")
  set (H5DIFF_TOOL_PLUGIN_LIB_NAME             "${HDF5_EXTERNAL_LIB_PREFIX}${H5DIFF_TOOL_PLUGIN_LIB_CORENAME}")
  set (H5DIFF_TOOL_PLUGIN_LIB_TARGET           ${H5DIFF_TOOL_PLUGIN_LIB_CORENAME})

  add_library (${H5DIFF_TOOL_PLUGIN_LIB_TARGET} SHARED dynlib_diff.c)
  target_include_directories (${H5DIFF_TOOL_PLUGIN_LIB_TARGET} PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
  TARGET_C_PROPERTIES (${H5DIFF_TOOL_PLUGIN_LIB_TARGET} SHARED)
  target_link_libraries (${H5DIFF_TOOL_PLUGIN_LIB_TARGET} PRIVATE ${HDF5_TEST_LIBSH_TARGET})
  H5_SET_LIB_OPTIONS (${H5DIFF_TOOL_PLUGIN_LIB_TARGET} ${H5DIFF_TOOL_PLUGIN_LIB_NAME} SHARED "LIB")

  #-----------------------------------------------------------------------------
  # Add Target to clang-format
  #-----------------------------------------------------------------------------
  if (HDF5_ENABLE_FORMATTERS)
    clang_format (HDF5_TOOLS_TEST_H5DIFF_PLUGIN_FORMAT ${H5DIFF_TOOL_PLUGIN_LIB_TARGET})
  endif ()

  # make plugins dir
  file (MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/plugins")
  #-----------------------------------------------------------------------------
  # Copy plugin library to a plugins folder
  #-----------------------------------------------------------------------------
  add_custom_command (
      TARGET     ${H5DIFF_TOOL_PLUGIN_LIB_TARGET}
      POST_BUILD
      COMMAND    ${CMAKE_COMMAND}
      ARGS       -E copy_if_different
          "$<TARGET_FILE:${H5DIFF_TOOL_PLUGIN_LIB_TARGET}>"
          "${CMAKE_BINARY_DIR}/plugins/$<TARGET_FILE_NAME:${H5DIFF_TOOL_PLUGIN_LIB_TARGET}>"
  )
endif ()

if (HDF5_TEST_TOOLS)
  include (CMakeTests.cmake)
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
