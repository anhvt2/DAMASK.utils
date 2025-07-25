cmake_minimum_required (VERSION 3.12)
project (HDF5_FORTRAN_TESTPAR C Fortran)

#-----------------------------------------------------------------------------
# Setup include Directories
#-----------------------------------------------------------------------------
set (TESTPAR_INCLUDES ${MPI_Fortran_INCLUDE_DIRS} ${HDF5_F90_BINARY_DIR} ${HDF5_F90_SRC_DIR}/src))
if (NOT ONLY_SHARED_LIBS)
  set (TESTPAR_INCLUDES ${TESTPAR_INCLUDES} ${CMAKE_Fortran_MODULE_DIRECTORY}/static)
else ()
  set (TESTPAR_INCLUDES ${TESTPAR_INCLUDES} ${CMAKE_Fortran_MODULE_DIRECTORY}/shared)
endif ()

#-----------------------------------------------------------------------------
# Add Tests
#-----------------------------------------------------------------------------

#-- Adding test for parallel_test
add_executable (parallel_test
    ptest.f90
    hyper.f90
    mdset.f90
)
target_include_directories (parallel_test
    PRIVATE ${TESTPAR_INCLUDES}
)
target_compile_options(parallel_test
    PRIVATE
        "${HDF5_CMAKE_Fortran_FLAGS}"
        $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:${WIN_COMPILE_FLAGS}>
)
if (NOT ONLY_SHARED_LIBS)
  target_link_libraries (parallel_test
      PRIVATE
          ${HDF5_F90_TEST_LIB_TARGET} ${HDF5_F90_LIB_TARGET} ${HDF5_LIB_TARGET} ${LINK_Fortran_LIBS}
          $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:"ws2_32.lib">
  )
  set_target_properties (parallel_test PROPERTIES
      FOLDER test/fortran
      LINKER_LANGUAGE Fortran
      Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/static
  )
else ()
  target_link_libraries (parallel_test
      PRIVATE
          ${HDF5_F90_TEST_LIBSH_TARGET} ${HDF5_F90_LIBSH_TARGET} ${HDF5_LIBSH_TARGET} ${LINK_Fortran_LIBS}
          $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:"ws2_32.lib">
  )
  set_target_properties (parallel_test PROPERTIES
      FOLDER test/fortran
      LINKER_LANGUAGE Fortran
      Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/shared
  )
endif ()

#set_property(TARGET parallel_test APPEND PROPERTY LINK_FLAGS $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:"-SUBSYSTEM:CONSOLE">)
#set_property(TARGET parallel_test APPEND PROPERTY LINK_FLAGS $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:${WIN_LINK_FLAGS}>)
if(MSVC)
  set_property(TARGET parallel_test PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE ${WIN_LINK_FLAGS}")
endif()

if (HDF5_TEST_FORTRAN AND HDF5_TEST_PARALLEL)
  include (CMakeTests.cmake)
endif ()
