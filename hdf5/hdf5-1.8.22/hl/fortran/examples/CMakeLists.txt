cmake_minimum_required (VERSION 3.12)
project (HDF5_HL_F90_EXAMPLES C Fortran)

set (examples
    exlite
    ex_ds1
)

foreach (example ${examples})
  add_executable (hl_f90_ex_${example} ${HDF5_HL_F90_EXAMPLES_SOURCE_DIR}/${example}.f90)
  target_compile_options(hl_f90_ex_${example}
      PRIVATE
          $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:${WIN_COMPILE_FLAGS}>
  )
#  set_property(TARGET hl_f90_ex_${example} APPEND PROPERTY LINK_FLAGS $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:"-SUBSYSTEM:CONSOLE">)
#  set_property(TARGET hl_f90_ex_${example} APPEND PROPERTY LINK_FLAGS $<$<STREQUAL:"x${CMAKE_Fortran_SIMULATE_ID}","xMSVC">:${WIN_LINK_FLAGS}>)
  if(MSVC)
    set_property(TARGET hl_f90_ex_${example} PROPERTY LINK_FLAGS "/SUBSYSTEM:CONSOLE ${WIN_LINK_FLAGS}")
  endif()
  if (BUILD_SHARED_LIBS AND NOT SKIP_HDF5_FORTRAN_SHARED)
    target_include_directories (hl_f90_ex_${example} PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}/shared;${HDF5_F90_BINARY_DIR};${HDF5_F90_SRC_DIR}/src")
    target_link_libraries (hl_f90_ex_${example} PRIVATE ${HDF5_HL_F90_LIBSH_TARGET} ${HDF5_F90_LIBSH_TARGET} ${HDF5_LIBSH_TARGET})
    set_target_properties (hl_f90_ex_${example} PROPERTIES
        LINKER_LANGUAGE Fortran
        FOLDER examples/hl/fortran
        Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/shared
    )
  else ()
    target_include_directories (hl_f90_ex_${example} PRIVATE "${CMAKE_Fortran_MODULE_DIRECTORY}/static;${HDF5_F90_BINARY_DIR};${HDF5_F90_SRC_DIR}/src")
    target_link_libraries (hl_f90_ex_${example} PRIVATE ${HDF5_HL_F90_LIB_TARGET} ${HDF5_F90_LIB_TARGET} ${HDF5_LIB_TARGET})
    set_target_properties (hl_f90_ex_${example} PROPERTIES
        LINKER_LANGUAGE Fortran
        FOLDER examples/hl/fortran
        Fortran_MODULE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/static
    )
  endif ()
endforeach ()

if (BUILD_TESTING AND HDF5_TEST_FORTRAN AND HDF5_TEST_EXAMPLES AND HDF5_TEST_SERIAL)
  include (CMakeTests.cmake)
endif ()
