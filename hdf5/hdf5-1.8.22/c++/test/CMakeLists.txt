cmake_minimum_required (VERSION 3.12)
project (HDF5_CPP_TEST CXX)

# --------------------------------------------------------------------
# Notes: When creating unit test executables they should be prefixed
# with "cpp_". This allows for easier filtering of the test suite when
# using ctest. An example would be
#  ctest -R cpp_
# which would only run the C++ based unit tests.
# --------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Define Sources
#-----------------------------------------------------------------------------
set (CPP_TEST_SOURCES
    ${HDF5_CPP_TEST_SOURCE_DIR}/testhdf5.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tarray.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tattr.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tcompound.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tdspl.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tfile.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tfilter.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/th5s.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tlinks.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tobject.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/trefer.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/ttypes.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/tvlstr.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/dsets.cpp
    ${HDF5_CPP_TEST_SOURCE_DIR}/h5cpputil.cpp
)

#-----------------------------------------------------------------------------
# Generate the H5srcdir_str.h file containing user settings needed by compilation
#-----------------------------------------------------------------------------
set (srcdir ${CMAKE_CURRENT_SOURCE_DIR})
configure_file (${HDF5_CPP_TEST_SOURCE_DIR}/H5srcdir_str.h.in H5srcdir_str.h  @ONLY)

add_executable (cpp_testhdf5 ${CPP_TEST_SOURCES} ${HDF5_CPP_TEST_SOURCE_DIR}/h5cpputil.h)
target_include_directories (cpp_testhdf5 PRIVATE "${HDF5_SRC_DIR};${HDF5_SRC_BINARY_DIR};${HDF5_TEST_BINARY_DIR};$<$<BOOL:${HDF5_ENABLE_PARALLEL}>:${MPI_C_INCLUDE_DIRS}>")
target_compile_options(cpp_testhdf5 PRIVATE "${HDF5_CMAKE_CXX_FLAGS}")
target_compile_definitions(cpp_testhdf5
    PRIVATE $<$<BOOL:${HDF5_ENABLE_PARALLEL}>:MPICH_SKIP_MPICXX;MPICH_IGNORE_CXX_SEEK># Parallel/MPI, prevent spurious cpp/cxx warnings
)
if (NOT BUILD_SHARED_LIBS)
  TARGET_C_PROPERTIES (cpp_testhdf5 STATIC)
  target_link_libraries (cpp_testhdf5 PRIVATE ${HDF5_CPP_LIB_TARGET} ${HDF5_LIB_TARGET} ${HDF5_TEST_LIB_TARGET})
else ()
  TARGET_C_PROPERTIES (cpp_testhdf5 SHARED)
  target_link_libraries (cpp_testhdf5 PRIVATE ${HDF5_CPP_LIBSH_TARGET} ${HDF5_LIBSH_TARGET} ${HDF5_TEST_LIBSH_TARGET})
  if (MINGW AND HDF5_MINGW_STATIC_GCC_LIBS)
    target_link_options (${HDF5_CPP_LIBSH_TARGET}
        PRIVATE -static-libgcc -static-libstdc++
    )
  endif ()
endif ()
set_target_properties (cpp_testhdf5 PROPERTIES FOLDER test/cpp)

#-----------------------------------------------------------------------------
# Add Target to clang-format
#-----------------------------------------------------------------------------
if (HDF5_ENABLE_FORMATTERS)
  clang_format (HDF5_CPP_TEST_cpp_testhdf5_FORMAT cpp_testhdf5)
endif ()

if (HDF5_TEST_CPP AND HDF5_TEST_SERIAL)
  include (CMakeTests.cmake)
endif ()
