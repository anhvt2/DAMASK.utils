# Add this directory for internal users.
include_directories(.)
include_directories ("${PROJECT_BINARY_DIR}/include")
# Find sources.
file(GLOB metis_sources *.c)
# Build libmetis.
add_library(metis ${METIS_LIBRARY_TYPE} ${GKlib_sources} ${metis_sources})
target_link_libraries(metis ${MATH_LIB})

if(METIS_INSTALL)
  install(TARGETS metis
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
  install(FILES gklib_defs.h DESTINATION include)
  install(FILES gklib_rename.h DESTINATION include)
endif()
