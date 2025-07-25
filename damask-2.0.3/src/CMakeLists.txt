# special flags for some files
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    SET_SOURCE_FILES_PROPERTIES( "lattice.f90" PROPERTIES
                                COMPILE_FLAGS "-ffree-line-length-240")
    SET_SOURCE_FILES_PROPERTIES( "DAMASK_interface.f90" PROPERTIES
                                COMPILE_FLAGS "-ffree-line-length-164")
    # long lines for interaction matrix
endif()

# The dependency detection in CMake is not functioning for Fortran,
# hence we declare the dependencies from top to bottom in the following

add_library(C_ROUTINES OBJECT "C_routines.c")
set(OBJECTFILES $<TARGET_OBJECTS:C_ROUTINES>)

add_library(SYSTEM_ROUTINES OBJECT "system_routines.f90")
add_dependencies(SYSTEM_ROUTINES C_ROUTINES)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:SYSTEM_ROUTINES>)

add_library(PREC OBJECT "prec.f90")
list(APPEND OBJECTFILES $<TARGET_OBJECTS:PREC>)

add_library(ELEMENT OBJECT "element.f90")
add_dependencies(ELEMENT IO)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:ELEMENT>)

add_library(QUIT OBJECT "quit.f90")
add_dependencies(QUIT PREC)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:QUIT>)

add_library(DAMASK_INTERFACE OBJECT "DAMASK_interface.f90")
add_dependencies(DAMASK_INTERFACE QUIT SYSTEM_ROUTINES)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DAMASK_INTERFACE>)

add_library(IO OBJECT "IO.f90")
add_dependencies(IO DAMASK_INTERFACE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:IO>)

add_library(NUMERICS OBJECT "numerics.f90")
add_dependencies(NUMERICS IO)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:NUMERICS>)

add_library(DEBUG OBJECT "debug.f90")
add_dependencies(DEBUG IO)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DEBUG>)

add_library(DAMASK_CONFIG OBJECT "config.f90")
add_dependencies(DAMASK_CONFIG DEBUG)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DAMASK_CONFIG>)

add_library(HDF5_UTILITIES OBJECT "HDF5_utilities.f90")
add_dependencies(HDF5_UTILITIES DAMASK_CONFIG NUMERICS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:HDF5_UTILITIES>)

add_library(RESULTS OBJECT "results.f90")
add_dependencies(RESULTS HDF5_UTILITIES)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:RESULTS>)

add_library(FEsolving OBJECT "FEsolving.f90")
add_dependencies(FEsolving DEBUG)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:FEsolving>)

add_library(MATH OBJECT "math.f90")
add_dependencies(MATH NUMERICS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:MATH>)

add_library(QUATERNIONS OBJECT "quaternions.f90")
add_dependencies(QUATERNIONS MATH)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:QUATERNIONS>)

add_library(LAMBERT OBJECT "Lambert.f90")
add_dependencies(LAMBERT MATH)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:LAMBERT>)

add_library(ROTATIONS OBJECT "rotations.f90")
add_dependencies(ROTATIONS LAMBERT QUATERNIONS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:ROTATIONS>)

add_library(MESH_BASE OBJECT "mesh_base.f90")
add_dependencies(MESH_BASE ELEMENT)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:MESH_BASE>)

# SPECTRAL solver and FEM solver use different mesh files
if (PROJECT_NAME STREQUAL "DAMASK_spectral")
      add_library(MESH OBJECT "mesh_grid.f90")
      add_dependencies(MESH MESH_BASE MATH FEsolving)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:MESH>)
elseif (PROJECT_NAME STREQUAL "DAMASK_FEM")
      add_library(FEZoo OBJECT "FEM_zoo.f90")
      add_dependencies(FEZoo IO)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:FEZoo>)
      add_library(MESH OBJECT "mesh_FEM.f90")
      add_dependencies(MESH FEZoo MESH_BASE MATH FEsolving)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:MESH>)
endif()

add_library(MATERIAL OBJECT "material.f90")
add_dependencies(MATERIAL MESH DAMASK_CONFIG ROTATIONS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:MATERIAL>)

add_library(LATTICE OBJECT "lattice.f90")
add_dependencies(LATTICE MATERIAL)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:LATTICE>)

# For each modular section
add_library (PLASTIC OBJECT
  "plastic_dislotwin.f90"
  "plastic_disloUCLA.f90"
  "plastic_isotropic.f90"
  "plastic_phenopowerlaw.f90"
  "plastic_kinematichardening.f90"
  "plastic_nonlocal.f90"
  "plastic_none.f90")
add_dependencies(PLASTIC LATTICE RESULTS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:PLASTIC>)

add_library (KINEMATICS OBJECT
  "kinematics_cleavage_opening.f90"
  "kinematics_slipplane_opening.f90"
  "kinematics_thermal_expansion.f90")
add_dependencies(KINEMATICS LATTICE RESULTS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:KINEMATICS>)

add_library (SOURCE OBJECT
  "source_thermal_dissipation.f90"
  "source_thermal_externalheat.f90"
  "source_damage_isoBrittle.f90"
  "source_damage_isoDuctile.f90"
  "source_damage_anisoBrittle.f90"
  "source_damage_anisoDuctile.f90")
add_dependencies(SOURCE LATTICE RESULTS)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:SOURCE>)

add_library(CONSTITUTIVE OBJECT "constitutive.f90")
add_dependencies(CONSTITUTIVE PLASTIC KINEMATICS SOURCE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:CONSTITUTIVE>)

add_library(CRYSTALLITE OBJECT "crystallite.f90")
add_dependencies(CRYSTALLITE CONSTITUTIVE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:CRYSTALLITE>)

add_library(HOMOGENIZATION OBJECT
  "homogenization_RGC.f90"
  "homogenization_isostrain.f90"
  "homogenization_none.f90")
add_dependencies(HOMOGENIZATION CRYSTALLITE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:HOMOGENIZATION>)

add_library(DAMAGE OBJECT
  "damage_none.f90"
  "damage_local.f90"
  "damage_nonlocal.f90")
add_dependencies(DAMAGE CRYSTALLITE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DAMAGE>)

add_library(THERMAL OBJECT
  "thermal_isothermal.f90"
  "thermal_adiabatic.f90"
  "thermal_conduction.f90")
add_dependencies(THERMAL CRYSTALLITE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:THERMAL>)

add_library(DAMASK_ENGINE OBJECT "homogenization.f90")
add_dependencies(DAMASK_ENGINE THERMAL DAMAGE HOMOGENIZATION)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DAMASK_ENGINE>)

add_library(DAMASK_CPFE OBJECT "CPFEM2.f90")
add_dependencies(DAMASK_CPFE DAMASK_ENGINE)
list(APPEND OBJECTFILES $<TARGET_OBJECTS:DAMASK_CPFE>)

if (PROJECT_NAME STREQUAL "DAMASK_spectral")
      add_library(SPECTRAL_UTILITIES OBJECT "spectral_utilities.f90")
      add_dependencies(SPECTRAL_UTILITIES DAMASK_CPFE)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:SPECTRAL_UTILITIES>)

      add_library(SPECTRAL_SOLVER OBJECT
      "grid_thermal_spectral.f90"
      "grid_damage_spectral.f90"
      "grid_mech_spectral_basic.f90"
      "grid_mech_spectral_polarisation.f90")
      add_dependencies(SPECTRAL_SOLVER SPECTRAL_UTILITIES)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:SPECTRAL_SOLVER>)

      if(NOT CMAKE_BUILD_TYPE STREQUAL "SYNTAXONLY")
        add_executable(DAMASK_spectral "DAMASK_grid.f90" ${OBJECTFILES})
      else()
        add_library(DAMASK_spectral OBJECT "DAMASK_grid.f90")
      endif()

      add_dependencies(DAMASK_spectral SPECTRAL_SOLVER)
elseif (PROJECT_NAME STREQUAL "DAMASK_FEM")
      add_library(FEM_UTILITIES OBJECT "FEM_utilities.f90")
      add_dependencies(FEM_UTILITIES DAMASK_CPFE)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:FEM_UTILITIES>)

      add_library(FEM_SOLVER OBJECT
      "FEM_mech.f90")
      add_dependencies(FEM_SOLVER FEM_UTILITIES)
      list(APPEND OBJECTFILES $<TARGET_OBJECTS:FEM_SOLVER>)

      add_executable(DAMASK_FEM "DAMASK_FEM.f90" ${OBJECTFILES})
      add_dependencies(DAMASK_FEM FEM_SOLVER)
endif()
