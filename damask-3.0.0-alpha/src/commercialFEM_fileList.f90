! Copyright 2011-20 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program. If not, see <http://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief all DAMASK files without solver
!> @details List of files needed by MSC.Marc
!--------------------------------------------------------------------------------------------------
#include "IO.f90"
#include "YAML_types.f90"
#include "YAML_parse.f90"
#include "future.f90"
#include "config.f90"
#include "LAPACK_interface.f90"
#include "math.f90"
#include "quaternions.f90"
#include "rotations.f90"
#include "FEsolving.f90"
#include "element.f90"
#include "HDF5_utilities.f90"
#include "results.f90"
#include "geometry_plastic_nonlocal.f90"
#include "discretization.f90"
#ifdef Marc4DAMASK
#include "marc/discretization_marc.f90"
#endif
#include "material.f90"
#include "lattice.f90"
#include "constitutive.f90"
#include "constitutive_plastic.f90"
#include "constitutive_plastic_none.f90"
#include "constitutive_plastic_isotropic.f90"
#include "constitutive_plastic_phenopowerlaw.f90"
#include "constitutive_plastic_kinehardening.f90"
#include "constitutive_plastic_dislotwin.f90"
#include "constitutive_plastic_disloTungsten.f90"
#include "constitutive_plastic_nonlocal.f90"
#include "constitutive_thermal.f90"
#include "source_thermal_dissipation.f90"
#include "source_thermal_externalheat.f90"
#include "kinematics_thermal_expansion.f90"
#include "constitutive_damage.f90"
#include "source_damage_isoBrittle.f90"
#include "source_damage_isoDuctile.f90"
#include "source_damage_anisoBrittle.f90"
#include "source_damage_anisoDuctile.f90"
#include "kinematics_cleavage_opening.f90"
#include "kinematics_slipplane_opening.f90"
#include "crystallite.f90"
#include "thermal_isothermal.f90"
#include "thermal_adiabatic.f90"
#include "thermal_conduction.f90"
#include "damage_none.f90"
#include "damage_local.f90"
#include "damage_nonlocal.f90"
#include "homogenization.f90"
#include "homogenization_mech_none.f90"
#include "homogenization_mech_isostrain.f90"
#include "homogenization_mech_RGC.f90"
#include "CPFEM.f90"
