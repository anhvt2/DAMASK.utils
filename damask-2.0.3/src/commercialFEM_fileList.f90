! Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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
!> @details List of files needed by MSC.Marc, Abaqus/Explicit, and Abaqus/Standard
!--------------------------------------------------------------------------------------------------
#include "IO.f90"
#include "numerics.f90"
#include "debug.f90"
#include "config.f90"
#ifdef DAMASKHDF5
#include "HDF5_utilities.f90"
#include "results.f90"
#endif
#include "math.f90"
#include "quaternions.f90"
#include "Lambert.f90"
#include "rotations.f90"
#include "FEsolving.f90"
#include "element.f90"
#include "mesh_base.f90"
#ifdef Abaqus
#include "mesh_abaqus.f90"
#endif
#ifdef Marc4DAMASK
#include "mesh_marc.f90"
#endif
#include "material.f90"
#include "lattice.f90"
#include "source_thermal_dissipation.f90"
#include "source_thermal_externalheat.f90"
#include "source_damage_isoBrittle.f90"
#include "source_damage_isoDuctile.f90"
#include "source_damage_anisoBrittle.f90"
#include "source_damage_anisoDuctile.f90"
#include "kinematics_cleavage_opening.f90"
#include "kinematics_slipplane_opening.f90"
#include "kinematics_thermal_expansion.f90"
#include "plastic_none.f90"
#include "plastic_isotropic.f90"
#include "plastic_phenopowerlaw.f90"
#include "plastic_kinematichardening.f90"
#include "plastic_dislotwin.f90"
#include "plastic_disloUCLA.f90"
#include "plastic_nonlocal.f90"
#include "constitutive.f90"
#include "crystallite.f90"
#include "homogenization_none.f90"
#include "homogenization_isostrain.f90"
#include "homogenization_RGC.f90"
#include "thermal_isothermal.f90"
#include "thermal_adiabatic.f90"
#include "thermal_conduction.f90"
#include "damage_none.f90"
#include "damage_local.f90"
#include "damage_nonlocal.f90"
#include "homogenization.f90"
#include "CPFEM.f90"
