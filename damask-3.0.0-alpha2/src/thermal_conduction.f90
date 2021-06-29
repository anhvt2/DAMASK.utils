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
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine for temperature evolution from heat conduction
!--------------------------------------------------------------------------------------------------
module thermal_conduction
  use prec
  use material
  use config
  use lattice
  use results
  use crystallite
  use constitutive
  use YAML_types

  implicit none
  private

  type :: tParameters
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
  end type tParameters

  type(tparameters),             dimension(:), allocatable :: &
    param

  public :: &
    thermal_conduction_init, &
    thermal_conduction_getSourceAndItsTangent, &
    thermal_conduction_getConductivity, &
    thermal_conduction_getSpecificHeat, &
    thermal_conduction_getMassDensity, &
    thermal_conduction_putTemperatureAndItsRate, &
    thermal_conduction_results

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_init

  integer :: Ninstances,Nmaterialpoints,h
  class(tNode), pointer :: &
    material_homogenization, &
    homog, &
    homogThermal
 
  print'(/,a)', ' <<<+-  thermal_conduction init  -+>>>'; flush(6)

  Ninstances = count(thermal_type == THERMAL_conduction_ID)
  allocate(param(Ninstances))

  material_homogenization => config_material%get('homogenization')
  do h = 1, size(material_name_homogenization)
    if (thermal_type(h) /= THERMAL_conduction_ID) cycle
    homog => material_homogenization%get(h)
    homogThermal => homog%get('thermal')
    associate(prm => param(thermal_typeInstance(h)))

#if defined (__GFORTRAN__)
    prm%output = output_asStrings(homogThermal)
#else
    prm%output = homogThermal%get_asStrings('output',defaultVal=emptyStringArray)
#endif

    Nmaterialpoints=count(material_homogenizationAt==h)
    thermalState(h)%sizeState = 0
    allocate(thermalState(h)%state0   (0,Nmaterialpoints))
    allocate(thermalState(h)%subState0(0,Nmaterialpoints))
    allocate(thermalState(h)%state    (0,Nmaterialpoints))

    thermalMapping(h)%p => material_homogenizationMemberAt
    deallocate(temperature    (h)%p)
    allocate  (temperature    (h)%p(Nmaterialpoints), source=thermal_initialT(h))
    deallocate(temperatureRate(h)%p)
    allocate  (temperatureRate(h)%p(Nmaterialpoints), source=0.0_pReal)

    end associate
  enddo

end subroutine thermal_conduction_init


!--------------------------------------------------------------------------------------------------
!> @brief return heat generation rate
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_getSourceAndItsTangent(Tdot, dTdot_dT, T, ip, el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal), intent(in) :: &
    T
  real(pReal), intent(out) :: &
    Tdot, dTdot_dT
  integer :: &
    homog
 
  Tdot = 0.0_pReal
  dTdot_dT = 0.0_pReal

  homog  = material_homogenizationAt(el)
  call constitutive_thermal_getRateAndItsTangents(TDot, dTDot_dT, T, crystallite_S,crystallite_Lp ,ip, el) 

  Tdot = Tdot/real(homogenization_Nconstituents(homog),pReal)
  dTdot_dT = dTdot_dT/real(homogenization_Nconstituents(homog),pReal)

end subroutine thermal_conduction_getSourceAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief return homogenized thermal conductivity in reference configuration
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getConductivity(ip,el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal), dimension(3,3) :: &
    thermal_conduction_getConductivity
  integer :: &
    grain


  thermal_conduction_getConductivity = 0.0_pReal
  do grain = 1, homogenization_Nconstituents(material_homogenizationAt(el))
    thermal_conduction_getConductivity = thermal_conduction_getConductivity + &
     crystallite_push33ToRef(grain,ip,el,lattice_K(:,:,material_phaseAt(grain,el)))
  enddo

  thermal_conduction_getConductivity = thermal_conduction_getConductivity &
                                     / real(homogenization_Nconstituents(material_homogenizationAt(el)),pReal)

end function thermal_conduction_getConductivity


!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized specific heat capacity
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getSpecificHeat(ip,el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal) :: &
    thermal_conduction_getSpecificHeat
  integer :: &
    grain

  thermal_conduction_getSpecificHeat = 0.0_pReal

  do grain = 1, homogenization_Nconstituents(material_homogenizationAt(el))
    thermal_conduction_getSpecificHeat = thermal_conduction_getSpecificHeat &
                                       + lattice_c_p(material_phaseAt(grain,el))
  enddo

  thermal_conduction_getSpecificHeat = thermal_conduction_getSpecificHeat &
                                     / real(homogenization_Nconstituents(material_homogenizationAt(el)),pReal)

end function thermal_conduction_getSpecificHeat


!--------------------------------------------------------------------------------------------------
!> @brief returns homogenized mass density
!--------------------------------------------------------------------------------------------------
function thermal_conduction_getMassDensity(ip,el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal) :: &
    thermal_conduction_getMassDensity
  integer :: &
    grain

  thermal_conduction_getMassDensity = 0.0_pReal


  do grain = 1, homogenization_Nconstituents(material_homogenizationAt(el))
    thermal_conduction_getMassDensity = thermal_conduction_getMassDensity &
                                      + lattice_rho(material_phaseAt(grain,el))
  enddo

  thermal_conduction_getMassDensity = thermal_conduction_getMassDensity &
                                    / real(homogenization_Nconstituents(material_homogenizationAt(el)),pReal)

end function thermal_conduction_getMassDensity


!--------------------------------------------------------------------------------------------------
!> @brief updates thermal state with solution from heat conduction PDE
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_putTemperatureAndItsRate(T,Tdot,ip,el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal),   intent(in) :: &
    T, &
    Tdot
  integer :: &
    homog, &
    offset

  homog  = material_homogenizationAt(el)
  offset = thermalMapping(homog)%p(ip,el)
  temperature    (homog)%p(offset) = T
  temperatureRate(homog)%p(offset) = Tdot

end subroutine thermal_conduction_putTemperatureAndItsRate


!--------------------------------------------------------------------------------------------------
!> @brief writes results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine thermal_conduction_results(homog,group)

  integer,          intent(in) :: homog
  character(len=*), intent(in) :: group

  integer :: o

  associate(prm => param(damage_typeInstance(homog)))
  outputsLoop: do o = 1,size(prm%output)
    select case(trim(prm%output(o)))
      case('T')
        call results_writeDataset(group,temperature(homog)%p,'T',&
                                  'temperature','K')
    end select
  enddo outputsLoop
  end associate

end subroutine thermal_conduction_results

end module thermal_conduction
