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
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Michigan State University
!> @brief material subroutine for variable heat source
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_thermal) source_thermal_externalheat


  integer,           dimension(:),   allocatable :: &
    source_thermal_externalheat_offset, &                                                           !< which source is my current thermal dissipation mechanism?
    source_thermal_externalheat_instance                                                            !< instance of thermal dissipation source mechanism

  type :: tParameters                                                                               !< container type for internal constitutive parameters
    real(pReal), dimension(:), allocatable :: &
      t_n, & 
      f_T
    integer :: &
     nIntervals
  end type tParameters

  type(tParameters), dimension(:), allocatable  :: param                                            !< containers of constitutive parameters (len Ninstances)


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function source_thermal_externalheat_init(source_length) result(mySources)

  integer, intent(in)                  :: source_length  
  logical, dimension(:,:), allocatable :: mySources

  class(tNode), pointer :: &
    phases, &
    phase, &
    sources, &
    src 
  integer :: Ninstances,sourceOffset,Nconstituents,p

  print'(/,a)', ' <<<+-  source_thermal_externalHeat init  -+>>>'

  mySources = source_active('thermal_externalheat',source_length)
  Ninstances = count(mySources)
  print'(a,i2)', ' # instances: ',Ninstances; flush(IO_STDOUT)
  if(Ninstances == 0) return

  phases => config_material%get('phase')
  allocate(param(Ninstances))
  allocate(source_thermal_externalheat_offset  (phases%length), source=0)
  allocate(source_thermal_externalheat_instance(phases%length), source=0)

  do p = 1, phases%length
    phase => phases%get(p) 
    if(any(mySources(:,p))) source_thermal_externalheat_instance(p) = count(mySources(:,1:p))
    if(count(mySources(:,p)) == 0) cycle
    sources => phase%get('source')
    do sourceOffset = 1, sources%length
      if(mySources(sourceOffset,p)) then
        source_thermal_externalheat_offset(p) = sourceOffset
        associate(prm  => param(source_thermal_externalheat_instance(p)))
        src => sources%get(sourceOffset) 

        prm%t_n = src%get_asFloats('t_n')
        prm%nIntervals = size(prm%t_n) - 1

        prm%f_T = src%get_asFloats('f_T',requiredSize = size(prm%t_n))

        Nconstituents = count(material_phaseAt==p) * discretization_nIPs
        call constitutive_allocateState(sourceState(p)%p(sourceOffset),Nconstituents,1,1,0)
        end associate

      endif
    enddo
  enddo

end function source_thermal_externalheat_init


!--------------------------------------------------------------------------------------------------
!> @brief rate of change of state
!> @details state only contains current time to linearly interpolate given heat powers
!--------------------------------------------------------------------------------------------------
module subroutine source_thermal_externalheat_dotState(phase, of)

  integer, intent(in) :: &
    phase, &
    of

  integer :: &
    sourceOffset

  sourceOffset = source_thermal_externalheat_offset(phase)

  sourceState(phase)%p(sourceOffset)%dotState(1,of) = 1.0_pReal                                     ! state is current time

end subroutine source_thermal_externalheat_dotState


!--------------------------------------------------------------------------------------------------
!> @brief returns local heat generation rate
!--------------------------------------------------------------------------------------------------
module subroutine source_thermal_externalheat_getRateAndItsTangent(TDot, dTDot_dT, phase, of)

  integer, intent(in) :: &
    phase, &
    of
  real(pReal),  intent(out) :: &
    TDot, &
    dTDot_dT

  integer :: &
    sourceOffset, interval
  real(pReal) :: &
    frac_time

  sourceOffset = source_thermal_externalheat_offset(phase)

  associate(prm => param(source_thermal_externalheat_instance(phase)))
  do interval = 1, prm%nIntervals                                                                   ! scan through all rate segments
    frac_time = (sourceState(phase)%p(sourceOffset)%state(1,of) - prm%t_n(interval)) &
              / (prm%t_n(interval+1) - prm%t_n(interval))                                           ! fractional time within segment
    if (     (frac_time <  0.0_pReal .and. interval == 1) &
        .or. (frac_time >= 1.0_pReal .and. interval == prm%nIntervals) &
        .or. (frac_time >= 0.0_pReal .and. frac_time < 1.0_pReal) ) &
      TDot = prm%f_T(interval  ) * (1.0_pReal - frac_time) + &
             prm%f_T(interval+1) * frac_time                                                        ! interpolate heat rate between segment boundaries...
                                                                                                    ! ...or extrapolate if outside of bounds
  enddo
  dTDot_dT = 0.0
  end associate

end subroutine source_thermal_externalheat_getRateAndItsTangent

end submodule source_thermal_externalheat
