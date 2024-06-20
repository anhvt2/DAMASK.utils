! Copyright 2011-2024 Max-Planck-Institut für Eisenforschung GmbH
! 
! DAMASK is free software: you can redistribute it and/or modify
! it under the terms of the GNU Affero General Public License as
! published by the Free Software Foundation, either version 3 of the
! License, or (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Affero General Public License for more details.
! 
! You should have received a copy of the GNU Affero General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Michigan State University
!> @brief material subroutine for variable heat source
!--------------------------------------------------------------------------------------------------
submodule(phase:thermal) source_externalheat


  integer,           dimension(:),   allocatable :: &
    source_ID                                                                                       !< index in phase source list corresponding to this source

  type :: tParameters                                                                               !< container type for internal constitutive parameters
    type(tTable) :: f                                                                               !< external heat power as (tabulated) function of time
  end type tParameters

  type(tParameters), dimension(:), allocatable  :: param                                            !< containers of constitutive parameters (len Ninstances)


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function source_externalheat_init(maxNsources) result(isMySource)

  integer, intent(in)                  :: maxNsources
  logical, dimension(:,:), allocatable :: isMySource

  type(tDict), pointer :: &
    phases, &
    phase, &
    thermal, &
    src
  type(tList), pointer :: &
    sources
  character(len=:), allocatable :: refs
  integer :: ph,Nmembers,so,Nsources


  isMySource = thermal_active('externalheat',maxNsources)
  if (count(isMySource) == 0) return

  print'(/,1x,a)', '<<<+-  phase:thermal:source_externalheat init  -+>>>'
  print'(/,1x,a,1x,i0)', '# phases:',count(isMySource); flush(IO_STDOUT)


  phases => config_material%get_dict('phase')
  allocate(param(phases%length))
  allocate(source_ID(phases%length), source=0)

  do ph = 1, phases%length
    Nsources = count(isMySource(:,ph))
    if (Nsources == 0) cycle
    if (Nsources > 1) call IO_error(600,ext_msg='externalheat')
    Nmembers = count(material_ID_phase == ph)
    phase => phases%get_dict(ph)
    thermal => phase%get_dict('thermal')
    sources => thermal%get_list('source')
    do so = 1, sources%length
      if (isMySource(so,ph)) then
        source_ID(ph) = so
        associate(prm  => param(ph))
          src => sources%get_dict(so)
          print'(/,1x,a,1x,i0,1x,a,1x,a,1x,i0)', 'phase',ph,'('//phases%key(ph)//')','source',so
          refs = config_listReferences(src,indent=3)
          if (len(refs) > 0) print'(/,1x,a)', refs

          prm%f = table(src,'t','f')
          call phase_allocateState(thermalState(ph)%p(so),Nmembers,1,1,0)
        end associate
        exit
      end if
    end do
  end do

end function source_externalheat_init


!--------------------------------------------------------------------------------------------------
!> @brief rate of change of state
!> @details state only contains current time to linearly interpolate given heat powers
!--------------------------------------------------------------------------------------------------
module subroutine source_externalheat_dotState(ph, en)

  integer, intent(in) :: &
    ph, &
    en


  thermalState(ph)%p(source_ID(ph))%dotState(1,en) = 1.0_pREAL                                         ! state is current time

end subroutine source_externalheat_dotState


!--------------------------------------------------------------------------------------------------
!> @brief returns local heat generation rate
!--------------------------------------------------------------------------------------------------
module function source_externalheat_f_T(ph,en) result(f_T)

  integer, intent(in) :: &
    ph, &
    en
  real(pREAL) :: &
    f_T


  associate(prm => param(ph))
    f_T = prm%f%at(thermalState(ph)%p(source_ID(ph))%state(1,en))
  end associate

end function source_externalheat_f_T

end submodule source_externalheat
