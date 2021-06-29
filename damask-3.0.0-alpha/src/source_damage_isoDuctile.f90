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
!> @author Luv Sharma, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine incorporating isotropic ductile damage source mechanism
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule (constitutive:constitutive_damage) source_damage_isoDuctile

  integer,                       dimension(:),           allocatable :: &
    source_damage_isoDuctile_offset, &                                                              !< which source is my current damage mechanism?
    source_damage_isoDuctile_instance                                                               !< instance of damage source mechanism

  type:: tParameters                                                                                !< container type for internal constitutive parameters
    real(pReal) :: &
      critPlasticStrain, &                                                                          !< critical plastic strain
      N
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
  end type tParameters

  type(tParameters), dimension(:), allocatable :: param                                             !< containers of constitutive parameters (len Ninstance)


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function source_damage_isoDuctile_init(source_length) result(mySources)

  integer, intent(in)                  :: source_length  
  logical, dimension(:,:), allocatable :: mySources

  class(tNode), pointer :: &
    phases, &
    phase, &
    sources, &
    src
  integer :: Ninstance,sourceOffset,NipcMyPhase,p
  character(len=pStringLen) :: extmsg = ''

  write(6,'(/,a)') ' <<<+-  source_damage_isoDuctile init  -+>>>'

  mySources = source_active('damage_isoDuctile',source_length)

  Ninstance = count(mySources)
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
  if(Ninstance == 0) return

  phases => material_root%get('phase')
  allocate(param(Ninstance))
  allocate(source_damage_isoDuctile_offset  (phases%length), source=0)
  allocate(source_damage_isoDuctile_instance(phases%length), source=0)

  do p = 1, phases%length
    phase => phases%get(p) 
    if(count(mySources(:,p)) == 0) cycle
    if(any(mySources(:,p))) source_damage_isoDuctile_instance(p) = count(mySources(:,1:p))
    sources => phase%get('source')
    do sourceOffset = 1, sources%length
      if(mySources(sourceOffset,p)) then
        source_damage_isoDuctile_offset(p) = sourceOffset
        associate(prm  => param(source_damage_isoDuctile_instance(p)))
        src => sources%get(sourceOffset) 

        prm%N                 = src%get_asFloat('q')
        prm%critPlasticStrain = src%get_asFloat('gamma_crit')

#if defined (__GFORTRAN__)
        prm%output = output_asStrings(src)
#else
        prm%output = src%get_asStrings('output',defaultVal=emptyStringArray)
#endif
 
        ! sanity checks
        if (prm%N                 <= 0.0_pReal) extmsg = trim(extmsg)//' q'
        if (prm%critPlasticStrain <= 0.0_pReal) extmsg = trim(extmsg)//' gamma_crit'

        NipcMyPhase=count(material_phaseAt==p) * discretization_nIP
        call constitutive_allocateState(sourceState(p)%p(sourceOffset),NipcMyPhase,1,1,0)
        sourceState(p)%p(sourceOffset)%atol = src%get_asFloat('isoDuctile_atol',defaultVal=1.0e-3_pReal)
        if(any(sourceState(p)%p(sourceOffset)%atol < 0.0_pReal)) extmsg = trim(extmsg)//' isoductile_atol'

        end associate

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
        if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg)//'(damage_isoDuctile)')
      endif
    enddo
  enddo


end function source_damage_isoDuctile_init


!--------------------------------------------------------------------------------------------------
!> @brief calculates derived quantities from state
!--------------------------------------------------------------------------------------------------
module subroutine source_damage_isoDuctile_dotState(ipc, ip, el)

  integer, intent(in) :: &
    ipc, &                                                                                          !< component-ID of integration point
    ip, &                                                                                           !< integration point
    el                                                                                              !< element

  integer :: &
    phase, &
    constituent, &
    sourceOffset, &
    damageOffset, &
    homog

  phase = material_phaseAt(ipc,el)
  constituent = material_phasememberAt(ipc,ip,el)
  sourceOffset = source_damage_isoDuctile_offset(phase)
  homog = material_homogenizationAt(el)
  damageOffset = damageMapping(homog)%p(ip,el)

  associate(prm => param(source_damage_isoDuctile_instance(phase)))
  sourceState(phase)%p(sourceOffset)%dotState(1,constituent) = &
    sum(plasticState(phase)%slipRate(:,constituent))/(damage(homog)%p(damageOffset)**prm%N)/prm%critPlasticStrain
  end associate

end subroutine source_damage_isoDuctile_dotState


!--------------------------------------------------------------------------------------------------
!> @brief returns local part of nonlocal damage driving force
!--------------------------------------------------------------------------------------------------
module subroutine source_damage_isoDuctile_getRateAndItsTangent(localphiDot, dLocalphiDot_dPhi, phi, phase, constituent)

  integer, intent(in) :: &
    phase, &
    constituent
  real(pReal),  intent(in) :: &
    phi
  real(pReal),  intent(out) :: &
    localphiDot, &
    dLocalphiDot_dPhi

  integer :: &
    sourceOffset

  sourceOffset = source_damage_isoDuctile_offset(phase)

  dLocalphiDot_dPhi = -sourceState(phase)%p(sourceOffset)%state(1,constituent)

  localphiDot = 1.0_pReal &
              + dLocalphiDot_dPhi*phi

end subroutine source_damage_isoDuctile_getRateAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief writes results to HDF5 output file
!--------------------------------------------------------------------------------------------------
module subroutine source_damage_isoDuctile_results(phase,group)

  integer,          intent(in) :: phase
  character(len=*), intent(in) :: group

  integer :: o

  associate(prm => param(source_damage_isoDuctile_instance(phase)), &
            stt => sourceState(phase)%p(source_damage_isoDuctile_offset(phase))%state)
  outputsLoop: do o = 1,size(prm%output)
    select case(trim(prm%output(o)))
      case ('f_phi')
        call results_writeDataset(group,stt,trim(prm%output(o)),'driving force','J/m³')
    end select
  enddo outputsLoop
  end associate

end subroutine source_damage_isoDuctile_results

end submodule source_damage_isoDuctile
