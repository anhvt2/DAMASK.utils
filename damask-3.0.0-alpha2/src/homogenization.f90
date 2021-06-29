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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Denny Tjahjanto, Max-Planck-Institut für Eisenforschung GmbH
!> @brief homogenization manager, organizing deformation partitioning and stress homogenization
!--------------------------------------------------------------------------------------------------
module homogenization
  use prec
  use IO
  use config
  use math
  use material
  use constitutive
  use crystallite
  use FEsolving
  use discretization
  use thermal_isothermal
  use thermal_adiabatic
  use thermal_conduction
  use damage_none
  use damage_local
  use damage_nonlocal
  use results

  implicit none
  private

  logical, public :: &
    terminallyIll = .false.                                                                         !< at least one material point is terminally ill

!--------------------------------------------------------------------------------------------------
! General variables for the homogenization at a  material point
  real(pReal),   dimension(:,:,:,:),     allocatable, public :: &
    homogenization_F0, &                                                                             !< def grad of IP at start of FE increment
    homogenization_F                                                                                 !< def grad of IP to be reached at end of FE increment
  real(pReal),   dimension(:,:,:,:),     allocatable, public, protected :: &
    homogenization_P                                                                                 !< first P--K stress of IP
  real(pReal),   dimension(:,:,:,:,:,:), allocatable, public, protected ::  &
    homogenization_dPdF                                                                              !< tangent of first P--K stress at IP

  type :: tNumerics
    integer :: &
      nMPstate                                                                                      !< materialpoint state loop limit
    real(pReal) :: &
      subStepMinHomog, &                                                                            !< minimum (relative) size of sub-step allowed during cutback in homogenization
      subStepSizeHomog, &                                                                           !< size of first substep when cutback in homogenization
      stepIncreaseHomog                                                                             !< increase of next substep size when previous substep converged in homogenization
  end type tNumerics

  type(tNumerics) :: num

  type :: tDebugOptions
    logical :: &
      basic, &
      extensive, &
      selective
    integer :: &
      element, &
      ip, &
      grain
  end type tDebugOptions

  type(tDebugOptions) :: debugHomog

  interface

    module subroutine mech_none_init
    end subroutine mech_none_init

    module subroutine mech_isostrain_init
    end subroutine mech_isostrain_init

    module subroutine mech_RGC_init(num_homogMech)
      class(tNode), pointer, intent(in) :: &
        num_homogMech                                                                               !< pointer to mechanical homogenization numerics data
    end subroutine mech_RGC_init


    module subroutine mech_isostrain_partitionDeformation(F,avgF)
      real(pReal),   dimension (:,:,:), intent(out) :: F                                            !< partitioned deformation gradient
      real(pReal),   dimension (3,3),   intent(in)  :: avgF                                         !< average deformation gradient at material point
    end subroutine mech_isostrain_partitionDeformation

    module subroutine mech_RGC_partitionDeformation(F,avgF,instance,of)
      real(pReal),   dimension (:,:,:), intent(out) :: F                                            !< partitioned deformation gradient
      real(pReal),   dimension (3,3),   intent(in)  :: avgF                                         !< average deformation gradient at material point
      integer,                          intent(in)  :: &
        instance, &
        of
    end subroutine mech_RGC_partitionDeformation


    module subroutine mech_isostrain_averageStressAndItsTangent(avgP,dAvgPdAvgF,P,dPdF,instance)
      real(pReal),   dimension (3,3),       intent(out) :: avgP                                     !< average stress at material point
      real(pReal),   dimension (3,3,3,3),   intent(out) :: dAvgPdAvgF                               !< average stiffness at material point

      real(pReal),   dimension (:,:,:),     intent(in)  :: P                                        !< partitioned stresses
      real(pReal),   dimension (:,:,:,:,:), intent(in)  :: dPdF                                     !< partitioned stiffnesses
      integer,                              intent(in)  :: instance
    end subroutine mech_isostrain_averageStressAndItsTangent

    module subroutine mech_RGC_averageStressAndItsTangent(avgP,dAvgPdAvgF,P,dPdF,instance)
      real(pReal),   dimension (3,3),       intent(out) :: avgP                                     !< average stress at material point
      real(pReal),   dimension (3,3,3,3),   intent(out) :: dAvgPdAvgF                               !< average stiffness at material point

      real(pReal),   dimension (:,:,:),     intent(in)  :: P                                        !< partitioned stresses
      real(pReal),   dimension (:,:,:,:,:), intent(in)  :: dPdF                                     !< partitioned stiffnesses
      integer,                              intent(in)  :: instance
    end subroutine mech_RGC_averageStressAndItsTangent

    module function mech_RGC_updateState(P,F,F0,avgF,dt,dPdF,ip,el)
      logical, dimension(2) :: mech_RGC_updateState
      real(pReal), dimension(:,:,:),     intent(in)    :: &
        P,&                                                                                         !< partitioned stresses
        F,&                                                                                         !< partitioned deformation gradients
        F0                                                                                          !< partitioned initial deformation gradients
      real(pReal), dimension(:,:,:,:,:), intent(in) :: dPdF                                         !< partitioned stiffnesses
      real(pReal), dimension(3,3),       intent(in) :: avgF                                         !< average F
      real(pReal),                       intent(in) :: dt                                           !< time increment
      integer,                           intent(in) :: &
        ip, &                                                                                       !< integration point number
        el                                                                                          !< element number
    end function mech_RGC_updateState


    module subroutine mech_RGC_results(instance,group)
      integer,          intent(in) :: instance                                                      !< homogenization instance
      character(len=*), intent(in) :: group                                                         !< group name in HDF5 file
    end subroutine mech_RGC_results

  end interface

  public ::  &
    homogenization_init, &
    materialpoint_stressAndItsTangent, &
    homogenization_results

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!--------------------------------------------------------------------------------------------------
subroutine homogenization_init

  class (tNode) , pointer :: &
    num_homog, &
    num_homogMech, &
    num_homogGeneric, &
    debug_homogenization

  debug_homogenization => config_debug%get('homogenization', defaultVal=emptyList)
  debugHomog%basic       =  debug_homogenization%contains('basic')
  debugHomog%extensive   =  debug_homogenization%contains('extensive')
  debugHomog%selective   =  debug_homogenization%contains('selective')
  debugHomog%element     =  config_debug%get_asInt('element',defaultVal = 1)
  debugHomog%ip          =  config_debug%get_asInt('integrationpoint',defaultVal = 1)
  debugHomog%grain       =  config_debug%get_asInt('grain',defaultVal = 1)

  if (debugHomog%grain < 1 &
    .or. debugHomog%grain > homogenization_Nconstituents(material_homogenizationAt(debugHomog%element))) &
    call IO_error(602,ext_msg='constituent', el=debugHomog%element, g=debugHomog%grain)


  num_homog        => config_numerics%get('homogenization',defaultVal=emptyDict)
  num_homogMech    => num_homog%get('mech',defaultVal=emptyDict)
  num_homogGeneric => num_homog%get('generic',defaultVal=emptyDict)

  if (any(homogenization_type == HOMOGENIZATION_NONE_ID))      call mech_none_init
  if (any(homogenization_type == HOMOGENIZATION_ISOSTRAIN_ID)) call mech_isostrain_init
  if (any(homogenization_type == HOMOGENIZATION_RGC_ID))       call mech_RGC_init(num_homogMech)

  if (any(thermal_type == THERMAL_isothermal_ID)) call thermal_isothermal_init
  if (any(thermal_type == THERMAL_adiabatic_ID))  call thermal_adiabatic_init
  if (any(thermal_type == THERMAL_conduction_ID)) call thermal_conduction_init

  if (any(damage_type == DAMAGE_none_ID))      call damage_none_init
  if (any(damage_type == DAMAGE_local_ID))     call damage_local_init
  if (any(damage_type == DAMAGE_nonlocal_ID))  call damage_nonlocal_init


!--------------------------------------------------------------------------------------------------
! allocate and initialize global variables
  allocate(homogenization_dPdF(3,3,3,3,discretization_nIPs,discretization_Nelems),       source=0.0_pReal)
  homogenization_F0 = spread(spread(math_I3,3,discretization_nIPs),4,discretization_Nelems)            ! initialize to identity
  homogenization_F = homogenization_F0                                                                ! initialize to identity
  allocate(homogenization_P(3,3,discretization_nIPs,discretization_Nelems),              source=0.0_pReal)

  print'(/,a)', ' <<<+-  homogenization init  -+>>>'; flush(IO_STDOUT)

  num%nMPstate          = num_homogGeneric%get_asInt  ('nMPstate',     defaultVal=10)
  num%subStepMinHomog   = num_homogGeneric%get_asFloat('subStepMin',   defaultVal=1.0e-3_pReal)
  num%subStepSizeHomog  = num_homogGeneric%get_asFloat('subStepSize',  defaultVal=0.25_pReal)
  num%stepIncreaseHomog = num_homogGeneric%get_asFloat('stepIncrease', defaultVal=1.5_pReal)

  if (num%nMPstate < 1)                   call IO_error(301,ext_msg='nMPstate')
  if (num%subStepMinHomog <= 0.0_pReal)   call IO_error(301,ext_msg='subStepMinHomog')
  if (num%subStepSizeHomog <= 0.0_pReal)  call IO_error(301,ext_msg='subStepSizeHomog')
  if (num%stepIncreaseHomog <= 0.0_pReal) call IO_error(301,ext_msg='stepIncreaseHomog')

end subroutine homogenization_init


!--------------------------------------------------------------------------------------------------
!> @brief  parallelized calculation of stress and corresponding tangent at material points
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_stressAndItsTangent(dt)

  real(pReal), intent(in) :: dt                                                                     !< time increment
  integer :: &
    NiterationHomog, &
    NiterationMPstate, &
    i, &                                                                                            !< integration point number
    e, &                                                                                            !< element number
    myNgrains
  real(pReal), dimension(discretization_nIPs,discretization_Nelems) :: &
    subFrac, &
    subStep
  logical,     dimension(discretization_nIPs,discretization_Nelems) :: &
    requested, &
    converged
  logical,     dimension(2,discretization_nIPs,discretization_Nelems) :: &
    doneAndHappy


!--------------------------------------------------------------------------------------------------
! initialize restoration points
  do e = FEsolving_execElem(1),FEsolving_execElem(2)
    do i = FEsolving_execIP(1),FEsolving_execIP(2);

      call crystallite_initializeRestorationPoints(i,e)

      subFrac(i,e) = 0.0_pReal
      converged(i,e) = .false.                                                                      ! pretend failed step ...
      subStep(i,e) = 1.0_pReal/num%subStepSizeHomog                                                 ! ... larger then the requested calculation
      requested(i,e) = .true.                                                                       ! everybody requires calculation

      if (homogState(material_homogenizationAt(e))%sizeState > 0) &
          homogState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
          homogState(material_homogenizationAt(e))%State0(   :,material_homogenizationMemberAt(i,e))

      if (thermalState(material_homogenizationAt(e))%sizeState > 0) &
          thermalState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
          thermalState(material_homogenizationAt(e))%State0(   :,material_homogenizationMemberAt(i,e))

      if (damageState(material_homogenizationAt(e))%sizeState > 0) &
          damageState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
          damageState(material_homogenizationAt(e))%State0(   :,material_homogenizationMemberAt(i,e))
    enddo
  enddo

  NiterationHomog = 0

  cutBackLooping: do while (.not. terminallyIll .and. &
       any(subStep(FEsolving_execIP(1):FEsolving_execIP(2),&
                   FEsolving_execElem(1):FEsolving_execElem(2)) > num%subStepMinHomog))

    !$OMP PARALLEL DO
    elementLooping1: do e = FEsolving_execElem(1),FEsolving_execElem(2)
      myNgrains = homogenization_Nconstituents(material_homogenizationAt(e))
      IpLooping1: do i = FEsolving_execIP(1),FEsolving_execIP(2)

        if (converged(i,e)) then
          subFrac(i,e) = subFrac(i,e) + subStep(i,e)
          subStep(i,e) = min(1.0_pReal-subFrac(i,e),num%stepIncreaseHomog*subStep(i,e))             ! introduce flexibility for step increase/acceleration

          steppingNeeded: if (subStep(i,e) > num%subStepMinHomog) then

            ! wind forward grain starting point
            call crystallite_windForward(i,e)

            if(homogState(material_homogenizationAt(e))%sizeState > 0) &
                homogState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
                homogState(material_homogenizationAt(e))%State    (:,material_homogenizationMemberAt(i,e))
            if(thermalState(material_homogenizationAt(e))%sizeState > 0) &
                thermalState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
                thermalState(material_homogenizationAt(e))%State    (:,material_homogenizationMemberAt(i,e))
            if(damageState(material_homogenizationAt(e))%sizeState > 0) &
                damageState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e)) = &
                damageState(material_homogenizationAt(e))%State    (:,material_homogenizationMemberAt(i,e))

          endif steppingNeeded

        else
          if ( (myNgrains == 1 .and. subStep(i,e) <= 1.0 ) .or. &                                   ! single grain already tried internal subStepping in crystallite
               num%subStepSizeHomog * subStep(i,e) <=  num%subStepMinHomog ) then                   ! would require too small subStep
                                                                                                    ! cutback makes no sense
            if (.not. terminallyIll) then                                                           ! so first signals terminally ill...
              print*, ' Integration point ', i,' at element ', e, ' terminally ill'
            endif
            terminallyIll = .true.                                                                  ! ...and kills all others
          else                                                                                      ! cutback makes sense
            subStep(i,e) = num%subStepSizeHomog * subStep(i,e)                                      ! crystallite had severe trouble, so do a significant cutback

            call crystallite_restore(i,e,subStep(i,e) < 1.0_pReal)

            if(homogState(material_homogenizationAt(e))%sizeState > 0) &
                homogState(material_homogenizationAt(e))%State(    :,material_homogenizationMemberAt(i,e)) = &
                homogState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e))
            if(thermalState(material_homogenizationAt(e))%sizeState > 0) &
                thermalState(material_homogenizationAt(e))%State(    :,material_homogenizationMemberAt(i,e)) = &
                thermalState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e))
            if(damageState(material_homogenizationAt(e))%sizeState > 0) &
                damageState(material_homogenizationAt(e))%State(    :,material_homogenizationMemberAt(i,e)) = &
                damageState(material_homogenizationAt(e))%subState0(:,material_homogenizationMemberAt(i,e))
          endif
        endif

        if (subStep(i,e) > num%subStepMinHomog) then
          requested(i,e) = .true.
          doneAndHappy(1:2,i,e) = [.false.,.true.]
        endif
      enddo IpLooping1
    enddo elementLooping1
    !$OMP END PARALLEL DO

    NiterationMPstate = 0

    convergenceLooping: do while (.not. terminallyIll .and. &
              any(                 requested(:,FEsolving_execELem(1):FEsolving_execElem(2)) &
                  .and. .not. doneAndHappy(1,:,FEsolving_execELem(1):FEsolving_execElem(2)) &
                 ) .and. &
              NiterationMPstate < num%nMPstate)
      NiterationMPstate = NiterationMPstate + 1

!--------------------------------------------------------------------------------------------------
! deformation partitioning
      !$OMP PARALLEL DO PRIVATE(myNgrains)
      elementLooping2: do e = FEsolving_execElem(1),FEsolving_execElem(2)
        myNgrains = homogenization_Nconstituents(material_homogenizationAt(e))
        IpLooping2: do i = FEsolving_execIP(1),FEsolving_execIP(2)
          if(requested(i,e) .and. .not. doneAndHappy(1,i,e)) then                                   ! requested but not yet done
            call partitionDeformation(homogenization_F0(1:3,1:3,i,e) &
                                      + (homogenization_F(1:3,1:3,i,e)-homogenization_F0(1:3,1:3,i,e))&
                                         *(subStep(i,e)+subFrac(i,e)), &
                                      i,e)
            crystallite_dt(1:myNgrains,i,e) = dt*subStep(i,e)                                       ! propagate materialpoint dt to grains
            crystallite_requested(1:myNgrains,i,e) = .true.                                         ! request calculation for constituents
          else
            crystallite_requested(1:myNgrains,i,e) = .false.                                        ! calculation for constituents not required anymore
          endif
        enddo IpLooping2
      enddo elementLooping2
      !$OMP END PARALLEL DO

!--------------------------------------------------------------------------------------------------
! crystallite integration
      converged = crystallite_stress() !ToDo: MD not sure if that is the best logic

!--------------------------------------------------------------------------------------------------
! state update
     !$OMP PARALLEL DO
      elementLooping3: do e = FEsolving_execElem(1),FEsolving_execElem(2)
        IpLooping3: do i = FEsolving_execIP(1),FEsolving_execIP(2)
          if (requested(i,e) .and. .not. doneAndHappy(1,i,e)) then
            if (.not. converged(i,e)) then
              doneAndHappy(1:2,i,e) = [.true.,.false.]
            else
              doneAndHappy(1:2,i,e) = updateState(dt*subStep(i,e), &
                                                  homogenization_F0(1:3,1:3,i,e) &
                                                  + (homogenization_F(1:3,1:3,i,e)-homogenization_F0(1:3,1:3,i,e)) &
                                                     *(subStep(i,e)+subFrac(i,e)), &
                                                 i,e)
              converged(i,e) = all(doneAndHappy(1:2,i,e))                                           ! converged if done and happy
            endif
          endif
        enddo IpLooping3
      enddo elementLooping3
      !$OMP END PARALLEL DO

    enddo convergenceLooping

    NiterationHomog = NiterationHomog + 1

  enddo cutBackLooping

  if (.not. terminallyIll ) then
    call crystallite_orientations()                                                                 ! calculate crystal orientations
    !$OMP PARALLEL DO
    elementLooping4: do e = FEsolving_execElem(1),FEsolving_execElem(2)
      IpLooping4: do i = FEsolving_execIP(1),FEsolving_execIP(2)
        call averageStressAndItsTangent(i,e)
      enddo IpLooping4
    enddo elementLooping4
    !$OMP END PARALLEL DO
  else
    print'(/,a,/)', ' << HOMOG >> Material Point terminally ill'
  endif

end subroutine materialpoint_stressAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief  partition material point def grad onto constituents
!--------------------------------------------------------------------------------------------------
subroutine partitionDeformation(subF,ip,el)

  real(pReal), intent(in), dimension(3,3) :: &
    subF
  integer,     intent(in) :: &
    ip, &                                                                                           !< integration point
    el                                                                                              !< element number

  chosenHomogenization: select case(homogenization_type(material_homogenizationAt(el)))

    case (HOMOGENIZATION_NONE_ID) chosenHomogenization
      crystallite_partitionedF(1:3,1:3,1,ip,el) = subF

    case (HOMOGENIZATION_ISOSTRAIN_ID) chosenHomogenization
      call mech_isostrain_partitionDeformation(&
                           crystallite_partitionedF(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
                           subF)

    case (HOMOGENIZATION_RGC_ID) chosenHomogenization
      call mech_RGC_partitionDeformation(&
                          crystallite_partitionedF(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
                          subF,&
                          ip, &
                          el)
  end select chosenHomogenization

end subroutine partitionDeformation


!--------------------------------------------------------------------------------------------------
!> @brief update the internal state of the homogenization scheme and tell whether "done" and
!> "happy" with result
!--------------------------------------------------------------------------------------------------
function updateState(subdt,subF,ip,el)

  real(pReal), intent(in) :: &
    subdt                                                                                           !< current time step
  real(pReal), intent(in), dimension(3,3) :: &
    subF
  integer,     intent(in) :: &
    ip, &                                                                                           !< integration point
    el                                                                                              !< element number
  integer :: c
  logical, dimension(2) :: updateState
  real(pReal) :: dPdFs(3,3,3,3,homogenization_Nconstituents(material_homogenizationAt(el)))

  updateState = .true.
  chosenHomogenization: select case(homogenization_type(material_homogenizationAt(el)))
    case (HOMOGENIZATION_RGC_ID) chosenHomogenization
      do c=1,homogenization_Nconstituents(material_homogenizationAt(el))
        dPdFs(:,:,:,:,c) = crystallite_stressTangent(c,ip,el)
      enddo
      updateState = &
        updateState .and. &
          mech_RGC_updateState(crystallite_P(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
                        crystallite_partitionedF(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
                        crystallite_partitionedF0(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el),&
                               subF,&
                               subdt, &
                               dPdFs, &
                               ip, &
                               el)
  end select chosenHomogenization

  chosenThermal: select case (thermal_type(material_homogenizationAt(el)))
    case (THERMAL_adiabatic_ID) chosenThermal
      updateState = &
        updateState .and. &
        thermal_adiabatic_updateState(subdt, &
                                      ip, &
                                      el)
  end select chosenThermal

  chosenDamage: select case (damage_type(material_homogenizationAt(el)))
    case (DAMAGE_local_ID) chosenDamage
      updateState = &
        updateState .and. &
        damage_local_updateState(subdt, &
                                 ip, &
                                 el)
  end select chosenDamage

end function updateState


!--------------------------------------------------------------------------------------------------
!> @brief derive average stress and stiffness from constituent quantities
!--------------------------------------------------------------------------------------------------
subroutine averageStressAndItsTangent(ip,el)

  integer, intent(in) :: &
    ip, &                                                                                           !< integration point
    el                                                                                              !< element number
  integer :: c
  real(pReal) :: dPdFs(3,3,3,3,homogenization_Nconstituents(material_homogenizationAt(el)))
 
  
  chosenHomogenization: select case(homogenization_type(material_homogenizationAt(el)))
    case (HOMOGENIZATION_NONE_ID) chosenHomogenization
        homogenization_P(1:3,1:3,ip,el)            = crystallite_P(1:3,1:3,1,ip,el)
        homogenization_dPdF(1:3,1:3,1:3,1:3,ip,el) = crystallite_stressTangent(1,ip,el)

    case (HOMOGENIZATION_ISOSTRAIN_ID) chosenHomogenization
      do c = 1, homogenization_Nconstituents(material_homogenizationAt(el))
        dPdFs(:,:,:,:,c) = crystallite_stressTangent(c,ip,el)
      enddo
      call mech_isostrain_averageStressAndItsTangent(&
        homogenization_P(1:3,1:3,ip,el), &
        homogenization_dPdF(1:3,1:3,1:3,1:3,ip,el),&
        crystallite_P(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
        dPdFs, &
        homogenization_typeInstance(material_homogenizationAt(el)))

    case (HOMOGENIZATION_RGC_ID) chosenHomogenization
      do c = 1, homogenization_Nconstituents(material_homogenizationAt(el))
        dPdFs(:,:,:,:,c) = crystallite_stressTangent(c,ip,el)
      enddo
      call mech_RGC_averageStressAndItsTangent(&
        homogenization_P(1:3,1:3,ip,el), &
        homogenization_dPdF(1:3,1:3,1:3,1:3,ip,el),&
        crystallite_P(1:3,1:3,1:homogenization_Nconstituents(material_homogenizationAt(el)),ip,el), &
        dPdFs, &
        homogenization_typeInstance(material_homogenizationAt(el)))
  end select chosenHomogenization

end subroutine averageStressAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief writes homogenization results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine homogenization_results
  use material, only: &
    material_homogenization_type => homogenization_type

  integer :: p
  character(len=:), allocatable :: group_base,group

  !real(pReal), dimension(:,:,:), allocatable :: temp

  do p=1,size(material_name_homogenization)
    group_base = 'current/homogenization/'//trim(material_name_homogenization(p))
    call results_closeGroup(results_addGroup(group_base))

    group = trim(group_base)//'/generic'
    call results_closeGroup(results_addGroup(group))
    !temp = reshape(homogenization_F,[3,3,discretization_nIPs*discretization_Nelems])
    !call results_writeDataset(group,temp,'F',&
    !                          'deformation gradient','1')
    !temp = reshape(homogenization_P,[3,3,discretization_nIPs*discretization_Nelems])
    !call results_writeDataset(group,temp,'P',&
    !                          '1st Piola-Kirchhoff stress','Pa')

    group = trim(group_base)//'/mech'
    call results_closeGroup(results_addGroup(group))
    select case(material_homogenization_type(p))
      case(HOMOGENIZATION_rgc_ID)
        call mech_RGC_results(homogenization_typeInstance(p),group)
    end select

    group = trim(group_base)//'/damage'
    call results_closeGroup(results_addGroup(group))
    select case(damage_type(p))
      case(DAMAGE_LOCAL_ID)
        call damage_local_results(p,group)
      case(DAMAGE_NONLOCAL_ID)
        call damage_nonlocal_results(p,group)
    end select

    group = trim(group_base)//'/thermal'
    call results_closeGroup(results_addGroup(group))
    select case(thermal_type(p))
      case(THERMAL_ADIABATIC_ID)
        call thermal_adiabatic_results(p,group)
      case(THERMAL_CONDUCTION_ID)
        call thermal_conduction_results(p,group)
    end select

 enddo

end subroutine homogenization_results

end module homogenization
