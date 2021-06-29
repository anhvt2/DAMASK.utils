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
!----------------------------------------------------------------------------------------------------
!> @brief internal microstructure state for all plasticity constitutive models
!----------------------------------------------------------------------------------------------------
submodule(constitutive) constitutive_mech

  interface

    module function plastic_none_init()          result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_none_init

    module function plastic_isotropic_init()     result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_isotropic_init

    module function plastic_phenopowerlaw_init() result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_phenopowerlaw_init

    module function plastic_kinehardening_init() result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_kinehardening_init

    module function plastic_dislotwin_init()     result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_dislotwin_init

    module function plastic_dislotungsten_init() result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_dislotungsten_init

    module function plastic_nonlocal_init()      result(myPlasticity)
      logical, dimension(:), allocatable :: &
        myPlasticity
    end function plastic_nonlocal_init


    module subroutine plastic_isotropic_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress

      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      integer,                         intent(in) :: &
        instance, &
        of
    end subroutine plastic_isotropic_LpAndItsTangent

    pure module subroutine plastic_phenopowerlaw_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress
      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      integer,                         intent(in) :: &
        instance, &
        of
    end subroutine plastic_phenopowerlaw_LpAndItsTangent

    pure module subroutine plastic_kinehardening_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress

      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      integer,                         intent(in) :: &
        instance, &
        of
    end subroutine plastic_kinehardening_LpAndItsTangent

    module subroutine plastic_dislotwin_LpAndItsTangent(Lp,dLp_dMp,Mp,T,instance,of)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress

      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      real(pReal),                     intent(in) :: &
        T
      integer,                         intent(in) :: &
        instance, &
        of
    end subroutine plastic_dislotwin_LpAndItsTangent

    pure module subroutine plastic_dislotungsten_LpAndItsTangent(Lp,dLp_dMp,Mp,T,instance,of)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress

      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      real(pReal),                     intent(in) :: &
        T
      integer,                         intent(in) :: &
        instance, &
        of
    end subroutine plastic_dislotungsten_LpAndItsTangent

    module subroutine plastic_nonlocal_LpAndItsTangent(Lp,dLp_dMp, &
                                                       Mp,Temperature,instance,of,ip,el)
      real(pReal), dimension(3,3),     intent(out) :: &
        Lp                                                                                          !< plastic velocity gradient
      real(pReal), dimension(3,3,3,3), intent(out) :: &
        dLp_dMp                                                                                     !< derivative of Lp with respect to the Mandel stress

      real(pReal), dimension(3,3),     intent(in) :: &
        Mp                                                                                          !< Mandel stress
      real(pReal),                     intent(in) :: &
        Temperature
      integer,                         intent(in) :: &
        instance, &
        of, &
        ip, &                                                                                       !< current integration point
        el                                                                                          !< current element number
    end subroutine plastic_nonlocal_LpAndItsTangent


    module subroutine plastic_dislotwin_dependentState(T,instance,of)
      integer,       intent(in) :: &
        instance, &
        of
      real(pReal),   intent(in) :: &
        T
    end subroutine plastic_dislotwin_dependentState

    module subroutine plastic_dislotungsten_dependentState(instance,of)
      integer,       intent(in) :: &
        instance, &
        of
    end subroutine plastic_dislotungsten_dependentState

    module subroutine plastic_nonlocal_dependentState(F, Fp, instance, of, ip, el)
      real(pReal), dimension(3,3), intent(in) :: &
        F, &                                                                                        !< deformation gradient
        Fp                                                                                          !< plastic deformation gradient
      integer, intent(in) :: &
        instance, &
        of, &
        ip, &                                                                                       !< current integration point
        el                                                                                          !< current element number
    end subroutine plastic_nonlocal_dependentState

    module subroutine plastic_isotropic_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_isotropic_results

    module subroutine plastic_phenopowerlaw_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_phenopowerlaw_results

    module subroutine plastic_kinehardening_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_kinehardening_results

    module subroutine plastic_dislotwin_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_dislotwin_results

    module subroutine plastic_dislotungsten_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_dislotungsten_results

    module subroutine plastic_nonlocal_results(instance,group)
      integer,          intent(in) :: instance
      character(len=*), intent(in) :: group
    end subroutine plastic_nonlocal_results


  end interface


contains


!--------------------------------------------------------------------------------------------------
!> @brief Initialize mechanical field related constitutive models
!> @details Initialize elasticity, plasticity and stiffness degradation models.
!--------------------------------------------------------------------------------------------------
module subroutine mech_init

  integer :: &
    p, &
    stiffDegradationCtr
  class(tNode), pointer :: &
    phases, &
    phase, &
    mech, &
    elastic, &
    stiffDegradation

  print'(/,a)', ' <<<+-  constitutive_mech init  -+>>>'

!-------------------------------------------------------------------------------------------------
! initialize elasticity (hooke)                         !ToDO: Maybe move to elastic submodule along with function homogenizedC?
  phases => config_material%get('phase')
  allocate(phase_elasticity(phases%length), source = ELASTICITY_undefined_ID)
  allocate(phase_elasticityInstance(phases%length), source = 0)
  allocate(phase_NstiffnessDegradations(phases%length),source=0)

  do p = 1, phases%length
    phase   => phases%get(p)
    mech    => phase%get('mechanics')
    elastic => mech%get('elasticity')
    if(elastic%get_asString('type') == 'hooke') then
      phase_elasticity(p) = ELASTICITY_HOOKE_ID
    else
      call IO_error(200,ext_msg=elastic%get_asString('type'))
    endif
    stiffDegradation => mech%get('stiffness_degradation',defaultVal=emptyList)                      ! check for stiffness degradation mechanisms
    phase_NstiffnessDegradations(p) = stiffDegradation%length
  enddo

  allocate(phase_stiffnessDegradation(maxval(phase_NstiffnessDegradations),phases%length), &
                        source=STIFFNESS_DEGRADATION_undefined_ID)

  if(maxVal(phase_NstiffnessDegradations)/=0) then
    do p = 1, phases%length
      phase => phases%get(p)
      mech    => phase%get('mechanics')
      stiffDegradation => mech%get('stiffness_degradation',defaultVal=emptyList)
      do stiffDegradationCtr = 1, stiffDegradation%length
        if(stiffDegradation%get_asString(stiffDegradationCtr) == 'damage') &
            phase_stiffnessDegradation(stiffDegradationCtr,p) = STIFFNESS_DEGRADATION_damage_ID
      enddo
    enddo
  endif


! initialize plasticity
  allocate(plasticState(phases%length))
  allocate(phase_plasticity(phases%length),source = PLASTICITY_undefined_ID)
  allocate(phase_plasticityInstance(phases%length),source = 0)
  allocate(phase_localPlasticity(phases%length),   source=.true.)

  where(plastic_none_init())              phase_plasticity = PLASTICITY_NONE_ID
  where(plastic_isotropic_init())         phase_plasticity = PLASTICITY_ISOTROPIC_ID
  where(plastic_phenopowerlaw_init())     phase_plasticity = PLASTICITY_PHENOPOWERLAW_ID
  where(plastic_kinehardening_init())     phase_plasticity = PLASTICITY_KINEHARDENING_ID
  where(plastic_dislotwin_init())         phase_plasticity = PLASTICITY_DISLOTWIN_ID
  where(plastic_dislotungsten_init())     phase_plasticity = PLASTICITY_DISLOTUNGSTEN_ID
  where(plastic_nonlocal_init())          phase_plasticity = PLASTICITY_NONLOCAL_ID

  do p = 1, phases%length
    phase_elasticityInstance(p) = count(phase_elasticity(1:p) == phase_elasticity(p))
    phase_plasticityInstance(p) = count(phase_plasticity(1:p) == phase_plasticity(p))
  enddo

end subroutine mech_init


!--------------------------------------------------------------------------------------------------
!> @brief checks if a plastic module is active or not
!--------------------------------------------------------------------------------------------------
module function plastic_active(plastic_label)  result(active_plastic)

  character(len=*), intent(in)       :: plastic_label                                               !< type of plasticity model
  logical, dimension(:), allocatable :: active_plastic

  class(tNode), pointer :: &
    phases, &
    phase, &
    mech, &
    pl
  integer :: p

  phases => config_material%get('phase')
  allocate(active_plastic(phases%length), source = .false. )
  do p = 1, phases%length
    phase => phases%get(p)
    mech  => phase%get('mechanics')
    pl    => mech%get('plasticity')
    if(pl%get_asString('type') == plastic_label) active_plastic(p) = .true.
  enddo

end function plastic_active


!--------------------------------------------------------------------------------------------------
!> @brief returns the 2nd Piola-Kirchhoff stress tensor and its tangent with respect to
!> the elastic and intermediate deformation gradients using Hooke's law
!--------------------------------------------------------------------------------------------------
module subroutine constitutive_hooke_SandItsTangents(S, dS_dFe, dS_dFi, &
                                              Fe, Fi, ipc, ip, el)

  integer, intent(in) :: &
    ipc, &                                                                                          !< component-ID of integration point
    ip, &                                                                                           !< integration point
    el                                                                                              !< element
  real(pReal),   intent(in),  dimension(3,3) :: &
    Fe, &                                                                                           !< elastic deformation gradient
    Fi                                                                                              !< intermediate deformation gradient
  real(pReal),   intent(out), dimension(3,3) :: &
    S                                                                                               !< 2nd Piola-Kirchhoff stress tensor in lattice configuration
  real(pReal),   intent(out), dimension(3,3,3,3) :: &
    dS_dFe, &                                                                                       !< derivative of 2nd P-K stress with respect to elastic deformation gradient
    dS_dFi                                                                                          !< derivative of 2nd P-K stress with respect to intermediate deformation gradient
  real(pReal), dimension(3,3) :: E
  real(pReal), dimension(3,3,3,3) :: C
  integer :: &
    ho, &                                                                                           !< homogenization
    d                                                                                               !< counter in degradation loop
  integer :: &
    i, j

  ho = material_homogenizationAt(el)
  C = math_66toSym3333(constitutive_homogenizedC(ipc,ip,el))

  DegradationLoop: do d = 1, phase_NstiffnessDegradations(material_phaseAt(ipc,el))
    degradationType: select case(phase_stiffnessDegradation(d,material_phaseAt(ipc,el)))
      case (STIFFNESS_DEGRADATION_damage_ID) degradationType
        C = C * damage(ho)%p(damageMapping(ho)%p(ip,el))**2
    end select degradationType
  enddo DegradationLoop

  E = 0.5_pReal*(matmul(transpose(Fe),Fe)-math_I3)                                                  !< Green-Lagrange strain in unloaded configuration
  S = math_mul3333xx33(C,matmul(matmul(transpose(Fi),E),Fi))                                        !< 2PK stress in lattice configuration in work conjugate with GL strain pulled back to lattice configuration

  do i =1, 3;do j=1,3
    dS_dFe(i,j,1:3,1:3) = matmul(Fe,matmul(matmul(Fi,C(i,j,1:3,1:3)),transpose(Fi)))                !< dS_ij/dFe_kl = C_ijmn * Fi_lm * Fi_on * Fe_ko
    dS_dFi(i,j,1:3,1:3) = 2.0_pReal*matmul(matmul(E,Fi),C(i,j,1:3,1:3))                             !< dS_ij/dFi_kl = C_ijln * E_km * Fe_mn
  enddo; enddo

end subroutine constitutive_hooke_SandItsTangents


!--------------------------------------------------------------------------------------------------
!> @brief calls microstructure function of the different plasticity constitutive models
!--------------------------------------------------------------------------------------------------
module subroutine constitutive_plastic_dependentState(F, Fp, ipc, ip, el)

  integer, intent(in) :: &
    ipc, &                                                                                          !< component-ID of integration point
    ip, &                                                                                           !< integration point
    el                                                                                              !< element
  real(pReal),   intent(in), dimension(3,3) :: &
    F, &                                                                                            !< elastic deformation gradient
    Fp                                                                                              !< plastic deformation gradient

  integer :: &
    ho, &                                                                                           !< homogenization
    tme, &                                                                                          !< thermal member position
    instance, of

  ho  = material_homogenizationAt(el)
  tme = thermalMapping(ho)%p(ip,el)
  of  = material_phasememberAt(ipc,ip,el)
  instance = phase_plasticityInstance(material_phaseAt(ipc,el))

  plasticityType: select case (phase_plasticity(material_phaseAt(ipc,el)))
    case (PLASTICITY_DISLOTWIN_ID) plasticityType
      call plastic_dislotwin_dependentState(temperature(ho)%p(tme),instance,of)
    case (PLASTICITY_DISLOTUNGSTEN_ID) plasticityType
      call plastic_dislotungsten_dependentState(instance,of)
    case (PLASTICITY_NONLOCAL_ID) plasticityType
      call plastic_nonlocal_dependentState (F,Fp,instance,of,ip,el)
  end select plasticityType

end subroutine constitutive_plastic_dependentState


!--------------------------------------------------------------------------------------------------
!> @brief  contains the constitutive equation for calculating the velocity gradient
! ToDo: Discuss whether it makes sense if crystallite handles the configuration conversion, i.e.
! Mp in, dLp_dMp out
!--------------------------------------------------------------------------------------------------
module subroutine constitutive_plastic_LpAndItsTangents(Lp, dLp_dS, dLp_dFi, &
                                     S, Fi, ipc, ip, el)
  integer, intent(in) :: &
    ipc, &                                                                                          !< component-ID of integration point
    ip, &                                                                                           !< integration point
    el                                                                                              !< element
  real(pReal),   intent(in),  dimension(3,3) :: &
    S, &                                                                                            !< 2nd Piola-Kirchhoff stress
    Fi                                                                                              !< intermediate deformation gradient
  real(pReal),   intent(out), dimension(3,3) :: &
    Lp                                                                                              !< plastic velocity gradient
  real(pReal),   intent(out), dimension(3,3,3,3) :: &
    dLp_dS, &
    dLp_dFi                                                                                         !< derivative of Lp with respect to Fi

  real(pReal), dimension(3,3,3,3) :: &
    dLp_dMp                                                                                         !< derivative of Lp with respect to Mandel stress
  real(pReal), dimension(3,3) :: &
    Mp                                                                                              !< Mandel stress work conjugate with Lp
  integer :: &
    ho, &                                                                                           !< homogenization
    tme                                                                                             !< thermal member position
  integer :: &
    i, j, instance, of

  ho = material_homogenizationAt(el)
  tme = thermalMapping(ho)%p(ip,el)

  Mp = matmul(matmul(transpose(Fi),Fi),S)
  of = material_phasememberAt(ipc,ip,el)
  instance = phase_plasticityInstance(material_phaseAt(ipc,el))

  plasticityType: select case (phase_plasticity(material_phaseAt(ipc,el)))

    case (PLASTICITY_NONE_ID) plasticityType
      Lp = 0.0_pReal
      dLp_dMp = 0.0_pReal

    case (PLASTICITY_ISOTROPIC_ID) plasticityType
      call plastic_isotropic_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)

    case (PLASTICITY_PHENOPOWERLAW_ID) plasticityType
      call plastic_phenopowerlaw_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)

    case (PLASTICITY_KINEHARDENING_ID) plasticityType
      call plastic_kinehardening_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)

    case (PLASTICITY_NONLOCAL_ID) plasticityType
      call plastic_nonlocal_LpAndItsTangent(Lp,dLp_dMp,Mp, temperature(ho)%p(tme),instance,of,ip,el)

    case (PLASTICITY_DISLOTWIN_ID) plasticityType
      call plastic_dislotwin_LpAndItsTangent(Lp,dLp_dMp,Mp,temperature(ho)%p(tme),instance,of)

    case (PLASTICITY_DISLOTUNGSTEN_ID) plasticityType
      call plastic_dislotungsten_LpAndItsTangent(Lp,dLp_dMp,Mp,temperature(ho)%p(tme),instance,of)

  end select plasticityType

  do i=1,3; do j=1,3
    dLp_dFi(i,j,1:3,1:3) = matmul(matmul(Fi,S),transpose(dLp_dMp(i,j,1:3,1:3))) + &
                           matmul(matmul(Fi,dLp_dMp(i,j,1:3,1:3)),S)
    dLp_dS(i,j,1:3,1:3)  = matmul(matmul(transpose(Fi),Fi),dLp_dMp(i,j,1:3,1:3))                     ! ToDo: @PS: why not:   dLp_dMp:(FiT Fi)
  enddo; enddo

end subroutine constitutive_plastic_LpAndItsTangents


!--------------------------------------------------------------------------------------------
!> @brief writes plasticity constitutive results to HDF5 output file
!--------------------------------------------------------------------------------------------
module subroutine plastic_results

  integer :: p
  character(len=pStringLen) :: group

  plasticityLoop:  do p=1,size(material_name_phase)
    group = trim('current/phase')//'/'//trim(material_name_phase(p))
    call results_closeGroup(results_addGroup(group))

    group = trim(group)//'/plastic'

    call results_closeGroup(results_addGroup(group))
    select case(phase_plasticity(p))

      case(PLASTICITY_ISOTROPIC_ID)
        call plastic_isotropic_results(phase_plasticityInstance(p),group)

      case(PLASTICITY_PHENOPOWERLAW_ID)
        call plastic_phenopowerlaw_results(phase_plasticityInstance(p),group)

      case(PLASTICITY_KINEHARDENING_ID)
        call plastic_kinehardening_results(phase_plasticityInstance(p),group)

      case(PLASTICITY_DISLOTWIN_ID)
        call plastic_dislotwin_results(phase_plasticityInstance(p),group)

      case(PLASTICITY_DISLOTUNGSTEN_ID)
        call plastic_dislotungsten_results(phase_plasticityInstance(p),group)

      case(PLASTICITY_NONLOCAL_ID)
        call plastic_nonlocal_results(phase_plasticityInstance(p),group)
    end select

  enddo plasticityLoop

end subroutine plastic_results

end submodule constitutive_mech

