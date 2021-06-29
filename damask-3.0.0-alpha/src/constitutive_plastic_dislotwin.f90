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
!> @author Su Leen Wong, Max-Planck-Institut für Eisenforschung GmbH
!> @author Nan Jia, Max-Planck-Institut für Eisenforschung GmbH
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine incoprorating dislocation and twinning physics
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_plastic) plastic_dislotwin

  real(pReal), parameter :: &
    kB = 1.38e-23_pReal                                                                             !< Boltzmann constant in J/Kelvin

  type :: tParameters
    real(pReal) :: &
      mu                  = 1.0_pReal, &                                                            !< equivalent shear modulus
      nu                  = 1.0_pReal, &                                                            !< equivalent shear Poisson's ratio
      D0                  = 1.0_pReal, &                                                            !< prefactor for self-diffusion coefficient
      Qsd                 = 1.0_pReal, &                                                            !< activation energy for dislocation climb
      omega               = 1.0_pReal, &                                                            !< frequency factor for dislocation climb
      D                   = 1.0_pReal, &                                                            !< grain size
      p_sb                = 1.0_pReal, &                                                            !< p-exponent in shear band velocity
      q_sb                = 1.0_pReal, &                                                            !< q-exponent in shear band velocity
      CEdgeDipMinDistance = 1.0_pReal, &                                                            !< adjustment parameter to calculate minimum dipole distance
      i_tw                = 1.0_pReal, &                                                            !< adjustment parameter to calculate MFP for twinning
      tau_0               = 1.0_pReal, &                                                            !< strength due to elements in solid solution
      L_tw                = 1.0_pReal, &                                                            !< Length of twin nuclei in Burgers vectors
      L_tr                = 1.0_pReal, &                                                            !< Length of trans nuclei in Burgers vectors
      xc_twin             = 1.0_pReal, &                                                            !< critical distance for formation of twin nucleus
      xc_trans            = 1.0_pReal, &                                                            !< critical distance for formation of trans nucleus
      V_cs                = 1.0_pReal, &                                                            !< cross slip volume
      sbResistance        = 1.0_pReal, &                                                            !< value for shearband resistance
      sbVelocity          = 1.0_pReal, &                                                            !< value for shearband velocity_0
      E_sb                = 1.0_pReal, &                                                            !< activation energy for shear bands
      SFE_0K              = 1.0_pReal, &                                                            !< stacking fault energy at zero K
      dSFE_dT             = 1.0_pReal, &                                                            !< temperature dependence of stacking fault energy
      gamma_fcc_hex       = 1.0_pReal, &                                                            !< Free energy difference between austensite and martensite
      i_tr                = 1.0_pReal, &                                                            !< adjustment parameter to calculate MFP for transformation
      h                   = 1.0_pReal                                                               !< Stack height of hex nucleus
    real(pReal),               allocatable, dimension(:) :: &
      b_sl, &                                                                                       !< absolute length of burgers vector [m] for each slip system
      b_tw, &                                                                                       !< absolute length of burgers vector [m] for each twin system
      b_tr, &                                                                                       !< absolute length of burgers vector [m] for each transformation system
      Delta_F,&                                                                                     !< activation energy for glide [J] for each slip system
      v0, &                                                                                         !< dislocation velocity prefactor [m/s] for each slip system
      dot_N_0_tw, &                                                                                 !< twin nucleation rate [1/m³s] for each twin system
      dot_N_0_tr, &                                                                                 !< trans nucleation rate [1/m³s] for each trans system
      t_tw, &                                                                                       !< twin thickness [m] for each twin system
      CLambdaSlip, &                                                                                !< Adj. parameter for distance between 2 forest dislocations for each slip system
      t_tr, &                                                                                       !< martensite lamellar thickness [m] for each trans system and instance
      p, &                                                                                          !< p-exponent in glide velocity
      q, &                                                                                          !< q-exponent in glide velocity
      r, &                                                                                          !< r-exponent in twin nucleation rate
      s, &                                                                                          !< s-exponent in trans nucleation rate
      gamma_char, &                                                                                 !< characteristic shear for twins
      B                                                                                             !< drag coefficient
    real(pReal),               allocatable, dimension(:,:) :: &
      h_sl_sl, &                                                                                    !< components of slip-slip interaction matrix
      h_sl_tw, &                                                                                    !< components of slip-twin interaction matrix
      h_tw_tw, &                                                                                    !< components of twin-twin interaction matrix
      h_sl_tr, &                                                                                    !< components of slip-trans interaction matrix
      h_tr_tr, &                                                                                    !< components of trans-trans interaction matrix
      n0_sl, &                                                                                      !< slip system normal
      forestProjection, &
      C66
    real(pReal),               allocatable, dimension(:,:,:) :: &
      P_sl, &
      P_tw, &
      P_tr, &
      C66_tw, &
      C66_tr
    integer :: &
      sum_N_sl, &                                                                                   !< total number of active slip system
      sum_N_tw, &                                                                                   !< total number of active twin system
      sum_N_tr                                                                                      !< total number of active transformation system
    integer,                   allocatable, dimension(:,:) :: &
      fcc_twinNucleationSlipPair                                                                    ! ToDo: Better name? Is also use for trans
    character(len=pStringLen), allocatable, dimension(:) :: &
      output
    logical :: &
      ExtendedDislocations, &                                                                       !< consider split into partials for climb calculation
      fccTwinTransNucleation, &                                                                     !< twinning and transformation models are for fcc
      dipoleFormation                                                                               !< flag indicating consideration of dipole formation
  end type                                                                                          !< container type for internal constitutive parameters

  type :: tDislotwinState
    real(pReal),                  dimension(:,:),   pointer :: &
      rho_mob, &
      rho_dip, &
      gamma_sl, &
      f_tw, &
      f_tr
  end type tDislotwinState

  type :: tDislotwinMicrostructure
    real(pReal),                  dimension(:,:),   allocatable :: &
      Lambda_sl, &                                                                                  !< mean free path between 2 obstacles seen by a moving dislocation
      Lambda_tw, &                                                                                  !< mean free path between 2 obstacles seen by a growing twin
      Lambda_tr, &                                                                                  !< mean free path between 2 obstacles seen by a growing martensite
      tau_pass, &                                                                                   !< threshold stress for slip
      tau_hat_tw, &                                                                                 !< threshold stress for twinning
      tau_hat_tr, &                                                                                 !< threshold stress for transformation
      V_tw, &                                                                                       !< volume of a new twin
      V_tr, &                                                                                       !< volume of a new martensite disc
      tau_r_tw, &                                                                                   !< stress to bring partials close together (twin)
      tau_r_tr                                                                                      !< stress to bring partials close together (trans)
  end type tDislotwinMicrostructure

!--------------------------------------------------------------------------------------------------
! containers for parameters and state
  type(tParameters),              allocatable, dimension(:) :: param
  type(tDislotwinState),          allocatable, dimension(:) :: &
    dotState, &
    state
  type(tDislotwinMicrostructure), allocatable, dimension(:) :: dependentState

contains


!--------------------------------------------------------------------------------------------------
!> @brief Perform module initialization.
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function plastic_dislotwin_init() result(myPlasticity)

  logical, dimension(:), allocatable :: myPlasticity
  integer :: &
    Ninstance, &
    p, i, &
    NipcMyPhase, &
    sizeState, sizeDotState, &
    startIndex, endIndex
  integer,     dimension(:), allocatable :: &
    N_sl, N_tw, N_tr
  real(pReal), allocatable, dimension(:) :: &
    rho_mob_0, &                                                                                    !< initial unipolar dislocation density per slip system
    rho_dip_0                                                                                       !< initial dipole dislocation density per slip system
  character(len=pStringLen) :: &
    extmsg = ''
  class(tNode), pointer :: &
    phases, &
    phase, &
    pl

  write(6,'(/,a)') ' <<<+-  constitutive_dislotwin init  -+>>>'

  write(6,'(/,a)') ' Ma and Roters, Acta Materialia 52(12):3603–3612, 2004'
  write(6,'(a)')   ' https://doi.org/10.1016/j.actamat.2004.04.012'

  write(6,'(/,a)') ' Roters et al., Computational Materials Science 39:91–95, 2007'
  write(6,'(a)')   ' https://doi.org/10.1016/j.commatsci.2006.04.014'

  write(6,'(/,a)') ' Wong et al., Acta Materialia 118:140–151, 2016'
  write(6,'(a,/)') ' https://doi.org/10.1016/j.actamat.2016.07.032'

  myPlasticity = plastic_active('dislotwin')

  Ninstance = count(myPlasticity)
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
  if(Ninstance == 0) return

  allocate(param(Ninstance))
  allocate(state(Ninstance))
  allocate(dotState(Ninstance))
  allocate(dependentState(Ninstance))

  phases => material_root%get('phase')
  i = 0
  do p = 1, phases%length
    phase => phases%get(p)

    if(.not. myPlasticity(p)) cycle
    i = i + 1
    associate(prm => param(i), &
              dot => dotState(i), &
              stt => state(i), &
              dst => dependentState(i))
    pl  => phase%get('plasticity')

#if defined (__GFORTRAN__)
    prm%output = output_asStrings(pl)
#else
    prm%output = pl%get_asStrings('output',defaultVal=emptyStringArray)
#endif

    ! This data is read in already in lattice
    prm%mu  = lattice_mu(p)
    prm%nu  = lattice_nu(p)
    prm%C66 = lattice_C66(1:6,1:6,p)

!--------------------------------------------------------------------------------------------------
! slip related parameters
    N_sl         = pl%get_asInts('N_sl',defaultVal=emptyIntArray)
    prm%sum_N_sl = sum(abs(N_sl))
    slipActive: if (prm%sum_N_sl > 0) then
      prm%P_sl    = lattice_SchmidMatrix_slip(N_sl,phase%get_asString('lattice'),&
                                              phase%get_asFloat('c/a',defaultVal=0.0_pReal))
      prm%h_sl_sl = lattice_interaction_SlipBySlip(N_sl,pl%get_asFloats('h_sl_sl'), &
                                                   phase%get_asString('lattice'))
      prm%forestProjection = lattice_forestProjection_edge(N_sl,phase%get_asString('lattice'),&
                                                           phase%get_asFloat('c/a',defaultVal=0.0_pReal))
      prm%forestProjection = transpose(prm%forestProjection)

      prm%n0_sl            = lattice_slip_normal(N_sl,phase%get_asString('lattice'),&
                                                 phase%get_asFloat('c/a',defaultVal=0.0_pReal))
      prm%fccTwinTransNucleation = merge(.true., .false., lattice_structure(p) == lattice_FCC_ID) &
                                 .and. (N_sl(1) == 12)
      if(prm%fccTwinTransNucleation) prm%fcc_twinNucleationSlipPair = lattice_FCC_TWINNUCLEATIONSLIPPAIR

      rho_mob_0                = pl%get_asFloats('rho_mob_0',   requiredSize=size(N_sl))
      rho_dip_0                = pl%get_asFloats('rho_dip_0',   requiredSize=size(N_sl))
      prm%v0                   = pl%get_asFloats('v_0',       requiredSize=size(N_sl))
      prm%b_sl                 = pl%get_asFloats('b_sl',      requiredSize=size(N_sl))
      prm%Delta_F              = pl%get_asFloats('Q_s',       requiredSize=size(N_sl))
      prm%CLambdaSlip          = pl%get_asFloats('i_sl',      requiredSize=size(N_sl))
      prm%p                    = pl%get_asFloats('p_sl',      requiredSize=size(N_sl))
      prm%q                    = pl%get_asFloats('q_sl',      requiredSize=size(N_sl))
      prm%B                    = pl%get_asFloats('B',         requiredSize=size(N_sl), &
                                                  defaultVal=[(0.0_pReal, i=1,size(N_sl))])

      prm%tau_0                = pl%get_asFloat('tau_0')
      prm%CEdgeDipMinDistance  = pl%get_asFloat('D_a')
      prm%D0                   = pl%get_asFloat('D_0')
      prm%Qsd                  = pl%get_asFloat('Q_cl')
      prm%ExtendedDislocations = pl%get_asBool('extend_dislocations',defaultVal = .false.)
      if (prm%ExtendedDislocations) then
        prm%SFE_0K             = pl%get_asFloat('Gamma_sf_0K')
        prm%dSFE_dT            = pl%get_asFloat('dGamma_sf_dT')
      endif

      prm%dipoleformation = .not. pl%get_asBool('no_dipole_formation',defaultVal = .false.)

      ! multiplication factor according to crystal structure (nearest neighbors bcc vs fcc/hex)
      ! details: Argon & Moffat, Acta Metallurgica, Vol. 29, pg 293 to 299, 1981
      prm%omega = pl%get_asFloat('omega',  defaultVal = 1000.0_pReal) &
                * merge(12.0_pReal,8.0_pReal,any(lattice_structure(p) == [lattice_FCC_ID,lattice_HEX_ID]))

      ! expand: family => system
      rho_mob_0        = math_expand(rho_mob_0,       N_sl)
      rho_dip_0        = math_expand(rho_dip_0,       N_sl)
      prm%v0           = math_expand(prm%v0,          N_sl)
      prm%b_sl         = math_expand(prm%b_sl,        N_sl)
      prm%Delta_F      = math_expand(prm%Delta_F,     N_sl)
      prm%CLambdaSlip  = math_expand(prm%CLambdaSlip, N_sl)
      prm%p            = math_expand(prm%p,           N_sl)
      prm%q            = math_expand(prm%q,           N_sl)
      prm%B            = math_expand(prm%B,           N_sl)

      ! sanity checks
      if (    prm%D0           <= 0.0_pReal)          extmsg = trim(extmsg)//' D_0'
      if (    prm%Qsd          <= 0.0_pReal)          extmsg = trim(extmsg)//' Q_cl'
      if (any(rho_mob_0        <  0.0_pReal))         extmsg = trim(extmsg)//' rho_mob_0'
      if (any(rho_dip_0        <  0.0_pReal))         extmsg = trim(extmsg)//' rho_dip_0'
      if (any(prm%v0           <  0.0_pReal))         extmsg = trim(extmsg)//' v_0'
      if (any(prm%b_sl         <= 0.0_pReal))         extmsg = trim(extmsg)//' b_sl'
      if (any(prm%Delta_F      <= 0.0_pReal))         extmsg = trim(extmsg)//' Q_s'
      if (any(prm%CLambdaSlip  <= 0.0_pReal))         extmsg = trim(extmsg)//' i_sl'
      if (any(prm%B            <  0.0_pReal))         extmsg = trim(extmsg)//' B'
      if (any(prm%p<=0.0_pReal .or. prm%p>1.0_pReal)) extmsg = trim(extmsg)//' p_sl'
      if (any(prm%q< 1.0_pReal .or. prm%q>2.0_pReal)) extmsg = trim(extmsg)//' q_sl'
    else slipActive
      rho_mob_0 = emptyRealArray; rho_dip_0 = emptyRealArray
      allocate(prm%b_sl,prm%Delta_F,prm%v0,prm%CLambdaSlip,prm%p,prm%q,prm%B,source=emptyRealArray)
      allocate(prm%forestProjection(0,0),prm%h_sl_sl(0,0))
    endif slipActive

!--------------------------------------------------------------------------------------------------
! twin related parameters
    N_tw         = pl%get_asInts('N_tw', defaultVal=emptyIntArray)
    prm%sum_N_tw = sum(abs(N_tw))
    twinActive: if (prm%sum_N_tw > 0) then
      prm%P_tw  = lattice_SchmidMatrix_twin(N_tw,phase%get_asString('lattice'),&
                                                   phase%get_asFloat('c/a',defaultVal=0.0_pReal))
      prm%h_tw_tw   = lattice_interaction_TwinByTwin(N_tw,&
                                                     pl%get_asFloats('h_tw_tw'), &
                                                     phase%get_asString('lattice'))

      prm%b_tw      = pl%get_asFloats('b_tw',     requiredSize=size(N_tw))
      prm%t_tw      = pl%get_asFloats('t_tw',     requiredSize=size(N_tw))
      prm%r         = pl%get_asFloats('p_tw',     requiredSize=size(N_tw))

      prm%xc_twin   = pl%get_asFloat('x_c_tw')
      prm%L_tw      = pl%get_asFloat('L_tw')
      prm%i_tw      = pl%get_asFloat('i_tw')

      prm%gamma_char= lattice_characteristicShear_Twin(N_tw,phase%get_asString('lattice'),&
                                                       phase%get_asFloat('c/a',defaultVal=0.0_pReal))

      prm%C66_tw    = lattice_C66_twin(N_tw,prm%C66,phase%get_asString('lattice'),&
                                       phase%get_asFloat('c/a',defaultVal=0.0_pReal))

      if (.not. prm%fccTwinTransNucleation) then
        prm%dot_N_0_tw = pl%get_asFloats('dot_N_0_tw')
        prm%dot_N_0_tw = math_expand(prm%dot_N_0_tw,N_tw)
      endif

      ! expand: family => system
      prm%b_tw = math_expand(prm%b_tw,N_tw)
      prm%t_tw = math_expand(prm%t_tw,N_tw)
      prm%r    = math_expand(prm%r,N_tw)

      ! sanity checks
      if (    prm%xc_twin       < 0.0_pReal)  extmsg = trim(extmsg)//' x_c_twin'
      if (    prm%L_tw          < 0.0_pReal)  extmsg = trim(extmsg)//' L_tw'
      if (    prm%i_tw          < 0.0_pReal)  extmsg = trim(extmsg)//' i_tw'
      if (any(prm%b_tw          < 0.0_pReal)) extmsg = trim(extmsg)//' b_tw'
      if (any(prm%t_tw          < 0.0_pReal)) extmsg = trim(extmsg)//' t_tw'
      if (any(prm%r             < 0.0_pReal)) extmsg = trim(extmsg)//' p_tw'
      if (.not. prm%fccTwinTransNucleation) then
        if (any(prm%dot_N_0_tw  < 0.0_pReal)) extmsg = trim(extmsg)//' dot_N_0_tw'
      endif
    else twinActive
      allocate(prm%gamma_char,prm%b_tw,prm%dot_N_0_tw,prm%t_tw,prm%r,source=emptyRealArray)
      allocate(prm%h_tw_tw(0,0))
    endif twinActive

!--------------------------------------------------------------------------------------------------
! transformation related parameters
    N_tr         = pl%get_asInts('N_tr', defaultVal=emptyIntArray)
    prm%sum_N_tr = sum(abs(N_tr))
    transActive: if (prm%sum_N_tr > 0) then
      prm%b_tr = pl%get_asFloats('b_tr')
      prm%b_tr = math_expand(prm%b_tr,N_tr)

      prm%h             = pl%get_asFloat('h',       defaultVal=0.0_pReal) ! ToDo: How to handle that???
      prm%i_tr          = pl%get_asFloat('i_tr',    defaultVal=0.0_pReal) ! ToDo: How to handle that???
      prm%gamma_fcc_hex = pl%get_asFloat('delta_G')
      prm%xc_trans      = pl%get_asFloat('x_c_tr',  defaultVal=0.0_pReal) ! ToDo: How to handle that???
      prm%L_tr          = pl%get_asFloat('L_tr')

      prm%h_tr_tr = lattice_interaction_TransByTrans(N_tr,pl%get_asFloats('h_tr_tr'), &
                                                     phase%get_asString('lattice'))

      prm%C66_tr  = lattice_C66_trans(N_tr,prm%C66,pl%get_asString('trans_lattice_structure'), &
                                      0.0_pReal, &
                                      pl%get_asFloat('a_bcc', defaultVal=0.0_pReal), &
                                      pl%get_asFloat('a_fcc', defaultVal=0.0_pReal))

      prm%P_tr    = lattice_SchmidMatrix_trans(N_tr,pl%get_asString('trans_lattice_structure'), &
                                               0.0_pReal, &
                                               pl%get_asFloat('a_bcc', defaultVal=0.0_pReal), &
                                               pl%get_asFloat('a_fcc', defaultVal=0.0_pReal))

      if (lattice_structure(p) /= lattice_FCC_ID) then
        prm%dot_N_0_tr = pl%get_asFloats('dot_N_0_tr')
        prm%dot_N_0_tr = math_expand(prm%dot_N_0_tr,N_tr)
      endif
      prm%t_tr = pl%get_asFloats('t_tr')
      prm%t_tr = math_expand(prm%t_tr,N_tr)
      prm%s    = pl%get_asFloats('p_tr',defaultVal=[0.0_pReal])
      prm%s    = math_expand(prm%s,N_tr)

      ! sanity checks
      if (    prm%xc_trans      < 0.0_pReal)  extmsg = trim(extmsg)//' x_c_trans'
      if (    prm%L_tr          < 0.0_pReal)  extmsg = trim(extmsg)//' L_tr'
      if (    prm%i_tr          < 0.0_pReal)  extmsg = trim(extmsg)//' i_tr'
      if (any(prm%t_tr          < 0.0_pReal)) extmsg = trim(extmsg)//' t_tr'
      if (any(prm%s             < 0.0_pReal)) extmsg = trim(extmsg)//' p_tr'
      if (lattice_structure(p) /= lattice_FCC_ID) then
        if (any(prm%dot_N_0_tr  < 0.0_pReal)) extmsg = trim(extmsg)//' dot_N_0_tr'
      endif
    else transActive
      allocate(prm%s,prm%b_tr,prm%t_tr,prm%dot_N_0_tr,source=emptyRealArray)
      allocate(prm%h_tr_tr(0,0))
    endif transActive

!--------------------------------------------------------------------------------------------------
! shearband related parameters
    prm%sbVelocity = pl%get_asFloat('v_sb',defaultVal=0.0_pReal)
    if (prm%sbVelocity > 0.0_pReal) then
      prm%sbResistance = pl%get_asFloat('xi_sb')
      prm%E_sb         = pl%get_asFloat('Q_sb')
      prm%p_sb         = pl%get_asFloat('p_sb')
      prm%q_sb         = pl%get_asFloat('q_sb')

      ! sanity checks
      if (prm%sbResistance  <  0.0_pReal) extmsg = trim(extmsg)//' xi_sb'
      if (prm%E_sb          <  0.0_pReal) extmsg = trim(extmsg)//' Q_sb'
      if (prm%p_sb          <= 0.0_pReal) extmsg = trim(extmsg)//' p_sb'
      if (prm%q_sb          <= 0.0_pReal) extmsg = trim(extmsg)//' q_sb'
    endif

!--------------------------------------------------------------------------------------------------
! parameters required for several mechanisms and their interactions
    if(prm%sum_N_sl + prm%sum_N_tw + prm%sum_N_tw > 0) &
      prm%D = pl%get_asFloat('D')

    twinOrSlipActive: if (prm%sum_N_tw + prm%sum_N_tr > 0) then
      prm%SFE_0K  = pl%get_asFloat('Gamma_sf_0K')
      prm%dSFE_dT = pl%get_asFloat('dGamma_sf_dT')
      prm%V_cs    = pl%get_asFloat('V_cs')
    endif twinOrSlipActive

    slipAndTwinActive: if (prm%sum_N_sl * prm%sum_N_tw > 0) then
      prm%h_sl_tw = lattice_interaction_SlipByTwin(N_sl,N_tw,&
                                                   pl%get_asFloats('h_sl_tw'), &
                                                   phase%get_asString('lattice'))
      if (prm%fccTwinTransNucleation .and. size(N_tw) /= 1) extmsg = trim(extmsg)//' interaction_sliptwin'
    endif slipAndTwinActive

    slipAndTransActive: if (prm%sum_N_sl * prm%sum_N_tr > 0) then
      prm%h_sl_tr = lattice_interaction_SlipByTrans(N_sl,N_tr,&
                                                    pl%get_asFloats('h_sl_tr'), &
                                                    phase%get_asString('lattice'))
      if (prm%fccTwinTransNucleation .and. size(N_tr) /= 1) extmsg = trim(extmsg)//' interaction_sliptrans'
    endif slipAndTransActive

!--------------------------------------------------------------------------------------------------
! allocate state arrays
    NipcMyPhase  = count(material_phaseAt == p) * discretization_nIP
    sizeDotState = size(['rho_mob ','rho_dip ','gamma_sl']) * prm%sum_N_sl &
                 + size(['f_tw'])                           * prm%sum_N_tw &
                 + size(['f_tr'])                           * prm%sum_N_tr
    sizeState = sizeDotState

    
    call constitutive_allocateState(plasticState(p),NipcMyPhase,sizeState,sizeDotState,0)

!--------------------------------------------------------------------------------------------------
! locally defined state aliases and initialization of state0 and atol
    startIndex = 1
    endIndex   = prm%sum_N_sl
    stt%rho_mob=>plasticState(p)%state(startIndex:endIndex,:)
    stt%rho_mob= spread(rho_mob_0,2,NipcMyPhase)
    dot%rho_mob=>plasticState(p)%dotState(startIndex:endIndex,:)
    plasticState(p)%atol(startIndex:endIndex) = pl%get_asFloat('atol_rho',defaultVal=1.0_pReal)
    if (any(plasticState(p)%atol(startIndex:endIndex) < 0.0_pReal)) extmsg = trim(extmsg)//' atol_rho'

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_sl
    stt%rho_dip=>plasticState(p)%state(startIndex:endIndex,:)
    stt%rho_dip= spread(rho_dip_0,2,NipcMyPhase)
    dot%rho_dip=>plasticState(p)%dotState(startIndex:endIndex,:)
    plasticState(p)%atol(startIndex:endIndex) = pl%get_asFloat('atol_rho',defaultVal=1.0_pReal)

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_sl
    stt%gamma_sl=>plasticState(p)%state(startIndex:endIndex,:)
    dot%gamma_sl=>plasticState(p)%dotState(startIndex:endIndex,:)
    plasticState(p)%atol(startIndex:endIndex) = 1.0e-2_pReal
    ! global alias
    plasticState(p)%slipRate        => plasticState(p)%dotState(startIndex:endIndex,:)

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_tw
    stt%f_tw=>plasticState(p)%state(startIndex:endIndex,:)
    dot%f_tw=>plasticState(p)%dotState(startIndex:endIndex,:)
    plasticState(p)%atol(startIndex:endIndex) = pl%get_asFloat('f_twin',defaultVal=1.0e-7_pReal)
    if (any(plasticState(p)%atol(startIndex:endIndex) < 0.0_pReal)) extmsg = trim(extmsg)//' f_twin'

    startIndex = endIndex + 1
    endIndex   = endIndex + prm%sum_N_tr
    stt%f_tr=>plasticState(p)%state(startIndex:endIndex,:)
    dot%f_tr=>plasticState(p)%dotState(startIndex:endIndex,:)
    plasticState(p)%atol(startIndex:endIndex) = pl%get_asFloat('f_trans',defaultVal=1.0e-6_pReal)
    if (any(plasticState(p)%atol(startIndex:endIndex) < 0.0_pReal)) extmsg = trim(extmsg)//' f_trans'

    allocate(dst%Lambda_sl             (prm%sum_N_sl,NipcMyPhase),source=0.0_pReal)
    allocate(dst%tau_pass              (prm%sum_N_sl,NipcMyPhase),source=0.0_pReal)

    allocate(dst%Lambda_tw             (prm%sum_N_tw,NipcMyPhase),source=0.0_pReal)
    allocate(dst%tau_hat_tw            (prm%sum_N_tw,NipcMyPhase),source=0.0_pReal)
    allocate(dst%tau_r_tw              (prm%sum_N_tw,NipcMyPhase),source=0.0_pReal)
    allocate(dst%V_tw                  (prm%sum_N_tw,NipcMyPhase),source=0.0_pReal)

    allocate(dst%Lambda_tr             (prm%sum_N_tr,NipcMyPhase),source=0.0_pReal)
    allocate(dst%tau_hat_tr            (prm%sum_N_tr,NipcMyPhase),source=0.0_pReal)
    allocate(dst%tau_r_tr              (prm%sum_N_tr,NipcMyPhase),source=0.0_pReal)
    allocate(dst%V_tr                  (prm%sum_N_tr,NipcMyPhase),source=0.0_pReal)

    plasticState(p)%state0 = plasticState(p)%state                                                  ! ToDo: this could be done centrally

    end associate

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
    if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg)//'(dislotwin)')

  enddo

end function plastic_dislotwin_init


!--------------------------------------------------------------------------------------------------
!> @brief Return the homogenized elasticity matrix.
!--------------------------------------------------------------------------------------------------
module function plastic_dislotwin_homogenizedC(ipc,ip,el) result(homogenizedC)

  real(pReal), dimension(6,6) :: &
    homogenizedC
  integer,     intent(in) :: &
    ipc, &                                                                                          !< component-ID of integration point
    ip, &                                                                                           !< integration point
    el                                                                                              !< element

  integer :: i, &
             of
  real(pReal) :: f_unrotated

  of = material_phasememberAt(ipc,ip,el)
  associate(prm => param(phase_plasticityInstance(material_phaseAt(ipc,el))),&
            stt => state(phase_plasticityInstance(material_phaseAT(ipc,el))))

  f_unrotated = 1.0_pReal &
              - sum(stt%f_tw(1:prm%sum_N_tw,of)) &
              - sum(stt%f_tr(1:prm%sum_N_tr,of))

  homogenizedC = f_unrotated * prm%C66
  do i=1,prm%sum_N_tw
    homogenizedC = homogenizedC &
                 + stt%f_tw(i,of)*prm%C66_tw(1:6,1:6,i)
  enddo
  do i=1,prm%sum_N_tr
    homogenizedC = homogenizedC &
                 + stt%f_tr(i,of)*prm%C66_tr(1:6,1:6,i)
  enddo

  end associate

end function plastic_dislotwin_homogenizedC


!--------------------------------------------------------------------------------------------------
!> @brief Calculate plastic velocity gradient and its tangent.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_dislotwin_LpAndItsTangent(Lp,dLp_dMp,Mp,T,instance,of)

  real(pReal), dimension(3,3),     intent(out) :: Lp
  real(pReal), dimension(3,3,3,3), intent(out) :: dLp_dMp
  real(pReal), dimension(3,3),     intent(in)  :: Mp
  integer,                         intent(in)  :: instance,of
  real(pReal),                     intent(in)  :: T

  integer :: i,k,l,m,n
  real(pReal) :: &
     f_unrotated,StressRatio_p,&
     BoltzmannRatio, &
     ddot_gamma_dtau, &
     tau
  real(pReal), dimension(param(instance)%sum_N_sl) :: &
    dot_gamma_sl,ddot_gamma_dtau_slip
  real(pReal), dimension(param(instance)%sum_N_tw) :: &
    dot_gamma_twin,ddot_gamma_dtau_twin
  real(pReal), dimension(param(instance)%sum_N_tr) :: &
    dot_gamma_tr,ddot_gamma_dtau_trans
  real(pReal):: dot_gamma_sb
  real(pReal), dimension(3,3) :: eigVectors, P_sb
  real(pReal), dimension(3)   :: eigValues
  real(pReal), dimension(3,6), parameter :: &
    sb_sComposition = &
      reshape(real([&
         1, 0, 1, &
         1, 0,-1, &
         1, 1, 0, &
         1,-1, 0, &
         0, 1, 1, &
         0, 1,-1  &
         ],pReal),[ 3,6]), &
    sb_mComposition = &
      reshape(real([&
         1, 0,-1, &
         1, 0,+1, &
         1,-1, 0, &
         1, 1, 0, &
         0, 1,-1, &
         0, 1, 1  &
         ],pReal),[ 3,6])

  associate(prm => param(instance), stt => state(instance))

  f_unrotated = 1.0_pReal &
              - sum(stt%f_tw(1:prm%sum_N_tw,of)) &
              - sum(stt%f_tr(1:prm%sum_N_tr,of))

  Lp = 0.0_pReal
  dLp_dMp = 0.0_pReal

  call kinetics_slip(Mp,T,instance,of,dot_gamma_sl,ddot_gamma_dtau_slip)
  slipContribution: do i = 1, prm%sum_N_sl
    Lp = Lp + dot_gamma_sl(i)*prm%P_sl(1:3,1:3,i)
    forall (k=1:3,l=1:3,m=1:3,n=1:3) &
      dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                       + ddot_gamma_dtau_slip(i) * prm%P_sl(k,l,i) * prm%P_sl(m,n,i)
  enddo slipContribution

  call kinetics_twin(Mp,T,dot_gamma_sl,instance,of,dot_gamma_twin,ddot_gamma_dtau_twin)
  twinContibution: do i = 1, prm%sum_N_tw
    Lp = Lp + dot_gamma_twin(i)*prm%P_tw(1:3,1:3,i)
    forall (k=1:3,l=1:3,m=1:3,n=1:3) &
      dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                       + ddot_gamma_dtau_twin(i)* prm%P_tw(k,l,i)*prm%P_tw(m,n,i)
  enddo twinContibution

  call kinetics_trans(Mp,T,dot_gamma_sl,instance,of,dot_gamma_tr,ddot_gamma_dtau_trans)
  transContibution: do i = 1, prm%sum_N_tr
    Lp = Lp + dot_gamma_tr(i)*prm%P_tr(1:3,1:3,i)
    forall (k=1:3,l=1:3,m=1:3,n=1:3) &
      dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                       + ddot_gamma_dtau_trans(i)* prm%P_tr(k,l,i)*prm%P_tr(m,n,i)
  enddo transContibution

  Lp      = Lp      * f_unrotated
  dLp_dMp = dLp_dMp * f_unrotated

  shearBandingContribution: if(dNeq0(prm%sbVelocity)) then

    BoltzmannRatio = prm%E_sb/(kB*T)
    call math_eigh33(eigValues,eigVectors,Mp)                                                       ! is Mp symmetric by design?

    do i = 1,6
      P_sb = 0.5_pReal * math_outer(matmul(eigVectors,sb_sComposition(1:3,i)),&
                                    matmul(eigVectors,sb_mComposition(1:3,i)))
      tau = math_tensordot(Mp,P_sb)

      significantShearBandStress: if (abs(tau) > tol_math_check) then
        StressRatio_p = (abs(tau)/prm%sbResistance)**prm%p_sb
        dot_gamma_sb = sign(prm%sbVelocity*exp(-BoltzmannRatio*(1-StressRatio_p)**prm%q_sb), tau)
        ddot_gamma_dtau = abs(dot_gamma_sb)*BoltzmannRatio* prm%p_sb*prm%q_sb/ prm%sbResistance &
                   * (abs(tau)/prm%sbResistance)**(prm%p_sb-1.0_pReal) &
                   * (1.0_pReal-StressRatio_p)**(prm%q_sb-1.0_pReal)

        Lp = Lp + dot_gamma_sb * P_sb
        forall (k=1:3,l=1:3,m=1:3,n=1:3) &
          dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                           + ddot_gamma_dtau * P_sb(k,l) * P_sb(m,n)
      endif significantShearBandStress
    enddo

  endif shearBandingContribution

  end associate

end subroutine plastic_dislotwin_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief Calculate the rate of change of microstructure.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_dislotwin_dotState(Mp,T,instance,of)

  real(pReal), dimension(3,3),  intent(in):: &
    Mp                                                                                              !< Mandel stress
  real(pReal),                  intent(in) :: &
    T                                                                                               !< temperature at integration point
  integer,                      intent(in) :: &
    instance, &
    of

  integer :: i
  real(pReal) :: &
    f_unrotated, &
    rho_dip_distance, &
    v_cl, &                                                                                         !< climb velocity
    Gamma, &                                                                                        !< stacking fault energy
    tau, &
    sigma_cl, &                                                                                     !< climb stress
    b_d                                                                                             !< ratio of burgers vector to stacking fault width
  real(pReal), dimension(param(instance)%sum_N_sl) :: &
    dot_rho_dip_formation, &
    dot_rho_dip_climb, &
    rho_dip_distance_min, &
    dot_gamma_sl
  real(pReal), dimension(param(instance)%sum_N_tw) :: &
    dot_gamma_twin
  real(pReal), dimension(param(instance)%sum_N_tr) :: &
    dot_gamma_tr

  associate(prm => param(instance),    stt => state(instance), &
            dot => dotState(instance), dst => dependentState(instance))

  f_unrotated = 1.0_pReal &
              - sum(stt%f_tw(1:prm%sum_N_tw,of)) &
              - sum(stt%f_tr(1:prm%sum_N_tr,of))

  call kinetics_slip(Mp,T,instance,of,dot_gamma_sl)
  dot%gamma_sl(:,of) = abs(dot_gamma_sl)

  rho_dip_distance_min = prm%CEdgeDipMinDistance*prm%b_sl

  slipState: do i = 1, prm%sum_N_sl
    tau = math_tensordot(Mp,prm%P_sl(1:3,1:3,i))

    significantSlipStress: if (dEq0(tau)) then
      dot_rho_dip_formation(i) = 0.0_pReal
      dot_rho_dip_climb(i) = 0.0_pReal
    else significantSlipStress
      rho_dip_distance = 3.0_pReal*prm%mu*prm%b_sl(i)/(16.0_pReal*PI*abs(tau))
      rho_dip_distance = math_clip(rho_dip_distance, right = dst%Lambda_sl(i,of))
      rho_dip_distance = math_clip(rho_dip_distance, left  = rho_dip_distance_min(i))

      if (prm%dipoleFormation) then
        dot_rho_dip_formation(i) = 2.0_pReal*(rho_dip_distance-rho_dip_distance_min(i))/prm%b_sl(i) &
                                 * stt%rho_mob(i,of)*abs(dot_gamma_sl(i))
      else
        dot_rho_dip_formation(i) = 0.0_pReal
      endif

      if (dEq(rho_dip_distance,rho_dip_distance_min(i))) then
        dot_rho_dip_climb(i) = 0.0_pReal
      else
      !@details: Refer: Argon & Moffat, Acta Metallurgica, Vol. 29, pg 293 to 299, 1981
        sigma_cl = dot_product(prm%n0_sl(1:3,i),matmul(Mp,prm%n0_sl(1:3,i)))
        if (prm%ExtendedDislocations) then
          Gamma = prm%SFE_0K + prm%dSFE_dT * T
          b_d = 24.0_pReal*PI*(1.0_pReal - prm%nu)/(2.0_pReal + prm%nu)* Gamma/(prm%mu*prm%b_sl(i))
        else
          b_d = 1.0_pReal
        endif
        v_cl = 2.0_pReal*prm%omega*b_d**2.0_pReal*exp(-prm%Qsd/(kB*T)) &
             * (exp(abs(sigma_cl)*prm%b_sl(i)**3.0_pReal/(kB*T)) - 1.0_pReal)

        dot_rho_dip_climb(i) = 4.0_pReal*v_cl*stt%rho_dip(i,of) &
                             / (rho_dip_distance-rho_dip_distance_min(i))
      endif
    endif significantSlipStress
  enddo slipState

  dot%rho_mob(:,of) = abs(dot_gamma_sl)/(prm%b_sl*dst%Lambda_sl(:,of)) &
                    - dot_rho_dip_formation &
                    - 2.0_pReal*rho_dip_distance_min/prm%b_sl * stt%rho_mob(:,of)*abs(dot_gamma_sl)

  dot%rho_dip(:,of) = dot_rho_dip_formation &
                    - 2.0_pReal*rho_dip_distance_min/prm%b_sl * stt%rho_dip(:,of)*abs(dot_gamma_sl) &
                    - dot_rho_dip_climb

  call kinetics_twin(Mp,T,dot_gamma_sl,instance,of,dot_gamma_twin)
  dot%f_tw(:,of) = f_unrotated*dot_gamma_twin/prm%gamma_char

  call kinetics_trans(Mp,T,dot_gamma_sl,instance,of,dot_gamma_tr)
  dot%f_tr(:,of) = f_unrotated*dot_gamma_tr

  end associate

end subroutine plastic_dislotwin_dotState


!--------------------------------------------------------------------------------------------------
!> @brief Calculate derived quantities from state.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_dislotwin_dependentState(T,instance,of)

  integer,       intent(in) :: &
    instance, &
    of
  real(pReal),   intent(in) :: &
    T

  real(pReal) :: &
    sumf_twin,Gamma,sumf_trans
  real(pReal), dimension(param(instance)%sum_N_sl) :: &
    inv_lambda_sl_sl, &                                                                             !< 1/mean free distance between 2 forest dislocations seen by a moving dislocation
    inv_lambda_sl_tw, &                                                                             !< 1/mean free distance between 2 twin stacks from different systems seen by a moving dislocation
    inv_lambda_sl_tr                                                                                !< 1/mean free distance between 2 martensite lamellar from different systems seen by a moving dislocation
  real(pReal), dimension(param(instance)%sum_N_tw) :: &
    inv_lambda_tw_tw, &                                                                             !< 1/mean free distance between 2 twin stacks from different systems seen by a growing twin
    f_over_t_tw
   real(pReal), dimension(param(instance)%sum_N_tr) :: &
    inv_lambda_tr_tr, &                                                                             !< 1/mean free distance between 2 martensite stacks from different systems seen by a growing martensite
    f_over_t_tr
  real(pReal), dimension(:), allocatable :: &
    x0


  associate(prm => param(instance),&
            stt => state(instance),&
            dst => dependentState(instance))

  sumf_twin  = sum(stt%f_tw(1:prm%sum_N_tw,of))
  sumf_trans = sum(stt%f_tr(1:prm%sum_N_tr,of))

  Gamma = prm%SFE_0K + prm%dSFE_dT * T

  !* rescaled volume fraction for topology
  f_over_t_tw = stt%f_tw(1:prm%sum_N_tw,of)/prm%t_tw                                                ! this is per system ...
  f_over_t_tr = sumf_trans/prm%t_tr                                                                 ! but this not
                                                                                                    ! ToDo ...Physically correct, but naming could be adjusted

  inv_lambda_sl_sl = sqrt(matmul(prm%forestProjection, &
                                 stt%rho_mob(:,of)+stt%rho_dip(:,of)))/prm%CLambdaSlip

  if (prm%sum_N_tw > 0 .and. prm%sum_N_sl > 0) &
    inv_lambda_sl_tw = matmul(prm%h_sl_tw,f_over_t_tw)/(1.0_pReal-sumf_twin)

  inv_lambda_tw_tw = matmul(prm%h_tw_tw,f_over_t_tw)/(1.0_pReal-sumf_twin)

  if (prm%sum_N_tr > 0 .and. prm%sum_N_sl > 0) &
    inv_lambda_sl_tr = matmul(prm%h_sl_tr,f_over_t_tr)/(1.0_pReal-sumf_trans)

  inv_lambda_tr_tr = matmul(prm%h_tr_tr,f_over_t_tr)/(1.0_pReal-sumf_trans)

  if ((prm%sum_N_tw > 0) .or. (prm%sum_N_tr > 0)) then                                              ! ToDo: better logic needed here
    dst%Lambda_sl(:,of) = prm%D &
                        / (1.0_pReal+prm%D*(inv_lambda_sl_sl + inv_lambda_sl_tw + inv_lambda_sl_tr))
  else
    dst%Lambda_sl(:,of) = prm%D &
                        / (1.0_pReal+prm%D*inv_lambda_sl_sl) !!!!!! correct?
  endif

  dst%Lambda_tw(:,of) = prm%i_tw*prm%D/(1.0_pReal+prm%D*inv_lambda_tw_tw)
  dst%Lambda_tr(:,of) = prm%i_tr*prm%D/(1.0_pReal+prm%D*inv_lambda_tr_tr)

  !* threshold stress for dislocation motion
  dst%tau_pass(:,of) = prm%mu*prm%b_sl* sqrt(matmul(prm%h_sl_sl,stt%rho_mob(:,of)+stt%rho_dip(:,of)))

  !* threshold stress for growing twin/martensite
  if(prm%sum_N_tw == prm%sum_N_sl) &
    dst%tau_hat_tw(:,of) = Gamma/(3.0_pReal*prm%b_tw) &
                         + 3.0_pReal*prm%b_tw*prm%mu/(prm%L_tw*prm%b_sl) ! slip burgers here correct?
  if(prm%sum_N_tr == prm%sum_N_sl) &
    dst%tau_hat_tr(:,of) = Gamma/(3.0_pReal*prm%b_tr) &
                         + 3.0_pReal*prm%b_tr*prm%mu/(prm%L_tr*prm%b_sl) & ! slip burgers here correct?
                         + prm%h*prm%gamma_fcc_hex/ (3.0_pReal*prm%b_tr)

  dst%V_tw(:,of) = (PI/4.0_pReal)*prm%t_tw*dst%Lambda_tw(:,of)**2.0_pReal
  dst%V_tr(:,of) = (PI/4.0_pReal)*prm%t_tr*dst%Lambda_tr(:,of)**2.0_pReal


  x0 = prm%mu*prm%b_tw**2.0_pReal/(Gamma*8.0_pReal*PI)*(2.0_pReal+prm%nu)/(1.0_pReal-prm%nu)        ! ToDo: In the paper, this is the burgers vector for slip and is the same for twin and trans
  dst%tau_r_tw(:,of) = prm%mu*prm%b_tw/(2.0_pReal*PI)*(1.0_pReal/(x0+prm%xc_twin)+cos(pi/3.0_pReal)/x0)

  x0 = prm%mu*prm%b_tr**2.0_pReal/(Gamma*8.0_pReal*PI)*(2.0_pReal+prm%nu)/(1.0_pReal-prm%nu)        ! ToDo: In the paper, this is the burgers vector for slip
  dst%tau_r_tr(:,of) = prm%mu*prm%b_tr/(2.0_pReal*PI)*(1.0_pReal/(x0+prm%xc_trans)+cos(pi/3.0_pReal)/x0)

  end associate

end subroutine plastic_dislotwin_dependentState


!--------------------------------------------------------------------------------------------------
!> @brief Write results to HDF5 output file.
!--------------------------------------------------------------------------------------------------
module subroutine plastic_dislotwin_results(instance,group)

  integer,          intent(in) :: instance
  character(len=*), intent(in) :: group

  integer :: o

 associate(prm => param(instance), stt => state(instance), dst => dependentState(instance))
  outputsLoop: do o = 1,size(prm%output)
    select case(trim(prm%output(o)))

      case('rho_mob')
        if(prm%sum_N_sl>0) call results_writeDataset(group,stt%rho_mob,trim(prm%output(o)), &
                                                     'mobile dislocation density','1/m²')
      case('rho_dip')
        if(prm%sum_N_sl>0) call results_writeDataset(group,stt%rho_dip,trim(prm%output(o)), &
                                                     'dislocation dipole density','1/m²')
      case('gamma_sl')
        if(prm%sum_N_sl>0) call results_writeDataset(group,stt%gamma_sl,trim(prm%output(o)), &
                                                     'plastic shear','1')
      case('Lambda_sl')
        if(prm%sum_N_sl>0) call results_writeDataset(group,dst%Lambda_sl,trim(prm%output(o)), &
                                                     'mean free path for slip','m')
      case('tau_pass')
        if(prm%sum_N_sl>0) call results_writeDataset(group,dst%tau_pass,trim(prm%output(o)), &
                                                     'passing stress for slip','Pa')

      case('f_tw')
        if(prm%sum_N_tw>0) call results_writeDataset(group,stt%f_tw,trim(prm%output(o)), &
                                                     'twinned volume fraction','m³/m³')
      case('Lambda_tw')
        if(prm%sum_N_tw>0) call results_writeDataset(group,dst%Lambda_tw,trim(prm%output(o)), &
                                                     'mean free path for twinning','m')
      case('tau_hat_tw')
        if(prm%sum_N_tw>0) call results_writeDataset(group,dst%tau_hat_tw,trim(prm%output(o)), &
                                                     'threshold stress for twinning','Pa')

      case('f_tr')
        if(prm%sum_N_tr>0) call results_writeDataset(group,stt%f_tr,trim(prm%output(o)), &
                                                     'martensite volume fraction','m³/m³')

    end select
  enddo outputsLoop
  end associate

end subroutine plastic_dislotwin_results


!--------------------------------------------------------------------------------------------------
!> @brief Calculate shear rates on slip systems, their derivatives with respect to resolved
!         stress, and the resolved stress.
!> @details Derivatives and resolved stress are calculated only optionally.
! NOTE: Against the common convention, the result (i.e. intent(out)) variables are the last to
! have the optional arguments at the end
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_slip(Mp,T,instance,of, &
                              dot_gamma_sl,ddot_gamma_dtau_slip,tau_slip)

  real(pReal), dimension(3,3),  intent(in) :: &
    Mp                                                                                              !< Mandel stress
  real(pReal),                  intent(in) :: &
    T                                                                                               !< temperature
  integer,                      intent(in) :: &
    instance, &
    of

  real(pReal), dimension(param(instance)%sum_N_sl), intent(out) :: &
    dot_gamma_sl
  real(pReal), dimension(param(instance)%sum_N_sl), optional, intent(out) :: &
    ddot_gamma_dtau_slip, &
    tau_slip
  real(pReal), dimension(param(instance)%sum_N_sl) :: &
    ddot_gamma_dtau

  real(pReal), dimension(param(instance)%sum_N_sl) :: &
    tau, &
    stressRatio, &
    StressRatio_p, &
    BoltzmannRatio, &
    v_wait_inverse, &                                                                               !< inverse of the effective velocity of a dislocation waiting at obstacles (unsigned)
    v_run_inverse, &                                                                                !< inverse of the velocity of a free moving dislocation (unsigned)
    dV_wait_inverse_dTau, &
    dV_run_inverse_dTau, &
    dV_dTau, &
    tau_eff                                                                                         !< effective resolved stress
  integer :: i

  associate(prm => param(instance), stt => state(instance), dst => dependentState(instance))

  do i = 1, prm%sum_N_sl
    tau(i) = math_tensordot(Mp,prm%P_sl(1:3,1:3,i))
  enddo

  tau_eff = abs(tau)-dst%tau_pass(:,of)

  significantStress: where(tau_eff > tol_math_check)
    stressRatio    = tau_eff/prm%tau_0
    StressRatio_p  = stressRatio** prm%p
    BoltzmannRatio = prm%Delta_F/(kB*T)
    v_wait_inverse = prm%v0**(-1.0_pReal) * exp(BoltzmannRatio*(1.0_pReal-StressRatio_p)** prm%q)
    v_run_inverse  = prm%B/(tau_eff*prm%b_sl)

    dot_gamma_sl = sign(stt%rho_mob(:,of)*prm%b_sl/(v_wait_inverse+v_run_inverse),tau)

    dV_wait_inverse_dTau = -1.0_pReal * v_wait_inverse * prm%p * prm%q * BoltzmannRatio &
                         * (stressRatio**(prm%p-1.0_pReal)) &
                         * (1.0_pReal-StressRatio_p)**(prm%q-1.0_pReal) &
                         / prm%tau_0
    dV_run_inverse_dTau  = -1.0_pReal * v_run_inverse/tau_eff
    dV_dTau              = -1.0_pReal * (dV_wait_inverse_dTau+dV_run_inverse_dTau) &
                         / (v_wait_inverse+v_run_inverse)**2.0_pReal
    ddot_gamma_dtau = dV_dTau*stt%rho_mob(:,of)*prm%b_sl
  else where significantStress
    dot_gamma_sl    = 0.0_pReal
    ddot_gamma_dtau = 0.0_pReal
  end where significantStress

  end associate

  if(present(ddot_gamma_dtau_slip)) ddot_gamma_dtau_slip = ddot_gamma_dtau
  if(present(tau_slip))             tau_slip             = tau

end subroutine kinetics_slip


!--------------------------------------------------------------------------------------------------
!> @brief Calculate shear rates on twin systems and their derivatives with respect to resolved
!         stress.
!> @details Derivatives are calculated only optionally.
! NOTE: Against the common convention, the result (i.e. intent(out)) variables are the last to
! have the optional arguments at the end.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_twin(Mp,T,dot_gamma_sl,instance,of,&
                              dot_gamma_twin,ddot_gamma_dtau_twin)

  real(pReal), dimension(3,3),  intent(in) :: &
    Mp                                                                                              !< Mandel stress
  real(pReal),                  intent(in) :: &
    T                                                                                               !< temperature
  integer,                      intent(in) :: &
    instance, &
    of
  real(pReal), dimension(param(instance)%sum_N_sl), intent(in) :: &
    dot_gamma_sl

  real(pReal), dimension(param(instance)%sum_N_tw), intent(out) :: &
    dot_gamma_twin
  real(pReal), dimension(param(instance)%sum_N_tw), optional, intent(out) :: &
    ddot_gamma_dtau_twin

  real, dimension(param(instance)%sum_N_tw) :: &
    tau, &
    Ndot0, &
    stressRatio_r, &
    ddot_gamma_dtau

  integer :: i,s1,s2

  associate(prm => param(instance), stt => state(instance), dst => dependentState(instance))

  do i = 1, prm%sum_N_tw
    tau(i) = math_tensordot(Mp,prm%P_tw(1:3,1:3,i))
    isFCC: if (prm%fccTwinTransNucleation) then
      s1=prm%fcc_twinNucleationSlipPair(1,i)
      s2=prm%fcc_twinNucleationSlipPair(2,i)
      if (tau(i) < dst%tau_r_tw(i,of)) then                                                         ! ToDo: correct?
        Ndot0=(abs(dot_gamma_sl(s1))*(stt%rho_mob(s2,of)+stt%rho_dip(s2,of))+&
               abs(dot_gamma_sl(s2))*(stt%rho_mob(s1,of)+stt%rho_dip(s1,of)))/&                     ! ToDo: MD: it would be more consistent to use shearrates from state
                (prm%L_tw*prm%b_sl(i))*&
                (1.0_pReal-exp(-prm%V_cs/(kB*T)*(dst%tau_r_tw(i,of)-tau(i))))                       ! P_ncs
      else
        Ndot0=0.0_pReal
      end if
    else isFCC
      Ndot0=prm%dot_N_0_tw(i)
    endif isFCC
  enddo

  significantStress: where(tau > tol_math_check)
    StressRatio_r   = (dst%tau_hat_tw(:,of)/tau)**prm%r
    dot_gamma_twin  = prm%gamma_char * dst%V_tw(:,of) * Ndot0*exp(-StressRatio_r)
    ddot_gamma_dtau = (dot_gamma_twin*prm%r/tau)*StressRatio_r
  else where significantStress
    dot_gamma_twin  = 0.0_pReal
    ddot_gamma_dtau = 0.0_pReal
  end where significantStress

  end associate

  if(present(ddot_gamma_dtau_twin)) ddot_gamma_dtau_twin = ddot_gamma_dtau

end subroutine kinetics_twin


!--------------------------------------------------------------------------------------------------
!> @brief Calculate shear rates on transformation systems and their derivatives with respect to
!         resolved stress.
!> @details Derivatives are calculated only optionally.
! NOTE: Against the common convention, the result (i.e. intent(out)) variables are the last to
! have the optional arguments at the end.
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics_trans(Mp,T,dot_gamma_sl,instance,of,&
                              dot_gamma_tr,ddot_gamma_dtau_trans)

  real(pReal), dimension(3,3),  intent(in) :: &
    Mp                                                                                              !< Mandel stress
  real(pReal),                  intent(in) :: &
    T                                                                                               !< temperature
  integer,                      intent(in) :: &
    instance, &
    of
  real(pReal), dimension(param(instance)%sum_N_sl), intent(in) :: &
    dot_gamma_sl

  real(pReal), dimension(param(instance)%sum_N_tr), intent(out) :: &
    dot_gamma_tr
  real(pReal), dimension(param(instance)%sum_N_tr), optional, intent(out) :: &
    ddot_gamma_dtau_trans

  real, dimension(param(instance)%sum_N_tr) :: &
    tau, &
    Ndot0, &
    stressRatio_s, &
    ddot_gamma_dtau

  integer :: i,s1,s2
  associate(prm => param(instance), stt => state(instance), dst => dependentState(instance))

  do i = 1, prm%sum_N_tr
    tau(i) = math_tensordot(Mp,prm%P_tr(1:3,1:3,i))
    isFCC: if (prm%fccTwinTransNucleation) then
      s1=prm%fcc_twinNucleationSlipPair(1,i)
      s2=prm%fcc_twinNucleationSlipPair(2,i)
      if (tau(i) < dst%tau_r_tr(i,of)) then                                                         ! ToDo: correct?
        Ndot0=(abs(dot_gamma_sl(s1))*(stt%rho_mob(s2,of)+stt%rho_dip(s2,of))+&
               abs(dot_gamma_sl(s2))*(stt%rho_mob(s1,of)+stt%rho_dip(s1,of)))/&                     ! ToDo: MD: it would be more consistent to use shearrates from state
                (prm%L_tr*prm%b_sl(i))*&
                (1.0_pReal-exp(-prm%V_cs/(kB*T)*(dst%tau_r_tr(i,of)-tau(i))))                       ! P_ncs
      else
        Ndot0=0.0_pReal
      end if
    else isFCC
      Ndot0=prm%dot_N_0_tr(i)
    endif isFCC
  enddo

  significantStress: where(tau > tol_math_check)
    StressRatio_s   = (dst%tau_hat_tr(:,of)/tau)**prm%s
    dot_gamma_tr    = dst%V_tr(:,of) * Ndot0*exp(-StressRatio_s)
    ddot_gamma_dtau = (dot_gamma_tr*prm%s/tau)*StressRatio_s
  else where significantStress
    dot_gamma_tr  = 0.0_pReal
    ddot_gamma_dtau = 0.0_pReal
  end where significantStress

  end associate

  if(present(ddot_gamma_dtau_trans)) ddot_gamma_dtau_trans = ddot_gamma_dtau

end subroutine kinetics_trans

end submodule plastic_dislotwin
