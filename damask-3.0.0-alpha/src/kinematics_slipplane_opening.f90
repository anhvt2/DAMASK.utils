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
!> @author Luv Sharma, Max-Planck-Institut für Eisenforschung GmbH
!> @author Pratheek Shanthraj, Max-Planck-Institut für Eisenforschung GmbH
!> @brief material subroutine incorporating kinematics resulting from opening of slip planes
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_damage) kinematics_slipplane_opening

  integer, dimension(:), allocatable :: kinematics_slipplane_opening_instance

  type :: tParameters                                                                               !< container type for internal constitutive parameters
    integer :: &
      sum_N_sl                                                                                      !< total number of cleavage planes
    real(pReal) :: &
      sdot0, &                                                                                      !< opening rate of cleavage planes
      n                                                                                             !< damage rate sensitivity
    real(pReal), dimension(:),   allocatable :: &
      critLoad
    real(pReal), dimension(:,:,:), allocatable     :: &
      P_d, &
      P_t, &
      P_n
  end type tParameters

  type(tParameters), dimension(:), allocatable :: param                                             !< containers of constitutive parameters (len Ninstance)


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function kinematics_slipplane_opening_init(kinematics_length) result(myKinematics)

  integer, intent(in)                  :: kinematics_length  
  logical, dimension(:,:), allocatable :: myKinematics

  integer :: Ninstance,p,i,k
  character(len=pStringLen) :: extmsg = ''
  integer,     dimension(:),   allocatable :: N_sl
  real(pReal), dimension(:,:), allocatable :: d,n,t
  class(tNode), pointer :: &
    phases, &
    phase, &
    pl, &
    kinematics, &
    kinematic_type 
 
  write(6,'(/,a)') ' <<<+-  kinematics_slipplane init  -+>>>'

  myKinematics = kinematics_active('slipplane_opening',kinematics_length)
 
  Ninstance = count(myKinematics)
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
  if(Ninstance == 0) return

  phases => material_root%get('phase')
  allocate(kinematics_slipplane_opening_instance(phases%length), source=0)
  allocate(param(Ninstance))

  do p = 1, phases%length
    if(any(myKinematics(:,p))) kinematics_slipplane_opening_instance(p) = count(myKinematics(:,1:p))
    phase => phases%get(p) 
    pl => phase%get('plasticity')
    if(count(myKinematics(:,p)) == 0) cycle
    kinematics => phase%get('kinematics')
    do k = 1, kinematics%length
      if(myKinematics(k,p)) then
        associate(prm  => param(kinematics_slipplane_opening_instance(p)))
        kinematic_type => kinematics%get(k) 

        prm%sdot0    = kinematic_type%get_asFloat('dot_o')
        prm%n        = kinematic_type%get_asFloat('q')
        N_sl         = pl%get_asInts('N_sl')
        prm%sum_N_sl = sum(abs(N_sl))

        d = lattice_slip_direction (N_sl,phase%get_asString('lattice'),&
                                    phase%get_asFloat('c/a',defaultVal=0.0_pReal))
        t = lattice_slip_transverse(N_sl,phase%get_asString('lattice'),&
                                    phase%get_asFloat('c/a',defaultVal=0.0_pReal))
        n = lattice_slip_normal    (N_sl,phase%get_asString('lattice'),&
                                    phase%get_asFloat('c/a',defaultVal=0.0_pReal))
        allocate(prm%P_d(3,3,size(d,2)),prm%P_t(3,3,size(t,2)),prm%P_n(3,3,size(n,2)))

        do i=1, size(n,2)
          prm%P_d(1:3,1:3,i) = math_outer(d(1:3,i), n(1:3,i))
          prm%P_t(1:3,1:3,i) = math_outer(t(1:3,i), n(1:3,i))
          prm%P_n(1:3,1:3,i) = math_outer(n(1:3,i), n(1:3,i))
        enddo

        prm%critLoad = kinematic_type%get_asFloats('g_crit',requiredSize=size(N_sl))

        ! expand: family => system
        prm%critLoad = math_expand(prm%critLoad,N_sl)

        ! sanity checks
        if (prm%n            <= 0.0_pReal)  extmsg = trim(extmsg)//' anisoDuctile_n'
        if (prm%sdot0        <= 0.0_pReal)  extmsg = trim(extmsg)//' anisoDuctile_sdot0'
        if (any(prm%critLoad <  0.0_pReal)) extmsg = trim(extmsg)//' anisoDuctile_critLoad'

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
        if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg)//'(slipplane_opening)')

        end associate
      endif
    enddo
  enddo


end function kinematics_slipplane_opening_init


!--------------------------------------------------------------------------------------------------
!> @brief  contains the constitutive equation for calculating the velocity gradient
!--------------------------------------------------------------------------------------------------
module subroutine kinematics_slipplane_opening_LiAndItsTangent(Ld, dLd_dTstar, S, ipc, ip, el)

  integer, intent(in) :: &
    ipc, &                                                                                          !< grain number
    ip, &                                                                                           !< integration point number
    el                                                                                              !< element number
  real(pReal),   intent(in),  dimension(3,3) :: &
    S
  real(pReal),   intent(out), dimension(3,3) :: &
    Ld                                                                                              !< damage velocity gradient
  real(pReal),   intent(out), dimension(3,3,3,3) :: &
    dLd_dTstar                                                                                      !< derivative of Ld with respect to Tstar (4th-order tensor)

  integer :: &
    instance, phase, &
    homog, damageOffset, &
    i, k, l, m, n
  real(pReal) :: &
    traction_d, traction_t, traction_n, traction_crit, &
    udotd, dudotd_dt, udott, dudott_dt, udotn, dudotn_dt

  phase = material_phaseAt(ipc,el)
  instance = kinematics_slipplane_opening_instance(phase)
  homog = material_homogenizationAt(el)
  damageOffset = damageMapping(homog)%p(ip,el)

  associate(prm => param(instance))
  Ld = 0.0_pReal
  dLd_dTstar = 0.0_pReal
  do i = 1, prm%sum_N_sl

    traction_d = math_tensordot(S,prm%P_d(1:3,1:3,i))
    traction_t = math_tensordot(S,prm%P_t(1:3,1:3,i))
    traction_n = math_tensordot(S,prm%P_n(1:3,1:3,i))

    traction_crit = prm%critLoad(i)* damage(homog)%p(damageOffset)                                  ! degrading critical load carrying capacity by damage

    udotd = sign(1.0_pReal,traction_d)* prm%sdot0* (  abs(traction_d)/traction_crit &
                                                    - abs(traction_d)/prm%critLoad(i))**prm%n
    udott = sign(1.0_pReal,traction_t)* prm%sdot0* (  abs(traction_t)/traction_crit &
                                                    - abs(traction_t)/prm%critLoad(i))**prm%n
    udotn = prm%sdot0* (  max(0.0_pReal,traction_n)/traction_crit &
                        - max(0.0_pReal,traction_n)/prm%critLoad(i))**prm%n

    if (dNeq0(traction_d)) then
      dudotd_dt = udotd*prm%n/traction_d
    else
      dudotd_dt = 0.0_pReal
    endif
    if (dNeq0(traction_t)) then
      dudott_dt = udott*prm%n/traction_t
    else
      dudott_dt = 0.0_pReal
    endif
    if (dNeq0(traction_n)) then
      dudotn_dt = udotn*prm%n/traction_n
    else
      dudotn_dt = 0.0_pReal
    endif

    forall (k=1:3,l=1:3,m=1:3,n=1:3) &
      dLd_dTstar(k,l,m,n) = dLd_dTstar(k,l,m,n) &
                          + dudotd_dt*prm%P_d(k,l,i)*prm%P_d(m,n,i) &
                          + dudott_dt*prm%P_t(k,l,i)*prm%P_t(m,n,i) &
                          + dudotn_dt*prm%P_n(k,l,i)*prm%P_n(m,n,i)

    Ld = Ld &
       + udotd*prm%P_d(1:3,1:3,i) &
       + udott*prm%P_t(1:3,1:3,i) &
       + udotn*prm%P_n(1:3,1:3,i)
  enddo

  end associate

end subroutine kinematics_slipplane_opening_LiAndItsTangent

end submodule kinematics_slipplane_opening
