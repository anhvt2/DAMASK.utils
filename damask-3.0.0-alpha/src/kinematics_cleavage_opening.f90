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
!> @brief material subroutine incorporating kinematics resulting from opening of cleavage planes
!> @details to be done
!--------------------------------------------------------------------------------------------------
submodule(constitutive:constitutive_damage) kinematics_cleavage_opening

  integer, dimension(:), allocatable :: kinematics_cleavage_opening_instance

  type :: tParameters                                                                               !< container type for internal constitutive parameters
    integer :: &
      sum_N_cl                                                                                      !< total number of cleavage planes
    real(pReal) :: &
      sdot0, &                                                                                      !< opening rate of cleavage planes
      n                                                                                             !< damage rate sensitivity
    real(pReal),   dimension(:),   allocatable :: &
      critLoad
    real(pReal), dimension(:,:,:,:), allocatable :: &
      cleavage_systems
  end type tParameters

  type(tParameters), dimension(:), allocatable :: param                                             !< containers of constitutive parameters (len Ninstance)


contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
module function kinematics_cleavage_opening_init(kinematics_length) result(myKinematics)
 
  integer, intent(in)                  :: kinematics_length  
  logical, dimension(:,:), allocatable :: myKinematics

  integer :: Ninstance,p,k
  integer, dimension(:), allocatable :: N_cl                                                        !< active number of cleavage systems per family
  character(len=pStringLen) :: extmsg = ''
  class(tNode), pointer :: &
    phases, &
    phase, &
    pl, &
    kinematics, &
    kinematic_type 
       
  write(6,'(/,a)') ' <<<+-  kinematics_cleavage_opening init  -+>>>'

  myKinematics = kinematics_active('cleavage_opening',kinematics_length)
  
  Ninstance = count(myKinematics)
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
  if(Ninstance == 0) return

  phases => material_root%get('phase')
  allocate(param(Ninstance))
  allocate(kinematics_cleavage_opening_instance(phases%length), source=0)

  do p = 1, phases%length
    if(any(myKinematics(:,p))) kinematics_cleavage_opening_instance(p) = count(myKinematics(:,1:p))
    phase => phases%get(p) 
    pl => phase%get('plasticity')
    if(count(myKinematics(:,p)) == 0) cycle
    kinematics => phase%get('kinematics')
    do k = 1, kinematics%length
      if(myKinematics(k,p)) then
        associate(prm  => param(kinematics_cleavage_opening_instance(p)))
        kinematic_type => kinematics%get(k) 

        N_cl = kinematic_type%get_asInts('N_cl')
        prm%sum_N_cl = sum(abs(N_cl))

        prm%n         = kinematic_type%get_asFloat('q')
        prm%sdot0     = kinematic_type%get_asFloat('dot_o')

        prm%critLoad  = kinematic_type%get_asFloats('g_crit',requiredSize=size(N_cl))

        prm%cleavage_systems  = lattice_SchmidMatrix_cleavage(N_cl,phase%get_asString('lattice'),&
                                                        phase%get_asFloat('c/a',defaultVal=0.0_pReal))

  ! expand: family => system
        prm%critLoad = math_expand(prm%critLoad,N_cl)

  ! sanity checks
        if (prm%n            <= 0.0_pReal)  extmsg = trim(extmsg)//' q'
        if (prm%sdot0        <= 0.0_pReal)  extmsg = trim(extmsg)//' dot_o'
        if (any(prm%critLoad <  0.0_pReal)) extmsg = trim(extmsg)//' g_crit'

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
        if (extmsg /= '') call IO_error(211,ext_msg=trim(extmsg)//'(cleavage_opening)')
        end associate
      endif
    enddo
  enddo


end function kinematics_cleavage_opening_init


!--------------------------------------------------------------------------------------------------
!> @brief  contains the constitutive equation for calculating the velocity gradient
!--------------------------------------------------------------------------------------------------
module subroutine kinematics_cleavage_opening_LiAndItsTangent(Ld, dLd_dTstar, S, ipc, ip, el)

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
    homog, damageOffset, &
    i, k, l, m, n
  real(pReal) :: &
    traction_d, traction_t, traction_n, traction_crit, &
    udotd, dudotd_dt, udott, dudott_dt, udotn, dudotn_dt

  homog = material_homogenizationAt(el)
  damageOffset = damageMapping(homog)%p(ip,el)

  Ld = 0.0_pReal
  dLd_dTstar = 0.0_pReal
  associate(prm => param(kinematics_cleavage_opening_instance(material_phaseAt(ipc,el))))
  do i = 1,prm%sum_N_cl
    traction_crit = prm%critLoad(i)* damage(homog)%p(damageOffset)**2.0_pReal

    traction_d = math_tensordot(S,prm%cleavage_systems(1:3,1:3,1,i))
    if (abs(traction_d) > traction_crit + tol_math_check) then
      udotd = sign(1.0_pReal,traction_d)* prm%sdot0 * ((abs(traction_d) - traction_crit)/traction_crit)**prm%n
      Ld = Ld + udotd*prm%cleavage_systems(1:3,1:3,1,i)
      dudotd_dt = sign(1.0_pReal,traction_d)*udotd*prm%n / (abs(traction_d) - traction_crit)
      forall (k=1:3,l=1:3,m=1:3,n=1:3) &
        dLd_dTstar(k,l,m,n) = dLd_dTstar(k,l,m,n) &
                            + dudotd_dt*prm%cleavage_systems(k,l,1,i) * prm%cleavage_systems(m,n,1,i)
    endif

    traction_t = math_tensordot(S,prm%cleavage_systems(1:3,1:3,2,i))
    if (abs(traction_t) > traction_crit + tol_math_check) then
      udott = sign(1.0_pReal,traction_t)* prm%sdot0 * ((abs(traction_t) - traction_crit)/traction_crit)**prm%n
      Ld = Ld + udott*prm%cleavage_systems(1:3,1:3,2,i)
      dudott_dt = sign(1.0_pReal,traction_t)*udott*prm%n / (abs(traction_t) - traction_crit)
      forall (k=1:3,l=1:3,m=1:3,n=1:3) &
        dLd_dTstar(k,l,m,n) = dLd_dTstar(k,l,m,n) &
                            + dudott_dt*prm%cleavage_systems(k,l,2,i) * prm%cleavage_systems(m,n,2,i)
    endif

    traction_n = math_tensordot(S,prm%cleavage_systems(1:3,1:3,3,i))
    if (abs(traction_n) > traction_crit + tol_math_check) then
      udotn = sign(1.0_pReal,traction_n)* prm%sdot0 * ((abs(traction_n) - traction_crit)/traction_crit)**prm%n
      Ld = Ld + udotn*prm%cleavage_systems(1:3,1:3,3,i)
      dudotn_dt = sign(1.0_pReal,traction_n)*udotn*prm%n / (abs(traction_n) - traction_crit)
      forall (k=1:3,l=1:3,m=1:3,n=1:3) &
        dLd_dTstar(k,l,m,n) = dLd_dTstar(k,l,m,n) &
                            + dudotn_dt*prm%cleavage_systems(k,l,3,i) * prm%cleavage_systems(m,n,3,i)
    endif
  enddo
  end associate

end subroutine kinematics_cleavage_opening_LiAndItsTangent

end submodule kinematics_cleavage_opening
