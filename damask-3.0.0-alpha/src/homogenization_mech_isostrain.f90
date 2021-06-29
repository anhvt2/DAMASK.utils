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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Isostrain (full constraint Taylor assuption) homogenization scheme
!--------------------------------------------------------------------------------------------------
submodule(homogenization) homogenization_mech_isostrain

  enum, bind(c); enumerator :: &
    parallel_ID, &
    average_ID
  end enum
 
  type :: tParameters                                                                               !< container type for internal constitutive parameters
    integer :: &
      Nconstituents
    integer(kind(average_ID)) :: &
      mapping
  end type
 
  type(tParameters), dimension(:), allocatable :: param                                             !< containers of constitutive parameters (len Ninstance)
  

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
module subroutine mech_isostrain_init

  integer :: &
    Ninstance, &
    h, &
    NofMyHomog
  class(tNode), pointer :: &
    material_homogenization, &
    homog, &
    homogMech
 
  write(6,'(/,a)')   ' <<<+-  homogenization_mech_isostrain init  -+>>>'
 
  Ninstance = count(homogenization_type == HOMOGENIZATION_ISOSTRAIN_ID)
  write(6,'(a16,1x,i5,/)') '# instances:',Ninstance; flush(6)
 
  allocate(param(Ninstance))                                                                        ! one container of parameters per instance

  material_homogenization => material_root%get('homogenization') 
  do h = 1, size(homogenization_type)
    if (homogenization_type(h) /= HOMOGENIZATION_ISOSTRAIN_ID) cycle
    homog => material_homogenization%get(h)
    homogMech => homog%get('mech')
    associate(prm => param(homogenization_typeInstance(h)))
   
    prm%Nconstituents = homogMech%get_asInt('N_constituents')
    select case(homogMech%get_asString('mapping',defaultVal = 'sum'))
      case ('sum')
        prm%mapping = parallel_ID
      case ('avg')
        prm%mapping = average_ID
      case default
        call IO_error(211,ext_msg='sum'//' (mech_isostrain)')
    end select
 
    NofMyHomog = count(material_homogenizationAt == h)
    homogState(h)%sizeState       = 0
    allocate(homogState(h)%state0   (0,NofMyHomog))
    allocate(homogState(h)%subState0(0,NofMyHomog))
    allocate(homogState(h)%state    (0,NofMyHomog))
    
    end associate
 
  enddo

end subroutine mech_isostrain_init


!--------------------------------------------------------------------------------------------------
!> @brief partitions the deformation gradient onto the constituents
!--------------------------------------------------------------------------------------------------
module subroutine mech_isostrain_partitionDeformation(F,avgF)
  
  real(pReal),   dimension (:,:,:), intent(out) :: F                                                !< partitioned deformation gradient
  
  real(pReal),   dimension (3,3),   intent(in)  :: avgF                                             !< average deformation gradient at material point
 
  F = spread(avgF,3,size(F,3))

end subroutine mech_isostrain_partitionDeformation


!--------------------------------------------------------------------------------------------------
!> @brief derive average stress and stiffness from constituent quantities 
!--------------------------------------------------------------------------------------------------
module subroutine mech_isostrain_averageStressAndItsTangent(avgP,dAvgPdAvgF,P,dPdF,instance)
   
  real(pReal),   dimension (3,3),       intent(out) :: avgP                                         !< average stress at material point
  real(pReal),   dimension (3,3,3,3),   intent(out) :: dAvgPdAvgF                                   !< average stiffness at material point
 
  real(pReal),   dimension (:,:,:),     intent(in)  :: P                                            !< partitioned stresses
  real(pReal),   dimension (:,:,:,:,:), intent(in)  :: dPdF                                         !< partitioned stiffnesses
  integer,                              intent(in)  :: instance 
   
  associate(prm => param(instance))
  
  select case (prm%mapping)
    case (parallel_ID)
      avgP       = sum(P,3)
      dAvgPdAvgF = sum(dPdF,5)
    case (average_ID)
      avgP       = sum(P,3)   /real(prm%Nconstituents,pReal)
      dAvgPdAvgF = sum(dPdF,5)/real(prm%Nconstituents,pReal)
  end select
  
  end associate

end subroutine mech_isostrain_averageStressAndItsTangent

end submodule homogenization_mech_isostrain
