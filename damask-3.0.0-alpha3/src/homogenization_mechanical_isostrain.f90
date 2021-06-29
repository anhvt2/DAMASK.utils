! Copyright 2011-2021 Max-Planck-Institut für Eisenforschung GmbH
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
submodule(homogenization:mechanical) isostrain

contains

!--------------------------------------------------------------------------------------------------
!> @brief allocates all neccessary fields, reads information from material configuration file
!--------------------------------------------------------------------------------------------------
module subroutine isostrain_init

  integer :: &
    ho, &
    Nmaterialpoints

  print'(/,a)', ' <<<+-  homogenization:mechanical:isostrain init  -+>>>'

  print'(a,i2)', ' # instances: ',count(homogenization_type == HOMOGENIZATION_ISOSTRAIN_ID)
  flush(IO_STDOUT)

  do ho = 1, size(homogenization_type)
    if (homogenization_type(ho) /= HOMOGENIZATION_ISOSTRAIN_ID) cycle

    Nmaterialpoints = count(material_homogenizationAt == ho)
    homogState(ho)%sizeState = 0
    allocate(homogState(ho)%state0(0,Nmaterialpoints))
    allocate(homogState(ho)%state (0,Nmaterialpoints))

  enddo

end subroutine isostrain_init


!--------------------------------------------------------------------------------------------------
!> @brief partitions the deformation gradient onto the constituents
!--------------------------------------------------------------------------------------------------
module subroutine isostrain_partitionDeformation(F,avgF)

  real(pReal),   dimension (:,:,:), intent(out) :: F                                                !< partitioned deformation gradient

  real(pReal),   dimension (3,3),   intent(in)  :: avgF                                             !< average deformation gradient at material point


  F = spread(avgF,3,size(F,3))

end subroutine isostrain_partitionDeformation

end submodule isostrain
