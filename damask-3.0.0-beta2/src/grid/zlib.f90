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
!> @brief  Inflate zlib compressed data
!--------------------------------------------------------------------------------------------------
module zlib
  use prec

  implicit none(type,external)
  private

  public :: &
    zlib_inflate

  interface

    subroutine inflate_C(s_deflated,s_inflated,deflated,inflated) bind(C)
      use, intrinsic :: ISO_C_Binding, only: C_SIGNED_CHAR, C_INT64_T
      implicit none(type,external)

      integer(C_INT64_T),                            intent(in)  :: s_deflated,s_inflated
      integer(C_SIGNED_CHAR), dimension(s_deflated), intent(in)  :: deflated
      integer(C_SIGNED_CHAR), dimension(s_inflated), intent(out) :: inflated
    end subroutine inflate_C

  end interface

contains

!--------------------------------------------------------------------------------------------------
!> @brief Inflate byte-wise representation
!--------------------------------------------------------------------------------------------------
function zlib_inflate(deflated,size_inflated)

  integer(C_SIGNED_CHAR), dimension(:), intent(in) :: deflated
  integer(pI64),                        intent(in) :: size_inflated

  integer(C_SIGNED_CHAR), dimension(size_inflated) :: zlib_inflate


  call inflate_C(size(deflated,kind=C_INT64_T),int(size_inflated,C_INT64_T),deflated,zlib_inflate)

end function zlib_inflate

end module zlib
