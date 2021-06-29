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
!> @author   Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author   Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author   Christoph Kords, Max-Planck-Institut für Eisenforschung GmbH
!> @author   Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @author   Luv Sharma, Max-Planck-Institut für Eisenforschung GmbH
!> @brief    setting precision for real and int type
!--------------------------------------------------------------------------------------------------
module prec
  use, intrinsic :: IEEE_arithmetic

  implicit none
  public

  ! https://software.intel.com/en-us/blogs/2017/03/27/doctor-fortran-in-it-takes-all-kinds
  integer,     parameter :: pReal      = IEEE_selected_real_kind(15,307)                            !< number with 15 significant digits, up to 1e+-307 (typically 64 bit)
#if(INT==8)
  integer,     parameter :: pInt       = selected_int_kind(18)                                      !< number with at least up to +-1e18 (typically 64 bit)
#else
  integer,     parameter :: pInt       = selected_int_kind(9)                                       !< number with at least up to +-1e9 (typically 32 bit)
#endif
  integer,     parameter :: pLongInt   = selected_int_kind(18)                                      !< number with at least up to +-1e18 (typically 64 bit)
  integer,     parameter :: pStringLen = 256                                                        !< default string length
  integer,     parameter :: pPathLen   = 4096                                                       !< maximum length of a path name on linux

  real(pReal), parameter :: tol_math_check = 1.0e-8_pReal                                           !< tolerance for internal math self-checks (rotation)


  type :: group_float                                                                               !< variable length datatype used for storage of state
    real(pReal), dimension(:), pointer :: p
  end type group_float

  ! http://stackoverflow.com/questions/3948210/can-i-have-a-pointer-to-an-item-in-an-allocatable-array
  type :: tState
    integer :: &
      sizeState        = 0, &                                                                       !< size of state
      sizeDotState     = 0, &                                                                       !< size of dot state, i.e. state(1:sizeDot) follows time evolution by dotState rates
      offsetDeltaState = 0, &                                                                       !< index offset of delta state
      sizeDeltaState   = 0                                                                          !< size of delta state, i.e. state(offset+1:offset+sizeDelta) follows time evolution by deltaState increments
    real(pReal), pointer,     dimension(:), contiguous :: &
      atol
    real(pReal), pointer,     dimension(:,:), contiguous :: &                                       ! a pointer is needed here because we might point to state/doState. However, they will never point to something, but are rather allocated and, hence, contiguous
      state0, &
      state, &                                                                                      !< state
      dotState, &                                                                                   !< rate of state change
      deltaState                                                                                    !< increment of state change
    real(pReal), allocatable, dimension(:,:) :: &
      partionedState0, &
      subState0
  end type

  type, extends(tState) :: tPlasticState
    logical :: &
      nonlocal = .false.
    real(pReal), pointer,     dimension(:,:) :: &
      slipRate                                                                                      !< slip rate
  end type

  type :: tSourceState
    type(tState), dimension(:), allocatable :: p                                                    !< tState for each active source mechanism in a phase
  end type

  type :: tHomogMapping
    integer, pointer, dimension(:,:) :: p
  end type

  real(pReal), private, parameter :: PREAL_EPSILON = epsilon(0.0_pReal)                             !< minimum positive number such that 1.0 + EPSILON /= 1.0.
  real(pReal), private, parameter :: PREAL_MIN     = tiny(0.0_pReal)                                !< smallest normalized floating point number

  integer,                   dimension(0), parameter :: &
    emptyIntArray    = [integer::]
  real(pReal),               dimension(0), parameter :: &
    emptyRealArray   = [real(pReal)::]
  character(len=pStringLen), dimension(0), parameter :: &
    emptyStringArray = [character(len=pStringLen)::]

  private :: &
    selfTest

contains


!--------------------------------------------------------------------------------------------------
!> @brief reporting precision
!--------------------------------------------------------------------------------------------------
subroutine prec_init

  write(6,'(/,a)') ' <<<+-  prec init  -+>>>'

  write(6,'(a,i3)')    ' Size of integer in bit: ',bit_size(0)
  write(6,'(a,i19)')   '   Maximum value:        ',huge(0)
  write(6,'(/,a,i3)')  ' Size of float in bit:   ',storage_size(0.0_pReal)
  write(6,'(a,e10.3)') '   Maximum value:        ',huge(0.0_pReal)
  write(6,'(a,e10.3)') '   Minimum value:        ',tiny(0.0_pReal)
  write(6,'(a,i3)')    '   Decimal precision:    ',precision(0.0_pReal)

  call selfTest

end subroutine prec_init


!--------------------------------------------------------------------------------------------------
!> @brief equality comparison for float with double precision
! replaces "==" but for certain (relative) tolerance. Counterpart to dNeq
! https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
! AlmostEqualRelative
!--------------------------------------------------------------------------------------------------
logical elemental pure function dEq(a,b,tol)

  real(pReal), intent(in)           :: a,b
  real(pReal), intent(in), optional :: tol
  real(pReal)                       :: eps

  if (present(tol)) then
    eps = tol
  else
    eps = PREAL_EPSILON * maxval(abs([a,b]))
  endif

  dEq = merge(.True.,.False.,abs(a-b) <= eps)

end function dEq


!--------------------------------------------------------------------------------------------------
!> @brief inequality comparison for float with double precision
! replaces "!=" but for certain (relative) tolerance. Counterpart to dEq
! https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
! AlmostEqualRelative NOT
!--------------------------------------------------------------------------------------------------
logical elemental pure function dNeq(a,b,tol)

  real(pReal), intent(in)           :: a,b
  real(pReal), intent(in), optional :: tol

  if (present(tol)) then
    dNeq = .not. dEq(a,b,tol)
  else
    dNeq = .not. dEq(a,b)
  endif

end function dNeq


!--------------------------------------------------------------------------------------------------
!> @brief equality to 0 comparison for float with double precision
! replaces "==0" but everything not representable as a normal number is treated as 0. Counterpart to dNeq0
! https://de.mathworks.com/help/matlab/ref/realmin.html
! https://docs.oracle.com/cd/E19957-01/806-3568/ncg_math.html
!--------------------------------------------------------------------------------------------------
logical elemental pure function dEq0(a,tol)

  real(pReal), intent(in)           :: a
  real(pReal), intent(in), optional :: tol
  real(pReal)                       :: eps

  if (present(tol)) then
    eps = tol
  else
    eps = PREAL_MIN * 10.0_pReal
  endif

  dEq0 = merge(.True.,.False.,abs(a) <= eps)

end function dEq0


!--------------------------------------------------------------------------------------------------
!> @brief inequality to 0 comparison for float with double precision
! replaces "!=0" but everything not representable as a normal number is treated as 0. Counterpart to dEq0
! https://de.mathworks.com/help/matlab/ref/realmin.html
! https://docs.oracle.com/cd/E19957-01/806-3568/ncg_math.html
!--------------------------------------------------------------------------------------------------
logical elemental pure function dNeq0(a,tol)

  real(pReal), intent(in)           :: a
  real(pReal), intent(in), optional :: tol

  if (present(tol)) then
    dNeq0 = .not. dEq0(a,tol)
  else
    dNeq0 = .not. dEq0(a)
  endif

end function dNeq0


!--------------------------------------------------------------------------------------------------
!> @brief equality comparison for complex with double precision
! replaces "==" but for certain (relative) tolerance. Counterpart to cNeq
! https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
! probably a component wise comparison would be more accurate than the comparsion of the absolute
! value
!--------------------------------------------------------------------------------------------------
logical elemental pure function cEq(a,b,tol)

  complex(pReal), intent(in)           :: a,b
  real(pReal),    intent(in), optional :: tol
  real(pReal)                          :: eps

  if (present(tol)) then
    eps = tol
  else
    eps = PREAL_EPSILON * maxval(abs([a,b]))
  endif

  cEq = merge(.True.,.False.,abs(a-b) <= eps)

end function cEq


!--------------------------------------------------------------------------------------------------
!> @brief inequality comparison for complex with double precision
! replaces "!=" but for certain (relative) tolerance. Counterpart to cEq
! https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
! probably a component wise comparison would be more accurate than the comparsion of the absolute
! value
!--------------------------------------------------------------------------------------------------
logical elemental pure function cNeq(a,b,tol)

  complex(pReal), intent(in)           :: a,b
  real(pReal),    intent(in), optional :: tol

  if (present(tol)) then
    cNeq = .not. cEq(a,b,tol)
  else
    cNeq = .not. cEq(a,b)
  endif

end function cNeq


!--------------------------------------------------------------------------------------------------
!> @brief check correctness of some prec functions
!--------------------------------------------------------------------------------------------------
subroutine selfTest

  integer, allocatable, dimension(:) :: realloc_lhs_test
  real(pReal), dimension(2) :: r
  external :: &
    quit

  call random_number(r)
  r = r/minval(r)
  if(.not. all(dEq(r,r+PREAL_EPSILON)))    call quit(9000)
  if(dEq(r(1),r(2)) .and. dNeq(r(1),r(2))) call quit(9000)
  if(.not. all(dEq0(r-(r+PREAL_MIN))))     call quit(9000)

  realloc_lhs_test = [1,2]
  if (any(realloc_lhs_test/=[1,2])) call quit(9000)

end subroutine selfTest

end module prec
