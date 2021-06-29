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
!-------------------------------------------------------------------------------------------------
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Linked list
!--------------------------------------------------------------------------------------------------
module list
  use prec
  use IO

  implicit none
  private
   type, private :: tPartitionedString
    character(len=:),      allocatable :: val
    integer, dimension(:), allocatable :: pos
  end type tPartitionedString

  type, public :: tPartitionedStringList
    type(tPartitionedString)               :: string
    type(tPartitionedStringList),  pointer :: next => null()
    contains
      procedure :: add            => add
      procedure :: show           => show
      procedure :: free           => free

 ! currently, a finalize is needed for all shapes of tPartitionedStringList.
 ! with Fortran 2015, we can define one recursive elemental function
 ! https://software.intel.com/en-us/forums/intel-visual-fortran-compiler-for-windows/topic/543326
      final     :: finalize, &
                   finalizeArray

      procedure :: keyExists      => keyExists
      procedure :: countKeys      => countKeys

      procedure :: getFloat       => getFloat
      procedure :: getInt         => getInt
      procedure :: getString      => getString

      procedure :: getFloats      => getFloats
      procedure :: getInts        => getInts
      procedure :: getStrings     => getStrings

  end type tPartitionedStringList

contains

!--------------------------------------------------------------------------------------------------
!> @brief add element
!> @details Adds a string together with the start/end position of chunks in this string. The new
!! element is added at the end of the list. Empty strings are not added. All strings are converted
!! to lower case. The data is not stored in the new element but in the current.
!--------------------------------------------------------------------------------------------------
subroutine add(this,string)

  class(tPartitionedStringList),  target, intent(in) :: this
  character(len=*),                       intent(in) :: string
  type(tPartitionedStringList),   pointer            :: new, temp

  if (IO_isBlank(string)) return

  allocate(new)
  temp => this
  do while (associated(temp%next))
    temp => temp%next
  enddo
  temp%string%val = IO_lc       (trim(string))
  temp%string%pos = IO_stringPos(trim(string))
  temp%next => new

end subroutine add


!--------------------------------------------------------------------------------------------------
!> @brief prints all elements
!> @details Strings are printed in order of insertion (FIFO)
!--------------------------------------------------------------------------------------------------
subroutine show(this)

  class(tPartitionedStringList), target, intent(in) :: this
  type(tPartitionedStringList),  pointer            :: item

  item => this
  do while (associated(item%next))
    write(6,'(a)') ' '//trim(item%string%val)
    item => item%next
  enddo

end subroutine show


!--------------------------------------------------------------------------------------------------
!> @brief empties list and frees associated memory
!> @details explicit interface to reset list. Triggers final statement (and following chain reaction)
!--------------------------------------------------------------------------------------------------
subroutine free(this)

  class(tPartitionedStringList),  intent(inout) :: this

  if(associated(this%next)) deallocate(this%next)

end subroutine free


!--------------------------------------------------------------------------------------------------
!> @brief empties list and frees associated memory
!> @details called when variable goes out of scope. Triggers chain reaction for list
!--------------------------------------------------------------------------------------------------
recursive subroutine finalize(this)

  type(tPartitionedStringList),  intent(inout) :: this

  if(associated(this%next)) deallocate(this%next)

end subroutine finalize


!--------------------------------------------------------------------------------------------------
!> @brief cleans entire array of linke lists
!> @details called when variable goes out of scope and deallocates the list at each array entry
!--------------------------------------------------------------------------------------------------
subroutine finalizeArray(this)

  integer :: i
  type(tPartitionedStringList),  intent(inout), dimension(:) :: this
  type(tPartitionedStringList),  pointer :: temp ! bug in Gfortran?

  do i=1, size(this)
    if (associated(this(i)%next)) then
      temp => this(i)%next
      !deallocate(this(i)) !internal compiler error: in gfc_build_final_call, at fortran/trans.c:975
      deallocate(temp)
    endif
  enddo

end subroutine finalizeArray


!--------------------------------------------------------------------------------------------------
!> @brief reports wether a given key (string value at first position) exists in the list
!--------------------------------------------------------------------------------------------------
logical function keyExists(this,key)

  class(tPartitionedStringList), target, intent(in) :: this
  character(len=*),                      intent(in) :: key
  type(tPartitionedStringList),  pointer            :: item

  keyExists = .false.

  item => this
  do while (associated(item%next) .and. .not. keyExists)
    keyExists = trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)
    item => item%next
  enddo

end function keyExists


!--------------------------------------------------------------------------------------------------
!> @brief count number of key appearances
!> @details traverses list and counts each occurrence of specified key
!--------------------------------------------------------------------------------------------------
integer function countKeys(this,key)

  class(tPartitionedStringList), target, intent(in) :: this
  character(len=*),                      intent(in) :: key
  type(tPartitionedStringList),  pointer            :: item

  countKeys = 0

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) &
      countKeys = countKeys + 1
    item => item%next
  enddo

end function countKeys


!--------------------------------------------------------------------------------------------------
!> @brief gets float value of for a given key from a linked list
!> @details gets the last value if the key occurs more than once. If key is not found exits with
!! error unless default is given
!--------------------------------------------------------------------------------------------------
real(pReal) function getFloat(this,key,defaultVal)

  class(tPartitionedStringList), target, intent(in)           :: this
  character(len=*),                      intent(in)           :: key
  real(pReal),                           intent(in), optional :: defaultVal
  type(tPartitionedStringList), pointer                       :: item
  logical                                                     :: found

  getFloat = huge(1.0)                                                                              ! suppress warning about unitialized value
  found = present(defaultVal)
  if (found) getFloat = defaultVal

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)
      getFloat = IO_FloatValue(item%string%val,item%string%pos,2)
    endif
    item => item%next
  enddo

  if (.not. found) call IO_error(140,ext_msg=key)

end function getFloat


!--------------------------------------------------------------------------------------------------
!> @brief gets integer value of for a given key from a linked list
!> @details gets the last value if the key occurs more than once. If key is not found exits with
!! error unless default is given
!--------------------------------------------------------------------------------------------------
integer function getInt(this,key,defaultVal)

  class(tPartitionedStringList), target, intent(in)           :: this
  character(len=*),                      intent(in)           :: key
  integer,                               intent(in), optional :: defaultVal
  type(tPartitionedStringList), pointer                       :: item
  logical                                                     :: found

  getInt = huge(1)                                                                                  ! suppress warning about unitialized value
  found = present(defaultVal)
  if (found) getInt = defaultVal

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)
      getInt = IO_IntValue(item%string%val,item%string%pos,2)
    endif
    item => item%next
  enddo

  if (.not. found) call IO_error(140,ext_msg=key)

end function getInt


!--------------------------------------------------------------------------------------------------
!> @brief gets string value of for a given key from a linked list
!> @details gets the last value if the key occurs more than once. If key is not found exits with
!! error unless default is given. If raw is true, the the complete string is returned, otherwise
!! the individual chunks are returned
!--------------------------------------------------------------------------------------------------
character(len=pStringLen) function getString(this,key,defaultVal,raw)

  class(tPartitionedStringList), target, intent(in)           :: this
  character(len=*),                      intent(in)           :: key
  character(len=*),                      intent(in), optional :: defaultVal
  logical,                               intent(in), optional :: raw
  type(tPartitionedStringList),  pointer                      :: item
  logical                                                     :: found, &
                                                                 whole
  if (present(raw)) then
    whole = raw
  else
    whole = .false.
  endif

  found = present(defaultVal)
  if (found) then
    if (len_trim(defaultVal) > len(getString)) call IO_error(0,ext_msg='getString')
    getString = trim(defaultVal)
  endif

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)

      if (whole) then
        getString = trim(item%string%val(item%string%pos(4):))                                      ! raw string starting a second chunk
      else
        getString = IO_StringValue(item%string%val,item%string%pos,2)
      endif
    endif
    item => item%next
  enddo

  if (.not. found) call IO_error(140,ext_msg=key)

end function getString


!--------------------------------------------------------------------------------------------------
!> @brief gets array of float values of for a given key from a linked list
!> @details for cumulative keys, "()", values from all occurrences are return. Otherwise only all
!! values from the last occurrence. If key is not found exits with error unless default is given.
!--------------------------------------------------------------------------------------------------
function getFloats(this,key,defaultVal,requiredSize)

  real(pReal),     dimension(:), allocatable          :: getFloats
  class(tPartitionedStringList), target, intent(in)   :: this
  character(len=*),              intent(in)           :: key
  real(pReal),   dimension(:),   intent(in), optional :: defaultVal
  integer,                 intent(in), optional :: requiredSize
  type(tPartitionedStringList),  pointer              :: item
  integer                                       :: i
  logical                                             :: found, &
                                                         cumulative

  cumulative = (key(1:1) == '(' .and. key(len_trim(key):len_trim(key)) == ')')
  found = .false.

  allocate(getFloats(0))

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (.not. cumulative) getFloats = [real(pReal)::]
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)
      do i = 2, item%string%pos(1)
        getFloats = [getFloats,IO_FloatValue(item%string%val,item%string%pos,i)]
      enddo
    endif
    item => item%next
  enddo

  if (.not. found) then
    if (present(defaultVal)) then; getFloats = defaultVal; else; call IO_error(140,ext_msg=key); endif
  endif
  if (present(requiredSize)) then
    if(requiredSize /= size(getFloats)) call IO_error(146,ext_msg=key)
  endif

end function getFloats


!--------------------------------------------------------------------------------------------------
!> @brief gets array of integer values of for a given key from a linked list
!> @details for cumulative keys, "()", values from all occurrences are return. Otherwise only all
!! values from the last occurrence. If key is not found exits with error unless default is given.
!--------------------------------------------------------------------------------------------------
function getInts(this,key,defaultVal,requiredSize)

  integer, dimension(:), allocatable                          :: getInts
  class(tPartitionedStringList), target, intent(in)           :: this
  character(len=*),                      intent(in)           :: key
  integer, dimension(:),                 intent(in), optional :: defaultVal
  integer,                               intent(in), optional :: requiredSize
  type(tPartitionedStringList),  pointer                      :: item
  integer                                                     :: i
  logical                                                     :: found, &
                                                                 cumulative

  cumulative = (key(1:1) == '(' .and. key(len_trim(key):len_trim(key)) == ')')
  found = .false.

  allocate(getInts(0))

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (.not. cumulative) getInts = [integer::]
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)
      do i = 2, item%string%pos(1)
        getInts = [getInts,IO_IntValue(item%string%val,item%string%pos,i)]
      enddo
    endif
    item => item%next
  enddo

  if (.not. found) then
    if (present(defaultVal)) then; getInts = defaultVal; else; call IO_error(140,ext_msg=key); endif
  endif
  if (present(requiredSize)) then
    if(requiredSize /= size(getInts)) call IO_error(146,ext_msg=key)
  endif

end function getInts


!--------------------------------------------------------------------------------------------------
!> @brief gets array of string values of for a given key from a linked list
!> @details for cumulative keys, "()", values from all occurrences are return. Otherwise only all
!! values from the last occurrence. If key is not found exits with error unless default is given.
!! If raw is true, the the complete string is returned, otherwise the individual chunks are returned
!--------------------------------------------------------------------------------------------------
function getStrings(this,key,defaultVal,raw)

  character(len=pStringLen),dimension(:), allocatable        :: getStrings
  class(tPartitionedStringList),target, intent(in)           :: this
  character(len=*),                     intent(in)           :: key
  character(len=*),    dimension(:),    intent(in), optional :: defaultVal
  logical,                              intent(in), optional :: raw
  type(tPartitionedStringList), pointer                      :: item
  character(len=pStringLen)                                  :: str
  integer                                                    :: i
  logical                                                    :: found, &
                                                                whole, &
                                                                cumulative

  cumulative = (key(1:1) == '(' .and. key(len_trim(key):len_trim(key)) == ')')
  if (present(raw)) then
    whole = raw
  else
    whole = .false.
  endif
  found = .false.

  item => this
  do while (associated(item%next))
    if (trim(IO_stringValue(item%string%val,item%string%pos,1)) == trim(key)) then
      found = .true.
      if (allocated(getStrings) .and. .not. cumulative) deallocate(getStrings)
      if (item%string%pos(1) < 2) call IO_error(143,ext_msg=key)

      notAllocated: if (.not. allocated(getStrings)) then
        if (whole) then
          str = item%string%val(item%string%pos(4):)
          getStrings = [str]
        else
          str = IO_StringValue(item%string%val,item%string%pos,2)
          allocate(getStrings(1),source=str)
          do i=3,item%string%pos(1)
            str = IO_StringValue(item%string%val,item%string%pos,i)
            getStrings = [getStrings,str]
          enddo
        endif
      else notAllocated
        if (whole) then
          str = item%string%val(item%string%pos(4):)
          getStrings = [getStrings,str]
        else
          do i=2,item%string%pos(1)
            str = IO_StringValue(item%string%val,item%string%pos,i)
            getStrings = [getStrings,str]
          enddo
        endif
      endif notAllocated
    endif
    item => item%next
  enddo

  if (.not. found) then
    if (present(defaultVal)) then
      if (len(defaultVal) > len(getStrings)) call IO_error(0,ext_msg='getStrings')
      getStrings = defaultVal
    else
      call IO_error(140,ext_msg=key)
    endif
  endif

end function getStrings


end module list
