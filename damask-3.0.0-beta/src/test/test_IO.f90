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
module test_IO
  use prec
  use parallelization
  use IO

  implicit none(type,external)

  private
  public :: test_IO_run

  contains

subroutine test_IO_run()

  real, dimension(30) :: rnd_real
  character(len=size(rnd_real)) :: rnd_str
  character(len=pSTRLEN), dimension(1) :: strarray_out
  character(len=:), allocatable :: str_out, fname
  integer :: u,i


  call IO_selfTest()

  call random_number(rnd_real)
  fname = 'test'//IO_intAsStr(worldrank)//'.txt'

  do i = 1, size(rnd_real)
    rnd_str(i:i) = char(32 + int(rnd_real(i)*(127.-32.)))
  end do
  open(newunit=u,file=fname,status='replace',form='formatted')
  write(u,'(a)') rnd_str
  close(u)

  str_out = IO_read(fname)
  if (rnd_str//IO_EOL /= str_out) error stop 'IO_read'
  strarray_out = IO_readlines(fname)
  if (rnd_str /= strarray_out(1)) error stop 'IO_readlines'

end subroutine test_IO_run

end module test_IO
