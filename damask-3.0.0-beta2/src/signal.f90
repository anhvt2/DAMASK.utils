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
!> @brief Handling of UNIX signals.
!--------------------------------------------------------------------------------------------------
module signal
  use prec
  use system_routines

  implicit none(type,external)
  private

  logical, volatile, public, protected :: &
    signal_SIGINT  = .false., &                                                                    !< interrupt signal
    signal_SIGUSR1 = .false., &                                                                    !< 1. user-defined signal
    signal_SIGUSR2 = .false.                                                                       !< 2. user-defined signal

  public :: &
    signal_init, &
    signal_setSIGINT, &
    signal_setSIGUSR1, &
    signal_setSIGUSR2

contains


!--------------------------------------------------------------------------------------------------
!> @brief Register signal handlers.
!--------------------------------------------------------------------------------------------------
subroutine signal_init()

  call signalint_c(c_funloc(catchSIGINT))
  call signalusr1_c(c_funloc(catchSIGUSR1))
  call signalusr2_c(c_funloc(catchSIGUSR2))

end subroutine signal_init


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGINT to .true.
!> @details This function can be registered to catch signals sent to the executable.
!--------------------------------------------------------------------------------------------------
subroutine catchSIGINT(sig) bind(C)

  integer(C_INT), value :: sig


  print'(a,i0)', ' received signal ',sig
  call signal_setSIGINT(.true.)

end subroutine catchSIGINT


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGUSR1 to .true.
!> @details This function can be registered to catch signals sent to the executable.
!--------------------------------------------------------------------------------------------------
subroutine catchSIGUSR1(sig) bind(C)

  integer(C_INT), value :: sig


  print'(a,i0)', ' received signal ',sig
  call signal_setSIGUSR1(.true.)

end subroutine catchSIGUSR1


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGUSR2 to .true.
!> @details This function can be registered to catch signals sent to the executable.
!--------------------------------------------------------------------------------------------------
subroutine catchSIGUSR2(sig) bind(C)

  integer(C_INT), value :: sig


  print'(a,i0,a)', ' received signal ',sig
  call signal_setSIGUSR2(.true.)

end subroutine catchSIGUSR2


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGINT.
!--------------------------------------------------------------------------------------------------
subroutine signal_setSIGINT(state)

  logical, intent(in) :: state


  signal_SIGINT = state
  print*, 'set SIGINT to',state

end subroutine signal_setSIGINT


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGUSR.
!--------------------------------------------------------------------------------------------------
subroutine signal_setSIGUSR1(state)

  logical, intent(in) :: state


  signal_SIGUSR1 = state
  print*, 'set SIGUSR1 to',state

end subroutine signal_setSIGUSR1


!--------------------------------------------------------------------------------------------------
!> @brief Set global variable signal_SIGUSR2.
!--------------------------------------------------------------------------------------------------
subroutine signal_setSIGUSR2(state)

  logical, intent(in) :: state


  signal_SIGUSR2 = state
  print*, 'set SIGUSR2 to',state

end subroutine signal_setSIGUSR2


end module signal
