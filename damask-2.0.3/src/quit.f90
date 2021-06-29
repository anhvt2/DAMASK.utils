! Copyright 2011-19 Max-Planck-Institut für Eisenforschung GmbH
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
!> @brief quit subroutine
!> @details exits the program and reports current time and duration. Exit code 0 signals
!> everything is fine. Exit code 1 signals an error, message according to IO_error. Exit code
!> 2 signals no severe problems, but some increments did not converge
!--------------------------------------------------------------------------------------------------
subroutine quit(stop_id)
#include <petsc/finclude/petscsys.h>
#ifdef _OPENMP
 use MPI, only: &
   MPI_finalize
#endif
 use PetscSys
 use hdf5

 implicit none
 integer, intent(in) :: stop_id
 integer, dimension(8) :: dateAndTime                                                               ! type default integer
 integer :: error
 PetscErrorCode :: ierr = 0

 call h5open_f(error)
 if (error /= 0) write(6,'(a,i5)') ' Error in h5open_f ',error                                      ! prevents error if not opened yet
 call h5close_f(error)
 if (error /= 0) write(6,'(a,i5)') ' Error in h5close_f ',error

 call PETScFinalize(ierr)
 CHKERRQ(ierr)

#ifdef _OPENMP
 call MPI_finalize(error)
 if (error /= 0) write(6,'(a,i5)') ' Error in MPI_finalize',error
#endif
 
 call date_and_time(values = dateAndTime)
 write(6,'(/,a)') 'DAMASK terminated on:'
 write(6,'(a,2(i2.2,a),i4.4)') 'Date:               ',dateAndTime(3),'/',&
                                                      dateAndTime(2),'/',&
                                                      dateAndTime(1)
 write(6,'(a,2(i2.2,a),i2.2)') 'Time:               ',dateAndTime(5),':',&
                                                      dateAndTime(6),':',&
                                                      dateAndTime(7)

 if (stop_id == 0 .and. ierr == 0 .and. error == 0) stop 0                                          ! normal termination
 if (stop_id == 2 .and. ierr == 0 .and. error == 0) stop 2                                          ! not all incs converged
 stop 1                                                                                             ! error (message from IO_error)

end subroutine quit
