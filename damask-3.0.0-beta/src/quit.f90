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
!> @brief quit subroutine
!> @details exits the program and reports current time and duration. Exit code 0 signals
!> everything is fine. Exit code 1 signals an error, message according to IO_error.
!--------------------------------------------------------------------------------------------------
subroutine quit(stop_id)
  use, intrinsic :: ISO_fortran_env, only: ERROR_UNIT
#include <petsc/finclude/petscsys.h>
  use PETScSys
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  use MPI_f08
#endif
  use HDF5

#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>14) && !defined(PETSC_HAVE_MPI_F90MODULE_VISIBILITY)
  implicit none(type,external)
#else
  implicit none
#endif

  integer, intent(in) :: stop_id

  integer, dimension(8) :: dateAndTime
  integer :: err_HDF5
  integer(MPI_INTEGER_KIND) :: err_MPI, worldsize
  PetscErrorCode :: err_PETSc


  call H5Open_f(err_HDF5)                                                                           ! prevents error if not opened yet
  if (err_HDF5 < 0) write(ERROR_UNIT,'(a,i5)') ' Error in H5Open_f ',err_HDF5
  call H5Close_f(err_HDF5)
  if (err_HDF5 < 0) write(ERROR_UNIT,'(a,i5)') ' Error in H5Close_f ',err_HDF5

  call PetscFinalize(err_PETSc)

  call date_and_time(values = dateAndTime)
  write(6,'(/,a)') ' DAMASK terminated on:'
  write(6,'(a,2(i2.2,a),i4.4)') ' Date:               ',dateAndTime(3),'/',&
                                                        dateAndTime(2),'/',&
                                                        dateAndTime(1)
  write(6,'(a,2(i2.2,a),i2.2)') ' Time:               ',dateAndTime(5),':',&
                                                        dateAndTime(6),':',&
                                                        dateAndTime(7)

  if (stop_id == 0 .and. err_HDF5 == 0 .and. err_PETSC == 0) then
    call MPI_Finalize(err_MPI)
    if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI_Finalize error'
    stop 0                                                                                          ! normal termination
  else
    call MPI_Comm_size(MPI_COMM_WORLD,worldsize,err_MPI)
    if (err_MPI /= 0_MPI_INTEGER_KIND) error stop 'MPI_Comm error'
    if (stop_id /= 0 .and. worldsize > 1) call MPI_Abort(MPI_COMM_WORLD,1,err_MPI)
    stop 1                                                                                          ! error (message from IO_error)
  endif

end subroutine quit
