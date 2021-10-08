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
!> @brief Inquires variables related to parallelization (openMP, MPI)
!--------------------------------------------------------------------------------------------------
module parallelization
  use, intrinsic :: ISO_fortran_env, only: &
    OUTPUT_UNIT

#ifdef PETSc
#include <petsc/finclude/petscsys.h>
   use petscsys
!$ use OMP_LIB
#endif
  use prec

  implicit none
  private

  integer, protected, public :: &
    worldrank = 0, &                                                                                !< MPI worldrank (/=0 for MPI simulations only)
    worldsize = 1                                                                                   !< MPI worldsize (/=1 for MPI simulations only)

#ifdef PETSc
  public :: &
    parallelization_init

contains

!--------------------------------------------------------------------------------------------------
!> @brief Initialize shared memory (openMP) and distributed memory (MPI) parallelization.
!--------------------------------------------------------------------------------------------------
subroutine parallelization_init

  integer :: err, typeSize
!$ integer :: got_env, threadLevel
!$ integer(pI32) :: OMP_NUM_THREADS
!$ character(len=6) NumThreadsString


  PetscErrorCode :: petsc_err
#ifdef _OPENMP
  ! If openMP is enabled, check if the MPI libary supports it and initialize accordingly.
  ! Otherwise, the first call to PETSc will do the initialization.
  call MPI_Init_Thread(MPI_THREAD_FUNNELED,threadLevel,err)
  if (err /= 0)                        error stop 'MPI init failed'
  if (threadLevel<MPI_THREAD_FUNNELED) error stop 'MPI library does not support OpenMP'
#endif

#if defined(DEBUG)
  call PetscInitialize(PETSC_NULL_CHARACTER,petsc_err)
#else
  call PetscInitializeNoArguments(petsc_err)
#endif
  CHKERRQ(petsc_err)

#if defined(DEBUG) && defined(__INTEL_COMPILER)
  call PetscSetFPTrap(PETSC_FP_TRAP_ON,petsc_err)
#else
  call PetscSetFPTrap(PETSC_FP_TRAP_OFF,petsc_err)
#endif
  CHKERRQ(petsc_err)

  call MPI_Comm_rank(PETSC_COMM_WORLD,worldrank,err)
  if (err /= 0)                              error stop 'Could not determine worldrank'

  if (worldrank == 0) print'(/,a)',  ' <<<+-  parallelization init  -+>>>'

  call MPI_Comm_size(PETSC_COMM_WORLD,worldsize,err)
  if (err /= 0)                              error stop 'Could not determine worldsize'
  if (worldrank == 0) print'(a,i3)', ' MPI processes: ',worldsize

  call MPI_Type_size(MPI_INTEGER,typeSize,err)
  if (err /= 0)                              error stop 'Could not determine MPI integer size'
  if (typeSize*8 /= bit_size(0))             error stop 'Mismatch between MPI and DAMASK integer'

  call MPI_Type_size(MPI_DOUBLE,typeSize,err)
  if (err /= 0)                              error stop 'Could not determine MPI real size'
  if (typeSize*8 /= storage_size(0.0_pReal)) error stop 'Mismatch between MPI and DAMASK real'

  if (worldrank /= 0) then
    close(OUTPUT_UNIT)                                                                              ! disable output
    open(OUTPUT_UNIT,file='/dev/null',status='replace')                                             ! close() alone will leave some temp files in cwd
  endif

!$ call get_environment_variable(name='OMP_NUM_THREADS',value=NumThreadsString,STATUS=got_env)
!$ if(got_env /= 0) then
!$   print*, 'Could not get $OMP_NUM_THREADS, using default'
!$   OMP_NUM_THREADS = 4_pI32
!$ else
!$   read(NumThreadsString,'(i6)') OMP_NUM_THREADS
!$   if (OMP_NUM_THREADS < 1_pI32) then
!$     print*, 'Invalid OMP_NUM_THREADS: "'//trim(NumThreadsString)//'", using default'
!$     OMP_NUM_THREADS = 4_pI32
!$   endif
!$ endif
!$ print'(a,i2)',   ' OMP_NUM_THREADS: ',OMP_NUM_THREADS
!$ call omp_set_num_threads(OMP_NUM_THREADS)

end subroutine parallelization_init
#endif

end module parallelization
