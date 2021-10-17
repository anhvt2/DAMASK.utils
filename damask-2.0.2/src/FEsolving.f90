! Copyright 2011-18 Max-Planck-Institut für Eisenforschung GmbH
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
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief triggering reading in of restart information when doing a restart
!> @todo Descriptions for public variables needed
!--------------------------------------------------------------------------------------------------
module FEsolving
 use prec, only: &
   pInt, &
   pReal
 
 implicit none
 private
 integer(pInt), public :: &
   restartInc   =  1_pInt                                                                           !< needs description

 logical, public :: & 
   symmetricSolver   = .false., &                                                                   !< use a symmetric FEM solver
   restartWrite      = .false., &                                                                   !< write current state to enable restart
   restartRead       = .false., &                                                                   !< restart information to continue calculation from saved state
   terminallyIll     = .false.                                                                      !< at least one material point is terminally ill

 integer(pInt), dimension(:,:), allocatable, public :: &
   FEsolving_execIP                                                                                 !< for ping-pong scheme always range to max IP, otherwise one specific IP
   
 integer(pInt), dimension(2), public :: &
   FEsolving_execElem                                                                               !< for ping-pong scheme always whole range, otherwise one specific element
   
 character(len=1024), public :: &
   modelName                                                                                        !< needs description
   
 logical, dimension(:,:), allocatable, public :: &
   calcMode                                                                                         !< do calculation or simply collect when using ping pong scheme

 public :: FE_init

contains


!--------------------------------------------------------------------------------------------------
!> @brief determine whether a symmetric solver is used and whether restart is requested
!> @details restart information is found in input file in case of FEM solvers, in case of spectal
!> solver the information is provided by the interface module
!--------------------------------------------------------------------------------------------------
subroutine FE_init
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif
 use debug, only: &
   debug_level, &
   debug_FEsolving, &
   debug_levelBasic
 use IO, only: &
   IO_stringPos, &
   IO_stringValue, &
   IO_intValue, &
   IO_lc, &
#if defined(Marc4DAMASK) || defined(Abaqus)
   IO_open_inputFile, &
   IO_open_logFile, &
#endif
   IO_warning, &
   IO_timeStamp
 use DAMASK_interface
 
 implicit none
#if defined(Marc4DAMASK) || defined(Abaqus)
 integer(pInt), parameter :: &
   FILEUNIT = 222_pInt
 integer(pInt) :: j
 character(len=65536) :: tag, line
 integer(pInt), allocatable, dimension(:) :: chunkPos
#endif

 write(6,'(/,a)')   ' <<<+-  FEsolving init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

 modelName = getSolverJobName()

#if defined(Spectral) || defined(FEM)

#ifdef Spectral
 restartInc = spectralRestartInc
#endif
#ifdef FEM
 restartInc = FEMRestartInc
#endif

 if(restartInc < 0_pInt) then
   call IO_warning(warning_ID=34_pInt)
   restartInc = 0_pInt
 endif
 restartRead = restartInc > 0_pInt                                                                  ! only read in if "true" restart requested

#else
 call IO_open_inputFile(FILEUNIT,modelName)
 rewind(FILEUNIT)
 do
   read (FILEUNIT,'(a1024)',END=100) line
   chunkPos = IO_stringPos(line)
   tag = IO_lc(IO_stringValue(line,chunkPos,1_pInt))                                               ! extract key
   select case(tag)
     case ('solver')
       read (FILEUNIT,'(a1024)',END=100) line                                                       ! next line
       chunkPos = IO_stringPos(line)
       symmetricSolver = (IO_intValue(line,chunkPos,2_pInt) /= 1_pInt)
     case ('restart')
       read (FILEUNIT,'(a1024)',END=100) line                                                       ! next line
       chunkPos = IO_stringPos(line)
       restartWrite = iand(IO_intValue(line,chunkPos,1_pInt),1_pInt) > 0_pInt
       restartRead  = iand(IO_intValue(line,chunkPos,1_pInt),2_pInt) > 0_pInt
     case ('*restart')
       do j=2_pInt,chunkPos(1)
         restartWrite = (IO_lc(IO_StringValue(line,chunkPos,j)) == 'write') .or. restartWrite
         restartRead  = (IO_lc(IO_StringValue(line,chunkPos,j)) == 'read')  .or. restartRead
       enddo
       if(restartWrite) then
         do j=2_pInt,chunkPos(1)
           restartWrite = (IO_lc(IO_StringValue(line,chunkPos,j)) /= 'frequency=0') .and. restartWrite
         enddo
       endif
   end select
 enddo
 100 close(FILEUNIT)

 if (restartRead) then
#ifdef Marc4DAMASK
   call IO_open_logFile(FILEUNIT)
   rewind(FILEUNIT)
   do
     read (FILEUNIT,'(a1024)',END=200) line
     chunkPos = IO_stringPos(line)
     if (   IO_lc(IO_stringValue(line,chunkPos,1_pInt)) == 'restart' &
      .and. IO_lc(IO_stringValue(line,chunkPos,2_pInt)) == 'file'    &
      .and. IO_lc(IO_stringValue(line,chunkPos,3_pInt)) == 'job'     &
      .and. IO_lc(IO_stringValue(line,chunkPos,4_pInt)) == 'id' )    &
        modelName = IO_StringValue(line,chunkPos,6_pInt)
   enddo
#else                                                                                               ! QUESTION: is this meaningful for the spectral/FEM case?
   call IO_open_inputFile(FILEUNIT,modelName)
   rewind(FILEUNIT)
   do
     read (FILEUNIT,'(a1024)',END=200) line
     chunkPos = IO_stringPos(line)
     if (IO_lc(IO_stringValue(line,chunkPos,1_pInt))=='*heading') then
       read (FILEUNIT,'(a1024)',END=200) line
       chunkPos = IO_stringPos(line)
       modelName = IO_StringValue(line,chunkPos,1_pInt)
     endif
   enddo
#endif
 200 close(FILEUNIT)
 endif

#endif
 if (iand(debug_level(debug_FEsolving),debug_levelBasic) /= 0_pInt) then
   write(6,'(a21,l1)') ' restart writing:    ', restartWrite
   write(6,'(a21,l1)') ' restart reading:    ', restartRead
   if (restartRead) write(6,'(a,/)') ' restart Job:        '//trim(modelName)
 endif

end subroutine FE_init

end module FEsolving
