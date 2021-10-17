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
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Christoph Kords, Max-Planck-Institut für Eisenforschung GmbH
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief Reading in and interpretating the debugging settings for the various modules
!--------------------------------------------------------------------------------------------------
module debug
 use prec, only: &
   pInt, &
   pReal, &
   pLongInt

 implicit none
 private
 integer(pInt), parameter, public :: &
   debug_LEVELSELECTIVE     = 2_pInt**0_pInt, &
   debug_LEVELBASIC         = 2_pInt**1_pInt, &
   debug_LEVELEXTENSIVE     = 2_pInt**2_pInt
 integer(pInt), parameter, private :: &
   debug_MAXGENERAL         = debug_LEVELEXTENSIVE                                                  ! must be set to the last bitcode used by (potentially) all debug types
 integer(pInt), parameter, public :: &
   debug_SPECTRALRESTART    = debug_MAXGENERAL*2_pInt**1_pInt, &
   debug_SPECTRALFFTW       = debug_MAXGENERAL*2_pInt**2_pInt, &
   debug_SPECTRALDIVERGENCE = debug_MAXGENERAL*2_pInt**3_pInt, &
   debug_SPECTRALROTATION   = debug_MAXGENERAL*2_pInt**4_pInt, &
   debug_SPECTRALPETSC      = debug_MAXGENERAL*2_pInt**5_pInt
   
 integer(pInt), parameter, public :: &
   debug_DEBUG                   =  1_pInt, &
   debug_MATH                    =  2_pInt, &
   debug_FESOLVING               =  3_pInt, &
   debug_MESH                    =  4_pInt, &                                                       !< stores debug level for mesh part of DAMASK bitwise coded
   debug_MATERIAL                =  5_pInt, &                                                       !< stores debug level for material part of DAMASK bitwise coded
   debug_LATTICE                 =  6_pInt, &                                                       !< stores debug level for lattice part of DAMASK bitwise coded
   debug_CONSTITUTIVE            =  7_pInt, &                                                       !< stores debug level for constitutive part of DAMASK bitwise coded
   debug_CRYSTALLITE             =  8_pInt, &
   debug_HOMOGENIZATION          =  9_pInt, &
   debug_CPFEM                   = 10_pInt, &
   debug_SPECTRAL                = 11_pInt, &
   debug_MARC                    = 12_pInt, &
   debug_ABAQUS                  = 13_pInt
 integer(pInt), parameter, private :: &
   debug_MAXNTYPE                = debug_ABAQUS                                                     !< must be set to the maximum defined debug type

 integer(pInt),protected, dimension(debug_maxNtype+2_pInt),  public :: &                            ! specific ones, and 2 for "all" and "other"
   debug_level                    = 0_pInt

 integer(pLongInt), public :: &
   debug_cumLpCalls              = 0_pLongInt, &                                                    !< total number of calls to LpAndItsTangent
   debug_cumDeltaStateCalls      = 0_pLongInt, &                                                    !< total number of calls to deltaState
   debug_cumDotStateCalls        = 0_pLongInt                                                       !< total number of calls to dotState

 integer(pInt), protected, public :: &
   debug_e                       = 1_pInt, &
   debug_i                       = 1_pInt, &
   debug_g                       = 1_pInt

 integer(pLongInt), public :: &
   debug_cumLpTicks              = 0_pLongInt, &                                                    !< total cpu ticks spent in LpAndItsTangent
   debug_cumDeltaStateTicks      = 0_pLongInt, &                                                    !< total cpu ticks spent in deltaState
   debug_cumDotStateTicks        = 0_pLongInt                                                       !< total cpu ticks spent in dotState

 integer(pInt), dimension(2), public :: &
   debug_stressMaxLocation       = 0_pInt, &
   debug_stressMinLocation       = 0_pInt, &
   debug_jacobianMaxLocation     = 0_pInt, &
   debug_jacobianMinLocation     = 0_pInt


 integer(pInt), dimension(:), allocatable, public :: &
   debug_CrystalliteLoopDistribution, &                                                             !< distribution of crystallite cutbacks
   debug_MaterialpointStateLoopDistribution, &
   debug_MaterialpointLoopDistribution

 integer(pInt), dimension(:,:), allocatable, public :: &
   debug_StressLoopLiDistribution, &                                                                !< distribution of stress iterations until convergence
   debug_StressLoopLpDistribution, &                                                                !< distribution of stress iterations until convergence
   debug_StateLoopDistribution                                                                      !< distribution of state iterations until convergence

 real(pReal), public :: &
   debug_stressMax               = -huge(1.0_pReal), &
   debug_stressMin               =  huge(1.0_pReal), &
   debug_jacobianMax             = -huge(1.0_pReal), &
   debug_jacobianMin             =  huge(1.0_pReal)

 character(len=64), parameter, private ::  &
   debug_CONFIGFILE         = 'debug.config'                                                        !< name of configuration file

#ifdef PETSc
 character(len=1024), parameter, public :: &
   PETSCDEBUG = ' -snes_view -snes_monitor '
#endif
 public :: debug_init, &
           debug_reset, &
           debug_info

contains


!--------------------------------------------------------------------------------------------------
!> @brief reads in parameters from debug.config and allocates arrays
!--------------------------------------------------------------------------------------------------
subroutine debug_init
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif
 use numerics, only: &
   nStress, &
   nState, &
   nCryst, &
   nMPstate, &
   nHomog
 use IO, only: &
   IO_read, &
   IO_error, &
   IO_open_file_stat, &
   IO_isBlank, &
   IO_stringPos, &
   IO_stringValue, &
   IO_lc, &
   IO_floatValue, &
   IO_intValue, &
   IO_timeStamp, &
   IO_EOF

 implicit none
 integer(pInt), parameter                 :: FILEUNIT    = 300_pInt

 integer(pInt)                            :: i, what
 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536)                     :: tag, line

 write(6,'(/,a)')   ' <<<+-  debug init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

 allocate(debug_StressLoopLpDistribution(nStress+1,2), source=0_pInt)
 allocate(debug_StressLoopLiDistribution(nStress+1,2), source=0_pInt)
 allocate(debug_StateLoopDistribution(nState+1,2), source=0_pInt)
 allocate(debug_CrystalliteLoopDistribution(nCryst+1), source=0_pInt)
 allocate(debug_MaterialpointStateLoopDistribution(nMPstate), source=0_pInt)
 allocate(debug_MaterialpointLoopDistribution(nHomog+1), source=0_pInt)

!--------------------------------------------------------------------------------------------------
! try to open the config file

 line = ''
 fileExists: if(IO_open_file_stat(FILEUNIT,debug_configFile)) then
   do while (trim(line) /= IO_EOF)                                                                  ! read thru sections of phase part
     line = IO_read(FILEUNIT)
     if (IO_isBlank(line)) cycle                                                                    ! skip empty lines
     chunkPos = IO_stringPos(line)
     tag = IO_lc(IO_stringValue(line,chunkPos,1_pInt))                                              ! extract key
     select case(tag)
       case ('element','e','el')
         debug_e = IO_intValue(line,chunkPos,2_pInt)
       case ('integrationpoint','i','ip')
         debug_i = IO_intValue(line,chunkPos,2_pInt)
       case ('grain','g','gr')
         debug_g = IO_intValue(line,chunkPos,2_pInt)
     end select

     what = 0_pInt
     select case(tag)
       case ('debug')
         what = debug_DEBUG
       case ('math')
         what = debug_MATH
       case ('fesolving', 'fe')
         what = debug_FESOLVING
       case ('mesh')
         what = debug_MESH
       case ('material')
         what = debug_MATERIAL
       case ('lattice')
         what = debug_LATTICE
       case ('constitutive')
         what = debug_CONSTITUTIVE
       case ('crystallite')
         what = debug_CRYSTALLITE
       case ('homogenization')
         what = debug_HOMOGENIZATION
       case ('cpfem')
         what = debug_CPFEM
       case ('spectral')
         what = debug_SPECTRAL
       case ('marc')
         what = debug_MARC
       case ('abaqus')
         what = debug_ABAQUS
       case ('all')
         what = debug_MAXNTYPE + 1_pInt
       case ('other')
         what = debug_MAXNTYPE + 2_pInt
     end select
     if (what /= 0) then
       do i = 2_pInt, chunkPos(1)
         select case(IO_lc(IO_stringValue(line,chunkPos,i)))
           case('basic')
             debug_level(what) = ior(debug_level(what), debug_LEVELBASIC)
           case('extensive')
             debug_level(what) = ior(debug_level(what), debug_LEVELEXTENSIVE)
           case('selective')
             debug_level(what) = ior(debug_level(what), debug_LEVELSELECTIVE)
           case('restart')
             debug_level(what) = ior(debug_level(what), debug_SPECTRALRESTART)
           case('fft','fftw')
             debug_level(what) = ior(debug_level(what), debug_SPECTRALFFTW)
           case('divergence')
             debug_level(what) = ior(debug_level(what), debug_SPECTRALDIVERGENCE)
           case('rotation')
             debug_level(what) = ior(debug_level(what), debug_SPECTRALROTATION)
           case('petsc')
             debug_level(what) = ior(debug_level(what), debug_SPECTRALPETSC)
         end select
       enddo
      endif
   enddo
   close(FILEUNIT)

   do i = 1_pInt, debug_maxNtype
     if (debug_level(i) == 0) &
       debug_level(i) = ior(debug_level(i), debug_level(debug_MAXNTYPE + 2_pInt))                   ! fill undefined debug types with levels specified by "other"

     debug_level(i) = ior(debug_level(i), debug_level(debug_MAXNTYPE + 1_pInt))                     ! fill all debug types with levels specified by "all"
   enddo

   if (iand(debug_level(debug_debug),debug_LEVELBASIC) /= 0) &
     write(6,'(a,/)') ' using values from config file'
 else fileExists
   if (iand(debug_level(debug_debug),debug_LEVELBASIC) /= 0) &
     write(6,'(a,/)') ' using standard values'
 endif fileExists

!--------------------------------------------------------------------------------------------------
! output switched on (debug level for debug must be extensive)
 if (iand(debug_level(debug_debug),debug_LEVELEXTENSIVE) /= 0) then
     do i = 1_pInt, debug_MAXNTYPE
       select case(i)
         case (debug_DEBUG)
           tag = ' Debug'
         case (debug_MATH)
           tag = ' Math'
         case (debug_FESOLVING)
           tag = ' FEsolving'
         case (debug_MESH)
           tag = ' Mesh'
         case (debug_MATERIAL)
           tag = ' Material'
         case (debug_LATTICE)
           tag = ' Lattice'
         case (debug_CONSTITUTIVE)
           tag = ' Constitutive'
         case (debug_CRYSTALLITE)
           tag = ' Crystallite'
         case (debug_HOMOGENIZATION)
           tag = ' Homogenizaiton'
         case (debug_CPFEM)
           tag = ' CPFEM'
         case (debug_SPECTRAL)
           tag = ' Spectral solver'
         case (debug_MARC)
           tag = ' MSC.MARC FEM solver'
         case (debug_ABAQUS)
           tag = ' ABAQUS FEM solver'
       end select

       if(debug_level(i) /= 0) then
         write(6,'(3a)') ' debug level for ', trim(tag), ':'
         if(iand(debug_level(i),debug_LEVELBASIC)        /= 0) write(6,'(a)') '  basic'
         if(iand(debug_level(i),debug_LEVELEXTENSIVE)    /= 0) write(6,'(a)') '  extensive'
         if(iand(debug_level(i),debug_LEVELSELECTIVE)    /= 0) then
           write(6,'(a)') ' selective on:'
           write(6,'(a24,1x,i8)') '  element:              ',debug_e
           write(6,'(a24,1x,i8)') '  ip:                   ',debug_i
           write(6,'(a24,1x,i8)') '  grain:                ',debug_g
         endif
         if(iand(debug_level(i),debug_SPECTRALRESTART)   /= 0) write(6,'(a)') '  restart'
         if(iand(debug_level(i),debug_SPECTRALFFTW)      /= 0) write(6,'(a)') '  FFTW'
         if(iand(debug_level(i),debug_SPECTRALDIVERGENCE)/= 0) write(6,'(a)') '  divergence'
         if(iand(debug_level(i),debug_SPECTRALROTATION)  /= 0) write(6,'(a)') '  rotation'
         if(iand(debug_level(i),debug_SPECTRALPETSC)     /= 0) write(6,'(a)') '  PETSc'
       endif
     enddo
 endif

end subroutine debug_init


!--------------------------------------------------------------------------------------------------
!> @brief resets all debug values
!--------------------------------------------------------------------------------------------------
subroutine debug_reset

 implicit none

 debug_StressLoopLpDistribution            = 0_pInt
 debug_StressLoopLiDistribution            = 0_pInt
 debug_StateLoopDistribution               = 0_pInt
 debug_CrystalliteLoopDistribution         = 0_pInt
 debug_MaterialpointStateLoopDistribution  = 0_pInt
 debug_MaterialpointLoopDistribution       = 0_pInt
 debug_cumLpTicks                          = 0_pLongInt
 debug_cumDeltaStateTicks                  = 0_pLongInt
 debug_cumDotStateTicks                    = 0_pLongInt
 debug_cumLpCalls                          = 0_pInt
 debug_cumDeltaStateCalls                  = 0_pInt
 debug_cumDotStateCalls                    = 0_pInt
 debug_stressMaxLocation                   = 0_pInt
 debug_stressMinLocation                   = 0_pInt
 debug_jacobianMaxLocation                 = 0_pInt
 debug_jacobianMinLocation                 = 0_pInt
 debug_stressMax                           = -huge(1.0_pReal)
 debug_stressMin                           =  huge(1.0_pReal)
 debug_jacobianMax                         = -huge(1.0_pReal)
 debug_jacobianMin                         =  huge(1.0_pReal)

end subroutine debug_reset


!--------------------------------------------------------------------------------------------------
!> @brief writes debug statements to standard out
!--------------------------------------------------------------------------------------------------
subroutine debug_info
 use numerics, only: &
   nStress, &
   nState, &
   nCryst, &
   nMPstate, &
   nHomog

 implicit none
 integer(pInt)     :: j,integral
 integer(pLongInt) :: tickrate
 character(len=1)  :: exceed

 call system_clock(count_rate=tickrate)

 !$OMP CRITICAL (write2out)
   debugOutputCryst: if (iand(debug_level(debug_CRYSTALLITE),debug_LEVELBASIC) /= 0) then
     write(6,'(/,a,/)') ' DEBUG Info (from previous cycle)'
     write(6,'(a33,1x,i12)')      'total calls to LpAndItsTangent  :',debug_cumLpCalls
     if (debug_cumLpCalls > 0_pInt) then
       write(6,'(a33,1x,f12.3)')  'total CPU time/s                :',&
         real(debug_cumLpTicks,pReal)/real(tickrate,pReal)
       write(6,'(a33,1x,f12.6)')  'avg CPU time/microsecs per call :',&
         real(debug_cumLpTicks,pReal)*1.0e6_pReal/real(tickrate*debug_cumLpCalls,pReal)
     endif
     write(6,'(/,a33,1x,i12)')     'total calls to collectDotState  :',debug_cumDotStateCalls
     if (debug_cumdotStateCalls > 0_pInt) then
       write(6,'(a33,1x,f12.3)')  'total CPU time/s                :',&
         real(debug_cumDotStateTicks,pReal)/real(tickrate,pReal)
       write(6,'(a33,1x,f12.6)')  'avg CPU time/microsecs per call :',&
         real(debug_cumDotStateTicks,pReal)*1.0e6_pReal/real(tickrate*debug_cumDotStateCalls,pReal)
     endif
     write(6,'(/,a33,1x,i12)')    'total calls to collectDeltaState:',debug_cumDeltaStateCalls
     if (debug_cumDeltaStateCalls > 0_pInt) then
       write(6,'(a33,1x,f12.3)')  'total CPU time/s                :',&
         real(debug_cumDeltaStateTicks,pReal)/real(tickrate,pReal)
       write(6,'(a33,1x,f12.6)')  'avg CPU time/microsecs per call :',&
         real(debug_cumDeltaStateTicks,pReal)*1.0e6_pReal/real(tickrate*debug_cumDeltaStateCalls,pReal)
     endif

     integral = 0_pInt
     write(6,'(3/,a)') 'distribution_StressLoopLp :    stress  stiffness'
     do j=1_pInt,nStress+1_pInt
       if (any(debug_StressLoopLpDistribution(j,:)     /= 0_pInt )) then
         integral = integral + j*(debug_StressLoopLpDistribution(j,1) + debug_StressLoopLpDistribution(j,2))
         exceed = ' '
         if (j > nStress) exceed = '+'                                                              ! last entry gets "+"
         write(6,'(i25,a1,i10,1x,i10)') min(nStress,j),exceed,debug_StressLoopLpDistribution(j,1),&
                                                              debug_StressLoopLpDistribution(j,2)
       endif
     enddo
     write(6,'(a15,i10,2(1x,i10))') '          total',integral,sum(debug_StressLoopLpDistribution(:,1)), &
                                                               sum(debug_StressLoopLpDistribution(:,2))

     integral = 0_pInt
     write(6,'(3/,a)') 'distribution_StressLoopLi :    stress  stiffness'
     do j=1_pInt,nStress+1_pInt
       if (any(debug_StressLoopLiDistribution(j,:)     /= 0_pInt )) then
         integral = integral + j*(debug_StressLoopLiDistribution(j,1) + debug_StressLoopLiDistribution(j,2))
         exceed = ' '
         if (j > nStress) exceed = '+'                                                              ! last entry gets "+"
         write(6,'(i25,a1,i10,1x,i10)') min(nStress,j),exceed,debug_StressLoopLiDistribution(j,1),&
                                                              debug_StressLoopLiDistribution(j,2)
       endif
     enddo
     write(6,'(a15,i10,2(1x,i10))') '          total',integral,sum(debug_StressLoopLiDistribution(:,1)), &
                                                               sum(debug_StressLoopLiDistribution(:,2))

     integral = 0_pInt
     write(6,'(2/,a)') 'distribution_CrystalliteStateLoop :'
     do j=1_pInt,nState+1_pInt
       if (any(debug_StateLoopDistribution(j,:) /= 0)) then
         integral = integral + j*(debug_StateLoopDistribution(j,1) + debug_StateLoopDistribution(j,2))
         exceed = ' '
         if (j > nState) exceed = '+'                                                               ! last entry gets "+"
         write(6,'(i25,a1,i10,1x,i10)') min(nState,j),exceed,debug_StateLoopDistribution(j,1),&
                                                             debug_StateLoopDistribution(j,2)
       endif
     enddo
     write(6,'(a15,i10,2(1x,i10))') '          total',integral,sum(debug_StateLoopDistribution(:,1)), &
                                                               sum(debug_StateLoopDistribution(:,2))

     integral = 0_pInt
     write(6,'(2/,a)') 'distribution_CrystalliteCutbackLoop :'
     do j=1_pInt,nCryst+1_pInt
       if (debug_CrystalliteLoopDistribution(j) /= 0) then
         integral = integral + j*debug_CrystalliteLoopDistribution(j)
         exceed = ' '
         if (j > nCryst) exceed = '+'
         write(6,'(i25,a1,i10)') min(nCryst,j),exceed,debug_CrystalliteLoopDistribution(j)
       endif
     enddo
     write(6,'(a15,i10,1x,i10)') '          total',integral,sum(debug_CrystalliteLoopDistribution)
   endif debugOutputCryst

   debugOutputHomog: if (iand(debug_level(debug_HOMOGENIZATION),debug_LEVELBASIC) /= 0) then
     integral = 0_pInt
     write(6,'(2/,a)') 'distribution_MaterialpointStateLoop :'
     do j=1_pInt,nMPstate
       if (debug_MaterialpointStateLoopDistribution(j) /= 0) then
         integral = integral + j*debug_MaterialpointStateLoopDistribution(j)
         write(6,'(i25,1x,i10)') j,debug_MaterialpointStateLoopDistribution(j)
       endif
     enddo
     write(6,'(a15,i10,1x,i10)') '          total',integral,sum(debug_MaterialpointStateLoopDistribution)

     integral = 0_pInt
     write(6,'(2/,a)') 'distribution_MaterialpointCutbackLoop :'
     do j=1_pInt,nHomog+1_pInt
       if (debug_MaterialpointLoopDistribution(j) /= 0) then
         integral = integral + j*debug_MaterialpointLoopDistribution(j)
         exceed = ' '
         if (j > nHomog) exceed = '+'
         write(6,'(i25,a1,i10)') min(nHomog,j),exceed,debug_MaterialpointLoopDistribution(j)
       endif
     enddo
     write(6,'(a15,i10,1x,i10)') '          total',integral,sum(debug_MaterialpointLoopDistribution)
   endif debugOutputHomog

   debugOutputCPFEM: if (iand(debug_level(debug_CPFEM),debug_LEVELBASIC) /= 0 &
                      .and. any(debug_stressMinLocation /= 0_pInt) &
                      .and. any(debug_stressMaxLocation /= 0_pInt) ) then
     write(6,'(2/,a,/)') ' Extreme values of returned stress and Jacobian'
     write(6,'(a39)')                      '                      value     el   ip'
     write(6,'(a14,1x,e12.3,1x,i8,1x,i4)')   ' stress   min :', debug_stressMin, debug_stressMinLocation
     write(6,'(a14,1x,e12.3,1x,i8,1x,i4)')   '          max :', debug_stressMax, debug_stressMaxLocation
     write(6,'(a14,1x,e12.3,1x,i8,1x,i4)')   ' Jacobian min :', debug_jacobianMin, debug_jacobianMinLocation
     write(6,'(a14,1x,e12.3,1x,i8,1x,i4,/)') '          max :', debug_jacobianMax, debug_jacobianMaxLocation
   endif debugOutputCPFEM
 !$OMP END CRITICAL (write2out)

end subroutine debug_info

end module debug
