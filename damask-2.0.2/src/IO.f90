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
!> @brief  input/output functions, partly depending on chosen solver
!--------------------------------------------------------------------------------------------------
module IO
 use prec, only: &
   pInt, &
   pReal

 implicit none
 private
 character(len=5), parameter, public :: &
   IO_EOF = '#EOF#'                                                                                 !< end of file string
 character(len=207), parameter, private :: &
   IO_DIVIDER = '───────────────────'//&
                '───────────────────'//&
                '───────────────────'//&
                '────────────'
 public :: &
   IO_init, &
   IO_read, &
   IO_checkAndRewind, &
   IO_open_file_stat, &
   IO_open_jobFile_stat, &
   IO_open_file, &
   IO_open_jobFile, &
   IO_write_jobFile, &
   IO_write_jobRealFile, &
   IO_write_jobIntFile, &
   IO_read_realFile, &
   IO_read_intFile, &
   IO_hybridIA, &
   IO_isBlank, &
   IO_getTag, &
   IO_countSections, &
   IO_countTagInPart, &
   IO_spotTagInPart, &
   IO_globalTagInPart, &
   IO_stringPos, &
   IO_stringValue, &
   IO_fixedStringValue ,&
   IO_floatValue, &
   IO_fixedNoEFloatValue, &
   IO_intValue, &
   IO_fixedIntValue, &
   IO_lc, &
   IO_skipChunks, &
   IO_extractValue, &
   IO_countDataLines, &
   IO_countNumericalDataLines, &
   IO_countContinuousIntValues, &
   IO_continuousIntValues, &
   IO_error, &
   IO_warning, &
   IO_intOut, &
   IO_timeStamp
#if defined(Marc4DAMASK) || defined(Abaqus)
 public :: &
   IO_open_inputFile, &
   IO_open_logFile
#endif
#ifdef Abaqus
 public :: &
   IO_abaqus_hasNoPart
#endif
 private :: &
   IO_fixedFloatValue, &
   IO_verifyFloatValue, &
   IO_verifyIntValue
#ifdef Abaqus
 private :: &
   abaqus_assembleInputFile
#endif

contains


!--------------------------------------------------------------------------------------------------
!> @brief only outputs revision number
!--------------------------------------------------------------------------------------------------
subroutine IO_init
#if defined(__GFORTRAN__) || __INTEL_COMPILER >= 1800
 use, intrinsic :: iso_fortran_env, only: &
   compiler_version, &
   compiler_options
#endif

 implicit none

 write(6,'(/,a)')   ' <<<+-  IO init  -+>>>'
 write(6,'(a15,a)') ' Current time: ',IO_timeStamp()
#include "compilation_info.f90"

end subroutine IO_init


!--------------------------------------------------------------------------------------------------
!> @brief recursively reads a line from a text file.
!!        Recursion is triggered by "{path/to/inputfile}" in a line
!--------------------------------------------------------------------------------------------------
recursive function IO_read(fileUnit,reset) result(line)

 implicit none
 integer(pInt), intent(in)           :: fileUnit                                                    !< file unit
 logical,       intent(in), optional :: reset

 integer(pInt), dimension(10) :: unitOn = 0_pInt                                                    ! save the stack of recursive file units
 integer(pInt)                :: stack = 1_pInt                                                     ! current stack position
 character(len=8192), dimension(10) :: pathOn = ''
 character(len=512)           :: path,input
 integer(pInt)                :: myStat
 character(len=65536)         :: line

 character(len=*), parameter  :: SEP = achar(47)//achar(92)                                         ! forward and backward slash ("/", "\")

!--------------------------------------------------------------------------------------------------
! reset case
 if(present(reset)) then; if (reset) then                                                           ! do not short circuit here
   do while (stack > 1_pInt)                                                                        ! can go back to former file
     close(unitOn(stack))
     stack = stack-1_pInt
   enddo
   return
 endif; endif


!--------------------------------------------------------------------------------------------------
! read from file
 unitOn(1) = fileUnit

 read(unitOn(stack),'(a65536)',END=100) line

 input = IO_getTag(line,'{','}')

!--------------------------------------------------------------------------------------------------
! normal case
 if (input == '') return                                                                            ! regular line

!--------------------------------------------------------------------------------------------------
! recursion case
 if (stack >= 10_pInt) call IO_error(104_pInt,ext_msg=input)                                        ! recursion limit reached

 inquire(UNIT=unitOn(stack),NAME=path)                                                              ! path of current file
 stack = stack+1_pInt
 if(scan(input,SEP) == 1) then                                                                      ! absolut path given (UNIX only)
   pathOn(stack) = input
 else
   pathOn(stack) = path(1:scan(path,SEP,.true.))//input                                             ! glue include to current file's dir
 endif

 open(newunit=unitOn(stack),iostat=myStat,file=pathOn(stack),action='read')                         ! open included file
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=pathOn(stack))

 line = IO_read(fileUnit)

 return

!--------------------------------------------------------------------------------------------------
! end of file case
100 if (stack > 1_pInt) then                                                                        ! can go back to former file
   close(unitOn(stack))
   stack = stack-1_pInt
   line = IO_read(fileUnit)
 else                                                                                               ! top-most file reached
   line = IO_EOF
 endif

end function IO_read


!--------------------------------------------------------------------------------------------------
!> @brief checks if unit is opened for reading, if true rewinds. Otherwise stops with
!!        error message
!--------------------------------------------------------------------------------------------------
subroutine IO_checkAndRewind(fileUnit)

 implicit none
 integer(pInt), intent(in) :: fileUnit                                                                !< file unit
 logical                   :: fileOpened
 character(len=15)         :: fileRead

 inquire(unit=fileUnit, opened=fileOpened, read=fileRead)
 if (.not. fileOpened .or. trim(fileRead)/='YES') call IO_error(102_pInt)
 rewind(fileUnit)

end subroutine IO_checkAndRewind


!--------------------------------------------------------------------------------------------------
!> @brief   opens existing file for reading to given unit. Path to file is relative to working
!!          directory
!> @details like IO_open_file_stat, but error is handled via call to IO_error and not via return
!!          value
!--------------------------------------------------------------------------------------------------
subroutine IO_open_file(fileUnit,relPath)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: relPath                                                          !< relative path from working directory

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//relPath
 open(fileUnit,status='old',iostat=myStat,file=path)
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_open_file


!--------------------------------------------------------------------------------------------------
!> @brief   opens existing file for reading to given unit. Path to file is relative to working
!!          directory
!> @details Like IO_open_file, but error is handled via return value and not via call to IO_error
!--------------------------------------------------------------------------------------------------
logical function IO_open_file_stat(fileUnit,relPath)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: relPath                                                          !< relative path from working directory

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//relPath
 open(fileUnit,status='old',iostat=myStat,file=path)
 IO_open_file_stat = (myStat == 0_pInt)

end function IO_open_file_stat


!--------------------------------------------------------------------------------------------------
!> @brief   opens existing file for reading to given unit. File is named after solver job name
!!          plus given extension and located in current working directory
!> @details like IO_open_jobFile_stat, but error is handled via call to IO_error and not via return
!!          value
!--------------------------------------------------------------------------------------------------
subroutine IO_open_jobFile(fileUnit,ext)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: ext                                                              !< extension of file

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//'.'//ext
 open(fileUnit,status='old',iostat=myStat,file=path)
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_open_jobFile


!--------------------------------------------------------------------------------------------------
!> @brief   opens existing file for reading to given unit. File is named after solver job name
!!          plus given extension and located in current working directory
!> @details Like IO_open_jobFile, but error is handled via return value and not via call to
!!          IO_error
!--------------------------------------------------------------------------------------------------
logical function IO_open_jobFile_stat(fileUnit,ext)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: ext                                                              !< extension of file

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//'.'//ext
 open(fileUnit,status='old',iostat=myStat,file=path)
 IO_open_jobFile_stat = (myStat == 0_pInt)

end function IO_open_JobFile_stat


#if defined(Marc4DAMASK) || defined(Abaqus)
!--------------------------------------------------------------------------------------------------
!> @brief opens FEM input file for reading located in current working directory to given unit
!--------------------------------------------------------------------------------------------------
subroutine IO_open_inputFile(fileUnit,modelName)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName,&
   getSolverJobName, &
   inputFileExtension

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: modelName                                                        !< model name, in case of restart not solver job name

 integer(pInt)                  :: myStat
 character(len=1024)            :: path
#ifdef Abaqus
 integer(pInt)                  :: fileType

 fileType = 1_pInt                                                                                  ! assume .pes
 path = trim(getSolverWorkingDirectoryName())//trim(modelName)//inputFileExtension(fileType)        ! attempt .pes, if it exists: it should be used
 open(fileUnit+1,status='old',iostat=myStat,file=path)
 if(myStat /= 0_pInt) then                                                                          ! if .pes does not work / exist; use conventional extension, i.e.".inp"
    fileType = 2_pInt
    path = trim(getSolverWorkingDirectoryName())//trim(modelName)//inputFileExtension(fileType)
    open(fileUnit+1,status='old',iostat=myStat,file=path)
 endif
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

 path = trim(getSolverWorkingDirectoryName())//trim(modelName)//inputFileExtension(fileType)//'_assembly'
 open(fileUnit,iostat=myStat,file=path)
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)
    if (.not.abaqus_assembleInputFile(fileUnit,fileUnit+1_pInt)) call IO_error(103_pInt)            ! strip comments and concatenate any "include"s
 close(fileUnit+1_pInt)
#endif
#ifdef Marc4DAMASK
   path = trim(getSolverWorkingDirectoryName())//trim(modelName)//inputFileExtension
   open(fileUnit,status='old',iostat=myStat,file=path)
   if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)
#endif

end subroutine IO_open_inputFile


!--------------------------------------------------------------------------------------------------
!> @brief opens existing FEM log file for reading to given unit. File is named after solver job
!!        name and located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_open_logFile(fileUnit)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName, &
   LogFileExtension

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                           !< file unit

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//LogFileExtension
 open(fileUnit,status='old',iostat=myStat,file=path)
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_open_logFile
#endif


!--------------------------------------------------------------------------------------------------
!> @brief opens ASCII file to given unit for writing. File is named after solver job name plus
!!        given extension and located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_write_jobFile(fileUnit,ext)
 use DAMASK_interface,  only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file unit
 character(len=*),   intent(in) :: ext                                                              !< extension of file

 integer(pInt)                  :: myStat
 character(len=1024)            :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//'.'//ext
 open(fileUnit,status='replace',iostat=myStat,file=path)
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_write_jobFile


!--------------------------------------------------------------------------------------------------
!> @brief opens binary file containing array of pReal numbers to given unit for writing. File is
!!        named after solver job name plus given extension and located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_write_jobRealFile(fileUnit,ext,recMultiplier)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName

 implicit none
 integer(pInt),      intent(in)           :: fileUnit                                               !< file unit
 character(len=*),   intent(in)           :: ext                                                    !< extension of file
 integer(pInt),      intent(in), optional :: recMultiplier                                          !< record length (multiple of pReal Numbers, if not given set to one)

 integer(pInt)                            :: myStat
 character(len=1024)                      :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//'.'//ext
 if (present(recMultiplier)) then
   open(fileUnit,status='replace',form='unformatted',access='direct', &
                                                   recl=pReal*recMultiplier,iostat=myStat,file=path)
 else
   open(fileUnit,status='replace',form='unformatted',access='direct', &
                                                   recl=pReal,iostat=myStat,file=path)
 endif

 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_write_jobRealFile


!--------------------------------------------------------------------------------------------------
!> @brief opens binary file containing array of pInt numbers to given unit for writing. File is
!!        named after solver job name plus given extension and located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_write_jobIntFile(fileUnit,ext,recMultiplier)
 use DAMASK_interface,  only: &
   getSolverWorkingDirectoryName, &
   getSolverJobName

 implicit none
 integer(pInt),      intent(in)           :: fileUnit                                               !< file unit
 character(len=*),   intent(in)           :: ext                                                    !< extension of file
 integer(pInt),      intent(in), optional :: recMultiplier                                          !< record length (multiple of pReal Numbers, if not given set to one)

 integer(pInt)                            :: myStat
 character(len=1024)                      :: path

 path = trim(getSolverWorkingDirectoryName())//trim(getSolverJobName())//'.'//ext
 if (present(recMultiplier)) then
   open(fileUnit,status='replace',form='unformatted',access='direct', &
                 recl=pInt*recMultiplier,iostat=myStat,file=path)
 else
   open(fileUnit,status='replace',form='unformatted',access='direct', &
                 recl=pInt,iostat=myStat,file=path)
 endif

 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_write_jobIntFile


!--------------------------------------------------------------------------------------------------
!> @brief opens binary file containing array of pReal numbers to given unit for reading. File is
!!        located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_read_realFile(fileUnit,ext,modelName,recMultiplier)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName

 implicit none
 integer(pInt),      intent(in)           :: fileUnit                                               !< file unit
 character(len=*),   intent(in)           :: ext, &                                                 !< extension of file
                                             modelName                                              !< model name, in case of restart not solver job name
 integer(pInt),      intent(in), optional :: recMultiplier                                          !< record length (multiple of pReal Numbers, if not given set to one)

 integer(pInt)                            :: myStat
 character(len=1024)                      :: path

 path = trim(getSolverWorkingDirectoryName())//trim(modelName)//'.'//ext
 if (present(recMultiplier)) then
   open(fileUnit,status='old',form='unformatted',access='direct', &
                 recl=pReal*recMultiplier,iostat=myStat,file=path)
 else
   open(fileUnit,status='old',form='unformatted',access='direct', &
                 recl=pReal,iostat=myStat,file=path)
 endif
 if (myStat /= 0_pInt) call IO_error(100_pInt,el=myStat,ext_msg=path)

end subroutine IO_read_realFile


!--------------------------------------------------------------------------------------------------
!> @brief opens binary file containing array of pInt numbers to given unit for reading. File is
!!        located in current working directory
!--------------------------------------------------------------------------------------------------
subroutine IO_read_intFile(fileUnit,ext,modelName,recMultiplier)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName

 implicit none
 integer(pInt),      intent(in)           :: fileUnit                                               !< file unit
 character(len=*),   intent(in)           :: ext, &                                                 !< extension of file
                                             modelName                                              !< model name, in case of restart not solver job name
 integer(pInt),      intent(in), optional :: recMultiplier                                          !< record length (multiple of pReal Numbers, if not given set to one)

 integer(pInt)                            :: myStat
 character(len=1024)                      :: path

 path = trim(getSolverWorkingDirectoryName())//trim(modelName)//'.'//ext
 if (present(recMultiplier)) then
   open(fileUnit,status='old',form='unformatted',access='direct', &
                 recl=pInt*recMultiplier,iostat=myStat,file=path)
 else
   open(fileUnit,status='old',form='unformatted',access='direct', &
                 recl=pInt,iostat=myStat,file=path)
 endif
 if (myStat /= 0) call IO_error(100_pInt,ext_msg=path)

end subroutine IO_read_intFile


#ifdef Abaqus
!--------------------------------------------------------------------------------------------------
!> @brief check if the input file for Abaqus contains part info
!--------------------------------------------------------------------------------------------------
logical function IO_abaqus_hasNoPart(fileUnit)

 implicit none
 integer(pInt),    intent(in)                :: fileUnit

 integer(pInt), allocatable, dimension(:)    :: chunkPos
 character(len=65536)                        :: line

 IO_abaqus_hasNoPart = .true.

610 FORMAT(A65536)
 rewind(fileUnit)
 do
   read(fileUnit,610,END=620) line
   chunkPos = IO_stringPos(line)
   if (IO_lc(IO_stringValue(line,chunkPos,1_pInt)) == '*part' ) then
     IO_abaqus_hasNoPart = .false.
     exit
   endif
 enddo

620 end function IO_abaqus_hasNoPart
#endif

!--------------------------------------------------------------------------------------------------
!> @brief hybrid IA sampling of ODFfile
!--------------------------------------------------------------------------------------------------
function IO_hybridIA(Nast,ODFfileName)
 use prec, only: &
   tol_math_check

 implicit none
 integer(pInt),                 intent(in)   :: Nast                                                !< number of samples?
 real(pReal), dimension(3,Nast)              :: IO_hybridIA
 character(len=*),              intent(in)   :: ODFfileName                                         !< name of ODF file including total path

!--------------------------------------------------------------------------------------------------
! math module is not available
 real(pReal),      parameter  :: PI = 3.141592653589793_pReal
 real(pReal),      parameter  :: INRAD = PI/180.0_pReal

 integer(pInt) :: i,j,bin,NnonZero,Nset,Nreps,reps,phi1,Phi,phi2
 integer(pInt), allocatable, dimension(:)   :: chunkPos
 integer(pInt), dimension(3)                :: steps                                                !< number of steps in phi1, Phi, and phi2 direction
 integer(pInt), dimension(4)                :: columns                                              !< columns in linearODF file where eulerangles and density are located
 integer(pInt), dimension(:), allocatable   :: binSet
 real(pReal) :: center,sum_dV_V,prob,dg_0,C,lowerC,upperC,rnd
 real(pReal), dimension(2,3)                :: limits                                               !< starting and end values for eulerangles
 real(pReal), dimension(3)                  :: deltas, &                                            !< angular step size in phi1, Phi, and phi2 direction
                                               eulers                                               !< euler angles when reading from file
 real(pReal), dimension(:,:,:), allocatable :: dV_V
 character(len=65536) :: line, keyword
 integer(pInt)                                :: headerLength
 integer(pInt),               parameter       :: FILEUNIT = 999_pInt

 IO_hybridIA = 0.0_pReal                                                                           ! initialize return value for case of error
 write(6,'(/,a,/)',advance='no') ' Using linear ODF file: '//trim(ODFfileName)
 write(6,'(/,a)') 'Eisenlohr et al.,  Computational Materials Science, 42(4):670–678, 2008'
 write(6,'(/,a)') 'https://doi.org/10.1016/j.commatsci.2007.09.015'


!--------------------------------------------------------------------------------------------------
! parse header of ODF file
 call IO_open_file(FILEUNIT,ODFfileName)
 headerLength = 0_pInt
 line=IO_read(FILEUNIT)
 chunkPos = IO_stringPos(line)
 keyword = IO_lc(IO_StringValue(line,chunkPos,2_pInt,.true.))
 if (keyword(1:4) == 'head') then
   headerLength = IO_intValue(line,chunkPos,1_pInt) + 1_pInt
 else
   call IO_error(error_ID=156_pInt, ext_msg='no header found')
 endif

!--------------------------------------------------------------------------------------------------
! figure out columns containing data
 do i = 1_pInt, headerLength-1_pInt
   line=IO_read(FILEUNIT)
 enddo
 columns = 0_pInt
 chunkPos = IO_stringPos(line)
 do i = 1_pInt, chunkPos(1)
   select case ( IO_lc(IO_StringValue(line,chunkPos,i,.true.)) )
     case ('phi1')
      columns(1) = i
     case ('phi')
      columns(2) = i
     case ('phi2')
      columns(3) = i
     case ('intensity')
      columns(4) = i
   end select
 enddo

 if (any(columns<1)) call IO_error(error_ID = 156_pInt, ext_msg='could not find expected header')

!--------------------------------------------------------------------------------------------------
! determine limits, number of steps and step size
 limits(1,1:3) = 721.0_pReal
 limits(2,1:3) =  -1.0_pReal
 steps  = 0_pInt

 line=IO_read(FILEUNIT)
 do while (trim(line) /= IO_EOF)
   chunkPos = IO_stringPos(line)
   eulers=[IO_floatValue(line,chunkPos,columns(1)),&
           IO_floatValue(line,chunkPos,columns(2)),&
           IO_floatValue(line,chunkPos,columns(3))]
   steps = steps + merge(1,0,eulers>limits(2,1:3))
   limits(1,1:3) = min(limits(1,1:3),eulers)
   limits(2,1:3) = max(limits(2,1:3),eulers)
   line=IO_read(FILEUNIT)
 enddo

 deltas = (limits(2,1:3)-limits(1,1:3))/real(steps-1_pInt,pReal)

 write(6,'(/,a,/,3(2x,f12.4,1x))',advance='no') ' Starting angles / ° = ',limits(1,1:3)
 write(6,'(/,a,/,3(2x,f12.4,1x))',advance='no') ' Ending angles / ° =   ',limits(2,1:3)
 write(6,'(/,a,/,3(2x,f12.4,1x))',advance='no') ' Angular steps / ° =   ',deltas

 if (all(abs(limits(1,1:3)) < tol_math_check)) then
   write(6,'(/,a,/)',advance='no') ' assuming vertex centered data'
   center = 0.0_pReal                                                                               ! no need to shift
   if (any(mod(int(limits(2,1:3),pInt),90)==0)) &
     call IO_error(error_ID = 156_pInt, ext_msg='linear ODF data repeated at right boundary')
 else
   write(6,'(/,a,/)',advance='no') ' assuming cell centered data'
   center = 0.5_pReal                                                                               ! shift data by half of a bin
 endif

 limits = limits*INRAD
 deltas = deltas*INRAD

!--------------------------------------------------------------------------------------------------
! read in data
 allocate(dV_V(steps(3),steps(2),steps(1)),source=0.0_pReal)
 sum_dV_V = 0.0_pReal
 dg_0 = deltas(1)*deltas(3)*2.0_pReal*sin(deltas(2)/2.0_pReal)
 NnonZero = 0_pInt

 call IO_checkAndRewind(FILEUNIT)                                                                   ! forward
 do i = 1_pInt, headerLength
   line=IO_read(FILEUNIT)
 enddo

 do phi1=1_pInt,steps(1); do Phi=1_pInt,steps(2); do phi2=1_pInt,steps(3)
   line=IO_read(FILEUNIT)
   chunkPos = IO_stringPos(line)
   eulers=[IO_floatValue(line,chunkPos,columns(1)),&                                                ! read in again for consistency check only
           IO_floatValue(line,chunkPos,columns(2)),&
           IO_floatValue(line,chunkPos,columns(3))]*INRAD
   if (any(abs((real([phi1,phi,phi2],pReal) -1.0_pReal + center)*deltas-eulers)>tol_math_check)) &  ! check if data is in expected order (phi2 fast) and correct for Fortran starting at 1
     call IO_error(error_ID = 156_pInt, ext_msg='linear ODF data not in expected order')

   prob = IO_floatValue(line,chunkPos,columns(4))
   if (prob > 0.0_pReal) then
      NnonZero = NnonZero+1_pInt
      sum_dV_V = sum_dV_V+prob
    else
      prob = 0.0_pReal
    endif
    dV_V(phi2,Phi,phi1) = prob*dg_0*sin((real(Phi-1_pInt,pReal)+center)*deltas(2))
 enddo; enddo; enddo
 close(FILEUNIT)
 dV_V = dV_V/sum_dV_V                                                                               ! normalize to 1

!--------------------------------------------------------------------------------------------------
! now fix bounds
 Nset = max(Nast,NnonZero)                                                                          ! if less than non-zero voxel count requested, sample at least that much
 lowerC = 0.0_pReal
 upperC = real(Nset, pReal)

 do while (hybridIA_reps(dV_V,steps,upperC) < Nset)
   lowerC = upperC
   upperC = upperC*2.0_pReal
 enddo

!--------------------------------------------------------------------------------------------------
! binary search for best C
 do
   C = (upperC+lowerC)/2.0_pReal
   Nreps = hybridIA_reps(dV_V,steps,C)
   if (abs(upperC-lowerC) < upperC*1.0e-14_pReal) then
     C = upperC
     Nreps = hybridIA_reps(dV_V,steps,C)
     exit
   elseif (Nreps < Nset) then
     lowerC = C
   elseif (Nreps > Nset) then
     upperC = C
   else
     exit
   endif
 enddo

 allocate(binSet(Nreps))
 bin = 0_pInt                                                                                       ! bin counter
 i = 1_pInt                                                                                         ! set counter
 do phi1=1_pInt,steps(1); do Phi=1_pInt,steps(2) ;do phi2=1_pInt,steps(3)
   reps = nint(C*dV_V(phi2,Phi,phi1), pInt)
   binSet(i:i+reps-1) = bin
   bin = bin+1_pInt                                                                                 ! advance bin
   i = i+reps                                                                                       ! advance set
 enddo; enddo; enddo

 do i=1_pInt,Nast
   if (i < Nast) then
     call random_number(rnd)
     j = nint(rnd*real(Nreps-i,pReal)+real(i,pReal)+0.5_pReal,pInt)
   else
     j = i
   endif
   bin = binSet(j)
   IO_hybridIA(1,i) = deltas(1)*(real(mod(bin/(steps(3)*steps(2)),steps(1)),pReal)+center)          ! phi1
   IO_hybridIA(2,i) = deltas(2)*(real(mod(bin/ steps(3)          ,steps(2)),pReal)+center)          ! Phi
   IO_hybridIA(3,i) = deltas(3)*(real(mod(bin                    ,steps(3)),pReal)+center)          ! phi2
   binSet(j) = binSet(i)
 enddo

 contains
 !--------------------------------------------------------------------------------------------------
 !> @brief counts hybrid IA repetitions
 !--------------------------------------------------------------------------------------------------
 integer(pInt) pure function hybridIA_reps(dV_V,steps,C)

  implicit none
  integer(pInt), intent(in), dimension(3)                          :: steps                          !< number of bins in Euler space
  real(pReal),   intent(in), dimension(steps(3),steps(2),steps(1)) :: dV_V                           !< needs description
  real(pReal),   intent(in)                                        :: C                              !< needs description

  integer(pInt) :: phi1,Phi,phi2

  hybridIA_reps = 0_pInt
  do phi1=1_pInt,steps(1); do Phi =1_pInt,steps(2); do phi2=1_pInt,steps(3)
    hybridIA_reps = hybridIA_reps+nint(C*dV_V(phi2,Phi,phi1), pInt)
  enddo; enddo; enddo

 end function hybridIA_reps

end function IO_hybridIA


!--------------------------------------------------------------------------------------------------
!> @brief identifies strings without content
!--------------------------------------------------------------------------------------------------
logical pure function IO_isBlank(string)

 implicit none
 character(len=*), intent(in) :: string                                                             !< string to check for content

 character(len=*),  parameter :: blankChar = achar(32)//achar(9)//achar(10)//achar(13)              ! whitespaces
 character(len=*),  parameter :: comment = achar(35)                                                ! comment id '#'

 integer :: posNonBlank, posComment                                                                 ! no pInt

 posNonBlank = verify(string,blankChar)
 posComment  = scan(string,comment)
 IO_isBlank = posNonBlank == 0 .or. posNonBlank == posComment

end function IO_isBlank


!--------------------------------------------------------------------------------------------------
!> @brief get tagged content of string
!--------------------------------------------------------------------------------------------------
pure function IO_getTag(string,openChar,closeChar)

 implicit none
 character(len=*), intent(in)  :: string                                                            !< string to check for tag
 character(len=len_trim(string)) :: IO_getTag

 character(len=*), intent(in)  :: openChar, &                                                       !< indicates beginning of tag
                                  closeChar                                                         !< indicates end of tag

 character(len=*), parameter   :: SEP=achar(32)//achar(9)//achar(10)//achar(13)                     ! whitespaces

 integer :: left,right                                                                              ! no pInt

 IO_getTag = ''
 left = scan(string,openChar)
 right = scan(string,closeChar)

 if (left == verify(string,SEP) .and. right > left) &                                               ! openChar is first and closeChar occurs
   IO_getTag = string(left+1:right-1)

end function IO_getTag


!--------------------------------------------------------------------------------------------------
!> @brief count number of [sections] in <part> for given file handle
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_countSections(fileUnit,part)

 implicit none
 integer(pInt),      intent(in) :: fileUnit                                                         !< file handle
 character(len=*),   intent(in) :: part                                                             !< part name in which sections are counted

 character(len=65536)           :: line

 line = ''
 IO_countSections = 0_pInt
 rewind(fileUnit)

 do while (trim(line) /= IO_EOF .and. IO_lc(IO_getTag(line,'<','>')) /= part)                       ! search for part
   line = IO_read(fileUnit)
 enddo

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   if (IO_isBlank(line)) cycle                                                                      ! skip empty lines
   if (IO_getTag(line,'<','>') /= '') then                                                          ! stop at next part
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   endif
   if (IO_getTag(line,'[',']') /= '') &                                                             ! found [section] identifier
     IO_countSections = IO_countSections + 1_pInt
 enddo

end function IO_countSections


!--------------------------------------------------------------------------------------------------
!> @brief returns array of tag counts within <part> for at most N [sections]
!--------------------------------------------------------------------------------------------------
function IO_countTagInPart(fileUnit,part,tag,Nsections)

 implicit none
 integer(pInt),   intent(in)                :: Nsections                                            !< maximum number of sections in which tag is searched for
 integer(pInt),   dimension(Nsections)      :: IO_countTagInPart
 integer(pInt),   intent(in)                :: fileUnit                                             !< file handle
 character(len=*),intent(in)                :: part, &                                              !< part in which tag is searched for
                                               tag                                                  !< tag to search for


 integer(pInt),   dimension(Nsections)      :: counter
 integer(pInt), allocatable, dimension(:)   :: chunkPos
 integer(pInt)                              :: section
 character(len=65536)                       :: line

 line = ''
 counter = 0_pInt
 section = 0_pInt

 rewind(fileUnit)
 do while (trim(line) /= IO_EOF .and. IO_lc(IO_getTag(line,'<','>')) /= part)                       ! search for part
   line = IO_read(fileUnit)
 enddo

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   if (IO_isBlank(line)) cycle                                                                      ! skip empty lines
   if (IO_getTag(line,'<','>') /= '') then                                                          ! stop at next part
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   endif
   if (IO_getTag(line,'[',']') /= '') section = section + 1_pInt                                    ! found [section] identifier
   if (section > 0) then
     chunkPos = IO_stringPos(line)
     if (tag == trim(IO_lc(IO_stringValue(line,chunkPos,1_pInt)))) &                                ! match
       counter(section) = counter(section) + 1_pInt
   endif
 enddo

 IO_countTagInPart = counter

end function IO_countTagInPart


!--------------------------------------------------------------------------------------------------
!> @brief returns array of tag presence within <part> for at most N [sections]
!--------------------------------------------------------------------------------------------------
function IO_spotTagInPart(fileUnit,part,tag,Nsections)

 implicit none
 integer(pInt),   intent(in)                :: Nsections                                            !< maximum number of sections in which tag is searched for
 logical,         dimension(Nsections)      :: IO_spotTagInPart
 integer(pInt),   intent(in)                :: fileUnit                                             !< file handle
 character(len=*),intent(in)                :: part, &                                              !< part in which tag is searched for
                                               tag                                                  !< tag to search for


 integer(pInt), allocatable, dimension(:) :: chunkPos
 integer(pInt)                            :: section
 character(len=65536)                     :: line

 IO_spotTagInPart = .false.                                                                         ! assume to nowhere spot tag
 section = 0_pInt
 line = ''

 rewind(fileUnit)
 do while (trim(line) /= IO_EOF .and. IO_lc(IO_getTag(line,'<','>')) /= part)                       ! search for part
   line = IO_read(fileUnit)
 enddo

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   if (IO_isBlank(line)) cycle                                                                      ! skip empty lines
   if (IO_getTag(line,'<','>') /= '') then                                                          ! stop at next part
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   endif
   if (IO_getTag(line,'[',']') /= '') section = section + 1_pInt                                    ! found [section] identifier
   if (section > 0_pInt) then
     chunkPos = IO_stringPos(line)
     if (tag == trim(IO_lc(IO_stringValue(line,chunkPos,1_pInt)))) &                                ! match
       IO_spotTagInPart(section) = .true.
   endif
 enddo

 end function IO_spotTagInPart


!--------------------------------------------------------------------------------------------------
!> @brief return logical whether tag is present within <part> before any [sections]
!--------------------------------------------------------------------------------------------------
logical function IO_globalTagInPart(fileUnit,part,tag)

 implicit none
 integer(pInt),   intent(in)                :: fileUnit                                             !< file handle
 character(len=*),intent(in)                :: part, &                                              !< part in which tag is searched for
                                               tag                                                  !< tag to search for


 integer(pInt), allocatable, dimension(:) :: chunkPos
 integer(pInt)                            :: section
 character(len=65536)                     :: line

 IO_globalTagInPart = .false.                                                                       ! assume to nowhere spot tag
 section = 0_pInt
 line =''

 rewind(fileUnit)
 do while (trim(line) /= IO_EOF .and. IO_lc(IO_getTag(line,'<','>')) /= part)                       ! search for part
   line = IO_read(fileUnit)
 enddo

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   if (IO_isBlank(line)) cycle                                                                      ! skip empty lines
   if (IO_getTag(line,'<','>') /= '') then                                                          ! stop at next part
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   endif
   if (IO_getTag(line,'[',']') /= '') section = section + 1_pInt                                    ! found [section] identifier
   if (section == 0_pInt) then
     chunkPos = IO_stringPos(line)
     if (tag == trim(IO_lc(IO_stringValue(line,chunkPos,1_pInt)))) &                                ! match
       IO_globalTagInPart = .true.
   endif
 enddo

end function IO_globalTagInPart


!--------------------------------------------------------------------------------------------------
!> @brief locates all space-separated chunks in given string and returns array containing number
!! them and the left/right position to be used by IO_xxxVal
!! Array size is dynamically adjusted to number of chunks found in string
!! IMPORTANT: first element contains number of chunks!
!--------------------------------------------------------------------------------------------------
pure function IO_stringPos(string)

 implicit none
 integer(pInt), dimension(:), allocatable            :: IO_stringPos
 character(len=*),                        intent(in) :: string                                      !< string in which chunk positions are searched for

 character(len=*), parameter  :: SEP=achar(44)//achar(32)//achar(9)//achar(10)//achar(13)           ! comma and whitespaces
 integer                      :: left, right                                                        ! no pInt (verify and scan return default integer)

 allocate(IO_stringPos(1), source=0_pInt)
 right = 0

 do while (verify(string(right+1:),SEP)>0)
   left  = right + verify(string(right+1:),SEP)
   right = left + scan(string(left:),SEP) - 2
   if ( string(left:left) == '#' ) exit
   IO_stringPos = [IO_stringPos,int(left, pInt), int(right, pInt)]
   IO_stringPos(1) = IO_stringPos(1)+1_pInt
 enddo

end function IO_stringPos


!--------------------------------------------------------------------------------------------------
!> @brief reads string value at myChunk from string
!--------------------------------------------------------------------------------------------------
function IO_stringValue(string,chunkPos,myChunk,silent)

 implicit none
 integer(pInt),   dimension(:),                intent(in) :: chunkPos                               !< positions of start and end of each tag/chunk in given string
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 character(len=*),                             intent(in) :: string                                 !< raw input with known start and end of each chunk
 character(len=:), allocatable                            :: IO_stringValue

 logical,                             optional,intent(in) :: silent                                 !< switch to trigger verbosity
 character(len=16), parameter                             :: MYNAME = 'IO_stringValue: '

 logical                                                  :: warn

 if (.not. present(silent)) then
   warn = .false.
 else
   warn = silent
 endif

 IO_stringValue = ''
 valuePresent: if (myChunk > chunkPos(1) .or. myChunk < 1_pInt) then
   if (warn) call IO_warning(201,el=myChunk,ext_msg=MYNAME//trim(string))
 else valuePresent
   IO_stringValue = string(chunkPos(myChunk*2):chunkPos(myChunk*2+1))
 endif valuePresent

end function IO_stringValue


!--------------------------------------------------------------------------------------------------
!> @brief reads string value at myChunk from fixed format string
!--------------------------------------------------------------------------------------------------
pure function IO_fixedStringValue (string,ends,myChunk)

 implicit none
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 integer(pInt),   dimension(:),  intent(in)               :: ends                                   !< positions of end of each tag/chunk in given string
 character(len=ends(myChunk+1)-ends(myChunk))             :: IO_fixedStringValue
 character(len=*),               intent(in)               :: string                                 !< raw input with known ends of each chunk

 IO_fixedStringValue = string(ends(myChunk)+1:ends(myChunk+1))

end function IO_fixedStringValue


!--------------------------------------------------------------------------------------------------
!> @brief reads float value at myChunk from string
!--------------------------------------------------------------------------------------------------
real(pReal) function IO_floatValue (string,chunkPos,myChunk)

 implicit none
 integer(pInt),   dimension(:),                intent(in) :: chunkPos                               !< positions of start and end of each tag/chunk in given string
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 character(len=*),                             intent(in) :: string                                 !< raw input with known start and end of each chunk
 character(len=15),              parameter  :: MYNAME = 'IO_floatValue: '
 character(len=17),              parameter  :: VALIDCHARACTERS = '0123456789eEdD.+-'

 IO_floatValue = 0.0_pReal

 valuePresent: if (myChunk > chunkPos(1) .or. myChunk < 1_pInt) then
   call IO_warning(201,el=myChunk,ext_msg=MYNAME//trim(string))
 else  valuePresent
   IO_floatValue = &
               IO_verifyFloatValue(trim(adjustl(string(chunkPos(myChunk*2):chunkPos(myChunk*2+1)))),&
                                       VALIDCHARACTERS,MYNAME)
 endif  valuePresent

end function IO_floatValue


!--------------------------------------------------------------------------------------------------
!> @brief reads float value at myChunk from fixed format string
!--------------------------------------------------------------------------------------------------
real(pReal) function IO_fixedFloatValue (string,ends,myChunk)

 implicit none
 character(len=*),               intent(in) :: string                                               !< raw input with known ends of each chunk
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 integer(pInt),   dimension(:),  intent(in) :: ends                                                 !< positions of end of each tag/chunk in given string
 character(len=20),              parameter  :: MYNAME = 'IO_fixedFloatValue: '
 character(len=17),              parameter  :: VALIDCHARACTERS = '0123456789eEdD.+-'

 IO_fixedFloatValue = &
                  IO_verifyFloatValue(trim(adjustl(string(ends(myChunk)+1_pInt:ends(myChunk+1_pInt)))),&
                                          VALIDCHARACTERS,MYNAME)

end function IO_fixedFloatValue


!--------------------------------------------------------------------------------------------------
!> @brief reads float x.y+z value at myChunk from format string
!--------------------------------------------------------------------------------------------------
real(pReal) function IO_fixedNoEFloatValue (string,ends,myChunk)

 implicit none
 character(len=*),               intent(in) :: string                                               !< raw input with known ends of each chunk
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 integer(pInt),   dimension(:),  intent(in) :: ends                                                 !< positions of end of each tag/chunk in given string
 character(len=22),              parameter  :: MYNAME = 'IO_fixedNoEFloatValue '
 character(len=13),              parameter  :: VALIDBASE = '0123456789.+-'
 character(len=12),              parameter  :: VALIDEXP  = '0123456789+-'

 real(pReal)   :: base
 integer(pInt) :: expon
 integer       :: pos_exp

 pos_exp = scan(string(ends(myChunk)+1:ends(myChunk+1)),'+-',back=.true.)
 hasExponent: if (pos_exp > 1) then
   base  = IO_verifyFloatValue(trim(adjustl(string(ends(myChunk)+1_pInt:ends(myChunk)+pos_exp-1_pInt))),&
                               VALIDBASE,MYNAME//'(base): ')
   expon = IO_verifyIntValue(trim(adjustl(string(ends(myChunk)+pos_exp:ends(myChunk+1_pInt)))),&
                               VALIDEXP,MYNAME//'(exp): ')
 else hasExponent
   base  = IO_verifyFloatValue(trim(adjustl(string(ends(myChunk)+1_pInt:ends(myChunk+1_pInt)))),&
                               VALIDBASE,MYNAME//'(base): ')
   expon = 0_pInt
 endif hasExponent
 IO_fixedNoEFloatValue = base*10.0_pReal**real(expon,pReal)

end function IO_fixedNoEFloatValue


!--------------------------------------------------------------------------------------------------
!> @brief reads integer value at myChunk from string
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_intValue(string,chunkPos,myChunk)

 implicit none
 character(len=*),                             intent(in) :: string                                 !< raw input with known start and end of each chunk
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 integer(pInt),   dimension(:),                intent(in) :: chunkPos                               !< positions of start and end of each tag/chunk in given string
 character(len=13),              parameter  :: MYNAME = 'IO_intValue: '
 character(len=12),              parameter  :: VALIDCHARACTERS = '0123456789+-'

 IO_intValue = 0_pInt

 valuePresent: if (myChunk > chunkPos(1) .or. myChunk < 1_pInt) then
   call IO_warning(201,el=myChunk,ext_msg=MYNAME//trim(string))
 else valuePresent
   IO_intValue = IO_verifyIntValue(trim(adjustl(string(chunkPos(myChunk*2):chunkPos(myChunk*2+1)))),&
                                   VALIDCHARACTERS,MYNAME)
 endif valuePresent

end function IO_intValue


!--------------------------------------------------------------------------------------------------
!> @brief reads integer value at myChunk from fixed format string
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_fixedIntValue(string,ends,myChunk)

 implicit none
 character(len=*),               intent(in) :: string                                               !< raw input with known ends of each chunk
 integer(pInt),                                intent(in) :: myChunk                                !< position number of desired chunk
 integer(pInt),   dimension(:),  intent(in) :: ends                                                 !< positions of end of each tag/chunk in given string
 character(len=20),              parameter  :: MYNAME = 'IO_fixedIntValue: '
 character(len=12),              parameter  :: VALIDCHARACTERS = '0123456789+-'

 IO_fixedIntValue = IO_verifyIntValue(trim(adjustl(string(ends(myChunk)+1_pInt:ends(myChunk+1_pInt)))),&
                                      VALIDCHARACTERS,MYNAME)

end function IO_fixedIntValue


!--------------------------------------------------------------------------------------------------
!> @brief changes characters in string to lower case
!--------------------------------------------------------------------------------------------------
pure function IO_lc(string)

 implicit none
 character(len=*), intent(in) :: string                                                             !< string to convert
 character(len=len(string))   :: IO_lc

 character(26), parameter :: LOWER = 'abcdefghijklmnopqrstuvwxyz'
 character(26), parameter :: UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

 integer                      :: i,n                                                                ! no pInt (len returns default integer)

 IO_lc = string
 do i=1,len(string)
   n = index(UPPER,IO_lc(i:i))
   if (n/=0) IO_lc(i:i) = LOWER(n:n)
 enddo

end function IO_lc


!--------------------------------------------------------------------------------------------------
!> @brief reads file to skip (at least) N chunks (may be over multiple lines)
!--------------------------------------------------------------------------------------------------
subroutine IO_skipChunks(fileUnit,N)

 implicit none
 integer(pInt), intent(in)                :: fileUnit, &                                            !< file handle
                                             N                                                      !< minimum number of chunks to skip

 integer(pInt)                            :: remainingChunks
 character(len=65536)                     :: line

 line = ''
 remainingChunks = N

 do while (trim(line) /= IO_EOF .and. remainingChunks > 0)
   line = IO_read(fileUnit)
   remainingChunks = remainingChunks - (size(IO_stringPos(line))-1_pInt)/2_pInt
 enddo
end subroutine IO_skipChunks


!--------------------------------------------------------------------------------------------------
!> @brief extracts string value from key=value pair and check whether key matches
!--------------------------------------------------------------------------------------------------
character(len=300) pure function IO_extractValue(pair,key)

 implicit none
 character(len=*), intent(in) :: pair, &                                                            !< key=value pair
                                 key                                                                !< key to be expected

 character(len=*), parameter  :: SEP = achar(61)                                                    ! '='

 integer                      :: myChunk                                                            !< position number of desired chunk

 IO_extractValue = ''

 myChunk = scan(pair,SEP)
 if (myChunk > 0 .and. pair(:myChunk-1) == key) IO_extractValue = pair(myChunk+1:)                  ! extract value if key matches

end function IO_extractValue


!--------------------------------------------------------------------------------------------------
!> @brief count lines containig data up to next *keyword
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_countDataLines(fileUnit)

 implicit none
 integer(pInt), intent(in)                :: fileUnit                                               !< file handle


 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536)                     :: line, &
                                             tmp

 IO_countDataLines = 0_pInt
 line = ''

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   chunkPos = IO_stringPos(line)
   tmp = IO_lc(IO_stringValue(line,chunkPos,1_pInt))
   if (tmp(1:1) == '*' .and. tmp(2:2) /= '*') then                                                  ! found keyword
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   else
     if (tmp(2:2) /= '*') IO_countDataLines = IO_countDataLines + 1_pInt
   endif
 enddo
 backspace(fileUnit)

end function IO_countDataLines


!--------------------------------------------------------------------------------------------------
!> @brief count lines containig data up to next *keyword
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_countNumericalDataLines(fileUnit)

 implicit none
 integer(pInt), intent(in)                :: fileUnit                                               !< file handle


 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536)                     :: line, &
                                             tmp

 IO_countNumericalDataLines = 0_pInt
 line = ''

 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   chunkPos = IO_stringPos(line)
   tmp = IO_lc(IO_stringValue(line,chunkPos,1_pInt))
   if (verify(trim(tmp),'0123456789') == 0) then                                                    ! numerical values
     IO_countNumericalDataLines = IO_countNumericalDataLines + 1_pInt
   else
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   endif
 enddo
 backspace(fileUnit)

end function IO_countNumericalDataLines

!--------------------------------------------------------------------------------------------------
!> @brief count items in consecutive lines depending on lines
!> @details Marc:      ints concatenated by "c" as last char or range of values a "to" b
!> Abaqus:    triplet of start,stop,inc
!> Spectral:  ints concatenated range of a "to" b, multiple entries with a "of" b
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_countContinuousIntValues(fileUnit)

 implicit none
 integer(pInt), intent(in) :: fileUnit

#ifdef Abaqus
 integer(pInt)                            :: l,c
#endif
 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536)                     :: line

 IO_countContinuousIntValues = 0_pInt
 line = ''

#ifndef Abaqus
 do while (trim(line) /= IO_EOF)
   line = IO_read(fileUnit)
   chunkPos = IO_stringPos(line)
   if (chunkPos(1) < 1_pInt) then                                                                   ! empty line
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit
   elseif (IO_lc(IO_stringValue(line,chunkPos,2_pInt)) == 'to' ) then                               ! found range indicator
     IO_countContinuousIntValues = 1_pInt + abs(  IO_intValue(line,chunkPos,3_pInt) &
                                                - IO_intValue(line,chunkPos,1_pInt))
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit                                                                                           ! only one single range indicator allowed
   else if (IO_lc(IO_stringValue(line,chunkPos,2_pInt)) == 'of' ) then                              ! found multiple entries indicator
     IO_countContinuousIntValues = IO_intValue(line,chunkPos,1_pInt)
     line = IO_read(fileUnit, .true.)                                                               ! reset IO_read
     exit                                                                                           ! only one single multiplier allowed
   else
     IO_countContinuousIntValues = IO_countContinuousIntValues+chunkPos(1)-1_pInt                   ! add line's count when assuming 'c'
     if ( IO_lc(IO_stringValue(line,chunkPos,chunkPos(1))) /= 'c' ) then                            ! line finished, read last value
       IO_countContinuousIntValues = IO_countContinuousIntValues+1_pInt
       line = IO_read(fileUnit, .true.)                                                             ! reset IO_read
       exit                                                                                         ! data ended
     endif
   endif
 enddo
#else
 c = IO_countDataLines(fileUnit)
 do l = 1_pInt,c
   backspace(fileUnit)                                                                              ! ToDo: substitute by rewind?
 enddo

 l = 1_pInt
 do while (trim(line) /= IO_EOF .and. l <= c)                                                       ! ToDo: is this correct
   l = l + 1_pInt
   line = IO_read(fileUnit)
   chunkPos = IO_stringPos(line)
   IO_countContinuousIntValues = IO_countContinuousIntValues + 1_pInt + &                           ! assuming range generation
                            (IO_intValue(line,chunkPos,2_pInt)-IO_intValue(line,chunkPos,1_pInt))/&
                                                     max(1_pInt,IO_intValue(line,chunkPos,3_pInt))
 enddo
#endif

end function IO_countContinuousIntValues


!--------------------------------------------------------------------------------------------------
!> @brief return integer list corresponding to items in consecutive lines.
!! First integer in array is counter
!> @details Marc:      ints concatenated by "c" as last char, range of a "to" b, or named set
!! Abaqus:    triplet of start,stop,inc or named set
!! Spectral:  ints concatenated range of a "to" b, multiple entries with a "of" b
!--------------------------------------------------------------------------------------------------
function IO_continuousIntValues(fileUnit,maxN,lookupName,lookupMap,lookupMaxN)

 implicit none
 integer(pInt),                     intent(in) :: maxN
 integer(pInt),     dimension(1+maxN)          :: IO_continuousIntValues

 integer(pInt),                     intent(in) :: fileUnit, &
                                                  lookupMaxN
 integer(pInt),     dimension(:,:), intent(in) :: lookupMap
 character(len=64), dimension(:),   intent(in) :: lookupName
 integer(pInt) :: i,first,last
#ifdef Abaqus
 integer(pInt) :: j,l,c
#endif

 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536) line
 logical rangeGeneration

 IO_continuousIntValues = 0_pInt
 rangeGeneration = .false.

#ifndef Abaqus
 do
   read(fileUnit,'(A65536)',end=100) line
   chunkPos = IO_stringPos(line)
   if (chunkPos(1) < 1_pInt) then                                                                   ! empty line
     exit
   elseif (verify(IO_stringValue(line,chunkPos,1_pInt),'0123456789') > 0) then                      ! a non-int, i.e. set name
     do i = 1_pInt, lookupMaxN                                                                      ! loop over known set names
       if (IO_stringValue(line,chunkPos,1_pInt) == lookupName(i)) then                              ! found matching name
         IO_continuousIntValues = lookupMap(:,i)                                                    ! return resp. entity list
         exit
       endif
     enddo
     exit
   else if (chunkPos(1) > 2_pInt .and. IO_lc(IO_stringValue(line,chunkPos,2_pInt)) == 'to' ) then   ! found range indicator
     first = IO_intValue(line,chunkPos,1_pInt)
     last  = IO_intValue(line,chunkPos,3_pInt)
     do i = first, last, sign(1_pInt,last-first)
       IO_continuousIntValues(1) = IO_continuousIntValues(1) + 1_pInt
       IO_continuousIntValues(1+IO_continuousIntValues(1)) = i
     enddo
     exit
   else if (chunkPos(1) > 2_pInt .and. IO_lc(IO_stringValue(line,chunkPos,2_pInt)) == 'of' ) then   ! found multiple entries indicator
     IO_continuousIntValues(1) = IO_intValue(line,chunkPos,1_pInt)
     IO_continuousIntValues(2:IO_continuousIntValues(1)+1) = IO_intValue(line,chunkPos,3_pInt)
     exit
   else
     do i = 1_pInt,chunkPos(1)-1_pInt                                                               ! interpret up to second to last value
       IO_continuousIntValues(1) = IO_continuousIntValues(1) + 1_pInt
       IO_continuousIntValues(1+IO_continuousIntValues(1)) = IO_intValue(line,chunkPos,i)
     enddo
     if ( IO_lc(IO_stringValue(line,chunkPos,chunkPos(1))) /= 'c' ) then                            ! line finished, read last value
       IO_continuousIntValues(1) = IO_continuousIntValues(1) + 1_pInt
       IO_continuousIntValues(1+IO_continuousIntValues(1)) = IO_intValue(line,chunkPos,chunkPos(1))
       exit
     endif
   endif
 enddo
#else
 c = IO_countDataLines(fileUnit)
 do l = 1_pInt,c
   backspace(fileUnit)
 enddo

!--------------------------------------------------------------------------------------------------
! check if the element values in the elset are auto generated
 backspace(fileUnit)
 read(fileUnit,'(A65536)',end=100) line
 chunkPos = IO_stringPos(line)
 do i = 1_pInt,chunkPos(1)
   if (IO_lc(IO_stringValue(line,chunkPos,i)) == 'generate') rangeGeneration = .true.
 enddo

 do l = 1_pInt,c
   read(fileUnit,'(A65536)',end=100) line
   chunkPos = IO_stringPos(line)
   if (verify(IO_stringValue(line,chunkPos,1_pInt),'0123456789') > 0) then                          ! a non-int, i.e. set names follow on this line
     do i = 1_pInt,chunkPos(1)                                                                      ! loop over set names in line
       do j = 1_pInt,lookupMaxN                                                                     ! look through known set names
         if (IO_stringValue(line,chunkPos,i) == lookupName(j)) then                                 ! found matching name
           first = 2_pInt + IO_continuousIntValues(1)                                               ! where to start appending data
           last  = first + lookupMap(1,j) - 1_pInt                                                  ! up to where to append data
           IO_continuousIntValues(first:last) = lookupMap(2:1+lookupMap(1,j),j)                     ! add resp. entity list
           IO_continuousIntValues(1) = IO_continuousIntValues(1) + lookupMap(1,j)                   ! count them
         endif
       enddo
     enddo
   else if (rangeGeneration) then                                                                   ! range generation
     do i = IO_intValue(line,chunkPos,1_pInt),&
            IO_intValue(line,chunkPos,2_pInt),&
            max(1_pInt,IO_intValue(line,chunkPos,3_pInt))
       IO_continuousIntValues(1) = IO_continuousIntValues(1) + 1_pInt
       IO_continuousIntValues(1+IO_continuousIntValues(1)) = i
     enddo
   else                                                                                             ! read individual elem nums
     do i = 1_pInt,chunkPos(1)
       IO_continuousIntValues(1) = IO_continuousIntValues(1) + 1_pInt
       IO_continuousIntValues(1+IO_continuousIntValues(1)) = IO_intValue(line,chunkPos,i)
     enddo
   endif
 enddo
#endif

100 end function IO_continuousIntValues


!--------------------------------------------------------------------------------------------------
!> @brief returns format string for integer values without leading zeros
!--------------------------------------------------------------------------------------------------
pure function IO_intOut(intToPrint)

  implicit none
  character(len=19) :: N_Digits                                                                     ! maximum digits for 64 bit integer
  character(len=40) :: IO_intOut
  integer(pInt), intent(in) :: intToPrint

  write(N_Digits, '(I19.19)') 1_pInt + int(log10(real(intToPrint)),pInt)
  IO_intOut = 'I'//trim(N_Digits)//'.'//trim(N_Digits)

end function IO_intOut


!--------------------------------------------------------------------------------------------------
!> @brief returns time stamp
!--------------------------------------------------------------------------------------------------
function IO_timeStamp()

  implicit none
  character(len=10) :: IO_timeStamp
  integer(pInt), dimension(8) :: values

  call DATE_AND_TIME(VALUES=values)
  write(IO_timeStamp,'(i2.2,a1,i2.2,a1,i2.2)') values(5),':',values(6),':',values(7)

end function IO_timeStamp


!--------------------------------------------------------------------------------------------------
!> @brief write error statements to standard out and terminate the Marc/spectral run with exit #9xxx
!> in ABAQUS either time step is reduced or execution terminated
!--------------------------------------------------------------------------------------------------
subroutine IO_error(error_ID,el,ip,g,instance,ext_msg)

 implicit none
 integer(pInt),              intent(in) :: error_ID
 integer(pInt),    optional, intent(in) :: el,ip,g,instance
 character(len=*), optional, intent(in) :: ext_msg

 external                               :: quit
 character(len=1024)                    :: msg
 character(len=1024)                    :: formatString

 select case (error_ID)

!--------------------------------------------------------------------------------------------------
! internal errors
 case (0_pInt)
   msg = 'internal check failed:'

!--------------------------------------------------------------------------------------------------
! file handling errors
 case (100_pInt)
   msg = 'could not open file:'
 case (101_pInt)
   msg = 'write error for file:'
 case (102_pInt)
   msg = 'could not read file:'
 case (103_pInt)
   msg = 'could not assemble input files'
 case (104_pInt)
   msg = '{input} recursion limit reached'
 case (105_pInt)
   msg = 'unknown output:'

!--------------------------------------------------------------------------------------------------
! lattice error messages
 case (130_pInt)
   msg = 'unknown lattice structure encountered'
 case (131_pInt)
   msg = 'hex lattice structure with invalid c/a ratio'
 case (132_pInt)
   msg = 'trans_lattice_structure not possible'
 case (133_pInt)
   msg = 'transformed hex lattice structure with invalid c/a ratio'
 case (135_pInt)
   msg = 'zero entry on stiffness diagonal'
 case (136_pInt)
   msg = 'zero entry on stiffness diagonal for transformed phase'

!--------------------------------------------------------------------------------------------------
! material error messages and related messages in mesh
 case (150_pInt)
   msg = 'index out of bounds'
 case (151_pInt)
   msg = 'microstructure has no constituents'
 case (153_pInt)
   msg = 'sum of phase fractions differs from 1'
 case (154_pInt)
   msg = 'homogenization index out of bounds'
 case (155_pInt)
   msg = 'microstructure index out of bounds'
 case (156_pInt)
   msg = 'reading from ODF file'
 case (157_pInt)
   msg = 'illegal texture transformation specified'
 case (160_pInt)
   msg = 'no entries in config part'
 case (165_pInt)
   msg = 'homogenization configuration'
 case (170_pInt)
   msg = 'no homogenization specified via State Variable 2'
 case (180_pInt)
   msg = 'no microstructure specified via State Variable 3'
 case (190_pInt)
   msg = 'unknown element type:'

!--------------------------------------------------------------------------------------------------
! plasticity error messages
 case (200_pInt)
   msg = 'unknown elasticity specified:'
 case (201_pInt)
   msg = 'unknown plasticity specified:'

 case (210_pInt)
   msg = 'unknown material parameter:'
 case (211_pInt)
   msg = 'material parameter out of bounds:'

!--------------------------------------------------------------------------------------------------
! numerics error messages
 case (300_pInt)
   msg = 'unknown numerics parameter:'
 case (301_pInt)
   msg = 'numerics parameter out of bounds:'

!--------------------------------------------------------------------------------------------------
! math errors
 case (400_pInt)
   msg = 'matrix inversion error'
 case (401_pInt)
   msg = 'math_check failed'
 case (405_pInt)
   msg = 'I_TO_HALTON-error: an input base BASE is <= 1'
 case (406_pInt)
   msg = 'Prime-error: N must be between 0 and PRIME_MAX'
 case (407_pInt)
   msg = 'Polar decomposition error'
 case (409_pInt)
   msg = 'math_check: R*v == q*v failed'
 case (410_pInt)
   msg = 'eigenvalues computation error'

!-------------------------------------------------------------------------------------------------
! homogenization errors
 case (500_pInt)
   msg = 'unknown homogenization specified'

!--------------------------------------------------------------------------------------------------
! user errors
 case (600_pInt)
   msg = 'Ping-Pong not possible when using non-DAMASK elements'
 case (601_pInt)
   msg = 'Ping-Pong needed when using non-local plasticity'
 case (602_pInt)
   msg = 'invalid selection for debug'

!-------------------------------------------------------------------------------------------------
! DAMASK_marc errors
 case (700_pInt)
   msg = 'invalid materialpoint result requested'
 case (701_pInt)
   msg = 'not supported input file format, use Marc 2016 or earlier'

!-------------------------------------------------------------------------------------------------
! errors related to spectral solver
 case (809_pInt)
   msg = 'initializing FFTW'
 case (810_pInt)
   msg = 'FFTW plan creation'
 case (831_pInt)
   msg = 'mask consistency violated in spectral loadcase'
 case (832_pInt)
   msg = 'ill-defined L (line partly defined) in spectral loadcase'
 case (834_pInt)
   msg = 'negative time increment in spectral loadcase'
 case (835_pInt)
   msg = 'non-positive increments in spectral loadcase'
 case (836_pInt)
   msg = 'non-positive result frequency in spectral loadcase'
 case (837_pInt)
   msg = 'incomplete loadcase'
 case (838_pInt)
   msg = 'mixed boundary conditions allow rotation'
 case (841_pInt)
   msg = 'missing header length info in spectral mesh'
 case (842_pInt)
   msg = 'homogenization in spectral mesh'
 case (843_pInt)
   msg = 'grid in spectral mesh'
 case (844_pInt)
   msg = 'size in spectral mesh'
 case (845_pInt)
   msg = 'incomplete information in spectral mesh header'
 case (846_pInt)
   msg = 'not a rotation defined for loadcase rotation'
 case (847_pInt)
   msg = 'update of gamma operator not possible when pre-calculated'
 case (880_pInt)
   msg = 'mismatch of microstructure count and a*b*c in geom file'
 case (891_pInt)
   msg = 'unknown solver type selected'
 case (892_pInt)
   msg = 'unknown filter type selected'
 case (893_pInt)
   msg = 'PETSc: SNES_DIVERGED_FNORM_NAN'
 case (894_pInt)
   msg = 'MPI error'

!-------------------------------------------------------------------------------------------------
! error messages related to parsing of Abaqus input file
 case (900_pInt)
   msg = 'improper definition of nodes in input file (Nnodes < 2)'
 case (901_pInt)
   msg = 'no elements defined in input file (Nelems = 0)'
 case (902_pInt)
   msg = 'no element sets defined in input file (No *Elset exists)'
 case (903_pInt)
   msg = 'no materials defined in input file (Look into section assigments)'
 case (904_pInt)
   msg = 'no elements could be assigned for Elset: '
 case (905_pInt)
   msg = 'error in mesh_abaqus_map_materials'
 case (906_pInt)
   msg = 'error in mesh_abaqus_count_cpElements'
 case (907_pInt)
   msg = 'size of mesh_mapFEtoCPelem in mesh_abaqus_map_elements'
 case (908_pInt)
   msg = 'size of mesh_mapFEtoCPnode in mesh_abaqus_map_nodes'
 case (909_pInt)
   msg = 'size of mesh_node in mesh_abaqus_build_nodes not equal to mesh_Nnodes'


!-------------------------------------------------------------------------------------------------
! general error messages
 case (666_pInt)
   msg = 'memory leak detected'
 case default
   msg = 'unknown error number...'

 end select

 !$OMP CRITICAL (write2out)
 write(0,'(/,a)')                ' ┌'//IO_DIVIDER//'┐'
 write(0,'(a,24x,a,40x,a)')      ' │','error',                                             '│'
 write(0,'(a,24x,i3,42x,a)')     ' │',error_ID,                                            '│'
 write(0,'(a)')                  ' ├'//IO_DIVIDER//'┤'
 write(formatString,'(a,i6.6,a,i6.6,a)') '(1x,a4,a',max(1,len(trim(msg))),',',&
                                                    max(1,72-len(trim(msg))-4),'x,a)'
 write(0,formatString)            '│ ',trim(msg),                                          '│'
 if (present(ext_msg)) then
   write(formatString,'(a,i6.6,a,i6.6,a)') '(1x,a4,a',max(1,len(trim(ext_msg))),',',&
                                                      max(1,72-len(trim(ext_msg))-4),'x,a)'
   write(0,formatString)          '│ ',trim(ext_msg),                                      '│'
 endif
 if (present(el)) &
   write(0,'(a19,1x,i9,44x,a3)') ' │ at element    ',el,                                   '│'
 if (present(ip)) &
   write(0,'(a19,1x,i9,44x,a3)') ' │ at IP         ',ip,                                   '│'
 if (present(g)) &
   write(0,'(a19,1x,i9,44x,a3)') ' │ at constituent',g,                                    '│'
 if (present(instance)) &
   write(0,'(a19,1x,i9,44x,a3)') ' │ at instance   ',instance,                             '│'
 write(0,'(a,69x,a)')            ' │',                                                     '│'
 write(0,'(a)')                  ' └'//IO_DIVIDER//'┘'
 flush(0)
 call quit(9000_pInt+error_ID)
 !$OMP END CRITICAL (write2out)

end subroutine IO_error


!--------------------------------------------------------------------------------------------------
!> @brief writes warning statement to standard out
!--------------------------------------------------------------------------------------------------
subroutine IO_warning(warning_ID,el,ip,g,ext_msg)

 implicit none
 integer(pInt),              intent(in) :: warning_ID
 integer(pInt),    optional, intent(in) :: el,ip,g
 character(len=*), optional, intent(in) :: ext_msg

 character(len=1024)                    :: msg
 character(len=1024)                    :: formatString

 select case (warning_ID)
 case (1_pInt)
   msg = 'unknown key'
 case (34_pInt)
   msg = 'invalid restart increment given'
 case (35_pInt)
   msg = 'could not get $DAMASK_NUM_THREADS'
 case (40_pInt)
   msg = 'found spectral solver parameter'
 case (42_pInt)
   msg = 'parameter has no effect'
 case (43_pInt)
   msg = 'main diagonal of C66 close to zero'
 case (47_pInt)
   msg = 'no valid parameter for FFTW, using FFTW_PATIENT'
 case (50_pInt)
   msg = 'not all available slip system families are defined'
 case (51_pInt)
   msg = 'not all available twin system families are defined'
 case (52_pInt)
   msg = 'not all available parameters are defined'
 case (53_pInt)
   msg = 'not all available transformation system families are defined'
 case (101_pInt)
   msg = 'crystallite debugging off'
 case (201_pInt)
   msg = 'position not found when parsing line'
 case (202_pInt)
   msg = 'invalid character in string chunk'
 case (203_pInt)
   msg = 'interpretation of string chunk failed'
 case (600_pInt)
   msg = 'crystallite responds elastically'
 case (601_pInt)
   msg = 'stiffness close to zero'
 case (650_pInt)
   msg = 'polar decomposition failed'
 case (700_pInt)
   msg = 'unknown crystal symmetry'
 case (850_pInt)
   msg = 'max number of cut back exceeded, terminating'
 case default
   msg = 'unknown warning number'
 end select

 !$OMP CRITICAL (write2out)
 write(6,'(/,a)')                ' ┌'//IO_DIVIDER//'┐'
 write(6,'(a,24x,a,38x,a)')      ' │','warning',                                           '│'
 write(6,'(a,24x,i3,42x,a)')     ' │',warning_ID,                                          '│'
 write(6,'(a)')                  ' ├'//IO_DIVIDER//'┤'
 write(formatString,'(a,i6.6,a,i6.6,a)') '(1x,a4,a',max(1,len(trim(msg))),',',&
                                                    max(1,72-len(trim(msg))-4),'x,a)'
 write(6,formatString)            '│ ',trim(msg),                                          '│'
 if (present(ext_msg)) then
   write(formatString,'(a,i6.6,a,i6.6,a)') '(1x,a4,a',max(1,len(trim(ext_msg))),',',&
                                                      max(1,72-len(trim(ext_msg))-4),'x,a)'
   write(6,formatString)          '│ ',trim(ext_msg),                                      '│'
 endif
 if (present(el)) &
   write(6,'(a19,1x,i9,44x,a3)') ' │ at element    ',el,                                   '│'
 if (present(ip)) &
   write(6,'(a19,1x,i9,44x,a3)') ' │ at IP         ',ip,                                   '│'
 if (present(g)) &
   write(6,'(a19,1x,i9,44x,a3)') ' │ at constituent',g,                                    '│'
 write(6,'(a,69x,a)')            ' │',                                                     '│'
 write(6,'(a)')                  ' └'//IO_DIVIDER//'┘'
 flush(6)
 !$OMP END CRITICAL (write2out)

end subroutine IO_warning


!--------------------------------------------------------------------------------------------------
! internal helper functions

!--------------------------------------------------------------------------------------------------
!> @brief returns verified integer value in given string
!--------------------------------------------------------------------------------------------------
integer(pInt) function IO_verifyIntValue (string,validChars,myName)

 implicit none
 character(len=*), intent(in) :: string, &                                                            !< string for conversion to int value. Must not contain spaces!
                                 validChars, &                                                        !< valid characters in string
                                 myName                                                               !< name of caller function (for debugging)
 integer(pInt)                :: readStatus, invalidWhere
 !character(len=len(trim(string))) :: trimmed does not work with ifort 14.0.1

 IO_verifyIntValue = 0_pInt

 invalidWhere = verify(string,validChars)
 if (invalidWhere == 0_pInt) then
   read(UNIT=string,iostat=readStatus,FMT=*) IO_verifyIntValue                                        ! no offending chars found
   if (readStatus /= 0_pInt) &                                                                        ! error during string to integer conversion
     call IO_warning(203_pInt,ext_msg=myName//'"'//string//'"')
 else
   call IO_warning(202_pInt,ext_msg=myName//'"'//string//'"')                                         ! complain about offending characters
   read(UNIT=string(1_pInt:invalidWhere-1_pInt),iostat=readStatus,FMT=*) IO_verifyIntValue            ! interpret remaining string
   if (readStatus /= 0_pInt) &                                                                        ! error during string to integer conversion
     call IO_warning(203_pInt,ext_msg=myName//'"'//string(1_pInt:invalidWhere-1_pInt)//'"')
 endif

end function IO_verifyIntValue


!--------------------------------------------------------------------------------------------------
!> @brief returns verified float value in given string
!--------------------------------------------------------------------------------------------------
real(pReal) function IO_verifyFloatValue (string,validChars,myName)

 implicit none
 character(len=*), intent(in) :: string, &                                                            !< string for conversion to int value. Must not contain spaces!
                                 validChars, &                                                        !< valid characters in string
                                 myName                                                               !< name of caller function (for debugging)

 integer(pInt)                :: readStatus, invalidWhere
 !character(len=len(trim(string))) :: trimmed does not work with ifort 14.0.1

 IO_verifyFloatValue = 0.0_pReal

 invalidWhere = verify(string,validChars)
 if (invalidWhere == 0_pInt) then
   read(UNIT=string,iostat=readStatus,FMT=*) IO_verifyFloatValue                                      ! no offending chars found
   if (readStatus /= 0_pInt) &                                                                        ! error during string to float conversion
     call IO_warning(203_pInt,ext_msg=myName//'"'//string//'"')
 else
   call IO_warning(202_pInt,ext_msg=myName//'"'//string//'"')                                         ! complain about offending characters
   read(UNIT=string(1_pInt:invalidWhere-1_pInt),iostat=readStatus,FMT=*) IO_verifyFloatValue          ! interpret remaining string
   if (readStatus /= 0_pInt) &                                                                        ! error during string to float conversion
     call IO_warning(203_pInt,ext_msg=myName//'"'//string(1_pInt:invalidWhere-1_pInt)//'"')
 endif

end function IO_verifyFloatValue

#ifdef Abaqus
!--------------------------------------------------------------------------------------------------
!> @brief create a new input file for abaqus simulations by removing all comment lines and
!> including "include"s
!--------------------------------------------------------------------------------------------------
recursive function abaqus_assembleInputFile(unit1,unit2) result(createSuccess)
 use DAMASK_interface, only: &
   getSolverWorkingDirectoryName

 implicit none
 integer(pInt), intent(in)                :: unit1, &
                                             unit2


 integer(pInt), allocatable, dimension(:) :: chunkPos
 character(len=65536)                     :: line,fname
 logical                                  :: createSuccess,fexist


 do
   read(unit2,'(A65536)',END=220) line
   chunkPos = IO_stringPos(line)

   if (IO_lc(IO_StringValue(line,chunkPos,1_pInt))=='*include') then
     fname = trim(getSolverWorkingDirectoryName())//trim(line(9+scan(line(9:),'='):))
     inquire(file=fname, exist=fexist)
     if (.not.(fexist)) then
       !$OMP CRITICAL (write2out)
         write(6,*)'ERROR: file does not exist error in abaqus_assembleInputFile'
         write(6,*)'filename: ', trim(fname)
       !$OMP END CRITICAL (write2out)
       createSuccess = .false.
       return
     endif
     open(unit2+1,err=200,status='old',file=fname)
     if (abaqus_assembleInputFile(unit1,unit2+1_pInt)) then
       createSuccess=.true.
       close(unit2+1)
     else
       createSuccess=.false.
       return
     endif
   else if (line(1:2) /= '**' .OR. line(1:8)=='**damask') then
     write(unit1,'(A)') trim(line)
   endif
 enddo

220 createSuccess = .true.
 return

200 createSuccess =.false.

end function abaqus_assembleInputFile
#endif

end module IO
