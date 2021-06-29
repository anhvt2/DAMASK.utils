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
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Koen Janssens, Paul Scherrer Institut
!> @author Arun Prakash, Fraunhofer IWM
!> @brief interfaces DAMASK with Abaqus/Explicit
!> @details put the included file abaqus_v6.env in either your home or model directory, 
!> it is a minimum Abaqus environment file  containing all changes necessary to use the 
!> DAMASK subroutine (see Abaqus documentation for more information on the use of abaqus_v6.env)
!--------------------------------------------------------------------------------------------------

#ifndef INT
#define INT 4
#endif

#ifndef FLOAT
#define FLOAT 8
#endif

#define Abaqus

#include "prec.f90"

module DAMASK_interface

implicit none
character(len=4), dimension(2),  parameter :: INPUTFILEEXTENSION = ['.pes','.inp']
character(len=4),                parameter :: LOGFILEEXTENSION   =  '.log'

contains

!--------------------------------------------------------------------------------------------------
!> @brief just reporting 
!--------------------------------------------------------------------------------------------------
subroutine DAMASK_interface_init
 integer, dimension(8) :: &
   dateAndTime                                                                                      ! type default integer
 call date_and_time(values = dateAndTime)
 write(6,'(/,a)') ' <<<+-  DAMASK_abaqus_exp  -+>>>'
 write(6,'(/,a)') ' Roters et al., Computational Materials Science, 2018'
 write(6,'(/,a)')              ' Version: '//DAMASKVERSION
 write(6,'(a,2(i2.2,a),i4.4)') ' Date:    ',dateAndTime(3),'/',&
                                            dateAndTime(2),'/',&
                                            dateAndTime(1) 
 write(6,'(a,2(i2.2,a),i2.2)') ' Time:    ',dateAndTime(5),':',&
                                            dateAndTime(6),':',&
                                            dateAndTime(7)  
 write(6,'(/,a)') ' <<<+-  DAMASK_interface init  -+>>>'
#include "compilation_info.f90"  

end subroutine DAMASK_interface_init



!--------------------------------------------------------------------------------------------------
!> @brief using Abaqus/Explicit function to get working directory name
!--------------------------------------------------------------------------------------------------
character(1024) function getSolverWorkingDirectoryName()

 implicit none
 integer :: lenOutDir

 getSolverWorkingDirectoryName=''
 call vgetOutDir(getSolverWorkingDirectoryName, lenOutDir)
 getSolverWorkingDirectoryName=trim(getSolverWorkingDirectoryName)//'/'
 
end function getSolverWorkingDirectoryName


!--------------------------------------------------------------------------------------------------
!> @brief using Abaqus/Explicit function to get solver job name
!--------------------------------------------------------------------------------------------------
character(1024) function getSolverJobName()

 implicit none
 integer :: lenJobName

 getSolverJobName=''
 call vGetJobName(getSolverJobName, lenJobName)
 
end function getSolverJobName

end module DAMASK_interface

#include "commercialFEM_fileList.f90"

subroutine vumat(nBlock, nDir, nshr, nStateV, nFieldV, nProps, lAnneal, &
                 stepTime, totalTime, dt, cmName, coordMp, charLength, &
                 props, density, strainInc, relSpinInc, &
                 tempOld, stretchOld, defgradOld, fieldOld, &
                 stressOld, stateOld, enerInternOld, enerInelasOld, &
                 tempNew, stretchNew, defgradNew, fieldNew, &
                 stressNew, stateNew, enerInternNew, enerInelasNew)
 use prec, only: &
   pReal, &
   pInt
!$ use numerics, only: &
!$ DAMASK_NumThreadsInt
 use FEsolving, only: &
   symmetricSolver, &
   terminallyIll
 use math, only: &
   invnrmMandel
 use debug, only: &
   debug_info, &
   debug_reset, &
   debug_levelBasic, &
   debug_level, &
   debug_abaqus
 use mesh, only: &
   mesh_unitlength, &
   mesh_FEasCP, &
   mesh_ipCoordinates
 use CPFEM, only: &
   CPFEM_general, &
   CPFEM_init_done, &
   CPFEM_initAll, &
   CPFEM_CALCRESULTS, &
   CPFEM_AGERESULTS, &
   cycleCounter, &
   theTime, &
   outdatedByNewInc, &
   outdatedFFN1
 use homogenization, only: &
   materialpoint_sizeResults, &
   materialpoint_results

 implicit none
 integer(pInt),                                 intent(in) :: &
   nDir, &                                                                                          !< number of direct components in a symmetric tensor
   nshr, &                                                                                          !< number of indirect components in a symmetric tensor
   nStateV, &                                                                                       !< number of user-defined state variables that are associated with this material type
   nFieldV, &                                                                                       !< number of user-defined external field variables
   nprops, &                                                                                        !< user-specified number of user-defined material properties
   lAnneal                                                                                          !< indicating whether the routine is being called during an annealing process
 integer(pInt), dimension(*),                   intent(in) :: &
   nBlock                                                                                           !< 1: No of Materialpoints in this call, 2: No of Materialpoint (IP)
                                                                                                    !< 3: No of layer, 4: No of secPoint, 5+: element numbers
 character(len=80),                             intent(in) :: &
   cmname                                                                                           !< uses-specified material name, left justified
 real(pReal),   dimension(nprops),              intent(in) :: &
   props                                                                                            !< user-supplied material properties
 real(pReal),                                   intent(in) :: &
   stepTime, &                                                                                      !< value of time since the step began
   totalTime, &                                                                                     !< value of total time
   dt                                                                                               !< time increment size
 real(pReal),   dimension(nblock(1)),           intent(in) :: &
   density, &                                                                                       !< current density at material points in the midstep configuration
   charLength, &                                                                                    !< characteristic element length
   enerInternOld, &                                                                                 !< internal energy per unit mass at each material point at the beginning of the increment
   enerInelasOld, &                                                                                 !< dissipated inelastic energy per unit mass at each material point at the beginning of the increment
   tempOld, &                                                                                       !< temperature at each material point at the beginning of the increment
   tempNew                                                                                          !< temperature at each material point at the end of the increment (Temperature calculated in ABAQUS boundary conditions)
 real(pReal), dimension(nblock(1),*),           intent(in) :: &
   coordMp                                                                                          !< material point coordinates
 real(pReal), dimension(nblock(1),ndir+nshr),   intent(in) :: &
   strainInc, &                                                                                     !< strain increment tensor at each material point
   stretchOld, &                                                                                    !< stretch tensor U at each material point 
   stretchNew, &                                                                                    !< stretch tensor U at each material point 
   stressOld                                                                                        !< stress tensor at each material point
 real(pReal), dimension(nblock(1),nshr),        intent(in) ::  &
   relSpinInc                                                                                       !< incremental relative rotation vector
 real(pReal), dimension(nblock(1),nstatev),     intent(in) :: &
   stateOld                                                                                         !< state variables at each material point at the beginning of the increment
 real(pReal), dimension(nblock(1),nfieldv),     intent(in) :: &
   fieldOld, &                                                                                      !< user-defined field variables
   fieldNew                                                                                         !< user-defined field variables
 real(pReal), dimension(nblock(1),ndir+2*nshr), intent(in) :: &
   defgradOld, &
   defgradNew 
 real(pReal), dimension(nblock(1)),             intent(out) :: &
   enerInternNew, &                                                                                 !< internal energy per unit mass at each material point at the end of the increment
   enerInelasNew                                                                                    !< dissipated inelastic energy per unit mass at each material point at the end of the increment
 real(pReal), dimension(nblock(1),ndir+nshr),   intent(out) :: &
   stressNew                                                                                        !< stress tensor at each material point at the end of the increment
 real(pReal), dimension(nblock(1),nstatev),     intent(out) :: &
   stateNew                                                                                         !< state variables at each material point at the end of the increment

 real(pReal), dimension(3) :: coordinates
 real(pReal), dimension(3,3) :: defgrd0,defgrd1
 real(pReal), dimension(6) ::   stress
 real(pReal), dimension(6,6) :: ddsdde
 real(pReal) :: temp, timeInc, stresspower
 integer(pInt) :: computationMode, n, i, cp_en

#ifdef _OPENMP
 integer :: defaultNumThreadsInt                                                                    !< default value set by Abaqus
 include "omp_lib.h"

 defaultNumThreadsInt = omp_get_num_threads()                                                       ! remember number of threads set by Marc
 call omp_set_num_threads(DAMASK_NumThreadsInt)                                                     ! set number of threads for parallel execution set by DAMASK_NUM_THREADS
#endif

 computationMode = CPFEM_CALCRESULTS                                                                ! always calculate
 do n = 1,nblock(1)                                                                                 ! loop over vector of IPs
   temp    = tempOld(n)                                                                             ! temp is intent(in)
   if ( .not. CPFEM_init_done ) then
     call CPFEM_initAll(nBlock(4_pInt+n),nBlock(2))
     outdatedByNewInc = .false.

     if (iand(debug_level(debug_abaqus),debug_levelBasic) /= 0) then
       write(6,'(i8,1x,i2,1x,a)') nBlock(4_pInt+n),nBlock(2),'first call special case..!'; flush(6)
     endif
   else if (theTime < totalTime) then                                                               ! reached convergence
     outdatedByNewInc = .true.

     if (iand(debug_level(debug_abaqus),debug_levelBasic) /= 0) then
       write (6,'(i8,1x,i2,1x,a)') nBlock(4_pInt+n),nBlock(2),'lastIncConverged + outdated'; flush(6)
     endif

   endif
   outdatedFFN1 = .false.
   terminallyIll = .false.
   cycleCounter = 1_pInt
   if ( outdatedByNewInc ) then
     outdatedByNewInc = .false.
     call debug_info()                                                                              ! first after new inc reports debugging
     call debug_reset()                                                                             ! resets debugging
     computationMode = ior(computationMode, CPFEM_AGERESULTS)                                       ! age results
   endif

   theTime  = totalTime                                                                             ! record current starting time
   if (iand(debug_level(debug_abaqus),debug_levelBasic) /= 0) then
     write(6,'(a,i8,i2,a)') '(',nBlock(4_pInt+n),nBlock(2),')'; flush(6)
     write(6,'(a,l1)') 'Aging Results: ', iand(computationMode, CPFEM_AGERESULTS) /= 0_pInt
   endif
   defgrd0 = 0.0_pReal
   defgrd1 = 0.0_pReal
   timeInc = dt

  !     ABAQUS explicit:     deformation gradient as vector 11, 22, 33, 12, 23, 31, 21, 32, 13
  !     ABAQUS explicit:     deformation gradient as vector 11, 22, 33, 12, 21
  
   forall (i=1:ndir)
     defgrd0(i,i) = defgradOld(n,i)
     defgrd1(i,i) = defgradNew(n,i)
   end forall
   if (nshr == 1) then
     defgrd0(1,2) = defgradOld(n,4)
     defgrd1(1,2) = defgradNew(n,4)
     defgrd0(2,1) = defgradOld(n,5)
     defgrd1(2,1) = defgradNew(n,5)
   else
     defgrd0(1,2) = defgradOld(n,4)
     defgrd1(1,2) = defgradNew(n,4)
     defgrd0(1,3) = defgradOld(n,9)
     defgrd1(1,3) = defgradNew(n,9)
     defgrd0(2,1) = defgradOld(n,7)
     defgrd1(2,1) = defgradNew(n,7)
     defgrd0(2,3) = defgradOld(n,5)
     defgrd1(2,3) = defgradNew(n,5)
     defgrd0(3,1) = defgradOld(n,6)
     defgrd1(3,1) = defgradNew(n,6)
     defgrd0(3,2) = defgradOld(n,8)
     defgrd1(3,2) = defgradNew(n,8)

   endif
   cp_en = mesh_FEasCP('elem',nBlock(4_pInt+n))
   mesh_ipCoordinates(1:3,n,cp_en) = mesh_unitlength * coordMp(n,1:3)

   call CPFEM_general(computationMode,.false.,defgrd0,defgrd1,temp,timeInc,cp_en,nBlock(2),stress,ddsdde)
  
  !     Mandel:     11, 22, 33, SQRT(2)*12, SQRT(2)*23, SQRT(2)*13
  !     straight:   11, 22, 33, 12, 23, 13
  !     ABAQUS implicit:     11, 22, 33, 12, 13, 23
  !     ABAQUS explicit:     11, 22, 33, 12, 23, 13
  !     ABAQUS explicit:     11, 22, 33, 12

   stressNew(n,1:ndir+nshr) = stress(1:ndir+nshr)*invnrmMandel(1:ndir+nshr)
   stateNew(n,1:min(nstatev,materialpoint_sizeResults)) = &
                              materialpoint_results(1:min(nstatev,materialpoint_sizeResults),&
                              nBlock(2),mesh_FEasCP('elem', nBlock(4_pInt+n)))
  
   stresspower = 0.5_pReal*sum((stressOld(n,1:ndir)+stressNew(n,1:ndir))*straininc(n,1:ndir))+&
                 sum((stressOld(n,ndir+1:ndir+nshr)+stressNew(n,ndir+1:ndir+nshr))*straininc(n,ndir+1:ndir+nshr))
   enerInternNew(n) = enerInternOld(n) + stresspower / density(n)                                   ! Internal energy per unit mass
   enerInelasNew(n) = enerInternNew(n)                                                              ! Dissipated inelastic energy per unit mass(Temporary output)

 enddo
!$ call omp_set_num_threads(defaultNumThreadsInt)                                                  ! reset number of threads to stored default value

end subroutine vumat


!--------------------------------------------------------------------------------------------------
!> @brief calls the exit function of Abaqus/Explicit
!--------------------------------------------------------------------------------------------------
subroutine quit(mpie_error)
 use prec, only: &
   pInt
 
 implicit none
 integer(pInt) :: mpie_error
 
 flush(6)
 call xplb_exit
 
end subroutine quit
