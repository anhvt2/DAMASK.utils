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
!> @author Philip Eisenlohr, Michigan State University
!> @author Zhuowen Zhao, Michigan State University
!> @author Martin Diehl, Max-Planck-Institut für Eisenforschung GmbH
!> @brief  Phenomenological crystal plasticity using a power law formulation for the shear rates
!! and a Voce-type kinematic hardening rule
!--------------------------------------------------------------------------------------------------
module plastic_kinehardening
 use prec, only: &
   pReal, &
   pInt

 implicit none
 private
 integer(pInt),                       dimension(:,:),   allocatable, target, public :: &
   plastic_kinehardening_sizePostResult                                                             !< size of each post result output
 character(len=64),                   dimension(:,:),   allocatable, target, public :: &
   plastic_kinehardening_output                                                                     !< name of each post result output

 enum, bind(c)
   enumerator :: &
     undefined_ID, &
     crss_ID, &                                                                                     !< critical resolved stress
     crss_back_ID, &                                                                                !< critical resolved back stress
     sense_ID, &                                                                                    !< sense of acting shear stress (-1 or +1)
     chi0_ID, &                                                                                     !< backstress at last switch of stress sense (positive?)
     gamma0_ID, &                                                                                   !< accumulated shear at last switch of stress sense (at current switch?)
     accshear_ID, &
     shearrate_ID, &
     resolvedstress_ID
 end enum

 type, private :: tParameters
   real(pReal) :: &
     gdot0, &                                                                                       !< reference shear strain rate for slip
     n, &                                                                                           !< stress exponent for slip
     aTolResistance, &
     aTolShear
   real(pReal),                 allocatable, dimension(:) :: &
     crss0, &                                                                                       !< initial critical shear stress for slip
     theta0, &                                                                                      !< initial hardening rate of forward stress for each slip
     theta1, &                                                                                      !< asymptotic hardening rate of forward stress for each slip
     theta0_b, &                                                                                    !< initial hardening rate of back stress for each slip
     theta1_b, &                                                                                    !< asymptotic hardening rate of back stress for each slip
     tau1, &
     tau1_b, &
     nonSchmidCoeff
   real(pReal),                 allocatable, dimension(:,:) :: &
     interaction_slipslip                                                                           !< slip resistance from slip activity
   real(pReal),                 allocatable, dimension(:,:,:) :: &
     Schmid, &
     nonSchmid_pos, &
     nonSchmid_neg
   integer(pInt) :: &
     totalNslip, &                                                                                  !< total number of active slip system
     of_debug = 0_pInt
   integer(pInt),               allocatable, dimension(:) :: &
     Nslip                                                                                          !< number of active slip systems for each family
   integer(kind(undefined_ID)), allocatable, dimension(:) :: &
     outputID                                                                                       !< ID of each post result output
 end type tParameters

 type, private :: tKinehardeningState
   real(pReal), pointer, dimension(:,:) :: &                                                        !< vectors along NipcMyInstance
     crss, &                                                                                        !< critical resolved stress
     crss_back, &                                                                                   !< critical resolved back stress
     sense, &                                                                                       !< sense of acting shear stress (-1 or +1)
     chi0, &                                                                                        !< backstress at last switch of stress sense
     gamma0, &                                                                                      !< accumulated shear at last switch of stress sense
     accshear                                                                                       !< accumulated (absolute) shear
 end type tKinehardeningState

!--------------------------------------------------------------------------------------------------
! containers for parameters and state
 type(tParameters),         allocatable, dimension(:), private :: param
 type(tKinehardeningState), allocatable, dimension(:), private :: &
   dotState, &
   deltaState, &
   state

 public :: &
   plastic_kinehardening_init, &
   plastic_kinehardening_LpAndItsTangent, &
   plastic_kinehardening_dotState, &
   plastic_kinehardening_deltaState, &
   plastic_kinehardening_postResults, &
   plastic_kinehardening_results
 private :: &
   kinetics

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!> @details reads in material parameters, allocates arrays, and does sanity checks
!--------------------------------------------------------------------------------------------------
subroutine plastic_kinehardening_init
 use prec, only: &
   dEq0, &
   pStringLen
 use debug, only: &
#ifdef DEBUG
   debug_e, &
   debug_i, &
   debug_g, &
   debug_levelExtensive, &
#endif
   debug_level, &
   debug_constitutive,&
   debug_levelBasic
 use math, only: &
   math_expand
 use IO, only: &
   IO_error
 use material, only: &
#ifdef DEBUG
   phasememberAt, &
#endif
   phase_plasticity, &
   phase_plasticityInstance, &
   phase_Noutput, &
   material_allocatePlasticState, &
   PLASTICITY_kinehardening_label, &
   PLASTICITY_kinehardening_ID, &
   material_phase, &
   plasticState
 use config, only: &
   config_phase
 use lattice

 implicit none
 integer(pInt) :: &
   Ninstance, &
   p, i, o, &
   NipcMyPhase, &
   sizeState, sizeDeltaState, sizeDotState, &
   startIndex, endIndex

 integer(pInt),          dimension(0), parameter :: emptyIntArray    = [integer(pInt)::]
 real(pReal),            dimension(0), parameter :: emptyRealArray   = [real(pReal)::]
 character(len=65536),   dimension(0), parameter :: emptyStringArray = [character(len=65536)::]

 integer(kind(undefined_ID)) :: &
   outputID

 character(len=pStringLen) :: &
   extmsg = ''
 character(len=65536), dimension(:), allocatable :: &
   outputs

 write(6,'(/,a)')   ' <<<+-  plastic_'//PLASTICITY_KINEHARDENING_label//' init  -+>>>'

 Ninstance = count(phase_plasticity == PLASTICITY_KINEHARDENING_ID)
 if (iand(debug_level(debug_constitutive),debug_levelBasic) /= 0) &
   write(6,'(a16,1x,i5,/)') '# instances:',Ninstance

 allocate(plastic_kinehardening_sizePostResult(maxval(phase_Noutput),Ninstance),source=0_pInt)
 allocate(plastic_kinehardening_output(maxval(phase_Noutput),Ninstance))
          plastic_kinehardening_output = ''

 allocate(param(Ninstance))
 allocate(state(Ninstance))
 allocate(dotState(Ninstance))
 allocate(deltaState(Ninstance))

 do p = 1_pInt, size(phase_plasticityInstance)
   if (phase_plasticity(p) /= PLASTICITY_KINEHARDENING_ID) cycle
   associate(prm => param(phase_plasticityInstance(p)), &
             dot => dotState(phase_plasticityInstance(p)), &
             dlt => deltaState(phase_plasticityInstance(p)), &
             stt => state(phase_plasticityInstance(p)),&
             config => config_phase(p))

#ifdef DEBUG
   if  (p==material_phase(debug_g,debug_i,debug_e)) then
      prm%of_debug = phasememberAt(debug_g,debug_i,debug_e)
   endif
#endif

!--------------------------------------------------------------------------------------------------
!  optional parameters that need to be defined
   prm%aTolResistance = config%getFloat('atol_resistance',defaultVal=1.0_pReal)
   prm%aTolShear      = config%getFloat('atol_shear',     defaultVal=1.0e-6_pReal)

   ! sanity checks
   if (prm%aTolResistance <= 0.0_pReal) extmsg = trim(extmsg)//' aTolresistance'
   if (prm%aTolShear      <= 0.0_pReal) extmsg = trim(extmsg)//' aTolShear'

!--------------------------------------------------------------------------------------------------
! slip related parameters
   prm%Nslip      = config%getInts('nslip',defaultVal=emptyIntArray)
   prm%totalNslip = sum(prm%Nslip)
   slipActive: if (prm%totalNslip > 0_pInt) then
     prm%Schmid = lattice_SchmidMatrix_slip(prm%Nslip,config%getString('lattice_structure'),&
                                            config%getFloat('c/a',defaultVal=0.0_pReal))

     if(trim(config%getString('lattice_structure')) == 'bcc') then
       prm%nonSchmidCoeff = config%getFloats('nonschmid_coefficients',&
                                              defaultVal = emptyRealArray)
       prm%nonSchmid_pos  = lattice_nonSchmidMatrix(prm%Nslip,prm%nonSchmidCoeff,+1_pInt)
       prm%nonSchmid_neg  = lattice_nonSchmidMatrix(prm%Nslip,prm%nonSchmidCoeff,-1_pInt)
     else
       prm%nonSchmid_pos  = prm%Schmid
       prm%nonSchmid_neg  = prm%Schmid
     endif
     prm%interaction_SlipSlip = lattice_interaction_SlipBySlip(prm%Nslip, &
                                                               config%getFloats('interaction_slipslip'), &
                                                               config%getString('lattice_structure'))

     prm%crss0    = config%getFloats('crss0',    requiredSize=size(prm%Nslip))
     prm%tau1     = config%getFloats('tau1',     requiredSize=size(prm%Nslip))
     prm%tau1_b   = config%getFloats('tau1_b',   requiredSize=size(prm%Nslip))
     prm%theta0   = config%getFloats('theta0',   requiredSize=size(prm%Nslip))
     prm%theta1   = config%getFloats('theta1',   requiredSize=size(prm%Nslip))
     prm%theta0_b = config%getFloats('theta0_b', requiredSize=size(prm%Nslip))
     prm%theta1_b = config%getFloats('theta1_b', requiredSize=size(prm%Nslip))

     prm%gdot0  = config%getFloat('gdot0')
     prm%n = config%getFloat('n_slip')

     ! expand: family => system
     prm%crss0    = math_expand(prm%crss0,   prm%Nslip)
     prm%tau1     = math_expand(prm%tau1,    prm%Nslip)
     prm%tau1_b   = math_expand(prm%tau1_b,  prm%Nslip)
     prm%theta0   = math_expand(prm%theta0,  prm%Nslip)
     prm%theta1   = math_expand(prm%theta1,  prm%Nslip)
     prm%theta0_b = math_expand(prm%theta0_b,prm%Nslip)
     prm%theta1_b = math_expand(prm%theta1_b,prm%Nslip)



!--------------------------------------------------------------------------------------------------
!  sanity checks
     if (    prm%gdot0  <= 0.0_pReal)   extmsg = trim(extmsg)//' gdot0'
     if (    prm%n      <= 0.0_pReal)   extmsg = trim(extmsg)//' n_slip'
     if (any(prm%crss0  <= 0.0_pReal))  extmsg = trim(extmsg)//' crss0'
     if (any(prm%tau1   <= 0.0_pReal))  extmsg = trim(extmsg)//' tau1'
     if (any(prm%tau1_b <= 0.0_pReal))  extmsg = trim(extmsg)//' tau1_b'

     !ToDo: Any sensible checks for theta?

   endif slipActive

!--------------------------------------------------------------------------------------------------
!  exit if any parameter is out of range
   if (extmsg /= '') &
     call IO_error(211_pInt,ext_msg=trim(extmsg)//'('//PLASTICITY_KINEHARDENING_label//')')

!--------------------------------------------------------------------------------------------------
!  output pararameters
   outputs = config%getStrings('(output)',defaultVal=emptyStringArray)
   allocate(prm%outputID(0))
   do i=1_pInt, size(outputs)
     outputID = undefined_ID
     select case(outputs(i))

       case ('resistance')
         outputID = merge(crss_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('accumulatedshear')
         outputID = merge(accshear_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('shearrate')
         outputID = merge(shearrate_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('resolvedstress')
         outputID = merge(resolvedstress_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('backstress')
         outputID = merge(crss_back_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('sense')
         outputID = merge(sense_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('chi0')
         outputID = merge(chi0_ID,undefined_ID,prm%totalNslip>0_pInt)
       case ('gamma0')
         outputID = merge(gamma0_ID,undefined_ID,prm%totalNslip>0_pInt)

     end select

     if (outputID /= undefined_ID) then
       plastic_kinehardening_output(i,phase_plasticityInstance(p)) = outputs(i)
       plastic_kinehardening_sizePostResult(i,phase_plasticityInstance(p)) = prm%totalNslip
       prm%outputID = [prm%outputID , outputID]
     endif

   enddo

!--------------------------------------------------------------------------------------------------
! allocate state arrays
   NipcMyPhase = count(material_phase == p)
   sizeDotState   = size(['crss     ','crss_back', 'accshear ']) * prm%totalNslip
   sizeDeltaState = size(['sense ',   'chi0  ',    'gamma0'   ]) * prm%totalNslip
   sizeState = sizeDotState + sizeDeltaState

   call material_allocatePlasticState(p,NipcMyPhase,sizeState,sizeDotState,sizeDeltaState, &
                                      prm%totalNslip,0_pInt,0_pInt)
   plasticState(p)%sizePostResults = sum(plastic_kinehardening_sizePostResult(:,phase_plasticityInstance(p)))

!--------------------------------------------------------------------------------------------------
! locally defined state aliases and initialization of state0 and aTolState
   startIndex = 1_pInt
   endIndex   = prm%totalNslip
   stt%crss => plasticState(p)%state   (startIndex:endIndex,:)
   stt%crss = spread(prm%crss0, 2, NipcMyPhase)
   dot%crss => plasticState(p)%dotState(startIndex:endIndex,:)
   plasticState(p)%aTolState(startIndex:endIndex) = prm%aTolResistance

   startIndex = endIndex + 1_pInt
   endIndex   = endIndex + prm%totalNslip
   stt%crss_back => plasticState(p)%state   (startIndex:endIndex,:)
   dot%crss_back => plasticState(p)%dotState(startIndex:endIndex,:)
   plasticState(p)%aTolState(startIndex:endIndex) = prm%aTolResistance

   startIndex = endIndex + 1_pInt
   endIndex   = endIndex + prm%totalNslip
   stt%accshear => plasticState(p)%state   (startIndex:endIndex,:)
   dot%accshear => plasticState(p)%dotState(startIndex:endIndex,:)
   plasticState(p)%aTolState(startIndex:endIndex) = prm%aTolShear
   ! global alias
   plasticState(p)%slipRate        => plasticState(p)%dotState(startIndex:endIndex,:)
   plasticState(p)%accumulatedSlip => plasticState(p)%state(startIndex:endIndex,:)

   o = plasticState(p)%offsetDeltaState
   startIndex = endIndex + 1_pInt
   endIndex   = endIndex + prm%totalNslip
   stt%sense => plasticState(p)%state     (startIndex  :endIndex  ,:)
   dlt%sense => plasticState(p)%deltaState(startIndex-o:endIndex-o,:)

   startIndex = endIndex + 1_pInt
   endIndex   = endIndex +  prm%totalNslip
   stt%chi0 => plasticState(p)%state     (startIndex  :endIndex  ,:)
   dlt%chi0 => plasticState(p)%deltaState(startIndex-o:endIndex-o,:)

   startIndex = endIndex + 1_pInt
   endIndex   = endIndex +  prm%totalNslip
   stt%gamma0 => plasticState(p)%state     (startIndex  :endIndex  ,:)
   dlt%gamma0 => plasticState(p)%deltaState(startIndex-o:endIndex-o,:)

   plasticState(p)%state0 = plasticState(p)%state                                                   ! ToDo: this could be done centrally

   end associate

 enddo

end subroutine plastic_kinehardening_init


!--------------------------------------------------------------------------------------------------
!> @brief calculates plastic velocity gradient and its tangent
!--------------------------------------------------------------------------------------------------
pure subroutine plastic_kinehardening_LpAndItsTangent(Lp,dLp_dMp,Mp,instance,of)

 implicit none
 real(pReal), dimension(3,3),     intent(out) :: &
   Lp                                                                                               !< plastic velocity gradient
 real(pReal), dimension(3,3,3,3), intent(out) :: &
   dLp_dMp                                                                                          !< derivative of Lp with respect to the Mandel stress

 real(pReal), dimension(3,3), intent(in) :: &
   Mp                                                                                               !< Mandel stress
 integer(pInt),               intent(in) :: &
   instance, &
   of

 integer(pInt) :: &
   i,k,l,m,n
 real(pReal), dimension(param(instance)%totalNslip) :: &
   gdot_pos,gdot_neg, &
   dgdot_dtau_pos,dgdot_dtau_neg

 Lp = 0.0_pReal
 dLp_dMp = 0.0_pReal

 associate(prm => param(instance))

 call kinetics(Mp,instance,of,gdot_pos,gdot_neg,dgdot_dtau_pos,dgdot_dtau_neg)

 do i = 1_pInt, prm%totalNslip
   Lp = Lp + (gdot_pos(i)+gdot_neg(i))*prm%Schmid(1:3,1:3,i)
   forall (k=1_pInt:3_pInt,l=1_pInt:3_pInt,m=1_pInt:3_pInt,n=1_pInt:3_pInt) &
     dLp_dMp(k,l,m,n) = dLp_dMp(k,l,m,n) &
                      + dgdot_dtau_pos(i) * prm%Schmid(k,l,i) * prm%nonSchmid_pos(m,n,i) &
                      + dgdot_dtau_neg(i) * prm%Schmid(k,l,i) * prm%nonSchmid_neg(m,n,i)
 enddo

 end associate

end subroutine plastic_kinehardening_LpAndItsTangent


!--------------------------------------------------------------------------------------------------
!> @brief calculates the rate of change of microstructure
!--------------------------------------------------------------------------------------------------
subroutine plastic_kinehardening_dotState(Mp,instance,of)

 implicit none
 real(pReal), dimension(3,3),  intent(in) :: &
   Mp                                                                                               !< Mandel stress
 integer(pInt),                intent(in) :: &
   instance, &
   of

 integer(pInt) :: &
   i
 real(pReal) :: &
   sumGamma
 real(pReal), dimension(param(instance)%totalNslip) :: &
   gdot_pos,gdot_neg


 associate(prm => param(instance), stt => state(instance), dot => dotState(instance))

 call kinetics(Mp,instance,of,gdot_pos,gdot_neg)
 dot%accshear(:,of) = abs(gdot_pos+gdot_neg)
 sumGamma = sum(stt%accshear(:,of))

 do i = 1_pInt, prm%totalNslip
   dot%crss(i,of) = dot_product(prm%interaction_SlipSlip(:,i),dot%accshear(:,of)) &
                  * (  prm%theta1(i) &
                     + (prm%theta0(i) - prm%theta1(i) + prm%theta0(i)*prm%theta1(i)*sumGamma/prm%tau1(i)) &
                     * exp(-sumGamma*prm%theta0(i)/prm%tau1(i)) &
                    )
 enddo
 dot%crss_back(:,of) = stt%sense(:,of)*dot%accshear(:,of) * &
          ( prm%theta1_b + &
            (prm%theta0_b - prm%theta1_b &
              + prm%theta0_b*prm%theta1_b/(prm%tau1_b+stt%chi0(:,of))*(stt%accshear(:,of)-stt%gamma0(:,of))&
            ) *exp(-(stt%accshear(:,of)-stt%gamma0(:,of)) *prm%theta0_b/(prm%tau1_b+stt%chi0(:,of))) &
          )

 end associate

end subroutine plastic_kinehardening_dotState


!--------------------------------------------------------------------------------------------------
!> @brief calculates (instantaneous) incremental change of microstructure
!--------------------------------------------------------------------------------------------------
subroutine plastic_kinehardening_deltaState(Mp,instance,of)
 use prec, only: &
   dNeq, &
   dEq0
#ifdef DEBUG
 use debug, only: &
   debug_level, &
   debug_constitutive,&
   debug_levelExtensive, &
   debug_levelSelective
#endif

 implicit none
 real(pReal), dimension(3,3),  intent(in) :: &
   Mp                                                                                               !< Mandel stress
 integer(pInt),                intent(in) :: &
   instance, &
   of

 real(pReal), dimension(param(instance)%totalNslip) :: &
   gdot_pos,gdot_neg, &
   sense

 associate(prm => param(instance), stt => state(instance), dlt => deltaState(instance))

 call kinetics(Mp,instance,of,gdot_pos,gdot_neg)
 sense = merge(state(instance)%sense(:,of), &                                                       ! keep existing...
               sign(1.0_pReal,gdot_pos+gdot_neg), &                                                 ! ...or have a defined
               dEq0(gdot_pos+gdot_neg,1e-10_pReal))                                                 ! current sense of shear direction

#ifdef DEBUG
 if (iand(debug_level(debug_constitutive), debug_levelExtensive) /= 0_pInt &
            .and. (of == prm%of_debug &
                   .or. .not. iand(debug_level(debug_constitutive),debug_levelSelective) /= 0_pInt)) then
   write(6,'(a)') '======= kinehardening delta state ======='
   write(6,*) sense,state(instance)%sense(:,of)
 endif
#endif

!--------------------------------------------------------------------------------------------------
! switch in sense of shear?
 where(dNeq(sense,stt%sense(:,of),0.1_pReal))
   dlt%sense (:,of) = sense - stt%sense(:,of)                                                       ! switch sense
   dlt%chi0  (:,of) = abs(stt%crss_back(:,of)) - stt%chi0(:,of)                                     ! remember current backstress magnitude
   dlt%gamma0(:,of) = stt%accshear(:,of) - stt%gamma0(:,of)                                         ! remember current accumulated shear
 else where
   dlt%sense (:,of) = 0.0_pReal
   dlt%chi0  (:,of) = 0.0_pReal
   dlt%gamma0(:,of) = 0.0_pReal
 end where

 end associate

end subroutine plastic_kinehardening_deltaState


!--------------------------------------------------------------------------------------------------
!> @brief return array of constitutive results
!--------------------------------------------------------------------------------------------------
function plastic_kinehardening_postResults(Mp,instance,of) result(postResults)
 use math, only: &
   math_mul33xx33

 implicit none
 real(pReal), dimension(3,3), intent(in) :: &
   Mp                                                                                               !< Mandel stress
 integer(pInt),               intent(in) :: &
   instance, &
   of

 real(pReal), dimension(sum(plastic_kinehardening_sizePostResult(:,instance))) :: &
   postResults

 integer(pInt) :: &
   o,c,i
 real(pReal), dimension(param(instance)%totalNslip) :: &
   gdot_pos,gdot_neg

 c = 0_pInt

 associate(prm => param(instance), stt => state(instance))

 outputsLoop: do o = 1_pInt,size(prm%outputID)
   select case(prm%outputID(o))

     case (crss_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%crss(:,of)
     case(crss_back_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%crss_back(:,of)
     case (sense_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%sense(:,of)
     case (chi0_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%chi0(:,of)
     case (gamma0_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%gamma0(:,of)
     case (accshear_ID)
       postResults(c+1_pInt:c+prm%totalNslip) = stt%accshear(:,of)
     case (shearrate_ID)
       call kinetics(Mp,instance,of,gdot_pos,gdot_neg)
       postResults(c+1_pInt:c+prm%totalNslip) = gdot_pos+gdot_neg
     case (resolvedstress_ID)
       do i = 1_pInt, prm%totalNslip
         postResults(c+i) = math_mul33xx33(Mp,prm%Schmid(1:3,1:3,i))
       enddo

   end select

   c = c + prm%totalNslip

 enddo outputsLoop

 end associate

end function plastic_kinehardening_postResults


!--------------------------------------------------------------------------------------------------
!> @brief writes results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine plastic_kinehardening_results(instance,group)
#if defined(PETSc) || defined(DAMASKHDF5)
  use results

  implicit none
  integer, intent(in) :: instance
  character(len=*) :: group
  integer :: o

  associate(prm => param(instance), stt => state(instance))
  outputsLoop: do o = 1_pInt,size(prm%outputID)
    select case(prm%outputID(o))
    end select
  enddo outputsLoop
  end associate
#else
  integer, intent(in) :: instance
  character(len=*) :: group
#endif

end subroutine plastic_kinehardening_results


!--------------------------------------------------------------------------------------------------
!> @brief calculates shear rates on slip systems and derivatives with respect to resolved stress
!> @details: Shear rates are calculated only optionally.
! NOTE: Against the common convention, the result (i.e. intent(out)) variables are the last to
! have the optional arguments at the end
!--------------------------------------------------------------------------------------------------
pure subroutine kinetics(Mp,instance,of, &
                         gdot_pos,gdot_neg,dgdot_dtau_pos,dgdot_dtau_neg)
 use prec, only: &
   dNeq0
 use math, only: &
   math_mul33xx33

 implicit none
 real(pReal), dimension(3,3),  intent(in) :: &
   Mp                                                                                               !< Mandel stress
 integer(pInt),                intent(in) :: &
   instance, &
   of

 real(pReal),                  intent(out), dimension(param(instance)%totalNslip) :: &
   gdot_pos, &
   gdot_neg
 real(pReal),                  intent(out), optional, dimension(param(instance)%totalNslip) :: &
   dgdot_dtau_pos, &
   dgdot_dtau_neg

 real(pReal), dimension(param(instance)%totalNslip) :: &
   tau_pos, &
   tau_neg
 integer(pInt) :: i
 logical       :: nonSchmidActive

 associate(prm => param(instance), stt => state(instance))

 nonSchmidActive = size(prm%nonSchmidCoeff) > 0_pInt

 do i = 1_pInt, prm%totalNslip
   tau_pos(i) =       math_mul33xx33(Mp,prm%nonSchmid_pos(1:3,1:3,i)) - stt%crss_back(i,of)
   tau_neg(i) = merge(math_mul33xx33(Mp,prm%nonSchmid_neg(1:3,1:3,i)) - stt%crss_back(i,of), &
                      0.0_pReal, nonSchmidActive)
 enddo

 where(dNeq0(tau_pos))
   gdot_pos = prm%gdot0 * merge(0.5_pReal,1.0_pReal, nonSchmidActive) &                             ! 1/2 if non-Schmid active
            * sign(abs(tau_pos/stt%crss(:,of))**prm%n,  tau_pos)
 else where
   gdot_pos = 0.0_pReal
 end where

 where(dNeq0(tau_neg))
   gdot_neg = prm%gdot0 * 0.5_pReal &                                                               ! only used if non-Schmid active, always 1/2
            * sign(abs(tau_neg/stt%crss(:,of))**prm%n,  tau_neg)
 else where
   gdot_neg = 0.0_pReal
 end where

 if (present(dgdot_dtau_pos)) then
   where(dNeq0(gdot_pos))
     dgdot_dtau_pos = gdot_pos*prm%n/tau_pos
   else where
     dgdot_dtau_pos = 0.0_pReal
   end where
 endif
 if (present(dgdot_dtau_neg)) then
   where(dNeq0(gdot_neg))
     dgdot_dtau_neg = gdot_neg*prm%n/tau_neg
   else where
     dgdot_dtau_neg = 0.0_pReal
   end where
 endif
 end associate

end subroutine kinetics

end module plastic_kinehardening
