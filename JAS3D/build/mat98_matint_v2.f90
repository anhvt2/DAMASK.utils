! $Id: matint.f90,v 5.0 2007/12/19 16:54:10 mlblanf Exp $
!
SUBROUTINE MATINT
!
   USE contrl_
   USE frefld_
   USE funcs_
   USE params_
   USE iolib_
   USE rdata_
   USE xtvars_
   USE konmat_memory_
   USE alvars_
   USE smat61_props_
!
!***********************************************************************
!
!     DESCRIPTION:
!       This routine is the material interface for the code. This is
!       a dummy subroutine name which is never called. This routine is
!       accessed through entry points from RPASS1, RPASS2, and TELALL.
!
!       It allows constitutive modelers to add a new material model
!       very easily.  FOURTEEN steps are outlined below that must be
!       taken to add a new material model.
!
!***********************************************************************
!
!***********************************************************************
!       *                                                         *
!       * STEP 1: NMDEF is the number of materials defined.       *
!       *         Increment this by one for your new material.    *
!       *         The number you enter here will be the value     *
!       *         of MKIND by which JAS3D internally              *
!       *         identifies your constitutive model.             *
!       *                                                         *
!***********************************************************************
!
      PARAMETER (NMDEF=1)
!
!***********************************************************************
!       *                                                         *
!       * STEP 2: MCUES is the maximum number of cues allowed     *
!       *         ( i.e. the maximum number of material           *
!       *         constants to be read) for any material.         *
!       *         Increase this number if your will need more.    *
!       *         Do not mess with it if you need less.           *
!       *                                                         *
!***********************************************************************
!
      PARAMETER (MCUES=365)
!
!***********************************************************************
!       *                                                         *
!       * STEP 3: MAXSV is the maximum number of internal state   *
!       *         variables for all the material models defined   *
!       *         Increase this number if yours will need more.   *
!       *         Do not mess with it if you need less.           *
!       *         May require increasing MXCOMP in evlist_.f90    *
!       *                                                         *
!***********************************************************************
!
      PARAMETER (MAXSV=99)
!
!***********************************************************************
!       *                                                         *
!       * STEP 3a: MAXALSV is the maximum number of augmented     *
!       *          Lagrange state variables for all the material  *
!       *          models defined. Increase this number if yours  *
!       *          will need more.                                *
!       *          Do not mess with it if you need less.          *
!       *          May require increasing MXCOMP in evlist_.f90   *
!       *                                                         *
!***********************************************************************
!
      PARAMETER (MAXALSV=1)
!
!***********************************************************************
!       *                                                         *
!       * STEP 3a: MAXFV is the maximum number of control         *
!       *          failure state variables for all the material   *
!       *          models defined. Increase this number if yours  *
!       *          will need more.                                *
!       *          Do not mess with it if you need less.          *
!       *          May require increasing MXCOMP in evlist_.f90   *
!       *                                                         *
!***********************************************************************
!
      PARAMETER (MAXFV=1)
!
      INCLUDE 'precision.blk'
      INCLUDE 'numbers.blk'
!
      CHARACTER*(MAXSTG) SVNAME(MAXSV,NMDEF),ALSVNAME(MAXALSV,NMDEF),LIST(*)
      CHARACTER*(MAXSTG) CUES(MCUES,NMDEF),NAMES(NMDEF),FVNAME(MAXFV,NMDEF)
      CHARACTER*(*) CMND(MFIELD)
      DIMENSION PROP(*),NSTRNF(NMDEF)
      DIMENSION NUMSV(NMDEF),NUMALSV(NMDEF),NCUES(NMDEF),NCONS(NMDEF),NUMFV(NMDEF)

! Conversion from ATM to dyn/cm2
      PARAMETER( SCNVRS=1013250*PONE )
!
!***********************************************************************
!       *                                                         *
!       * STEP 4: NUMSV is the array which defines the number     *
!       *         of internal state variables for the material.   *
!       *         Enter the appropriate number for your material  *
!       *                                                         *
!***********************************************************************
!
! MTYPE           1
      DATA NUMSV/ 99 /
!
!***********************************************************************
!       *                                                         *
!       * STEP 4a: NUMALSV is the array which defines the number  *
!       *          of augmented lagrange state variables for the  *
!       *          material. Enter the appropriate number for     *
!       *          your material                                  *
!       *                                                         *
!***********************************************************************
!
! MTYPE             1
      DATA NUMALSV/ 0 /
!
!***********************************************************************
!       *                                                         *
!       * STEP 4b: NUMFV is the array which defines the number of *
!       *          failure state variables for the material.      *
!       *          Enter the appropriate number for your material *
!       *          model.                                         *
!       *                                                         *
!***********************************************************************
!
! MTYPE           1
      DATA NUMFV/ 0 /
!
!***********************************************************************
!       *                                                         *
!       * STEP 5: NCUES is the array that defines the number of   *
!       *         material constants that will be read in for     *
!       *         each material. Enter the appropriate value in   *
!       *         the data statement for your new material.       *
!       *                                                         *
!***********************************************************************
!
! MTYPE           1
      DATA NCUES/ 27 /
!
!***********************************************************************
!       *                                                         *
!       * STEP 6: NCONS is the array that defines the number of   *
!       *         material constants that are needed in each      *
!       *         constitutive routine.  Note that NCONS must     *
!       *         be greater than or equal to NCUES.  If NCONS    *
!       *         exceeds NCUES, this implies that you are        *
!       *         going to define extra constants in STEP 12      *
!       *         in addition to supplying lambda plus two mu     *
!       *                                                         *
!***********************************************************************
!
! MTYPE           1
      DATA NCONS/ 27 /
!
!***********************************************************************
!       *                                                         *
!       * STEP 7: CUES is the CHARACTER array that contains       *
!       *         the names of each material property that will   *
!       *         be read for each material type. Note that the   *
!       *         word length must be less than 32 characters     *
!       *         and the range of the first index must not       *
!       *         exceed MCUES defined above in STEP 2.  Enter    *
!       *         a DATA statement with the names for your        *
!       *         material constants to be read.  Blanks are      *
!       *         allowed in the cue names.                       *
!       *                                                         *
!       *     IMPORTANT: We allow the user to abbreviate the cues *
!       *         to three characters; so each word in your cue   *
!       *         name must be unique to three characters.  This  *
!       *         means that C1, C2, C3, etc. are three valid cue *
!       *         names but CON1, CON2, CON3, etc. are not!       *
!       *                                                         *
!***********************************************************************
!
! Remi Dingreville 's Local Deformation Gradient Crystal Plasticity
      DATA (CUES(I,1),I=1,27)/'C11','C12','C44','FLAG SYS'           ,  &
        'FLAG SLIP','FLAG HARDEN','FLOW RATE','1FLOW PARAM'          ,  &
        '2FLOW PARAM','ACTIVE RATE','TAUCR','CRSS0','1HARDEN PARAM'  ,  &
        '2HARDEN PARAM','3HARDEN PARAM','4HARDEN PARAM','5HARDEN PARAM',&
        '6HARDEN PARAM','7HARDEN PARAM','8HARDEN PARAM'              ,  &
        '9HARDEN PARAM','10HARDEN PARAM','C_1','C_2','C_3','C_4','C_5'/
!
!***********************************************************************
!       *                                                         *
!       * STEP 8: NAMES is the array that contains the names of   *
!       *         the material that the user inputs. Enter the    *
!       *         name of your new material in the DATA state-    *
!       *         ment. Note that the name cannot exceed 32       *
!       *         characters. should try to pick a name that      *
!       *         is representitive of your material.  Blanks     *
!       *         are allowed.                                    *
!       *                                                         *
!       *   IMPORTANT: Your material names must be unique to the  *
!       *         the first three characters in each word because *
!       *         we allow the user to abbreviate to three letters*
!       *                                                         *
!***********************************************************************
!                 -=[No more than 32 characters]=-
!
      DATA (NAMES(I),I=1,1) / 'LCP' /
!
!***********************************************************************
!       *                                                         *
!       * STEP 9: SVNAME is the character data which defines      *
!       *         the names of your internal state variables.     *
!       *         Enter appropriate names for your internal       *
!       *         state variables.  Please use the same spelling  *
!       *         for conventional quantities (ie EQPS for        *
!       *         equivalent plastic strain or ALPHA11 for the    *
!       *         11 component of back stress)  NOTE: only eight  *
!       *         characters are allowed and no blanks are        *
!       *         allowed.                                        *
!       *                                                         *
!***********************************************************************
!
! Remi Dingreville's Local Deformation Gradient Crystal Plasticity
      DATA (SVNAME(I,1),I=1,99)/ &
      'FE11','FE12','FE13','FE21','FE22','FE23','FE31','FE32','FE33' ,  &
      'FP11','FP12','FP13','FP21','FP22','FP23','FP31','FP32','FP33' ,  &
      'FT11_PREV','FT12_PREV','FT13_PREV','FT21_PREV','FT22_PREV'    ,  &
      'FT23_PREV','FT31_PREV','FT32_PREV','FT33_PREV'                ,  &
      'RE11','RE12','RE13','RE21','RE22','RE23','RE31','RE32','RE33' ,  &
      'Rinit11','Rinit12','Rinit13','Rinit21','Rinit22','Rinit23'    ,  &
      'Rinit31','Rinit32','Rinit33'                                  ,  &
      'RT11','RT12','RT13','RT21','RT22','RT23','RT31','RT32','RT33' ,  &
      'MISORIENT','EQPS','NSUB','SLIP_AVG','DD_AVG','CRSS_AVG'       ,  &
      'ORIENT','NUM_ACTIVE_SLIP','HP'                                ,  &
      'Slip1','Slip2','Slip3','Slip4','Slip5','Slip6','Slip7','Slip8',  &
      'Slip9','Slip10','Slip11','Slip12'                             ,  &
      'CRSS1','CRSS2','CRSS3','CRSS4','CRSS5','CRSS6','CRSS7','CRSS8',  &
      'CRSS9','CRSS10','CRSS11','CRSS12'                             ,  &
      'DD1','DD2','DD3','DD4','DD5','DD6','DD7','DD8','DD9'          ,  &
      'DD10','DD11','DD12'/
!
!***********************************************************************
!       *                                                         *
!       * STEP 9a: ALSVNAME is the character data which defines   *
!       *          the names of your augmented Lagrange state     *
!       *          variables. Enter appropriate names for your    *
!       *          augmented Lagrange state variables.  Please    *
!       *          use the same spelling for identical quantities *
!       *          (ie BULKSTRN for bulk strain or SXX_INCR for   *
!       *          xx-deviatoric stress increment).  NOTE: only   *
!       *          eight characters are allowed and no blanks are *
!       *          allowed.                                       *
!       *                                                         *
!***********************************************************************
!
! Crystal Plasticity 
!     (no augmented Lagrange state variables for this material)
!
!***********************************************************************
!       *                                                         *
!       * STEP 9b: FVNAME is the character data which defines     *
!       *          the names of your material failure state       *
!       *          variables. Enter appropriate names for these   *
!       *          failure state variables.  Please use the same  *
!       *          spelling for identical quantities (ie BULKSTRN *
!       *          for bulk strain or SXX_INCR for xx-deviatoric  *
!       *          stress increment).  NOTE: only eight characters*
!       *          are allowed and no blanks are allowed          *
!       *                                                         *
!***********************************************************************
!
! Crystal Plasticty 
!     (no CONTROL FAILURE variables for this material)
!
!***********************************************************************
!       *                                                         *
!       * STEP 10: NSTRNF is the array which flags whether the    *
!       *          constitutive model requires the total strains  *
!       *          for the material point.  NOTE: these strains   *
!       *          will be supplied to the constitutive routine   *
!       *          in the unrotated configuration. Set the        *
!       *          appropriate value for your model in the data   *
!       *          statement.                                     *
!       *             0 = does not require strains                *
!       *             1 = does require strains                    *
!       *                                                         *
!***********************************************************************
!
! MTYPE            1
      DATA NSTRNF/ 0 /
!
      ENTRY MATRP1( MKIND,NINSV,KSFLG,KDFIBER,CMND,NALSV,NFAILV )
!
!***********************************************************************
!
!     DESCRIPTION:
!       This entry point is called during the first pass through the
!       input file in RPASS1. In this pass we determine if the user has
!       specified a valid material name. If so, we save some integer
!       parameters.
!
!     FORMAL PARAMETERS:
!       MKIND    INTEGER    Material type number
!       NINSV    INTEGER    No. of internal state variables
!                           for this material kind
!       KSFLG    INTEGER    Strain allocation flag
!       KDFIBER  INTEGER    Flag for variable density in blocks with SMAT15
!                           and SMAT61
!       CMND     CHARACTER  Character data from free field reader
!       NALSV    INTEGER    No. of augmented Lagrange variables.
!       NFAIL    INTEGER    No. of control failure variables.
!
!     CALLED BY: RPASS1
!
!***********************************************************************
!
! Check command for a valid material model name.
!
      KN = 3
      KF = MFIELD - KN
      CALL QCHECK( MKIND,NMDEF,NWORD,KF,NAMES,CMND(KN),KVALUE(KN) )
      IF( MKIND .EQ. 0 ) THEN
         WRITE(KOUT,1050)
         CALL KFATAL('1.matint')
         IERR = IERR + 1
         NCUE = 0
      ELSE
! Define material parameters -
         MCONS = MAX( MCONS , NCONS(MKIND) )
         IF( NSTRNF(MKIND) .EQ. 1 ) KSFLG = 1
         NINSV  = NUMSV(MKIND)
         NCUE   = NCUES(MKIND)
         NALSV  = NUMALSV(MKIND)
         NFAILV = NUMFV(MKIND)
      END IF
!
! Check for valid material cues until and END indicator is found.
!
      KCUE = 0
      KSUM = 0
  100 CONTINUE
! Read a new record -
      CALL PFREFLD( KIN,KOUTFILE,'AUTO',CMND,TO_UPPER )
      IF( IOSTAT .LT. 0 ) THEN
         NFIELD  = 1
         CMND(1) = 'EXIT'
      ELSE IF( IOSTAT .GT. 0 ) THEN
         NFIELD  = 1
         CMND(1) = 'EXIT'
         WRITE(KOUT,1060) IOSTAT
         CALL KFATAL('2.matint')
         IERR = IERR + 1
      END IF
! Ignore a comment line -
      IF( NFIELD .EQ. 0 ) GO TO 100
      KN    = 0
      NWORD = 0
  150 CONTINUE
! Read another card if there are no more fields -
      KN = KN + NWORD + 1
      IF( KN .GT. NFIELD ) GO TO 100
! Search for another cue -
      KF = MFIELD - KN
      CALL QCHECK( MATCH,NCUE,NWORD,KF,CUES(1,MKIND),CMND(KN), &
                   KVALUE(KN) )
! Check for an end indicator
      IF( CMND(KN).EQ.'END' .OR. CMND(KN).EQ.'EXIT' ) THEN
         NSUM = ( NCUE * (NCUE + 1) ) / 2
         IF( KCUE .NE. NCUE .OR. KSUM .NE. NSUM ) THEN
         END IF
      ELSE IF( MATCH .EQ. 0 ) THEN
         WRITE(KOUT,1020) (CMND(I)(1:4), I= KN, KN+NWORD-1)
         CALL KFATAL('4.matint')
         IERR = IERR + 1
         GO TO 150
      ELSE
! Increment cue match counter and accumulate cue match sum check -
         KCUE = KCUE + 1
         KSUM = KSUM + MATCH
         GO TO 150
      END IF
!
      RETURN
!
      ENTRY MATRP2( MKIND,ITHFLG,TMPSCL,RHODFLT,RHO0,DATMOD,SHRMOD,PROP,CMND )
!
!***********************************************************************
!
!     DESCRIPTION:
!       This entry point is called during the second pass through the
!       input file in RPASS2. In this pass we actually store the data
!       the user has specified in the space which has been allocated.
!
!     FORMAL PARAMETERS:
!       MKIND    INTEGER    Material type number
!       ITHFLG   INTEGER    Thermal strain function flag
!       TMPSCL   REAL       Thermal strain function scale value
!       RHODFLT  REAL       Density default value
!       RHO0     REAL       Density
!       DATMOD   REAL       Lamda plus two mu
!       SHRMOD   REAL       Two mu
!       PROP     REAL       Array containing the material
!                           properties for this material
!       CMND     CHARACTER  Character data from free field reader
!
!     CALLED BY: RPASS2
!
!***********************************************************************
!
! Strip off material model name -
      KN = 3
      KF = MFIELD - KN
      CALL QCHECK( M,NMDEF,NWORD,KF,NAMES,CMND(KN),KVALUE(KN) )
!
! Save the density - if left zero, take the default
      IF( KVALUE(KN+NWORD) .GT. 0 )THEN
        RHO0 = RVALUE(KN+NWORD)
      ELSE IF( PROP(MCONS) .EQ. PZERO )THEN
        RHO0 = RHODFLT
      END IF
!
! Save the thermal strain function number and scale factor
      IF( KVALUE(KN+NWORD+1) .GT. 0 )THEN
        ITHFLG = IVALUE(KN+NWORD+1)
        IF( KVALUE(KN+NWORD+2) .GT. 0 )THEN
          TMPSCL = RVALUE(KN+NWORD+2)
        ELSE
          TMPSCL = PONE
        END IF
      END IF
!
! Read material data and store in PROP array.
!
      NCUE = NCUES(MKIND)
  200 CONTINUE
! Read a new record -
      CALL PFREFLD( KIN,0,' ',CMND,TO_UPPER )
      IF( IOSTAT .LT. 0 ) THEN
         NFIELD  = 1
         CMND(1) = 'EXIT'
      ELSE IF( IOSTAT .GT. 0 ) THEN
         NFIELD  = 1
         CMND(1) = 'EXIT'
         WRITE(KOUT,1060) IOSTAT
         CALL KFATAL('5.matint')
         IERR = IERR + 1
      END IF
! Ignore a comment line -
      IF( NFIELD .EQ. 0 ) GO TO 200
      KN    = 0
      NWORD = 0
  250 CONTINUE
! Read another card if there are no more fields -
      KN = KN + NWORD + 1
      IF( KN .GT. NFIELD ) GO TO 200
! Search for another cue -
      KF = MFIELD - KN
      CALL QCHECK( MATCH,NCUE,NWORD,KF,CUES(1,MKIND),CMND(KN), &
                   KVALUE(KN) )
! Check for an end indicator
      IF( CMND(KN).NE.'END' .AND. CMND(KN).NE.'EXIT' ) THEN
! Store material data value -
         IF( MATCH .GT. 0 ) PROP(MATCH) = RVALUE(KN+NWORD)
         GO TO 250
      END IF
!
!***********************************************************************
!       *                                                         *
!       * STEP 11: You must supply lambda plus two mu for your    *
!       *         material so the program can make an initial     *
!       *         time step estimate.  Store this value in the    *
!       *         variable, DATMOD. Also you must store the       *
!       *         quantity twice the shear modulus in SHRMOD.     *
!       *                                                         *
!       *         You may also define any other material con-     *
!       *         stants that you may want in your constitutive   *
!       *         routine.  Store them in the remaining space     *
!       *         in the PROP array after the properties that     *
!       *         were read in via your cues. The PROP array      *
!       *         positions you define should be above NCUES      *
!       *         (STEP 5) and below or equal to NCONS (STEP 6).  *
!       *         You may wait to define extra material con-      *
!       *         stants in STEP 12 but you must provide DATMOD   *
!       *         and SHRMOD here.                                *
!       *         PLEASE, PLEASE, PLEASE use COMMENT cards        *
!       *                                                         *
!***********************************************************************
!
! Crystal Plasticity
      IF( MKIND.EQ.1 )THEN
!  Two mu:
        TWOMU = PROP(1) - PROP(2)
!  Lambda:
        XLAM = PROP(2)
!  Lambda plus two mu:
        DATMOD = XLAM + TWOMU
        SHRMOD = TWOMU
!
      END IF
      RETURN
!
      ENTRY MATTEL( MKIND,MATID,RHO0,ITHX,ITHY,ITHZ,TSCX,TSCY,TSCZ,PROP, &
                    SHRMOD,DATMOD )
!
!
!***********************************************************************
!
!     DESCRIPTION:
!       This entry point is called when we are printing the problem
!       definition in TELALL.  Here we print the information that was
!       stored in the call to the entry points above.
!
!     FORMAL PARAMETERS:
!       MKIND    INTEGER    Material type number
!       MATID    INTEGER    Material ID
!       RHO0     REAL       Density
!       ITHX     INTEGER    Thermal strain function flag, x direction
!       ITHY     INTEGER    Thermal strain function flag, y direction
!       ITHZ     INTEGER    Thermal strain function flag, z direction
!       TSCX     REAL       Thermal strain function x scale value
!       TSCY     REAL       Thermal strain function y scale value
!       TSCZ     REAL       Thermal strain function z scale value
!       PROP     REAL       Array containing the material
!                           properties for this material
!       KFDAT    INTEGER    Functions integer data
!       FUNCS    REAL       Functions data array
!
!     CALLED BY: TELALL
!
!***********************************************************************
!
! Print material header and check density.
      IF( NTHERM .EQ. 0 )THEN
        WRITE(KOUT,1028) NAMES(MKIND),MATID,RHO0
        CALL KPRINT
      ELSE
        WRITE(KOUT,1029) NAMES(MKIND),MATID,RHO0,ITHX,ITHY,ITHZ, &
          TSCX,TSCY,TSCZ
        CALL KPRINT
        NUM = 0
        CALL CHLIST( ITHX,KFDAT,NUM,3,NFUNC )
        IF( NUM.NE.0 ) THEN
          ITHX = NUM
        ELSE
          IERR = IERR + 1
          WRITE(KOUT,1140)
          CALL KFATAL('9a.matint')
        END IF
        NUM = 0
        CALL CHLIST( ITHY,KFDAT,NUM,3,NFUNC )
        IF( NUM.NE.0 ) THEN
          ITHY = NUM
        ELSE
          IERR = IERR + 1
          WRITE(KOUT,1140)
          CALL KFATAL('9b.matint')
        END IF
        NUM = 0
        CALL CHLIST( ITHZ,KFDAT,NUM,3,NFUNC )
        IF( NUM.NE.0 ) THEN
          ITHZ = NUM
        ELSE
          IERR = IERR + 1
          WRITE(KOUT,1140)
          CALL KFATAL('9c.matint')
        END IF
      END IF
      IF( RHO0 .LE. PZERO ) THEN
         WRITE(KOUT,3020)
         CALL KFATAL('10.matint')
         IERR = IERR + 1
      END IF
!
! Print cues and input data.
      NCUE = NCUES(MKIND)
      WRITE(KOUT,1030) CUES(1,MKIND),PROP(1)
      CALL KPRINT
      DO I=2,NCUE
        WRITE(KOUT,1031) CUES(I,MKIND),PROP(I)
        CALL KPRINT
      ENDDO
!
!***********************************************************************
!       *                                                         *
!       * STEP 12: You may check your material data here and      *
!       *        print any data, diagnostics, warnings or         *
!       *        information you desire.  Note that you can de-   *
!       *        fine extra material properties values here       *
!       *        as in STEP 11.  Also, a format using an indent   *
!       *        of 14X will produce a nice output.               *
!       *                                                         *
!***********************************************************************
!
!  N/A
!
!***********************************************************************
!       *                                                         *
!       * STEP 13: Modify subroutine UPDSTR to call your material *
!       *          model subroutine.                              *
!       *                                                         *
!***********************************************************************
!
!  N/A
!
!**********************************************************************
!  STEP 14: For material models using an augmented Lagrange wrapper
!  and associated augmented Lagrange variables in conjunction with
!  the multi-level solver --
!
!  1) Initialize the IALER index, the ALERR array and the AL control
!     parameter pointer IALCP in VRESID, SRESID, VRESIDFI, VRESIDSD
!  2) Extract the element ID with the maximum material violation error
!     in VRESID, SRESID, VRESIDFI, VRESIDSD
!  3) Add EPMAGnn, NIDMAXnn, MXERRIDnn and NMATnn to maxerr_.f90
!  4) Extract maximum errors and element ID in errmatc.f90
!     Report maximum errors and element ID in rprterr.f90
!     Initialize NMATxx flag in xresid.f90
!  5) Check on AL variables with external-predictor values in
!     prdstat.f90, and use the values in getlag.f90
!  6) Add prediction coding to updlag.f90 for AL variables
!  7) Add AL energy coding to constrcal.f90
!  8) Repeat all of this for *fi.f90, *sd.f90 and ex_*.f90
!  8) Confirm compatibily with sresid.f90 and friends.
!  9) For thermal dependencies, add coding to evalth.f90 and create
!     a routine similar to thup47.f90 for AL variables. Add coding
!     to svinit.f90 so material model works in the non-thermal case.
! 10) In the module time_2g_scale_, add subroutine T2GS_SETnn and
!     its call in the CASE statement in subroutine T2GS_SET.
!     for your material model
!
!                             **NOTE**
!  The THERMO MOONEY RIVLIN material required changes in all of the
!  following routines. If your material model has features that are
!  comparable to material 47, it would be a good idea to visit every
!  one of these routines and insure that there is coding to support
!  your new material model:
!
!  Imakefile, xjas3d.mes, evalth.f90, findxt.f90, ex_stable.f90,
!  constrcal.f90, diag.f90, getlag.f90, stable.f90, stabso.f90,
!  stress_stiff.f90, updlag.f90, updstr.f90, vresid.f90, getlagfi.f90,
!  stablefi.f90, updlagfi.f90, vresidfi.f90, updstrfi.f90, diagsd.f90,
!  vresidsd.f90, matint.f90, svinit.f90, errmatc.f90, maxerr_.f90,
!  prdstat.f90, rprterr.f90, xresid.f90

!***********************************************************************
!
 1010 FORMAT(10X,'***** ERROR: INCONSISTENT NUMBER OF MATERIAL ', &
        'CONSTANTS DEFINED' )
 1015 FORMAT(10X,'***** ERROR: NONPOSITIVE MODULI DETECTED FOR ', &
        'MOONEY-RIVLIN MODEL' )
 1016 FORMAT(10X,'***** ERROR: NONPOSITIVE MODULI DETECTED FOR ', &
        'ALW MOONEY-RIVLIN MODEL' )
 1017 FORMAT(10X,'***** ERROR: NONPOSITIVE MODULI DETECTED FOR ', &
        'THERMO MOONEY-RIVLIN MODEL' )
 1018 FORMAT(10X,'***** ERROR: IMPROPER CONSTANTS DETECTED FOR ', &
        'ALW SWANSON RUBBER MODEL')
 1019 FORMAT(10X,'***** ERROR: IMPROPER CONSTANTS DETECTED FOR ', &
        'ALW SWANSON RUBBER MODEL')
 1020 FORMAT(10X,'***** ERROR: UNKNOWN MATERIAL CUE:',5(1X,A) )
 1028 FORMAT(//,14X,'MATERIAL TYPE ........................',A,/, &
                14X,'MATERIAL ID ..........................',I11,/, &
                14X,'DENSITY ..............................',1PE12.4)
 1029 FORMAT(//,14X,'MATERIAL TYPE ........................',A,/, &
                14X,'MATERIAL ID ..........................',I11,/, &
                14X,'DENSITY ..............................',1PE12.4,/ &
                6X,'THERMAL STRAIN FUNCTION IDs .......',3I11,/, &
                6X,'THERMAL STRAIN SCALE FACTORS ......',3(1PE11.3))
 1030 FORMAT(14X,'MATERIAL PROPERTIES:',/,(21X,A25,1PE14.6))
 1031 FORMAT((21X,A25,1PE14.6))
 1050 FORMAT(10X,'***** ERROR: UNKNOWN MATERIAL NAME *****')
 1060 FORMAT(10X,'***** ERROR: UNABLE TO READ INPUT IN FREE FIELD', &
        ' READER, FORTRAN ERROR NUMBER = ',I5,' *****')
 1140 FORMAT(10X,'***** ERROR: FUNCTION ID ON ABOVE LINE NOT FOUND', &
        ' IN FUNCTION DEFINITIONS *****')
 3040 FORMAT( 10X,'***** ERROR: ', &
           'ILLEGAL POISSONS RATIO - ASSUMED TO BE ZERO.' )
 3180 FORMAT( 10X,'***** ERROR: BAD VALUE FOR DAMAGE CONSTANT - ', &
       ' USE .5 IF YOU WANT NO DAMAGE CALCULATION')
 3020 FORMAT( 10X,'***** ERROR: DENSITY MUST BE POSITIVE *****' )
!
      RETURN
!
      ENTRY SVINFO( IBLK,NLIST,LIST )
!
!***********************************************************************
!
!     DESCRIPTION: This entry point gets the number of internal state
!                  variables for the block and their names
!
!     FORMAL PARAMETERS:
!       IBLK    INTEGER     Block number
!       NLIST   INTEGER     Number of internal state variables
!       LIST    CHARACTER   Internal state variable names for this block
!
!     CALLED BY: EVSIZE
!
!***********************************************************************
!
      MK = KONMAT(2,IBLK)
      NLIST = 0
      IF( MK.LT.1 .OR. MK.GT.NMDEF ) RETURN
      DO N = 1,NUMSV(MK)
        NLIST = NLIST + 1
        LIST(NLIST) = SVNAME(N,MK)
      END DO

      RETURN
!
      ENTRY ALSVINFO( IBLK,NLIST,LIST )
!
!***********************************************************************
!
!     DESCRIPTION: This entry point gets the number of augmented Lagrange
!                  state variables for the block and their names
!
!     FORMAL PARAMETERS:
!       IBLK    INTEGER     Block number
!       NLIST   INTEGER     Number of augmented Lagrange state variables
!       LIST    CHARACTER   Augmented Lagrange state variable names for
!                           this block
!
!     CALLED BY: EVSIZE
!
!***********************************************************************
!
      MK = KONMAT(2,IBLK)
      NLIST = 0
      IF( MK.LT.1 .OR. MK.GT.NMDEF ) RETURN
      DO N = 1,NUMALSV(MK)
        NLIST = NLIST + 1
        LIST(NLIST) = ALSVNAME(N,MK)
      END DO
!
      RETURN
!
      ENTRY CFSVINFO( IBLK,NLIST,LIST )
!
!***********************************************************************
!
!     DESCRIPTION: This entry point gets the number of control failure
!                  state variables for the block and their names
!
!     FORMAL PARAMETERS:
!       IBLK    INTEGER     Block number
!       NLIST   INTEGER     Number of control failure state variables
!       LIST    CHARACTER   Control failure state variable names for
!                           this block
!
!     CALLED BY: EVSIZE
!
!***********************************************************************
!
      MK = KONMAT(2,IBLK)
      NLIST = 0
      IF( MK.LT.1 .OR. MK.GT.NMDEF ) RETURN
      DO N = 1,NUMFV(MK)
        NLIST = NLIST + 1
        LIST(NLIST) = FVNAME(N,MK)
      END DO
!
END SUBROUTINE MATINT
