! $Id: init.f90,v 5.0.2.2 2009/03/12 13:40:53 mlblanf Exp $
!
SUBROUTINE INIT( LINK,COORD,                           &
        CUR,DISPL,VEL,CYCVEL,                          &
        DATMAT,KINT,ROTV,                              &
        KDISPL,KPNBC,                                  &
        PNBC,IMTCON,ISSCON,INSCON,                     &
        ISMCON,NTSCAL,TSCALE,                          &
        XTNPRV,XTNNXT,XTNOLD,XTNCUR,                   &
        XTGPRV,XTGNXT,XTGOLD,XTGCUR,                   &
        KPRDBC,PRDBC,                                  &
        DELETE,VELC,IIVAL,RIVAL,CIVAL,                 &
        ACCL,MAPNOD,MAPEL,                             &
        VOLNOD,XTSTEPS,ISEVXT,LXTN,LXTE,               &
        LXTIN,LXTIE,SCFXTN,SCFXTE,                     &
        NFLOC,NSFAC,                                   &
        NMAPRB,REACT,RREACT,                           &
        CONDAT,QVEC,QVECD,QVECDD,ROTA,RMASS,           &
        CFORCE )
!
!***********************************************************************
!
!     DESCRIPTION:
!       This routine performs the first part of the initialization of
!       internal data required for the QSOLVE/GSOLVE routines.
!
!     FORMAL PARAMETERS:
!       LINK     INTEGER    Connectivity array
!       COORD    REAL       Global nodal coordinates array
!       CUR      REAL       Global nodal position array
!       DISPL    REAL       Nodal point displacements
!       VEL      REAL       Nodal point velocities
!       CYCVEL   REAL       Saved velocities for cyclic predictor
!       DATMAT   REAL       Material properties array
!       KINT     INTEGER    Number and type of integration points
!       ROTV     REAL       Nodal rotational velocity for shell nodes
!       KDISPL   INTEGER    No displacement integer data structure
!       KPNBC    INTEGER    Prescribed nodal BC integer data array
!       PNBC     REAL       Prescribed nodal BC real data array
!       IMTCON   INTEGER    List of contact material IDs
!       ISSCON   INTEGER    List of contact side set IDs
!       INSCON   INTEGER    List of contact node set IDs
!       ISMCON   INTEGER    List of side sets to removed from contacts
!       NTSCAL   INTEGER    Thickness scale factor associated block ID
!       TSCALE   REAL       Thickness scale factor
!       XTNPRV   REAL       Nodal external field array, previous record
!       XTNNXT   REAL       Nodal external field array, next record
!       XTNOLD   REAL       Nodal external field array, last time step
!       XTNCUR   REAL       Nodal external field array, current time step
!       XTGPRV   REAL       Global external variables array, previous record
!       XTGNXT   REAL       Global external variables array, next record
!       XTGOLD   REAL       Global external variables array, last time step
!       XTGCUR   REAL       Global external variables array, current time step
!       KPRDBC   INTEGER    Periodic BC integer data array
!       PRDBC    REAL       Periodic BC real data array
!       DELETE   REAL       Element material block time
!                             insertion/deletion array
!       VELC     REAL       Nodal point velocities up to level 1 iterate
!       IIVAL    INTEGER    Initial value integer data
!       RIVAL    REAL       Initial value real data
!       CIVAL    CHARACTER  Initial value charater data
!       ACCL     REAL       Nodal acceleration array
!       MAPNOD   INTEGER    Maps local node number to global node number
!       MAPEL    INTEGER    Maps local element number to global
!       VOLNOD   REAL       Nodal Volumes
!       XTSTEPS  REAL       Array of time steps on the mesh file
!       ISEVXT   INTEGER    Truth table for element variables on mesh file
!       LXTN     INTEGER    Which nodal field to read from
!       LXTE     INTEGER    Which element field to read from
!       LXTIN    INTEGER    Which nodal variable array to interpolate from
!       LXTIE    INTEGER    Which element variable array to interpolate from
!       SCFXTN   REAL       Scale factor for nodal field function
!       SCFXTE   REAL       Scale factor for element field function
!       NFLOC    INTEGER    Pointer to first node in the NODES array
!                             having this flag
!       NSFAC    INTEGER    List of sides in all side sets
!       NMAPRB   INTEGER    Rigid body constraint material for each node
!       CFORCE   REAL       Contact force, explicit dynamics
!
!     CALLED BY: JASSUB
!
!***********************************************************************
!
      USE precision_
      USE params_
      USE alloc_
      USE bcsets_
      USE bsize_
      USE exodus_
      USE comdat_
      USE contrl_
      USE elem_conn_
      USE element_type_
      USE funcs_
      USE gcontac_
      USE globals_
      USE iolib_
      USE layers_
      USE mshpcon_
      USE orient_
      USE psize_
      USE rdata_
      USE rigid_
      USE timer_
      USE xtvars_
      USE shmap_
      USE constraint_memory_
      USE ev_memory_
      USE konmat_memory_
      USE namebl_memory_
      USE nodes_int_
      USE contact_interface_
      USE ics_
      USE contact_output_
      USE time_2g_scale_ ! time dependent 2G scaling
      USE mlevlin_
      USE diagnostics_
      USE solver_actions_
      USE legacycontact_
      USE smat61_props_
      USE sst_, ONLY: NSSTDEF

      INCLUDE 'precision.blk'
      INCLUDE 'numbers.blk'
!
      DIMENSION LINK(LEN_LINK),                                         &
        COORD(NNOD,NSPC),CUR(NNOD,NSPC),DISPL(NNOD,NSPC),               &
        VEL(NNOD,NSPC),CYCVEL(NNOD*NSPC*MAXCYC),                        &
        DATMAT(MCONS,NEMBLK),                                           &
        KINT(2,*),ROTV(NSNOD*NSPC),                                     &
        KDISPL(*),KPNBC(*),PNBC(*),                                     &
        IMTCON(*),ISSCON(*),ISMCON(*),                                  &
        INSCON(*),NTSCAL(NEMBLK,2),TSCALE(NEMBLK,2),                    &
        XTNPRV(NXTN,NNOD),XTNNXT(NXTN,NNOD),XTNOLD(NXTIN,NNOD),         &
        XTNCUR(NXTIN,NNOD),XTGPRV(NXTGNAM),XTGNXT(NXTGNAM),             &
        XTGOLD(NXTGNAM),XTGCUR(NXTGNAM),KPRDBC(12,NPRDBC),PRDBC(9,NPRDBC), &
        DELETE(2,*),                                                    &
        VELC(NNOD,NSPC),IIVAL(3,NIVAL),RIVAL(NIVAL),                    &
        ACCL(NNOD,NSPC),                                                &
        MAPNOD(NNOD),MAPEL(NUMEL),VOLNOD(NNOD),                         &
        XTSTEPS(NXTSTEPS),ISEVXT(NXTENAM,NEMBLK),                       &
        LXTN(NXTN),LXTE(NXTE),LXTIN(NXTIN),LXTIE(NXTIE),                &
        SCFXTN(NXTIN),SCFXTE(NXTIE),                                    &
        NFLOC(NBCNOD),NSFAC(NSLIST),                                    &
        NMAPRB(NNOD),                                                   &
        REACT(NNOD,NSPC),RREACT(NSNOD*NSPC),                            &
        CONDAT(NSIZCD,0:MCONDT,0:MCONDT),                               &
        IELHEX(NELERR,2), IELSHE(NELERR,2),                             &
        QVEC(NSNOD,4),QVECD(NSNOD,4),QVECDD(NSNOD,4),ROTA(NSNOD*NSPC),  &
        RMASS(NSNOD),CFORCE(NNOD,NSPC)
!
      CHARACTER*(MAXSTG) NAME,CIVAL(NIVAL)
      CHARACTER*80 CESTRG
      LOGICAL MPERR
      INTEGER :: LC_STATUS
      REAL( RTYPE), DIMENSION( 9, NPRDBC) :: LC_PRDBC
      INTEGER :: bCUR_SUBROUTINE_CALL_DEPTH

      bCUR_SUBROUTINE_CALL_DEPTH = CURRENT_SUBROUTINE_CALL_DEPTH
      CALL_TRACE = 'INIT: begin'
      IF( SUBROUTINE_TRACE_WAS_REQUESTED .AND. &
          CURRENT_SUBROUTINE_CALL_DEPTH.le.REQUESTED_SUBROUTINE_CALL_DEPTH ) &
        CALL PRINT_CALL_TRACE

      IF( SOLVER_ACTIONS_WERE_REQUESTED )&
        CALL DECLARE_SOLVER_ACTION(1,sa_Initialize_MLSolver)

      IF( LCON_USELCLIB )THEN
        IF( NPRDBC > 0 ) LC_PRDBC( 1:9, 1:NPRDBC) = PRDBC( 1:9, 1:NPRDBC)
      END IF
!
      IERR   = 0
      JERR   = 0
! Initialize variables in COMMON block GLOBALS
      RMX    = PZERO
      RMAG   = PZERO
      ALMAG  = PZERO
      RCTMAG = PZERO
      NISTEP = 0
      NTSTEP = 0
      NSTEPS = 0
      RX     = PZERO
      RY     = PZERO
      RZ     = PZERO
!
! Build shell nodal communication lists
      CALL SETUP_SWAPSHELL
!
! Read initial precalculated data records from mesh file, if needed
      IF( NXTN+NXTE+NXTG.NE.0 )THEN
        IF( NXTSTEPS.EQ.1 )THEN
! Only one time step on the mesh file; read it
          MSTP = 1
          IREADP = 0
          TIME2 = XTSTEPS(1)
        ELSE
! Read the first two time steps from the mesh file
          MSTP = 2
          IREADP = 1
          TIME1 = XTSTEPS(1)
          TIME2 = XTSTEPS(2)
        END IF
        CALL RSZMSH('SET_TO_REFERENCE')
        CALL MSHRD( XTNPRV,XTNNXT,XTGPRV,XTGNXT,IREADP,ISEVXT, &
		    LXTN,LXTE )
        IF( NXTSTEPS.EQ.1 )THEN
! Only one time step on the mesh file; use it as a constant value by
! duplicating it into the previous field array
          TIME1 = TIME2 - PONE
          DO 20 I = 1,NXTN
            DO 10 J = 1,NNOD_R
              XTNPRV(I,J) = XTNNXT(I,J)
   10       CONTINUE
   20     CONTINUE
          IF( NXTG.GT.0 )THEN
            DO 30 I = 1,NXTGNAM
              XTGPRV(I) = XTGNXT(I)
   30       CONTINUE
          END IF
        END IF
      END IF
! Initialize interpolated GLOBAL variables
      IF( NXTG.GT.0 )THEN
        DO 40 I = 1,NXTGNAM
          XTGCUR(I) = XTGPRV(I)
          XTGOLD(I) = XTGCUR(I)
   40   CONTINUE
      END IF
! Initialize interpolated NODAL variables
      DO 100 IXTIN = 1,NXTIN
        ISRC = LXTIN(IXTIN)/1000000
        IXTN = MOD( LXTIN(IXTIN),1000000 )
        IF( ISRC.EQ.0 )THEN
! External nodal field
          DO 50 INOD = 1,NNOD_R
            XTNCUR(IXTIN,INOD) = XTNPRV(IXTN,INOD)
            XTNOLD(IXTIN,INOD) = XTNCUR(IXTIN,INOD)
   50     CONTINUE
        ELSE IF( ISRC.EQ.1 )THEN
! Currently no support for nodal field obtained from element values
        ELSE IF( ISRC.EQ.2 )THEN
! Get values from internal function
          CALL GETVAL( IXTN,TIME,VALUE,IFLG )
          TVAL = VALUE*SCFXTN(IXTIN)
          DO 60 INOD = 1,NNOD_R
            XTNCUR(IXTIN,INOD) = TVAL
            XTNOLD(IXTIN,INOD) = TVAL
   60     CONTINUE
        ELSE IF( ISRC.EQ.3 )THEN
! Get values from user-supplied subroutine
!          CALL USRXTN( NNOD_R,TIME,DT,IXTN,XTNCUR,IXTIN )
!          DO 80 INOD = 1,NNOD_R
!            XTNOLD(IXTIN,INOD) = XTNCUR(IXTIN,INOD)
!   80     CONTINUE
        END IF
  100 CONTINUE
!
      IF( SOLVER_ACTIONS_WERE_REQUESTED )&
        CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_Elements)

! Loop over element blocks
      IOFF = 0
      DO 150 I=1,NEMBLK
        NAME = NAMEBL(I)
        MKIND  = KONMAT(2,I)
        IF( NAME.EQ.'NULL' )THEN
          IF( MKIND.EQ.15 .OR. MKIND.EQ.61 )THEN
! Clear the SHRMOD and DATMOD fields on processors that have no elements
! using the inextensible fiber model
            DATMAT(MCONS-2,I) = 0.0
            DATMAT(MCONS-1,I) = 0.0
          END IF
          GO TO 140
        END IF
        MATID  = KONMAT(1,I)
        NSTART = KONMAT(3,I)
        NEND   = KONMAT(4,I)
        NEL    = KONMAT(5,I)
        NINSV  = KONMAT(6,I)
        ITHX   = KONMAT(8,I)
        NELNOD = KONMAT(15,I)
        NPROST = KONMAT(16,I)
        NALSV  = KONMAT(18,I)
        NFAILV = KONMAT(19,I)
        DENSE  = DATMAT(MCONS,I)
        DM0    = DATMAT(MCONS-1,I)
        TWOMU  = DATMAT(MCONS-2,I)
        TSCX   = DATMAT(MCONS-3,I)
        NTINT  = KINT(1,I)
!
! Find pointers to element variables for this material block
!   Stress
        CALL EVFIND( I,IOFF,'USIG'  ,IUSIG  )
        CALL EVFIND( I,IOFF,'SIG'   ,ISIG   )
        CALL EVFIND( I,IOFF,'ALSIG' ,IALSIG )
!   Hourglass resistance
        CALL EVFIND( I,IOFF,'HGR'   ,IHGR   )
!   Hourglass energy density
        CALL EVFIND( I,IOFF,'EHGENG',IHENG  )
!   Strain
        CALL EVFIND( I,IOFF,'UEPS'  ,IUEPS  )
        CALL EVFIND( I,IOFF,'EPS'   ,IEPS   )
!   Stretch
        CALL EVFIND( I,IOFF,'STRECH',ISTR   )
!   Total strain
        CALL EVFIND( I,IOFF,'STRTOT',ISTO   )
!   Unrotated total strain
        CALL EVFIND( I,IOFF,'USTRTOT',ISTOU )
!   Rotation
        CALL EVFIND( I,IOFF,'ROTATE',IROT   )
!   Density 
        CALL EVFIND( I,IOFF,'RHO'   ,IRHO   )
!   Artificial bulk viscosities
        CALL EVFIND( I,IOFF,'VISPR' ,IVIS   )
!   Strain energy density
        CALL EVFIND( I,IOFF,'EINENG',IIENG  )
!   Shear modulus
        CALL EVFIND( I,IOFF,'SHRMOD',ISHRMO )
!   Effective modulus
        CALL EVFIND( I,IOFF,'EFFMOD',IEFFMO )
!   Element thickness
        CALL EVFIND( I,IOFF,'THICK' ,ITHI   )
!   Element thickness for layered shells
        CALL EVFIND( I,IOFF,'TOTALT',ITOT   )
!   Element offset of shell mid-surface
        CALL EVFIND( I,IOFF,'OFFSET',IOFS   )
!   Element basis vectors
        CALL EVFIND( I,IOFF,'BASEL' ,IBAS   )
!   Strain rate
        CALL EVFIND( I,IOFF,'DOPT'  ,IDOP   )
!   Status
        CALL EVFIND( I,IOFF,'STATUS',ISTAT  )
!   Secant modulus
        CALL EVFIND( I,IOFF,'CM'    ,ICM    )
!   Thermal strain rate
        CALL EVFIND( I,IOFF,'DTHS'  ,IDTHS  )
!   Total isotropic thermal strain for hyperelasticity
        CALL EVFIND( I,IOFF,'RJTH'  ,IDJTH )
!   Stress-free isotropic thermal strain for hyperelasticity
        CALL EVFIND( I,IOFF,'SFJTH',IDSFJTH )
!   Internal state variables
        CALL EVFIND( I,IOFF,'SV'    ,IPISV  )
!   Augmented Lagrange state variables
        CALL EVFIND( I,IOFF,'ALSV'  ,IALSV  )
!   CONTROL FAILURE state variables
        CALL EVFIND( I,IOFF,'CFSV'  ,ICFSV  )
!   External element variables from mesh file
        CALL EVFIND( I,IOFF,'XTEPRV',IPRV   )
        CALL EVFIND( I,IOFF,'XTENXT',INXT   )
!   Interpolated element variables
        CALL EVFIND( I,IOFF,'XTEOLD',IOLD   )
        CALL EVFIND( I,IOFF,'XTECUR',ICUR   )
!   Volume
        CALL EVFIND( I,IOFF,'VOLUME',IVOL   )
! Link array for this element block
        CALL LNKFIND( I,IOFF,ILNK )
!
        IF( NXTE.NE.0 .AND. NXTSTEPS.EQ.1 )THEN
! Only one time step on the mesh file; use it as a constant value by
! duplicating it into the previous field array
          KPRV = IPRV
          KNXT = INXT
          DO 110 J = 1,NEL*NXTE
            EV(KPRV) = EV(KNXT)
            KPRV = KPRV + 1
            KNXT = KNXT + 1
  110     CONTINUE
        END IF
!
! Initialize the element variables
! For hexadra
        IF( NAME(1:3) .EQ. 'HEX' )THEN
          IF( INDEX(NAME,'CONSTRAINT') .NE. 0 ) GO TO 150
          IF( INDEX(NAME,'RIGID') .NE. 0 )THEN
            IF( KSTAT.NE.0 )THEN
              ASTAT = PONE
              IF( TIME.LT.DELETE(2,I) .OR. TIME.GE.DELETE(1,I) .OR.  &
                  NPROST.NE.0 ) ASTAT = PZERO
              DO J = 1,NEL
                EV(ISTAT+J-1) = ASTAT
              END DO
            END IF
            GO TO 150
          END IF
          IF( INDEX(NAME,'VOID') .NE. 0 )THEN
!              print*,'INIT: setting volume to 0, IVOL =',IVOL
              DO J = 1,NEL
                EV(IVOL+J-1) = 0
              END DO
            GO TO 150
          END IF
          IF( NAME(4:6).EQ.'_FI' .OR. NAME(4:6).EQ.'_SD' )THEN
            CALL EVINITFI( NEL,NTINT,EV(IUSIG),EV(IUEPS),EV(ISTR),      &
              EV(IROT),EV(IRHO),EV(ISTAT),DENSE,TWOMU,EV(ISHRMO),       &
              DM0,EV(IEFFMO),EV(ICM),LINK(ILNK),XTNPRV,EV(IPRV),        &
              EV(IOLD),EV(ICUR),LXTIE,SCFXTE,EV(IIENG),EV(IDTHS),       &
              EV(IDJTH),EV(IDSFJTH),DELETE(1,I),DELETE(2,I),NPROST,     &
              ITHX,TSCX,MKIND,MATID,NINSV,EV(IPISV),DATMAT(1,I),COORD,  &
              DISPL,NSTART )
          ELSE                                                         
          CALL EVINIT( NEL,EV(IUSIG),EV(IHGR),EV(IHENG),EV(IUEPS),EV(ISTR),&
            EV(ISTO),EV(ISTOU),EV(IROT),EV(IRHO),EV(ISTAT),DENSE,TWOMU, &
            EV(ISHRMO),DM0,EV(IEFFMO),EV(ICM),LINK(ILNK),XTNPRV,        &
            EV(IPRV),EV(IOLD),EV(ICUR),LXTIE,SCFXTE,EV(IIENG),EV(IDTHS),&
            EV(IDJTH),EV(IDSFJTH),DELETE(1,I),DELETE(2,I),NPROST,       &
            EV(IVIS),ITHX,TSCX,MKIND,MATID,NINSV,NTINT,EV(IPISV),       &
            DATMAT(1,I),COORD,DISPL,NSTART  )
          END IF
!
! Reset inital value of hex element variable components from
! inital value commands to a constant value, to a function of
! location or with a user supplied subroutine
          CALL SETIVAL( I,IIVAL,RIVAL,CIVAL,COORD )
!
! For shell elements
        ELSE IF( NAME(1:5).EQ.'SHELL' )THEN
!
! For SMAT15, if added-stiffness has been specified in terms of 
! orthotropic moduli (KASFLG=2), check to insure that the shell 
! thickness has been deliberately specified.
          IF( MKIND.EQ.15 )THEN
            IF( NINT(DATMAT(21,I)).EQ.2 )THEN
              IFOUND = 0
! Check to see if a thickness value other than 1.0, has been supplied 
! by the mesh data input file.
              DO J = 0,NEL-1 
                IF( ABS(EV(ITHI+J)-PONE).GT.TOL1M6 )IFOUND = 1
              END DO
              IF( IFOUND.EQ.0 )THEN
! Check to see if this element block has a shell-scale-thickness input
! record. If it does, the shell thickness has been dilberately specified.
! (TSCALE(*,2) and NTSCAL(*,2) contain the user's original input data.)
                DO J = 1,NEMBLK
                  IF( NTSCAL(J,2).EQ.MATID )IFOUND = 1
                END DO
              END IF
! If this processor has an error, write message to KOUT
              IF( IFOUND.EQ.0 )THEN
                WRITE(KOUT,1050) MATID
                CALL KFATAL('3.init')
                IERR = IERR + 1
              END IF
            END IF
          END IF
!
          CALL SEVINI( NTINT,NEL,COORD,EV(ISIG),EV(IHGR),EV(IHENG),EV(IEPS),&
     &      EV(IRHO),EV(IDOP),EV(ITHI),EV(ITOT),EV(IOFS),TSCALE(I,1),   &
     &      EV(ISTAT),DENSE,EV(IALSIG),                                 &
     &      EV(IBAS),LINK(ILNK),XTNPRV,EV(IPRV),                        &
     &      EV(IOLD),EV(ICUR),LXTIE,SCFXTE,EV(IIENG),ITHX,TSCX,EV(IDTHS), &
     &      EV(IDJTH),DELETE(1,I),DELETE(2,I),NPROST,MATID,NAME,DISPL,NSTART )
!
! Reset inital value of shell element variable components from inital
! value commands to a constant value, to a function of location or
! with a user supplied subroutine
          CALL SETIVAL( I,IIVAL,RIVAL,CIVAL,COORD )
!
        END IF
        KONMAT(16,I) = NPROST
!
! Set initial PROGRAMMED STATUS if relevant
        IF( (NPROST.NE.0) .AND. (DELETE(2,I).LE.TIME .AND.              &
     &                           TIME.LT.DELETE(1,I)) )                 &
     &    CALL PRSTAT( NEL,NNOD,NSPC,NELNOD,NINSV,NSTART,NEND,          &
     &                 LINK(ILNK),I,DELETE(2,I),   &
     &                 DELETE(1,I),TIME,DT,EV(ISTAT),EV(IPISV),         &
     &                 EV(ICUR),CUR,DATMAT(1,I) )
!
! Initialize internal state variables
        IF( NINSV.NE.0 .OR. NALSV.NE.0 .OR. NFAILV.NE.0 )THEN
          KORID = KONMAT(13,I)
          CALL SVINIT( NEL,NELNOD,MKIND,NINSV,KORID,NTINT,NALSV,        &
     &      EV(IPISV),EV(IALSV),EV(IUSIG),DATMAT(1,I),COORD,LINK(ILNK), &
     &      EV(IBAS),EV(ICUR),EV(ITHI),EV(IRHO),MAPEL,I, &
     &      NFAILV,EV(ICFSV), &
! **********************************************************************
!  BEGIN  extra code nonlocal extension.
! **********************************************************************
     &      NSTART,NEND )
! **********************************************************************
!  END  extra code nonlocal extension.
! **********************************************************************
!
! Reset inital value of internal state variables from inital
! value commands to a constant value, to a function of
! location or with a user supplied subroutine
          CALL SETIVAL( I,IIVAL,RIVAL,CIVAL,COORD )
!
        END IF
!
  140   CONTINUE
!
  150 CONTINUE

! estalish if there are any active multilevel controls
      IF( SOLVER_ACTIONS_WERE_REQUESTED .AND. ACTIVE_MLControls )THEN
        CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_MLControls)
      ELSE IF(SOLVER_ACTIONS_WERE_REQUESTED )THEN
        CALL DECLARE_SOLVER_ACTION(2,-sa_Initialize_MLControls)
      END IF

      IF( ILCONL>0 )THEN
        IF( SOLVER_ACTIONS_WERE_REQUESTED )THEN
          CALL DECLARE_SOLVER_ACTION(3,sa_Initialize_Control_Stiffness)
          CALL DECLARE_SOLVER_ACTION(4,sa_Set_Time_Dependent_CS_Moduli)
        END IF
! MWH do we really need this here ?
! This call to T2GS_SET is also in NEWSTEP
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Initialize 2G scaling, DATMOD, and SHRMOD
  ! for material blocks that use time dependent Target E
        CALL T2GS_SET( DATMAT,KINT )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ELSE IF( SOLVER_ACTIONS_WERE_REQUESTED .AND. ACTIVE_MLControls )THEN
        CALL DECLARE_SOLVER_ACTION(3,-sa_Initialize_Control_Stiffness)
      END IF

!
! Find principal values and coordinates for mass moments of inertia 
!   at shell nodes (principal coordinates for the mass moments
!   will be used as nodal coordinates)
      IF( NSNOD .NE. 0 ) CALL LAYERS
!
! Initialize coordinate array for element-block reference nodes
! These reference points may or may not be declared as Rigid Bodies
      DO 212 J = 1,NSPC
        DO 210 I = NNOD-NNOD_RB+1,NNOD
          COORD(I,J) = PZERO
  210   CONTINUE
  212 CONTINUE
!
! Initialize rigid body data
      IF( NRBMB.GT.0 )THEN
        IF( SOLVER_ACTIONS_WERE_REQUESTED )&
          CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_Rigid_Bodies)
        CALL RBINIT( COORD,NMAPRB,KINT )
      ELSE IF( SOLVER_ACTIONS_WERE_REQUESTED )THEN
        CALL DECLARE_SOLVER_ACTION(2,-sa_Initialize_Rigid_Bodies)
      END IF
!
! Initialize all displacements, velocities, and accelerations to zero
      IF( SOLVER_ACTIONS_WERE_REQUESTED )&
        CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_Solver_Vars)

      DO 222 J = 1,NSPC
        DO 220 I = 1,NNOD
          CUR(I,J)   = COORD(I,J)
          DISPL(I,J) = PZERO
          VEL(I,J)   = PZERO
          VELC(I,J)  = PZERO
          ACCL(I,J)  = PZERO
  220   CONTINUE
  222 CONTINUE
      DO 223 I = 1,NNOD*NSPC*MAXCYC
        CYCVEL(I) = PZERO
  223 CONTINUE
!
! Initialize rotational DOFs and normals for shell nodes.
      DO 225 I=1,NSNOD*NSPC
        ROTV(I) = PZERO
        ROTA(I) = PZERO
  225 CONTINUE
!
! Initialize quaternion variables for shells assuming homogeneous
! mass moment of inertia tensor
      DO I = 1,NSNOD
        QVEC(I,1) = PONE
        QVEC(I,2) = PZERO
        QVEC(I,3) = PZERO
        QVEC(I,4) = PZERO
!
        QVECD(I,1) = PONE
        QVECD(I,2) = PZERO
        QVECD(I,3) = PZERO
        QVECD(I,4) = PZERO
!
        QVECDD(I,1) = PZERO
        QVECDD(I,2) = PZERO
        QVECDD(I,3) = PZERO
        QVECDD(I,4) = PZERO
      END DO
!
! Initialize inertia arrays for quasistatics to principal values
! Set RBROT to coincide with principal inertia arrays
      IF( KDYNAM .EQ. 0 )THEN
        DO I = 1,NRBMB
          RBINERTIA(1,I) = PONE
          RBINERTIA(2,I) = PONE
          RBINERTIA(3,I) = PONE
          RBINERTIA(4,I) = PZERO
          RBINERTIA(5,I) = PZERO
          RBINERTIA(6,I) = PZERO
!
          RBROT(I,1,1) = PONE
          RBROT(I,1,2) = PZERO
          RBROT(I,1,3) = PZERO
          RBROT(I,2,1) = PZERO
          RBROT(I,2,2) = PONE
          RBROT(I,2,3) = PZERO
          RBROT(I,3,1) = PZERO
          RBROT(I,3,2) = PZERO
          RBROT(I,3,3) = PONE
!
          NDS = NDSHIN(IRBSTR(I,3))
          RBQVEC(I,1) = QVEC(NDS,1)
          RBQVEC(I,2) = QVEC(NDS,2)
          RBQVEC(I,3) = QVEC(NDS,3)
          RBQVEC(I,4) = QVEC(NDS,4)
!
          RBQVECD(I,1) = QVEC(NDS,1)
          RBQVECD(I,2) = QVEC(NDS,2)
          RBQVECD(I,3) = QVEC(NDS,3)
          RBQVECD(I,4) = QVEC(NDS,4)
        END DO
!
        DO I = 1,NSNOD
          RMASS(I) = PONE
        END DO
      END IF
!
! Initialize nodal reactions to zero if requested
      IF( KREAC.NE.0 )THEN
        CALL ZERFILLR( REACT,NNOD*NSPC )
        CALL ZERFILLR( RREACT,NSNOD*NSPC )
      END IF
!
! Check for non-positive element volume
      IF( JCHECK.EQ.1 )THEN
        IVOLNOD = 0
        IPRTERR = 1
        CALL VOLCHK( IVOLNOD,IPRTERR,VOLNOD,CUR,      &
          DELETE,MAPEL,IELHEX,IELSHE,IVERRH,IVERRS )
        IERR = IERR + IVERRH + IVERRS
      END IF
! Initialize element volumes if requested
      IVOLNOD = 2
      IPRTERR = 1
      CALL VOLCHK2( IVOLNOD,IPRTERR,VOLNOD,COORD,     &
        DELETE,MAPEL,IELHEX,IELSHE,IVERRH,IVERRS )
      IERR = IERR + IVERRH + IVERRS

! Initialize and check kinematic constraints
      IF( SOLVER_ACTIONS_WERE_REQUESTED )&
        CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_BCs)
      CALL BCINIT( IERR,NODISP,NPNBC,KDISPL,KPNBC,PNBC,COORD,       &
     &  KNTBC,BCVEC,NPRDBC,KPRDBC,PRDBC,MAPNOD,NMAPRB )
!
! Enforce Initial Velocity Material
      IF( SOLVER_ACTIONS_WERE_REQUESTED .AND. NICS>0 )THEN
        CALL DECLARE_SOLVER_ACTION(2,sa_Set_ICs)
      ELSE IF( SOLVER_ACTIONS_WERE_REQUESTED )THEN
        CALL DECLARE_SOLVER_ACTION(2,-sa_Set_ICs)
      END IF
      DO IC=1,NICS
        IC_SPEC = KICS(IC,1)
        IC_TYPE = KICS(IC,2)
        IF( IC_TYPE == 0 )THEN
          CYCLE
        ELSE IF( IC_TYPE == 1 )THEN
! ...initial condition on a node set
          IC_NSID = KICS(IC,3)
        ELSE IF( IC_TYPE == 2 )THEN
! ...initial condition on an element block
          IC_BLKID = KICS(IC,3)
          IOFF = 0
          NEL    = KONMAT(5,IC_BLKID)
          NELNOD = KONMAT(15,IC_BLKID)
          CALL LNKFIND( IC_BLKID,IOFF,ILNK )
          VX = VICS(IC,1)
          VY = VICS(IC,2)
          VZ = VICS(IC,3)
          IF( IC_SPEC == 1 )THEN
            CALL INITIAL_VEL_MAT( NEL,NELNOD,LINK(ILNK),VEL,VX,VY,VZ )
          ELSE IF( IC_SPEC == 2 )THEN
            CALL INITIAL_ROTV_MAT( NEL,NDSHIN,NELNOD,LINK(ILNK),ROTV,VX,VY,VZ )
          END IF
        ELSE
          CYCLE
        END IF
      END DO

!
! Initialize contact force (EXPLICIT DYNAMICS)
      IF ( KDYNAM == 1 ) CFORCE = PZERO
!
! - - -  G l o b a l  C o n t a c t - - - -
!
      IF( CONTACTS )THEN

        IF( SOLVER_ACTIONS_WERE_REQUESTED )&
          CALL DECLARE_SOLVER_ACTION(2,sa_Initialize_Contact)

        ITRCLEV = 1
!        IF( IPROC .EQ. 3 )THEN
        IF( IPROC .EQ. EXPLICIT_DYNAMIC )THEN
          ISOLVR = 0
        ELSE
          ISOLVR = 1
        END IF
!** Set NLCTS = 0 for the time being.  When the local search is
!   operational, this will be read from the input deck.
        NLCTS = 0
!  Number of friction model state variables
!
!  set Contact global parameters
        CALL CSETGPAR(                                  &
          NNOD_R  ,NNOD_C  ,                            &
          KCONVAR ,ISOLVR  ,IRBFQUAT,NLCTS   ,KPLTRCB,  &
          REQUESTED_SUBROUTINE_CALL_DEPTH ,INITIAL_CSEARCH_ALL,        &
          MCC_ENFORCEMENT  ,CONTACT_RESTART_DATA             )
!
!  set Contact search parameters
        IF( KCSWINDOW.GT.0 )THEN
            IF( ICSWINDFX.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICSWINDFX )ICSWINDFX=J
               END DO
               CALL GETVAL( ICSWINDFX,TIME,CSWINDX,IFLG )
            END IF
            IF( ICSWINDFY.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICSWINDFY )ICSWINDFY=J
               END DO
               CALL GETVAL( ICSWINDFY,TIME,CSWINDY,IFLG )
            END IF
            IF( ICSWINDFZ.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICSWINDFZ )ICSWINDFZ=J
               END DO
               CALL GETVAL( ICSWINDFZ,TIME,CSWINDZ,IFLG )
            END IF
            IF( ICSWINDFR.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICSWINDFR )ICSWINDFR=J
               END DO
               CALL GETVAL( ICSWINDFR,TIME,CSWINDR,IFLG )
            END IF
            IF( TIME.ge.CSTSTART .and. TIME.le.CSTEND )THEN
               KCSW = 1
            ELSE
               KCSW = 0
            END IF
        ELSE
           KCSW = 0
           KCSWT = 0
           CSWINDX = PZERO
           CSWINDY = PZERO
           CSWINDZ = PZERO
           CSWINDR = PZERO
        END IF
        CALL CSETSPAR( CSWINDX,CSWINDY,CSWINDZ,CSWINDR,KCSW,KCSWT )
!
!  set Contact debug parameters
        IF( KCDWINDOW.GT.0 )THEN
            IF( ICDWINDFX.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICDWINDFX )ICDWINDFX=J
               END DO
               CALL GETVAL( ICDWINDFX,TIME,CDWINDX,IFLG )
            END IF
            IF( ICDWINDFY.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICDWINDFY )ICDWINDFY=J
               END DO
               CALL GETVAL( ICDWINDFY,TIME,CDWINDY,IFLG )
            END IF
            IF( ICDWINDFZ.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICDWINDFZ )ICDWINDFZ=J
               END DO
               CALL GETVAL( ICDWINDFZ,TIME,CDWINDZ,IFLG )
            END IF
            IF( ICDWINDFR.GT.0 )THEN
               DO J=1,NFUNC
                 IF( KFDAT(1,J).eq.ICDWINDFR )ICDWINDFR=J
               END DO
               CALL GETVAL( ICDWINDFR,TIME,CDWINDR,IFLG )
            END IF
            KCDW = 1
        ELSE
           KCDW = 0
           CDWINDX = PZERO
           CDWINDY = PZERO
           CDWINDZ = PZERO
           CDWINDR = PZERO
        END IF
        KCDN = 0
        IF( NCDNODES .GT. 0 )KCDN = 1
        KCDT = 0
        IF( TIME.ge.CDTSTART .and. TIME.le.CDTEND )KCDT = 1
        KCDEBUG = max( KCDW*KCDT , KCDN*KCDT )
        CALL CSETDPAR( CDWINDX,CDWINDY,CDWINDZ,CDWINDR,KCDEBUG,KCDW, &
                       NCDNODES,ICDNODES )
!
! Make IELTYP,IELMAT,IESTATUS arrays
        CALL Set_element_type( KSTAT )
!
! Make Temporary Link Array
        CALL MAKEICLINK
!
        CALL_TRACE = 'INIT: calling CONTACT_SURFACE_DEFINITION'
        IF( SUBROUTINE_TRACE_WAS_REQUESTED .AND. &
            CURRENT_SUBROUTINE_CALL_DEPTH.le.REQUESTED_SUBROUTINE_CALL_DEPTH ) &
          CALL PRINT_CALL_TRACE

        IF( .NOT. LCON_USELCLIB )THEN
! Use native contact implementation
          CALL CONTACT_SURFACE_DEFINITION (                               &
            NUMEL   ,NNOD    ,MAPNOD  ,MAPEL   ,MCONDT  ,CONDAT  ,        &
            NBCSID  ,NSFLG   ,NSLEN   ,NSPTR   ,NELEMS  ,NSFAC   ,        &
            NBCNOD_R,KFLAGS  ,NPFLAG  ,NFLOC   ,IBC     ,                 &
            NPRDBC  ,KPRDBC  ,                                   &
            NMTCON  ,NSSCON  ,NNSCON  ,NSMCON  ,                          &
            IMTCON  ,ISSCON  ,INSCON  ,ISMCON  ,                          &
            NUMCNOD ,MAX_NUM_CON_MULTIPLIER,ICEFLAG ,ICECFLG, CESTRG  ,COORD )
        ELSE
! Use Legacy Contact Library
!  Initialize the legacy contact library by performing initialization of its
!  internal modules
          LC_STATUS = LCON_INITIALIZATION( MAPNOD, KPRDBC, LC_PRDBC, COORD, &
            KDISPL, KPNBC, PNBC, XTSTEPS )
          IF( LC_STATUS /= 0 ) GOTO 999 ! error return (all processes)
!
!  Initialize the Analytic Rigid Surfaces capability of the legacy contact library
          LC_STATUS = LCON_ARSINITIALIZATION()
          IF( LC_STATUS /= 0 ) GOTO 999 ! error return (all processes)
!
!  Initialize the Steady State Transport capability of the legacy contact library
          IF( NSSTDEF > 0 )THEN
            LC_STATUS = LCON_SSTINITIALIZATION()
            IF( LC_STATUS /= 0 ) GOTO 999 ! error return (all processes)
          END IF
!
!  Define the contact surfaces in the legacy contact library
          LC_STATUS = LCON_CONTACTSURFACEDEFINITION( MAPEL, CONDAT, &
            IMTCON, ISSCON, INSCON, ISMCON, NUMCNOD, ICEFLAG, &
            ICECFLG, CESTRG)
          IF( LC_STATUS /= 0 ) GOTO 999 ! error return (all processes)
        END IF
!
! Initialize nodal contact arrays to zero
        NODAL  = 1
        IOFF   = 0
        IF( KCONDIRNOR.EQ.1 )THEN
          DO I=1,NUM_CONTACT_OUTPUT
            CALL NVFIND(NODAL,IOFF,'CDIRNOR'//CONTACT_NAMES(I) ,LCDN )
            CALL ZERFILLR( EV(LCDN),NNOD*NSPC )
          END DO
        END IF
        IF( KCONDIRTAN.EQ.1 )THEN
          DO I=1,NUM_CONTACT_OUTPUT
            CALL NVFIND(NODAL,IOFF,'CDIRTAN'//CONTACT_NAMES(I) ,LCDT )
            CALL ZERFILLR( EV(LCDT),NNOD*NSPC )
          END DO
        END IF
        IF( KCONDIRSLP.EQ.1 )THEN
          DO I=1,NUM_CONTACT_OUTPUT
            CALL NVFIND(NODAL,IOFF,'CDIRSLP'//CONTACT_NAMES(I) ,LCDS )
            CALL ZERFILLR( EV(LCDS),NNOD*NSPC )
          END DO
        END IF
        IF( KCONDIRISLP.EQ.1 )THEN
          DO I=1,NUM_CONTACT_OUTPUT
            CALL NVFIND(NODAL,IOFF,'CDIRISLP'//CONTACT_NAMES(I) ,LCDIS )
            CALL ZERFILLR( EV(LCDIS),NNOD*NSPC )
          END DO
        END IF
        IF( KCONVAR.EQ.1 )THEN
          DO I=1,NUM_CONTACT_OUTPUT           
            CALL NVFIND(NODAL,IOFF,'CVARS'//CONTACT_NAMES(I) ,LCVRS )
            CALL ZERFILLR( EV(LCVRS),NNOD*8 )
          END DO
        END IF
!
! For consistency with our restart philosophy (i.e., initialize
! CRSVARS so only the components that exist on the restart file
! will be overwritten and all others will be initialized to their
! default values) initialize CRSVARS
!
        NODAL = 1
        IOFF = 0
        CALL NVFIND( NODAL,IOFF,'CRSVARS_1',LCRSV1 )
        CALL NVFIND( NODAL,IOFF,'CRSVARS_2',LCRSV2 )
        CALL NVFIND( NODAL,IOFF,'CRSVARS_3',LCRSV3 )
        IF( LCON_USELCLIB )THEN
! Use Legacy Contact Library
          LC_STATUS = LCON_CONTACTRESTART( RETRIEVE, ALL_VARS, EV( LCRSV1), &
            EV( LCRSV2), EV( LCRSV3))
          IF( LC_STATUS /= 0 ) GOTO 999 ! error return (all processes)
        ELSE
! Use native contact implementation
          CALL CONTACT_RESTART( RETRIEVE,ALL_VARS, &
                                EV(LCRSV1),EV(LCRSV2),EV(LCRSV3),&
                                IERR,IERRVAL1,IERRVAL2 )
        END IF
        
! If CSRFSIZE encountered an error on any processor, exit
        IF( ICECFLG.GT.0 )THEN
! If this processor had an error, write message to KOUT
          IF( ICEFLAG.NE.0 )THEN
            WRITE(KOUT,'(A,/,A)') 'ERROR in init2 from CSRFINIT',CESTRG
            CALL KLOGERR('0.init')
          END IF
          CALL KLOGCK('1.init')
          STOP '1.init should not get here'
        END IF
!      ELSE IF( SOLVER_ACTIONS_WERE_REQUESTED .AND. ACTIVE_MLControls )THEN
!        CALL DECLARE_SOLVER_ACTION(2,-sa_Initialize_Contact)
      END IF

! If errors have been found, log an error.
 999  CALL KERRORS(JERR)
      IF( MPERR(IERR .GT. 0 .OR. JERR.NE.0) )THEN
        WRITE(KOUT,1040) IERR
        CALL KLOGERR('2.init')
      END IF

      CURRENT_SUBROUTINE_CALL_DEPTH = bCUR_SUBROUTINE_CALL_DEPTH
      CALL_TRACE = 'INIT: end'
      IF( SUBROUTINE_TRACE_WAS_REQUESTED .AND. &
          CURRENT_SUBROUTINE_CALL_DEPTH.le.REQUESTED_SUBROUTINE_CALL_DEPTH ) &
        CALL PRINT_CALL_TRACE

!
 1040 FORMAT(/////10X,'AT LEAST,',I4,' ERRORS HAVE BEEN DETECTED WHILE',&
     &  ' INITIALIZING DATA',/,10X,'COMPUTATION WILL BE TERMINATED' )
 1050 FORMAT(' ***** ERROR: A SHELL ELEMENT BLOCK WITH A DEFAULT THICKNESS '/ &
     &       '              OF 1.0 IS USING INEXTENSIBLE FIBER (SMAT15)'/     &
     &       '              MATERIAL WITH ADDED-STIFFNESS MODULI SPECIFIED.'/ &
     &       '              MATERIAL ID: ',I8,' REQUIRES EXPLICIT THICKNESS'/ &
     &       '              SPECIFICATION VIA "SHELL SCALE THICKNESS" INPUT'/ )
!
END SUBROUTINE INIT
