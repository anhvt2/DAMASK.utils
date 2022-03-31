! $Id: updstr.f90,v 5.0 2007/12/19 16:54:01 mlblanf Exp $
!
SUBROUTINE UPDSTR( NE,MKIND,NINSV,PROP,SIG,SV,D,DELSIG, &
                         STRAIN,STRECH,XTEOLD,XTECUR, &
                         DTHS,RJTH,SFJTH,CM,IQCM,ROTATE,IALFLG,ISOL, &
                         NALSV,ALSV,NFAILV,CFSV )
!
!***********************************************************************
!
!     DESCRIPTION:
!       This routine updates element stresses for a vector block of
!       elements. It manipulates stresses to account for finite rotation
!       effects and computes the hypo-elastic stress rates for use in
!       time step and hourglass control.
!
!     FORMAL PARAMETERS:
!       NE      INTEGER    Number of Elements in this Vector Block
!       MKIND   INTEGER    Material Type for this Material Block
!       NINSV   INTEGER    Number of Internal State Variables for the
!                          Material Type of this Material Block (if any)
!       PROP    REAL       Material Properties for this Material Block
!       SIG     REAL       Element Stresses (Relative to this Vector Block):
!                             (1) = XX-Normal
!                             (2) = YY-Normal
!                             (3) = ZZ-Normal
!                             (4) = XY-Shear
!                             (5) = YZ-Shear
!                             (6) = ZX=Shear
!       SV      REAL       Element Internal State Variables
!       D       REAL       Unrotated Deformation Rates:
!                             (1) = XX-Normal
!                             (2) = YY-Normal
!                             (3) = ZZ-Normal
!                             (4) = XY-Shear
!                             (5) = YZ-Shear
!                             (6) = ZX=Shear
!       DELSIG  REAL       Hypo-elastic Stress Increments
!       STRAIN  REAL       Element strains in un-rotated configuration
!                          (note: only used if KSFLG = 1 )
!       STRECH  REAL       Element stretches (total)
!       XTEOLD  REAL       Old value of interpolated element variables
!       XTECUR  REAL       Current value of interpolated element variables
!       DTHS    REAL       Thermal strain rate
!       RJTH    REAL       Total isotropic thermal strain for hyperelasticity
!       SFJTH   REAL       Stress-free isotropic thermal strain for hyperelasticity
!       CM      REAL       Tangent Modulus
!       IQCM    INTEGER    Flag to calculate Tangent Modulus
!       ROTATE  REAL       Rotation tensor at end of time step
!       F       REAL       Total deformation gradient tensor at end
!                          of time step (mechanical + thermal parts)
!       IALFLG  INTEGER    Flag to use Aug. Lag. formulation in MAT4
!       ISOL    INTEGER    Update control parameter
!                          = 0, All updates are temporary
!                          = 1. All updates are retained
!       NALSV   INTEGER    Number of aug. Lag. state variables for the
!                          Material Type of this Material Block (if any)
!       ALSV    REAL       Augmented Lagrange state variables
!       NFAILV  INTEGER    Number of control failure state variables for the
!                          Material Type of this Material Block (if any)
!       CFSV    REAL       Control failure state variables
!
!     CALLED BY: VRESID
!
!***********************************************************************
!
   USE params_
   USE alloc_
   USE timer_
   USE xtvars_
   USE numbers_
      INCLUDE 'precision.blk'
!
      DIMENSION PROP(*),SIG(NSYMM,NE),SV(NINSV,NE),D(NSYMM,NE), &
        DELSIG(NSYMM,NE),ALSV(NALSV,NE), &
        STRAIN(NSYMM,NE),STRECH(NSYMM,NE),ROTATE(NSPC,NSPC,NE), &
        XTEOLD(NXTIE,NE),XTECUR(NXTIE,NE),DTHS(3,NE),CM(NTNMD,NE), &
        RJTH(NE),SFJTH(NE),STRECHSAVE(NSYMM,NEBLK), &
        F(NONSYM,NE),CFSV(NFAILV,NE)
! Temporary storage for gathered values of external element variables
      DIMENSION TMPO(NE),TMPC(NE),RCTO(NE),RCTC(NE)
      DIMENSION USM(NE)
      LOGICAL :: STRAIN_REQUESTED
!
! Modify strain rates and stretch tensor for thermal strain
      IF( NTHERM.NE.0 .AND. MKIND.NE.10 .AND. MKIND.NE.48   &
                      .AND. MKIND.NE.49 )THEN
        DO 100 K = 1,NE
          D(1,K) = D(1,K) - DTHS(1,K)
          D(2,K) = D(2,K) - DTHS(2,K)
          D(3,K) = D(3,K) - DTHS(3,K)
          STRECHSAVE(1,K) = STRECH(1,K)
          STRECHSAVE(2,K) = STRECH(2,K)
          STRECHSAVE(3,K) = STRECH(3,K)
          STRECHSAVE(4,K) = STRECH(4,K)
          STRECHSAVE(5,K) = STRECH(5,K)
          STRECHSAVE(6,K) = STRECH(6,K)
          FAC = (SFJTH(K)/RJTH(K))**PTHIRD
          STRECH(1,K) = STRECH(1,K)*FAC
          STRECH(2,K) = STRECH(2,K)*FAC
          STRECH(3,K) = STRECH(3,K)*FAC
          STRECH(4,K) = STRECH(4,K)*FAC
          STRECH(5,K) = STRECH(5,K)*FAC
          STRECH(6,K) = STRECH(6,K)*FAC
  100   CONTINUE
      END IF
!
! Copy stresses at the begining of the time step
      DO 120 N = 1,NE
        DELSIG(1,N) = SIG(1,N)
        DELSIG(2,N) = SIG(2,N)
        DELSIG(3,N) = SIG(3,N)
        DELSIG(4,N) = SIG(4,N)
        DELSIG(5,N) = SIG(5,N)
        DELSIG(6,N) = SIG(6,N)
  120 CONTINUE

! Integrate strain (Material-kind equal to 30 uses the strain 
! array STRAIN but manages the strain computation itself.)
      STRAIN_REQUESTED = (KSFLG.NE.0 .AND. MKIND.NE.30)
      IF( STRAIN_REQUESTED ) THEN
        STRAIN(1:6,1:NE) = STRAIN(1:6,1:NE) + DT * D(1:6,1:NE)
      ENDIF

! Call requested constitutive model

      IF( MKIND.EQ.1 )THEN
! **********************************************************************
! BEGIN extra code for LCP extension.
! **********************************************************************
        CALL MAT96(NE,NINSV,DT,PROP,SIG,SV,CM,IQCM,STRECH,ROTATE,TIME)
! **********************************************************************
! END extra code for LCP extension.
! **********************************************************************

      END IF
!
! Compute hypo-elastic stress increments
      DO 150 N = 1,NE
        DELSIG(1,N) = SIG(1,N) - DELSIG(1,N)
        DELSIG(2,N) = SIG(2,N) - DELSIG(2,N)
        DELSIG(3,N) = SIG(3,N) - DELSIG(3,N)
        DELSIG(4,N) = SIG(4,N) - DELSIG(4,N)
        DELSIG(5,N) = SIG(5,N) - DELSIG(5,N)
        DELSIG(6,N) = SIG(6,N) - DELSIG(6,N)
  150 CONTINUE
!
! Restore the stretch tensor to be total stretch
      IF( NTHERM .NE. 0 )THEN
        DO 170 K = 1,NE
          STRECH(1,K) = STRECHSAVE(1,K)
          STRECH(2,K) = STRECHSAVE(2,K)
          STRECH(3,K) = STRECHSAVE(3,K)
          STRECH(4,K) = STRECHSAVE(4,K)
          STRECH(5,K) = STRECHSAVE(5,K)
          STRECH(6,K) = STRECHSAVE(6,K)
  170   CONTINUE
      END IF

!
END SUBROUTINE UPDSTR
