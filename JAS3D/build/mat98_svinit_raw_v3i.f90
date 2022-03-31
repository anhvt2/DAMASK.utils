! $Id: svinit.f90,v 5.0 2007/12/19 16:54:11 mlblanf Exp $
!
SUBROUTINE SVINIT( NEL,NELNOD,MKIND,NINSV,KORID,NTINT,NALSV,SV,ALSV,SIG, &
                   PROP,COORD,LINK,BASEL,XTECUR,EVTHCK,RHO,MAPEL,IBLK, &
                   NFAILV,CFSV, &
!***********************************************************************
! BEGIN extra code for LCP extension.
! **********************************************************************
                   MELSTART,MELEND )
!***********************************************************************
! END extra code for LCP extension.
! **********************************************************************
!
!***********************************************************************
!
!     DESCRIPTION:
!       This routine calls the initialization routines for
!       the constitutive models that require initialization
!       of internal state variables.
!
!     FORMAL PARAMETERS:
!       NEL      INTEGER   Number of elements in this element block
!       NELNOD   INTEGER   Number of nodes per elements in this block
!       MKIND    INTEGER   Material type number
!       NINSV    INTEGER   Number of internal state variables
!                          for this material type
!       KORID    INTEGER   orientation model id
!       NTINT    INTEGER   Number of integration points for this block
!       NALSV    INTEGER   Number of augmented Lagrange variables
!       SV       REAL      Array containing the internal state
!                          variables which must be initialized
!       ALSV     REAL      Array containing the augmented Lagrange 
!                          variables which must be initialized
!       SIG      REAL      Stresses in this material
!       PROP     REAL      Material properties for this material
!       COORD    REAL      Nodal point coordinates
!       LINK     INTEGER   Element connectivity for this material block
!       XTECUR   REAL      Current value of interpolated element variables
!       EVTHCK   REAL      Shell element thickness in EV array
!       MAPEL    INTEGER   Maps local element number to global
!       IBLK     INTEGER   Block ID
!       NFAILV   INTEGER   Number of CONTROL FAILURE variables
!       CFSV     REAL      Array containing CONTROL FAILURE varibles
!
!     CALLED BY: INIT
!
!***********************************************************************
!
   USE precision_
   USE params_
   USE contrl_
   USE funcs_
   USE orient_
   USE psize_
   USE rdata_
   USE xtvars_
      INCLUDE 'precision.blk'
      INCLUDE 'numbers.blk'
!
      DIMENSION SV(NINSV,NTINT,NEL),SIG(NSYMM,NEL),PROP(*), &
        COORD(NNOD,NSPC),LINK(NELNOD,NEL),BASEL(NSPC,NSPC,NEL), &
        XTECUR(NXTIE,NEL),EVTHCK(NEL),RHO(*),ALSV(NALSV,NTINT,NEL), &
        MAPEL(NUMEL),CFSV(NFAILV,NTINT,NEL)

!***********************************************************************
! BEGIN extra code for LCP extension.
! **********************************************************************
      INTEGER          MELSTART, MELEND
!
! Variables for Remi Dingreville's Deformation Gradient Crystal Plasticity Model
!
      DIMENSION        RINIT(10),RTMP(9,NEL),Rot_rand(3,3),Rlab(3,3)

      INTEGER          Flag_Slip, Flag_Harden
      REAL(8) 	       COORDS(NEL,3),TEMP,RTEMP(1000000,25),TEMP1

!
!  END variable declarations
!
      Em10 = 1.0E-10
!***********************************************************************
! END extra code for LCP extension.
! **********************************************************************
!
! Default initialization is all zero initial internal state variables
! and augmented Lagrange state variables
      IF( NINSV.GT.0 )THEN
        MSIZE = SIZE(SV)
        CALL ZERFILLR( SV,MSIZE )
      END IF
      IF( NALSV.GT.0 )THEN
        MSIZE = SIZE(ALSV)
        CALL ZERFILLR( ALSV,MSIZE )
      END IF
      IF( NFAILV.GT.0 )THEN
        MSIZE = SIZE(CFSV)
        CALL ZERFILLR( CFSV,MSIZE )
      END IF
!
!
! Local Crystal Plasticity Deformation Gradient Model
!
      IF( MKIND.EQ.1 )THEN
!***********************************************************************
! BEGIN extra code for LCP extension.
! **********************************************************************
!
! Initialize all variables
!
        DO J = 1,NINSV
           DO K  = 1,NTINT
              DO I  = 1,NEL
                 SV(J,K,I) = 0.0
              END DO
           END DO
        END DO
!
        DO I=1,NEL

          DO J=1,9
            RTMP(J,I) = 0.0
          END DO
!
        END DO
!
!
! Assign Materials Properties
!
        urC11       = PROP( 1)
        urC12       = PROP( 2)
        urC44       = PROP( 3)
        Flag_Sys    = PROP( 4)
        Flag_Slip   = int(PROP( 5) + 0.5)
        Flag_Harden = int(PROP( 6) + 0.5)
        T0          = PROP(12)
        H9          = PROP(21)

!
! Shear modulus in 110 direction
!
        denom       = (urC11 - urC12)*(urC11 + 2.0*urC12)
        S11         = (urC11 + urC12)/denom
        S12         = urC12/denom
        S44         = 1.0/urC44
        G_110       = 1.0/(0.5*S44 + S11 + S12)
!
! Burgers vector is in m. All dislocation density's are in 1/(m^2)
!
        burg_vect = PROP(22)
!
! Lab reference orientation
!
          DO I0=1,3
          DO J0=1,3
             Rlab(I0,J0) = 0.0
          END DO
          END DO
          Rlab(1,1) = 1.0
          Rlab(2,2) = 1.0
          Rlab(3,3) = 1.0

!
! Read orientation from potts.out file
!
!       OPEN (UNIT=59,FILE='potts.out',STATUS='OLD')
!        
! BEGIN Read the crystallographic orientations per block:
!

!        DO L = 1, IBLK
!          READ (59,*) (RINIT(J),J=1,9)
!        END DO
!        DO L = 1, NEL
!          DO J=1,9
!            RTMP(J,L) = RINIT(J)
!          END DO
!        END DO          
!        
!  END  Read the crystallographic orientations per block:
!
!        CLOSE(59)
!
!***********************************************************************
! READ INITIAL ORIENTATION PER ELEMENT

        RTEMP(:,:)=0
        OPEN (UNIT=59,FILE='output_EquiaxedSize.out',STATUS='OLD')
    DO L = 1,1000000
        READ (59,*) RTEMP(L,1:25)
    END DO 

    CLOSE(59)

! ELEMENT CENTER COORDINATES

        COORDS(1:NEL,1:3)=0 

        DO L = 1, NEL 
                DO J = 1, 3
                        DO K = 1, 8
                        COORDS(L,J)=COORDS(L,J)+COORD(LINK(K,L),J)
                        ENDDO
                COORDS(L,J)=NINT(COORDS(L,J)/8+49.5)
                ENDDO

                K=1+COORDS(L,1)+COORDS(L,2)*100+COORDS(L,3)*10000


                DO J = 1, 9
                RTMP(J,L)=RTEMP(K,J+4)
                ENDDO
                

                DO J = 1, 12
                    DO I=1,NTINT
                        SV(87+J,I,L)=RTEMP(K,J+13)
                    ENDDO
                ENDDO

                !TEMP=0.5455 * 1000 * (RTEMP(L,14))**(-0.5)
                !TEMP=TEMP/3.06

                !DO J = 1, NTINT
                !SV(63,J,L) = TEMP
                !ENDDO
        ENDDO
!***********************************************************************

        DO  K  = 1,NTINT
          DO  I  = 1,NEL

            DO J  = 1,9
              SV(J+36,K,I)  = RTMP(J,I) !RINIT
              SV(J+45,K,I) = RTMP(J,I) !RT
            END DO

              Rot_rand(1,1) = RTMP(1,I)
              Rot_rand(1,2) = RTMP(2,I)
              Rot_rand(1,3) = RTMP(3,I)
              Rot_rand(2,1) = RTMP(4,I)
              Rot_rand(2,2) = RTMP(5,I)
              Rot_rand(2,3) = RTMP(6,I)
              Rot_rand(3,1) = RTMP(7,I)
              Rot_rand(3,2) = RTMP(8,I)
              Rot_rand(3,3) = RTMP(9,I)
              call misorient(phi_orient,Rlab,Rot_rand)
!
! Assign initial element orient w.r.t lab reference
!
              SV(61,K,I) = phi_orient 
            
          END DO
        END DO
!
! Initialize non-zero SVs
!
        DO K  = 1,NTINT
          DO I  = 1,NEL
!
! Initialize Elastic Deformation Gradient Fe = I
!
               SV( 1,K,I) = 1.0
               SV( 5,K,I) = 1.0
               SV( 9,K,I) = 1.0
!
! Initialize Plastic Deformation Gradient Fp = I
!
               SV(10,K,I) = 1.0
               SV(14,K,I) = 1.0
               SV(18,K,I) = 1.0
!
! Initialize Total Deformation Gradient from last step Ftot = I
!
               SV(19,K,I) = 1.0
               SV(23,K,I) = 1.0
               SV(27,K,I) = 1.0
!
! Initialize Elastic Rotation matrix Re = I
!
               SV(28,K,I) = 1.0
               SV(32,K,I) = 1.0
               SV(36,K,I) = 1.0

! Initialize CRSS 

            DO J=1,12
               SV(75+J,K,I) = T0 !CRSS
               !SV(75+J,K,I) = T0+SV(63,K,I)
            END DO
!
! Initialize SSD density 
!          IF (Flag_Harden.eq.1) THEN
!            DO J=1,12
               !SV(87+J,K,I) = (T0/(H9*G_110*burg_vect))**2
!                SV(87+J,K,I) = 1.0E5
!            END DO
!         END IF

!open(1001, file = 'data1.dat', status = 'old')
!write(1001,*) T0, H9, G_110, burg_vect
!close(1001)

! Average slip
              SV(58,K,I) = (SV(64,K,I) + SV(65,K,I) + SV(66,K,I)     +  &
                            SV(67,K,I) + SV(68,K,I) + SV(69,K,I)     +  &
                            SV(70,K,I) + SV(71,K,I) + SV(72,K,I)     +  &
                            SV(73,K,I) + SV(74,K,I) + SV(75,K,I))/12.0 

! Average CRSS
              SV(60,K,I) = (SV(76,K,I) + SV(77,K,I) + SV(78,K,I)     +  &
                            SV(79,K,I) + SV(80,K,I) + SV(81,K,I)     +  &
                            SV(82,K,I) + SV(83,K,I) + SV(84,K,I)     +  &
                            SV(85,K,I) + SV(86,K,I) + SV(87,K,I))/12.0

! Average DD
              SV(59,K,I) = (SV(88,K,I) + SV(89,K,I) + SV(90,K,I)       +  &
                            SV(91,K,I) + SV(92,K,I) + SV(93,K,I)       +  &
                            SV(94,K,I) + SV(95,K,I) + SV(96,K,I)       +  &
                            SV(97,K,I) + SV(98,K,I) + SV(99,K,I))/12.0

! Check
        
        END DO
        END DO

!***********************************************************************
! END extra code for LCP extension.
! **********************************************************************
      END IF
!
END SUBROUTINE SVINIT
