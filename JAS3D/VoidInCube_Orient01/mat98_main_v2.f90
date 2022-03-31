! Remi Dingreville's Local Deformation Gradient Crystal Plasticity
! (a.k.a "The mat_Chandross subroutine" )
!
      SUBROUTINE  MAT96(NEL,NINSV,DT,PROP,SIG,SV,CM,IQCM,STRECH,ROTATE,TIME)
!
!------------------------------------------------------------------------
!
! DESCRIPTION: Crystal Plasticity Model 
!            
!
! FORMAL PARAMETERS:
!   NEL    INTEGER Number of Elements in this Block
!   NINSV  INTEGER Number of State Variables 
!   TIME   REAL    Time Value
!   DT     REAL    Time Increment
!   PROP   REAL    Material Properties (Room Temperature)
!                (      1  ) = Elastic Constant      - C11 (unrotated)
!                (      2  ) = Elastic Constant      - C12 (unrotated)
!                (      3  ) = Elastic Constant      - C44 (unrotated)
!                (      4  ) = Slip system type      - Flag_Sys
!                (      5  ) = Flag Flow rule        - Flag_Slip
!                (      6  ) = Flag Hardening law    - Flag_Harden
!                (      7  ) = Flow Rate             - GD0
!                (      8  ) = Flow Rule Coeff. 1    - XP
!                (      9  ) = Flow Rule Coeff. 2    - XQ
!                (     10  ) = Active Slip           - Active_slip (Cutoff active rate)
!	             (     11  ) = Critical Stress       - T_cr  
!                (     12  ) = Initial Flow Stress   - T0
!                (     13  ) = Hardening Coeff. 1    - H1 
!                (     14  ) = Hardening Coeff. 2    - H2 
!                (     15  ) = Hardening Coeff. 3    - H3 
!                (     16  ) = Hardening Coeff. 4    - H4 
!                (     17  ) = Hardening Coeff. 5    - H5 
!                (     18  ) = Hardening Coeff. 6    - H6 
!                (     19  ) = Hardening Coeff. 7    - H7 
!                (     20  ) = Hardening Coeff. 8    - H8 
!                (     21  ) = Hardening Coeff. 9    - H9  (alpha:Taylor Const.)
!                (     22  ) = Hardening Coeff.10    - H10 (Burgers Vector)
!	             (     23  ) = Non-Schmid Const. 1   - C_1
!                (     24  ) = Non-Schmid Const. 2   - C_2
!                (     25  ) = Non-Schmid Const. 3   - C_3
!                (     26  ) = Non-Schmid Const. 4   - C_4
!                (     27  ) = Non-Schmid Const. 5   - C_5

!   SIG    REAL    Stresses
!   D      REAL    Strain Rates
!   SV     REAL    Internal State Variables:
!                  (  1 -  9,*) = 3x3 elastic deformation gradient tensor
!                  ( 10 - 18,*) = 3x3 plastic deformation gradient tensor
!                  ( 19 - 27,*) = Total deformation gradient from last step
!                  ( 28 - 36,*) = Elastic Rotation Tensor 
!                  ( 37 - 45,*) = Initial Orientation Tensor
!                  ( 46 - 54,*) = Rotation Tensor Components - RT
!                  (      55,*) = Misorientation w.r.t. initial orientation 
!                  (      56,*) = Effective Plastic Strain
!                  (      57,*) = Number of subincrements
!                  (      58,*) = Average cumulative slip
!                  (      59,*) = Average dislocation Density
!                  (      60,*) = Average CRSS
!                  (      61,*) = Orientation w.r.t. Lab reference
!                  (      62,*) = Number of active slip systems
!                  (      63,*) = GrainSize Effect
!                  ( 64 - 75,*) = Accumulated slip on each slip
!                  ( 76 - 87,*) = Crit. Resolved Shear Stress on slip system
!                  ( 88 - 99,*) = Dislocation Density
!         
!   CM     REAL       Tangent Modulus
!   IQCM   INTEGER    Flag to Calculate Tangent Modulus
!   STRECH REAL       Element stretches
!   ROTATE REAL       Element rigid-body rotations
!
!   CALLED BY: UPDSTR
!
!------------------------------------------------------------------------

   USE params_
   USE xtvars_
   USE funcs_
   USE iolib_
      INCLUDE 'precision.blk'
      INCLUDE 'numbers.blk'

!
! Variables passed into MAT86 via the subroutine call
!
      DIMENSION        PROP(*),SV(NINSV,NEL),SIG(NSYMM,NEL)          ,  &
                       CM(NTNMD,NEL),STRECH(NSYMM,NEL)               ,  &
                       ROTATE(NSPC,NSPC,NEL)
!
! Variables in the UPDATE STRESS Loop   
!
      DIMENSION        vect_n(12,3),vect_d(12,3),F_tDt(3,3),Fel(3,3)   ,  &
                       Fplast(3,3),Fel_nxt(3,3),Fplast_nxt(3,3)      ,  &
                       curr_n(12,3),curr_d(12,3),Sym_schmid(12,3,3)    ,  &
                       Vplast(3,3),re_nxt(3,3),Schmid_int(12,3,3)    ,  &
                       E_CauchyGreen(6),Eplast(6),var_stiff(9)       ,  &
                       Celast(6,6),sig_cau(6),sig_pk2(6),sig_unrot(6),  &
                       Rot(3,3),Rel(3,3)
      DIMENSION        cauchy(6,NEL),cauchy_nxt(6,NEL),DF_DT(9,NEL)  ,  &
                       fe_nxt(3,3,NEL),fp_nxt(3,3,NEL),RSS(12,NEL)   ,  &
                       def_grad(3,3,NEL)
      LOGICAL          update,subinc
      INTEGER          check,num_sub_in,max_sub,inthj
!
! Variables in Slip Kinematic / Flow rule
!
      DIMENSION        var_slip(74),res_slip(13),slip_rate(12,NEL)   ,  &
                       cumul_slip(12,NEL),Eeff_plast_nxt(NEL)        ,  &
                       active_syst(NEL)
      INTEGER          Flag_Slip, Flag_sys
!
! Variables in Hardening law
!
      DIMENSION        var_harden(61),res_harden(24),ssd_nxt(12,NEL) ,  &
                       gnd_nxt(12,NEL),CRSS_nxt(12,NEL)
      INTEGER          Flag_Harden
!
! Variables for Stability 
!
      DIMENSION        ck_curr_n(12,3),ck_curr_d(12,3),ck_Ps(12,3,3)   ,  &
                       ck_RSS(12),ck_slip(13),ck_var_slip(74)        ,  &
                       slip(12),sta_slip(24),slip_1(12),Flow(12)       ,  &
                       dRSS(24),dCRSS(24),CRIT(NEL),dImm(24),dMob(24) ,  &
                       ck_Ptot(12,3,3),ck_Pns(12,3,3),                   &
                       TNS(24),ck_TS(12),ck_NS(12),ck_sym(12,3,3)
!
! Variables in misorientation calculation
!
      DIMENSION        sn(12,3),sd(12,3), Rinit(3,3)
!
! 
!Variables in BCC model
      DIMENSION	       Ptot(12,3,3), Pns(12,3,3), Ps(12,3,3),   &
                       TS(12,NEL),TS_NS(12,NEL),bcc_var(5)

      ZERONUM = 2.0E-13

!************************************************************************
!                       Variable Initialization
!************************************************************************
      C11          = PROP( 1)
      C12          = PROP( 2)
      C44          = PROP( 3)
      Flag_Sys     = int(PROP( 4)+0.5)
      Flag_Slip    = int(PROP( 5)+0.5)
      Flag_Harden  = int(PROP( 6)+0.5)
      GD0          = PROP( 7)
      XP           = PROP( 8)
      XQ           = PROP( 9)
      active_slip  = PROP(10) 
      T_cr         = PROP(11)
      T0           = PROP(12)
      H1           = PROP(13)
      H2           = PROP(14) 
      H3           = PROP(15)
      H4           = PROP(16)
      H5           = PROP(17)
      H6           = PROP(18)
      H7           = PROP(19) !Temperature
      H8           = PROP(20) !Qeff
      H9           = PROP(21) !Alpha (Taylor cons.)
      H10          = PROP(22) !Burgers vector
      C_1          = PROP(23)
      C_2          = PROP(24)
      C_3          = PROP(25)
      C_4          = PROP(26)
      C_5          = PROP(27)
!
! Shear modulus in 110 direction
!
      denom        = (C11 - C12)*(C11 + 2.0*C12)
      S11          = (C11 + C12)/denom
      S12          = C12/denom
      S44          = 1.0/C44
      G110         = 1.0/(S11 + S12 + 0.5*S44)      
!
! Assign elastic properties
!
      var_stiff(1) = C11
      var_stiff(2) = C11
      var_stiff(3) = C11
      var_stiff(4) = C12
      var_stiff(5) = C12
      var_stiff(6) = C12
      var_stiff(7) = C44
      var_stiff(8) = C44
      var_stiff(9) = C44
!
! Assign slip variables
!
      var_slip(1)      = GD0
      var_slip(2)      = XP
      var_slip(3)      = XQ
      var_slip(4)      = H1
      var_slip(5)      = H2
      var_slip(6)      = H3
      var_slip(7)      = H4
      var_slip(8)      = H5
      var_slip(9)      = H6
      var_slip(10)     = H7
      var_slip(11)     = H8
      var_slip(12)     = H9
      var_slip(13)     = H10 
      var_slip(14)     = active_slip
!     var_slip(15-26)  = RSS
!     var_slip(27-38)  = CRSS
!     var_slip(39-50)  = DD
!     var_slip(51-62)  = slip rate
!     var_slip(63-74)= Tns
            
      ck_var_slip(1)      = GD0
      ck_var_slip(2)      = XP
      ck_var_slip(3)      = XQ
      ck_var_slip(4)      = H1
      ck_var_slip(5)      = H2
      ck_var_slip(6)      = H3
      ck_var_slip(7)      = H4
      ck_var_slip(8)      = H5
      ck_var_slip(9)      = H6
      ck_var_slip(10)     = H7
      ck_var_slip(11)     = H8
      ck_var_slip(12)     = H9
      ck_var_slip(13)     = H10 
      ck_var_slip(14)     = active_slip
!     ck_var_slip(15-26)  = RSS
!     ck_var_slip(27-38)  = CRSS
!     ck_var_slip(39-50)  = DD
!     ck_var_slip(51-62)  = slip rate
!     ck_var_slip(63-74)= Tns

! Assign Hardening law variables
!
      var_harden(1)      = T0
      var_harden(2)      = G110
      var_harden(3)      = H1
      var_harden(4)      = H2
      var_harden(5)      = H3
      var_harden(6)      = H4
      var_harden(7)      = H5
      var_harden(8)      = H6
      var_harden(9)      = H7
      var_harden(10)     = H8
      var_harden(11)     = H9
      var_harden(12)     = H10
!     var_harden(13)     = Eeffe_plast (Effective plastic strain)
!     var_harden(14-25)  = CRSS
!     var_harden(26-37)  = Acc.Strain
!     var_harden(38-49)  = Dislocation density
!     var_harden(50-61) = Slip rate*time step
!
! Assign BCC variables
!
      bcc_var(1) = C_1
      bcc_var(2) = C_2
      bcc_var(3) = C_3
      bcc_var(4) = C_4
      bcc_var(5) = C_5

      call SLIPSYS(Flag_Sys,sd,sn)

      DO I=1, NEL
!
!************************************************************************
!                               For JAS Only
!               Rotate Stress from unrotated config to current config
!
!               cauchy(i,j)=Rotate(i,k)*SIG(k,l)*(Rotate(l,j))^T
!
!               Note: In explanation indices (i,j) are for stress tensor
!                     NOT for vector component and element
!
!       Cauchy stress Matrix = cauchy(1)   cauchy(4)   cauchy(6)
!                              cauchy(4)   cauchy(2)   cauchy(5)
!                              cauchy(6)   cauchy(5)   cauchy(3)
!************************************************************************
        cauchy(1,I) = ROTATE(1,1,I)*( SIG(1,I)*ROTATE(1,1,I)         +  &
             SIG(4,I)*ROTATE(1,2,I) + SIG(6,I)*ROTATE(1,3,I) )       +  &
                      ROTATE(1,2,I)*( SIG(4,I)*ROTATE(1,1,I)         +  &
             SIG(2,I)*ROTATE(1,2,I) + SIG(5,I)*ROTATE(1,3,I) )       +  &
                      ROTATE(1,3,I)*( SIG(6,I)*ROTATE(1,1,I)         +  &
             SIG(5,I)*ROTATE(1,2,I) + SIG(3,I)*ROTATE(1,3,I) )

        cauchy(2,I) = ROTATE(2,1,I)*( SIG(1,I)*ROTATE(2,1,I)         +  &
             SIG(4,I)*ROTATE(2,2,I) + SIG(6,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(2,2,I)*( SIG(4,I)*ROTATE(2,1,I)         +  &
             SIG(2,I)*ROTATE(2,2,I) + SIG(5,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(2,3,I)*( SIG(6,I)*ROTATE(2,1,I)         +  &
             SIG(5,I)*ROTATE(2,2,I) + SIG(3,I)*ROTATE(2,3,I) )

        cauchy(3,I) = ROTATE(3,1,I)*( SIG(1,I)*ROTATE(3,1,I)         +  &
             SIG(4,I)*ROTATE(3,2,I) + SIG(6,I)*ROTATE(3,3,I) )       +  &
                      ROTATE(3,2,I)*( SIG(4,I)*ROTATE(3,1,I)         +  &
             SIG(2,I)*ROTATE(3,2,I) + SIG(5,I)*ROTATE(3,3,I) )       +  &
                      ROTATE(3,3,I)*( SIG(6,I)*ROTATE(3,1,I)         +  &
             SIG(5,I)*ROTATE(3,2,I) + SIG(3,I)*ROTATE(3,3,I) ) 

        cauchy(4,I) = ROTATE(1,1,I)*( SIG(1,I)*ROTATE(2,1,I)         +  &
             SIG(4,I)*ROTATE(2,2,I) + SIG(6,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(1,2,I)*( SIG(4,I)*ROTATE(2,1,I)         +  &
             SIG(2,I)*ROTATE(2,2,I) + SIG(5,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(1,3,I)*( SIG(6,I)*ROTATE(2,1,I)         +  &
             SIG(5,I)*ROTATE(2,2,I) + SIG(3,I)*ROTATE(2,3,I) ) 

        cauchy(5,I) = ROTATE(3,1,I)*( SIG(1,I)*ROTATE(2,1,I)         +  &
             SIG(4,I)*ROTATE(2,2,I) + SIG(6,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(3,2,I)*( SIG(4,I)*ROTATE(2,1,I)         +  &
             SIG(2,I)*ROTATE(2,2,I) + SIG(5,I)*ROTATE(2,3,I) )       +  &
                      ROTATE(3,3,I)*( SIG(6,I)*ROTATE(2,1,I)         +  &
             SIG(5,I)*ROTATE(2,2,I) + SIG(3,I)*ROTATE(2,3,I) ) 

        cauchy(6,I) = ROTATE(3,1,I)*( SIG(1,I)*ROTATE(1,1,I)         +  &
             SIG(4,I)*ROTATE(1,2,I) + SIG(6,I)*ROTATE(1,3,I) )       +  &
                      ROTATE(3,2,I)*( SIG(4,I)*ROTATE(1,1,I)         +  &
             SIG(2,I)*ROTATE(1,2,I) + SIG(5,I)*ROTATE(1,3,I) )       +  &
                      ROTATE(3,3,I)*( SIG(6,I)*ROTATE(1,1,I)         +  &
             SIG(5,I)*ROTATE(1,2,I) + SIG(3,I)*ROTATE(1,3,I) )
             
!************************************************************************             
!                              For JAS Only
!       Form the deformation gradient at t + dt with left stretch tensor
!
!                           F(i,j)=U(i,k)*R(k,j)
!************************************************************************
        def_grad(1,1,I) = STRECH(1,I)*ROTATE(1,1,I)                  +  &
                          STRECH(4,I)*ROTATE(2,1,I)                  +  &
                          STRECH(6,I)*ROTATE(3,1,I)

        def_grad(1,2,I) = STRECH(1,I)*ROTATE(1,2,I)                  +  &
                          STRECH(4,I)*ROTATE(2,2,I)                  +  &
                          STRECH(6,I)*ROTATE(3,2,I)     

        def_grad(1,3,I) = STRECH(1,I)*ROTATE(1,3,I)                  +  &
                          STRECH(4,I)*ROTATE(2,3,I)                  +  &
                          STRECH(6,I)*ROTATE(3,3,I)

        def_grad(2,1,I) = STRECH(4,I)*ROTATE(1,1,I)                  +  &
                          STRECH(2,I)*ROTATE(2,1,I)                  +  &
                          STRECH(5,I)*ROTATE(3,1,I) 

        def_grad(2,2,I) = STRECH(4,I)*ROTATE(1,2,I)                  +  &
                          STRECH(2,I)*ROTATE(2,2,I)                  +  &
                          STRECH(5,I)*ROTATE(3,2,I)     

        def_grad(2,3,I) = STRECH(4,I)*ROTATE(1,3,I)                  +  &
                          STRECH(2,I)*ROTATE(2,3,I)                  +  &
                          STRECH(5,I)*ROTATE(3,3,I) 

        def_grad(3,1,I) = STRECH(6,I)*ROTATE(1,1,I)                  +  &
                          STRECH(5,I)*ROTATE(2,1,I)                  +  &
                          STRECH(3,I)*ROTATE(3,1,I) 

        def_grad(3,2,I) = STRECH(6,I)*ROTATE(1,2,I)                  +  &
                          STRECH(5,I)*ROTATE(2,2,I)                  +  &
                          STRECH(3,I)*ROTATE(3,2,I)     

        def_grad(3,3,I) = STRECH(6,I)*ROTATE(1,3,I)                  +  &
                          STRECH(5,I)*ROTATE(2,3,I)                  +  &
                          STRECH(3,I)*ROTATE(3,3,I) 
        DO J=1,3
        DO K=1,3
        IF (ABS(def_grad(J,K,I)).le.ZERONUM) THEN
                def_grad(J,K,I) = 0.0
        END IF
        END DO
        END DO
        
!************************************************************************
!                Linear interpolation of the deformation gradient
!                (needed for subincremented deformation gradient)
!
!                Sets direction of deformation increment
!
!                DF/Dt = [ F(t+Dt) - F(t) ]/Dt
!
!                DF_DT matrix  DF_DT(1) DF_DT(2) DF_DT(3)
!                              DF_DT(4) DF_DT(5) DF_DT(6)
!                              DF_DT(7) DF_DT(8) DF_DT(9)
!************************************************************************
        DF_DT(1,I) = ( def_grad(1,1,I) - SV(19,I) )/ DT
        DF_DT(2,I) = ( def_grad(1,2,I) - SV(20,I) )/ DT
        DF_DT(3,I) = ( def_grad(1,3,I) - SV(21,I) )/ DT
        DF_DT(4,I) = ( def_grad(2,1,I) - SV(22,I) )/ DT
        DF_DT(5,I) = ( def_grad(2,2,I) - SV(23,I) )/ DT
        DF_DT(6,I) = ( def_grad(2,3,I) - SV(24,I) )/ DT
        DF_DT(7,I) = ( def_grad(3,1,I) - SV(25,I) )/ DT
        DF_DT(8,I) = ( def_grad(3,2,I) - SV(26,I) )/ DT
        DF_DT(9,I) = ( def_grad(3,3,I) - SV(27,I) )/ DT
!     END LOOP OVER ALL THE ELEMENTS 	              
      END DO 
!************************************************************************
!                        Subincrementation parameters
!************************************************************************
      
      time_step  = DT
      sub_start  = 0.0
      sub_end    = 0.0
      update     = .true.
      num_sub_in = 0
      max_sub    = 15000 
      
! ***********************************************************************
!                        Subincrementation Loop
!        There are two loop contained within subincrementation loop
!                         1 - Update Stress Loop
!                         2 - Hardening/Stability Loop
! ***********************************************************************
      DO WHILE( (sub_end.lt.DT).and.(num_sub_in.lt.max_sub) )
!
! Number of sub-incrementation
!
      num_sub_in = num_sub_in + 1

!***********************************************************************
!                       Time step acceleration scheme  
!***********************************************************************
          IF( (update).and.(sub_start.ne.0) )THEN
          sub_start = sub_end
          ck_accel  = sub_end/DT
          check     = int(1/ck_accel)
          check     = mod(check,2)
          
          IF( check.eq.0 )THEN
            time_step = sub_end
          END IF
          
            sub_end   = sub_start + time_step
          
          IF( sub_end.gt.DT )THEN
            sub_end   = DT
            time_step = sub_end - sub_start
          END IF
          
          ELSE IF( (update).and.(sub_start.eq.0) )THEN
            sub_start = sub_end
            sub_end   = sub_start + time_step
          ELSE
            sub_end   = sub_start + time_step
          END IF
!1111111111111111111111111111111111111111111111111111111111111111111111111
!                     1 - Update Stress Loop
!         UPDATE STRESS LOOP OVER ALL THE ELEMENTS
        DO I=1,NEL
!                    
! Elastic deformation gradient
!        
          Fel(1,1)    = SV(1,I)
          Fel(1,2)    = SV(2,I)
          Fel(1,3)    = SV(3,I)
          Fel(2,1)    = SV(4,I)
          Fel(2,2)    = SV(5,I)
          Fel(2,3)    = SV(6,I)
          Fel(3,1)    = SV(7,I)
          Fel(3,2)    = SV(8,I)
          Fel(3,3)    = SV(9,I)
!
! Plastic deformation gradient
!
          Fplast(1,1) = SV(10,I)
          Fplast(1,2) = SV(11,I)
          Fplast(1,3) = SV(12,I)
          Fplast(2,1) = SV(13,I)
          Fplast(2,2) = SV(14,I)
          Fplast(2,3) = SV(15,I)
          Fplast(3,1) = SV(16,I)
          Fplast(3,2) = SV(17,I)
          Fplast(3,3) = SV(18,I)
!
! Initial element orientation
!
          Rinit(1,1)  = SV(37,I)
          Rinit(1,2)  = SV(38,I)
          Rinit(1,3)  = SV(39,I)
          Rinit(2,1)  = SV(40,I)
          Rinit(2,2)  = SV(41,I)
          Rinit(2,3)  = SV(42,I)
          Rinit(3,1)  = SV(43,I)
          Rinit(3,2)  = SV(44,I)
          Rinit(3,3)  = SV(45,I)
          
!************************************************************************
! Rotate each slip plane and slip direction into the proper orientation
!************************************************************************
       DO J=1, 3
           DO K=1, 12
            vect_d(K,J) = Rinit(J,1)*sd(K,1) + Rinit(J,2)*sd(K,2)        +  &
                          Rinit(J,3)*sd(K,3)
            vect_n(K,J) = Rinit(J,1)*sn(K,1)+ Rinit(J,2)*sn(K,2)       +  &
                          Rinit(J,3)*sn(K,3)
           END DO                            
       END DO      

! Elastic stiffness tensor for element "I"
!          
          call stiffness(Celast,var_stiff,Rinit)
!*************************************************************************
!         Linearly interoplate the deformation gradient
!
!         F(t+dt) = F(t) + DF/DT x Dt 
!*************************************************************************
          F_tDt(1,1)  = SV(19,I) + DF_DT(1,I)*(sub_end)
          F_tDt(1,2)  = SV(20,I) + DF_DT(2,I)*(sub_end)
          F_tDt(1,3)  = SV(21,I) + DF_DT(3,I)*(sub_end)
          F_tDt(2,1)  = SV(22,I) + DF_DT(4,I)*(sub_end)
          F_tDt(2,2)  = SV(23,I) + DF_DT(5,I)*(sub_end)
          F_tDt(2,3)  = SV(24,I) + DF_DT(6,I)*(sub_end)
          F_tDt(3,1)  = SV(25,I) + DF_DT(7,I)*(sub_end)
          F_tDt(3,2)  = SV(26,I) + DF_DT(8,I)*(sub_end)
          F_tDt(3,3)  = SV(27,I) + DF_DT(9,I)*(sub_end)
!************************************************************************
!         Push the slip normals forward to current config
!
!         N(i) =n(j) x Fe(j,i)^-1 for 4 normal vectors
!************************************************************************
          call push_forwN(curr_n,vect_n,Fel)
!************************************************************************
!         Push the slip directions forward to current config 
!
!         D(i) = Fe(i,j) x d(j) for 6 direction vectors
!************************************************************************
         call push_forwD(curr_d,vect_d,Fel)
!************************************************************************
!         Calculate the Symmetric part of Schmid Tensor
!         Sym( d X n) = 0.5[ d X n + n X d  ]
!         For 12 slip systems
!************************************************************************
          call Symm_Schmid(Ps,curr_d,curr_n)

          IF (C_1+C_2+C_3+C_4+C_5.ne.0) then 
           call Non_Schmid(Ptot,Pns,Ps,curr_d,curr_n,bcc_var)
          ENDIF
!************************************************************************
!         Project Cauchy stress onto each slip systemg          
!         at time "t" in current configuration
!
!         tau(a)= cauchy(i,j) x [ curr_d(i) ; curr_n(j)  ]_Sym
!************************************************************************
         DO J=1, 12
                RSS(J,I) = Ps(J,1,1)*cauchy(1,I)              +  &
                           Ps(J,2,2)*cauchy(2,I)              +  &
                           Ps(J,3,3)*cauchy(3,I)              +  &
                     2.0*( Ps(J,1,2)*cauchy(4,I)              +  &
                           Ps(J,2,3)*cauchy(5,I)              +  &
                           Ps(J,1,3)*cauchy(6,I) )
                TS_NS(J,I)=0

               IF (C_1+C_2+C_3+C_4+C_5.ne.0) THEN          
                TS_NS(J,I) =Pns(J,1,1)*cauchy(1,I)           +  &
                            Pns(J,2,2)*cauchy(2,I)           +  &
                            Pns(J,3,3)*cauchy(3,I)           +  &
                      2.0*( Pns(J,1,2)*cauchy(4,I)           +  &
                            Pns(J,2,3)*cauchy(5,I)           +  &
                            Pns(J,1,3)*cauchy(6,I) )
               ENDIF


!************************************************************************
!         Calculate slip rate for each slip system from flow rule
!************************************************************************
!
! Assign Resolved Shear Stress (RSS) 
!
            var_slip(14+J) = RSS(J,I)
!
! Assign Critical Resolved Shear Stress (CRSS)
!
            var_slip(26+J) = SV(75+J,I)
!
! Assign dislocation content
!
            var_slip(38+J) = SV(87+J,I)

! Assign Non-Schmid Stress (TS_NS)
!
            var_slip(62+J) = TS_NS(J,I)

! Assign average slip rate from the previous time step

            var_slip(50+J)=SV(63+J,I)/TIME  

          END DO          

!************************************************************************
! Evaluate slip rate
!
          call Slip_Kinem(res_slip,var_slip,Flag_Slip)

!************************************************************************
! Assign number of active slip systems
!          
          active_syst(I) = res_slip(13)
          
          DO J=1,12
!
! Assign slip rate on each slip system
!
             slip_rate(J,I) = res_slip(J)


!************************************************************************
!         Integrate the slip rate on each slip system
!
!         y = y(t) + y_dot(t+Dt)xDt
!         for 12 slip systems
!************************************************************************

             cumul_slip(J,I) = SV(63+J,I)+time_step*slip_rate(J,I)
           
         END DO

!************************************************************************
!        Calculate the intermediate config. plastic velocity gradient
!
!   Plast_vel_grad = sum (a=1..12) [ y_a ( n_intermed X d_intermed ) ]
!************************************************************************
!
! Calculate Schmid tensor
!
          call Schmid_tensor(Schmid_int,vect_d,vect_n)
!
! Calculate plastic velocity gradient
!
          DO J=1,3
            DO K=1,3
              Vplast(J,K)=slip_rate( 1,I)*Schmid_int( 1,J,K)         +  &
                          slip_rate( 2,I)*Schmid_int( 2,J,K)         +  &
                          slip_rate( 3,I)*Schmid_int( 3,J,K)         +  &
                          slip_rate( 4,I)*Schmid_int( 4,J,K)         +  &
                          slip_rate( 5,I)*Schmid_int( 5,J,K)         +  &
                          slip_rate( 6,I)*Schmid_int( 6,J,K)         +  &
                          slip_rate( 7,I)*Schmid_int( 7,J,K)         +  &
                          slip_rate( 8,I)*Schmid_int( 8,J,K)         +  &
                          slip_rate( 9,I)*Schmid_int( 9,J,K)         +  &
                          slip_rate(10,I)*Schmid_int(10,J,K)         +  &
                          slip_rate(11,I)*Schmid_int(11,J,K)         +  &
                          slip_rate(12,I)*Schmid_int(12,J,K)         
            END DO
          END DO
!************************************************************************
!               Find plastic gradient increment for next step:          
!   The numerical integration over the time step done using expon. fnc.
!
!   Increment found by integrating dF_plast = plast_Vel_grad x F_plast
!
!      --> F_plast(t+Dt) = exp{ plast_Vel_grad(t) x Dt } x F_plast(t)
!************************************************************************
          call calc_plast_grad(Fplast_nxt,Fplast,Vplast,time_step)

          DO K=1,3
            DO L=1,3
              fp_nxt(K,L,I) = Fplast_nxt(K,L)
            END DO
         END DO

!************************************************************************
!        Calculate effective plastic strain
!************************************************************************
!
! Plastic Strain
!          
          call cauchygreen_strn(Eplast,Fplast_nxt)
!
! Effective Plastic strain : Eeff_plast = [2/3*(Eplast:Eplast)]^(1/2)                            
!
          Eeff_plast_nxt(I) = SQRT( 2.0*(Eplast(1)*Eplast(1)         +  &
                                         Eplast(2)*Eplast(2)         +  &
                                         Eplast(3)*Eplast(3)         +  &
                                    2.0*(Eplast(4)*Eplast(4)         +  &
                                         Eplast(5)*Eplast(5)         +  &
                                         Eplast(6)*Eplast(6)))/3.0 )

!************************************************************************
!       Calculate the elastic deformation gradient at time t + dt
!
!               Fe(t+Dt) = F(t+Dt).[Fp(t+Dt)]^(-1)
!************************************************************************
          call calc_elast_grad(Fel_nxt,F_tDt,Fplast_nxt)
          
          DO K=1,3
            DO L=1,3
               fe_nxt(K,L,I) = Fel_nxt(K,L)
            END DO
          END DO
!************************************************************************
!       Calculate the Green Elastic strain tensor in the int. config.
!
!         E_CauchyGreen(i,j)=0.5[ (F_e(t+Dt))^T.F_e(t+Dt) - I ]
!
!       Cauchy Green matrix  E_CG(1) E_CG(4) E_CG(6)
!                            E_CG(4) E_CG(2) E_CG(5)
!                            E_CG(6) E_CG(5) E_CG(3)
!************************************************************************
          call cauchygreen_strn(E_CauchyGreen,Fel_nxt)

!************************************************************************
!       Calculate the 2nd Piola-Kirchoff Stress in the int. config.
!               sig_pk2=C(i,j,k,l):E_CauchyGreen(k,l)
!
! sig_pk2 matrix  = sig_pk2(1)   sig_pk2(4)   sig_pk2(6)
!                   sig_pk2(4)   sig_pk2(2)   sig_pk2(5)
!                   sig_pk2(6)   sig_pk2(5)   sig_pk2(3)  
!************************************************************************
          call Elasticity(sig_pk2,Celast,E_CauchyGreen)
!************************************************************************
!               Move the 2nd P-K stress to the current config. 
!               and convert to Cauchy stress
!
!    cauchy(i,j)=1/J(Fe(t+Dt) [ F_e(t+Dt).Sig_pk2(t+Dt).(F_e(t+Dt))^T ]
!
!       cauchy matrix  = cauchy(1)   cauchy(4)   cauchy(6)
!                        cauchy(4)   cauchy(2)   cauchy(5)
!                        cauchy(6)   cauchy(5)   cauchy(3)
!************************************************************************
          call pk2tocauchy(sig_cau,sig_pk2,Fel_nxt)

          DO J=1,6

          cauchy_nxt(J,I) = sig_cau(J)
          
          END DO

!************************************************************************
! Hardening law
!************************************************************************
        var_harden(13) = Eeff_plast_nxt(I)

        DO J=1,12
        var_harden(13+J) = SV(75+J,I) !CRSS              
        var_harden(37+J) = SV(87+J,I)!DD
        var_harden(25+J) = ABS(cumul_slip(J,I)-SV(63+J,I)) !delta slip
        END DO

          call Harden_law(res_harden,var_harden,Flag_Harden,Flag_Sys,sd,sn,res_slip) 

        DO J=1,12
             CRSS_nxt(J,I) = res_harden(J)          
             ssd_nxt(J,I)  = res_harden(12+J)
        END DO          
!
    END DO

!         END THE UPDATE STRESS LOOP OVER ALL THE ELEMENTS
!111111111111111111111111111111111111111111111111111111111111111111111111
          
!222222222222222222222222222222222222222222222222222222222222222222222222          
!                    2 - Stability Loop Parameters
!                 STABILITY CHECK FOR ALL ELEMENTS
!
          subinc = .false.
          update = .true.
          I = 1
          
          DO WHILE((subinc.eqv..false.).and.(I.le.NEL)) 
!************************************************************************
!                          STABILITY CHECK                  
!              check slip rates at t + dt to see if the    
!              Forward integration scheme will be stable.  
!************************************************************************
!
!************************************************************************
!           Push the slip normals and slip directions  
!           forward to current config with updated Fe
!************************************************************************
!
          Rinit(1,1)  = SV(37,I)
          Rinit(1,2)  = SV(38,I)
          Rinit(1,3)  = SV(39,I)
          Rinit(2,1)  = SV(40,I)
          Rinit(2,2)  = SV(41,I)
          Rinit(2,3)  = SV(42,I)
          Rinit(3,1)  = SV(43,I)
          Rinit(3,2)  = SV(44,I)
          Rinit(3,3)  = SV(45,I)

!************************************************************************
! Rotate each slip plane and slip direction into the proper orientation
!************************************************************************
          DO J=1, 3
!
! Slip directions
!
          DO K=1,12
            vect_d(K,J) = Rinit(J,1)*sd(K,1) +  Rinit(J,2)*sd(K,2)        +  &
                          Rinit(J,3)*sd(K,3)
          ENDDO
! Slip normals
!                          
          DO K=1,12
            vect_n(k,J) = Rinit(J,1)*sn(K,1) + Rinit(J,2)*sn(K,2)         +  &
                          Rinit(J,3)*sn(K,3)
          ENDDO
                             
          END DO      
!
          DO K=1,3
            DO L=1,3
               Fel_nxt(K,L) = fe_nxt(K,L,I)
            END DO
          END DO          
!
! Push forward normal directions
!          
          call push_forwN(ck_curr_n,vect_n,Fel_nxt)
!
! Push forward plane directions          
!          
          call push_forwD(ck_curr_d,vect_d,Fel_nxt)
!************************************************************************
!               Calculate the Symmetric part of Schmid Tensor
!               with updated directions and normals
!                       Sym( d X n) = 0.5[ d X n + n X d  ]
!                       For 12 slip systems
!************************************************************************
          call Symm_Schmid(ck_Ps,ck_curr_d,ck_curr_n)

          IF (C_1+C_2+C_3+C_4+C_5.ne.0) then 
          call Non_Schmid(ck_Ptot,ck_Pns,ck_Ps,ck_curr_d,ck_curr_n,bcc_var)
          ENDIF

!************************************************************************
!               Project Cauchy stress onto each slip systemg          
!               at time "t+Dt" in current configuration
!
!               tau(a)= cauchy(i,j) x [ curr_d(i) x curr_n(j)]_Sym
!************************************************************************

          DO J=1, 12
              ck_RSS(J) =  ck_Ps(J,1,1)*cauchy_nxt(1,I)             +  &
                           ck_Ps(J,2,2)*cauchy_nxt(2,I)             +  &
                           ck_Ps(J,3,3)*cauchy_nxt(3,I)             +  &
                     2.0*( ck_Ps(J,1,2)*cauchy_nxt(4,I)             +  &
                           ck_Ps(J,2,3)*cauchy_nxt(5,I)             +  &
                           ck_Ps(J,1,3)*cauchy_nxt(6,I) )
              ck_NS(J)=0

            IF (C_1+C_2+C_3+C_4+C_5.ne.0) THEN  
              ck_NS(J) =   ck_Pns(J,1,1)*cauchy_nxt(1,I)            +  &
                           ck_Pns(J,2,2)*cauchy_nxt(2,I)            +  &
                           ck_Pns(J,3,3)*cauchy_nxt(3,I)            +  &
                     2.0*( ck_Pns(J,1,2)*cauchy_nxt(4,I)            +  &
                           ck_Pns(J,2,3)*cauchy_nxt(5,I)            +  &
                           ck_Pns(J,1,3)*cauchy_nxt(6,I) )
            ENDIF
!
! Assign Resolved Shear Stress (RSS) for stability check 
!
              ck_var_slip(14+J) = ck_RSS(J)
!
! Assign Critical Resolved Shear Stress (CRSS) for stability check
!
              ck_var_slip(26+J) = CRSS_nxt(J,I)
              
! Assign dislocation content (DD) for stability check
!
              ck_var_slip(38+J) = ssd_nxt(J,I)

! Assign Non-Schmid Stress (TS_NS) for stability check
!              
              ck_var_slip(62+j)= ck_NS(J)

! Assign average slip rate from the previous time step for stability check
              
              cK_var_slip(50+j) = SV(63+J,I)/TIME

          END DO

! Evaluate slip rate for stability check

          call Slip_Kinem(ck_slip,ck_var_slip,Flag_Slip) 

!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Stability criterion:
! Check if strain rate calculated from linear predictor and strain rate
! calculated from update stress/flow rule are comparable...by looking at
! error in a (un)weighted level function
!
! ERR =(1/Nsys) SQRT( [( y(t+Dt)/ymax )*f ]^2 )
! f   = y_predictor(t+DT) - y0*ABS(RSS(t+Dt)/CRSS(t+DT))*sgn(RSS(t+Dt)          
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!         
          DO J=1,12
!
             slip(J)     = slip_rate(J,I)
             dRSS(J)     = RSS(J,I)
             dRSS(J+12)  = ck_RSS(J)
             dCRSS(J)    = SV(75+J,I)
             dCRSS(12+J) = CRSS_nxt(J,I)
             dImm(J)     = SV(87+J,I)
             dImm(J+12)  = ssd_nxt(J,I)
             dMob(J)     = 2.0D0*SV(87+J,I)
             dMob(J+12)  = 2.0D0*ssd_nxt(J,I)
             TNS(J)      = TS_NS(J,I)
             TNS(J+12)   = ck_NS(J)

          END DO
!               
             Flow(1)     = GD0
             Flow(2)     = XP
             Flow(3)     = XQ
             Flow(4)     = active_slip
             Flow(5)     = H7 !temperature
             Flow(6)     = H8 !Qeffe
             Flow(7)     = H10
             Flow(8)     = T_cr
             Flow(9)     = H3
             Flow(10)    = H4
             Flow(11)    = H5
             Flow(12)    = H6
!
! Taylor predictor of strain rate at time t+Dt
!             
          call predict_rate(slip_1,slip,dRSS,dCRSS,dImm,dMob,Flow,  &
                            Flag_Slip,TNS) 
!

          DO J=1,12
!         
             sta_slip(J)      = slip_1(J) 
             sta_slip(J+12)   = ck_slip(J)

          END DO

               
          call stable_inc2(subinc,sta_slip,active_slip)
          
! Increment element counter
!                                                                        
            I = I+1
!            
            END DO
!       END OF STABILITY CHECK FOR ALL ELEMENTS

!222222222222222222222222222222222222222222222222222222222222222222222222          
          
!333333333333333333333333333333333333333333333333333333333333333333333333
!                     3- UPDATE VARIABLES IF CONVERGENCE
!
!                         Subincrementation logic:
!          If the slip step magnitude of any of the slip systems
!          is out of tolerance, divide the time step in half (dichotomy)
!          and don't update any of the variables.
!          Otherwise, continue and update variables.    

!
! If time step unstable: Dt = Dt/2
!
            IF( subinc )THEN

            time_step = time_step/2.0
            update = .false.
            sub_end = sub_start
!
! If time step stabe
!            
            ELSE
              update = .true.
            END IF
!UPDATE_UPDATE_UPDATE_UPDATE_UPDATE_UPDATE_UPDATE_UPDATE_UPDATE_UPDATE
!            
!************************************************************************
! If the time step is stable: Update the values for the next subincrement
!************************************************************************
          IF( update )THEN
            DO I=1,NEL
!
! Update Elastic Deformation Gradient
!
                SV( 1,I) = fe_nxt(1,1,I)
                SV( 2,I) = fe_nxt(1,2,I)  
                SV( 3,I) = fe_nxt(1,3,I)
                SV( 4,I) = fe_nxt(2,1,I)
                SV( 5,I) = fe_nxt(2,2,I)
                SV( 6,I) = fe_nxt(2,3,I)
                SV( 7,I) = fe_nxt(3,1,I)
                SV( 8,I) = fe_nxt(3,2,I)
                SV( 9,I) = fe_nxt(3,3,I)
!
! Update Plastic Deformation Gradient
! 
                SV(10,I) = fp_nxt(1,1,I)
                SV(11,I) = fp_nxt(1,2,I)
                SV(12,I) = fp_nxt(1,3,I)
                SV(13,I) = fp_nxt(2,1,I)
                SV(14,I) = fp_nxt(2,2,I)
                SV(15,I) = fp_nxt(2,3,I)
                SV(16,I) = fp_nxt(3,1,I)
                SV(17,I) = fp_nxt(3,2,I)
                SV(18,I) = fp_nxt(3,3,I)
!
! Update effective plastic strain
!
                SV(56,I) = Eeff_plast_nxt(I)
!
               DO J=1,12
!
! Update cumulative plastic strain / DD / CRSS
              
                SV(63+J,I) = cumul_slip(J,I)
                SV(87+J,I) = ssd_nxt(J,I)
                SV(75+J,I) = CRSS_nxt(J,I)
  
               END DO
!
! Update cauchy stress
!              
              DO J=1,6
                cauchy(J,I) = cauchy_nxt(J,I)
              END DO
!
! Update number of active slip systems
!
                SV(62,I) = active_syst(I)
                            
!  
!END_UPDATE_END_UPDATE_END_UPDATE_END_UPDATE_END_UPDATE_END_UPDATE_END
              END DO  
          END IF
!
!END OF BIG LOOP
        END DO         
        
!************************************************************************
!              Error Message if material model can not converge
!************************************************************************
        IF (sub_end .ne. DT) THEN
           WRITE (6,*) 'System does not converge...'
           WRITE (6,*) 'DT          : ', DT 
           WRITE (6,*) 'sub_end     : ', sub_end
           WRITE (6,*) 'num_sub_inc : ', num_sub_in
           WRITE (6,*) 'time step = :', time_step
           WRITE (6,*) 'Element     :',I
           WRITE (6,*) 'Time        :',Time
        END IF
!************************************************************************
! Once subincremetation is completed additional SV's need to be updated
!************************************************************************
        IF (sub_end .eq. DT) THEN
          DO I=1,NEL
!
! Update deformation gradient
!          
            SV(19,I) = def_grad(1,1,I)
            SV(20,I) = def_grad(1,2,I)
            SV(21,I) = def_grad(1,3,I)
            SV(22,I) = def_grad(2,1,I)
            SV(23,I) = def_grad(2,2,I)
            SV(24,I) = def_grad(2,3,I)
            SV(25,I) = def_grad(3,1,I)
            SV(26,I) = def_grad(3,2,I)
            SV(27,I) = def_grad(3,3,I)
!
! Update average SSD, GND, CRSS
!
            SV(58,I)=0
            SV(59,I)=0
            SV(60,I)=0

          DO J=1,12
            SV(58,I)=SV(58,I)+SV(63+J,I) !AVG. SLIP
            SV(59,I)=SV(59,I)+SV(87+J,I) !AVG. SSD
            SV(60,I)=SV(60,I)+SV(75+J,I) !AVG. CRSS
          ENDDO

            SV(58,I)=SV(58,I)/12
            SV(59,I)=SV(59,I)/12
            SV(60,I)=SV(60,I)/12

!************************************************************************
! For JAS Only - Rotate Stress from current config to unrotated config 
!
!       sig(i,j) = Rot^T.cauchy.Rot
!            
!************************************************************************
            DO K=1,3
              DO L=1,3

              Rot(K,L) = ROTATE(K,L,I)
              
              END DO
            END DO 

            DO J=1,6

              sig_cau(J) = cauchy_nxt(J,I)
            
            END DO
            
            call cauchytosig(sig_unrot,sig_cau,Rot)

            DO J=1,6

            SIG(J,I) = sig_unrot(J)

            END DO
!************************************************************************
! Calculate elastic rotation
!************************************************************************
!
! Polar decomposition
!             
            DO K=1,3
              DO L=1,3
                                
              Fel_nxt(K,L) = fe_nxt(K,L,I)
            
              END DO
            END DO
!            
            call polar_decomp(re_nxt,Fel_nxt)
!
! Rotate the unit normals to the current configuration
!
            SV(28,I) = re_nxt(1,1)*SV(37,I) + re_nxt(1,2)*SV(40,I)   +  &
                       re_nxt(1,3)*SV(43,I)                    
            SV(29,I) = re_nxt(1,1)*SV(38,I) + re_nxt(1,2)*SV(41,I)   +  &
                       re_nxt(1,3)*SV(44,I)
            SV(30,I) = re_nxt(1,1)*SV(39,I) + re_nxt(1,2)*SV(42,I)   +  &
                       re_nxt(1,3)*SV(45,I)
            SV(31,I) = re_nxt(2,1)*SV(37,I) + re_nxt(2,2)*SV(40,I)   +  &
                       re_nxt(2,3)*SV(43,I)
            SV(32,I) = re_nxt(2,1)*SV(38,I) + re_nxt(2,2)*SV(41,I)   +  &
                       re_nxt(2,3)*SV(44,I)
            SV(33,I) = re_nxt(2,1)*SV(39,I) + re_nxt(2,2)*SV(42,I)   +  &
                       re_nxt(2,3)*SV(45,I)
            SV(34,I) = re_nxt(3,1)*SV(37,I) + re_nxt(3,2)*SV(40,I)   +  &
                       re_nxt(3,3)*SV(43,I)
            SV(35,I) = re_nxt(3,1)*SV(38,I) + re_nxt(3,2)*SV(41,I)   +  &
                       re_nxt(3,3)*SV(44,I)
            SV(36,I) = re_nxt(3,1)*SV(39,I) + re_nxt(3,2)*SV(42,I)   +  &
                       re_nxt(3,3)*SV(45,I)     

!************************************************************************
! Calculate  misorientation
!************************************************************************
!
! Elastic rotation
!
            Rel(1,1) = SV(28,I)
            Rel(1,2) = SV(29,I)
            Rel(1,3) = SV(30,I)
            Rel(2,1) = SV(31,I)
            Rel(2,2) = SV(32,I)
            Rel(2,3) = SV(33,I)
            Rel(3,1) = SV(34,I)
            Rel(3,2) = SV(35,I)
            Rel(3,3) = SV(36,I)
!
! Initial orientation of element
!
            Rinit(1,1) = SV(37,I)
            Rinit(1,2) = SV(38,I)
            Rinit(1,3) = SV(39,I)
            Rinit(2,1) = SV(40,I)
            Rinit(2,2) = SV(41,I)
            Rinit(2,3) = SV(42,I)
            Rinit(3,1) = SV(43,I)
            Rinit(3,2) = SV(44,I)
            Rinit(3,3) = SV(45,I)
!
! Crystallographic orientation tensor.
 
        SV(46,I) = (ROTATE(1,1,I)*Rel(1,1)) + &
                   (ROTATE(2,1,I)*Rel(2,1)) + &
                   (ROTATE(3,1,I)*Rel(3,1))
        SV(47,I) = (ROTATE(1,1,I)*Rel(1,2)) + &
                   (ROTATE(2,1,I)*Rel(2,2)) + &
                   (ROTATE(3,1,I)*Rel(3,2))
        SV(48,I) = (ROTATE(1,1,I)*Rel(1,3)) + &
                   (ROTATE(2,1,I)*Rel(2,3)) + &
                   (ROTATE(3,1,I)*Rel(3,3))
        SV(49,I) = (ROTATE(1,2,I)*Rel(1,1)) + &
                   (ROTATE(2,2,I)*Rel(2,1)) + &
                   (ROTATE(3,2,I)*Rel(3,1))
        SV(50,I) = (ROTATE(1,2,I)*Rel(1,2)) + &
                   (ROTATE(2,2,I)*Rel(2,2)) + &
                   (ROTATE(3,2,I)*Rel(3,2))
        SV(51,I) = (ROTATE(1,2,I)*Rel(1,3)) + &
                   (ROTATE(2,2,I)*Rel(2,3)) + &
                   (ROTATE(3,2,I)*Rel(3,3))
        SV(52,I) = (ROTATE(1,3,I)*Rel(1,1)) + &
                   (ROTATE(2,3,I)*Rel(2,1)) + &
                   (ROTATE(3,3,I)*Rel(3,1))
        SV(53,I) = (ROTATE(1,3,I)*Rel(1,2)) + &
                   (ROTATE(2,3,I)*Rel(2,2)) + &
                   (ROTATE(3,3,I)*Rel(3,2))
        SV(54,I) = (ROTATE(1,3,I)*Rel(1,3)) + &
                   (ROTATE(2,3,I)*Rel(2,3)) + &
                   (ROTATE(3,3,I)*Rel(3,3))

        call misorient(phi_mis,Rinit,Rel)
            
            SV(55,I) = phi_mis

            END DO
        END IF
!                      END OF UPDATE VARIABLES IF CONVERGENCE
!333333333333333333333333333333333333333333333333333333333333333333333333
!        END DO
!
        IF( IQCM.NE.0 ) THEN
          DO I=1, NEL
!************************************************************************
! Elastic Stiffness Matrix at End of the Step     
!************************************************************************
! Tangent Modulus X DT matrix is stored as
!       CM(1)    CM(2)    CM(6)    CM(8)    CM(11)   CM(16)
!       CM(2)    CM(3)    CM(7)    CM(9)    CM(12)   CM(17)
!       CM(6)    CM(7)    CM(5)    CM(10)   CM(13)   CM(18)
!       CM(8)    CM(9)    CM(10)   CM(4)    CM(14)   CM(19)
!       CM(11)   CM(12)   CM(13)   CM(14)   CM(15)   CM(20)
!       CM(16)   CM(17)   CM(18)   CM(19)   CM(20)   CM(21)
          
             CM( 1,I) =       PROP( 1)*DT
             CM( 2,I) = 2.0 * PROP( 1)*DT
             CM( 3,I) =       PROP( 2)*DT
             CM( 4,I) =       PROP( 3)*DT
             CM( 5,I) =       PROP( 3)*DT
             CM( 6,I) = 2.0 * PROP( 2)*DT
             CM( 7,I) = 2.0 * PROP( 3)*DT
             CM( 8,I) = 2.0 * PROP( 2)*DT
             CM( 9,I) = 2.0 * PROP( 3)*DT
             CM(10,I) = 2.0 * PROP( 3)*DT
             CM(11,I) = 2.0 * PROP( 3)*DT
             CM(12,I) = 2.0 * PROP( 1)*DT
             CM(13,I) = 2.0 * PROP( 1)*DT
             CM(14,I) = 2.0 * PROP( 2)*DT
             CM(15,I) =       PROP( 1)*DT
             CM(16,I) = 2.0 * PROP( 1)*DT
             CM(17,I) = 2.0 * PROP( 2)*DT
             CM(18,I) = 2.0 * PROP( 2)*DT
             CM(19,I) = 2.0 * PROP( 3)*DT
             CM(20,I) = 2.0 * PROP( 1)*DT
             CM(21,I) =       PROP( 2)*DT

          END DO
       END IF

      END
