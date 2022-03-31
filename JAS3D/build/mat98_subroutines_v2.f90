! Hojun Lim's Local Crystal Plasticity Model based on Dingreville's Formulation (MAT86)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Slip_Kinem
! Functionality: Defines the Slip Kinematic / Flow rule
!
! Input(s):      var_slip  : Slip Kinematics variables
!                   (     1)  = Initial Flow rate    - GD0
!                   (     2)  = Flow rule param. 1   - XP
!                   (     3)  = Flow rule param. 2   - XQ
!                   (  4-27)  = Resolv. Shear        - RSS
!                   ( 28-51)  = Crit. Resolv. Shear  - CRSS       
!                   ( 52-75)  =                      - GND
!                   ( 76-99)  = [Empty]              
!                   (   100)  = Active rate          - Active_slip             
!                   (   101)  = H10                  - Burgers Vector   
!                   (   102)  = [Empty]
!                   (   103)  = [Empty]
!                   (104-127) = Tns
!                   (    128) = T_cr
!
!                 
!                Flag_slip : Determines which flow rule to use
!                          1- Power Law Flow rule 
!                             y= y0 (Resolv_shear/CRSS )^Xp
!
! Output(s):     res_slip  : Slip Kinematics results
!                  (   1-12) = Slip rate       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       subroutine Slip_Kinem(res_slip,var_slip,Flag_slip) 

       double precision res_slip(13), var_slip(74),T_cr,Tcr1,Tcr2
       double precision RSS(12), CRSS(12), slip_rate(12)
       double precision active_slip, GD0, XP, XQ, Qeff, Temp, Boltzk,  &
                        AF, Burgers, TK, c1, c2, Em10,VA,WA,E6,Em20,AF1,AF2
       double precision Vdisloc,rho_mob(12),rho_imm(12),TS_NS(12),TS_fric
       double precision SIG_EI0,SIG_LT0,SIG_EI,SIG_LT,HK,EDOT0
       integer I, J, K, L, L0, Flag_slip

            Em10 = 1.0E-10
            Em20 = 1.0E-20
            E6   = 1.0E6

! Initialize variables
       
        DO L0=1,12
        res_slip(L0)  = 0.0D0
        slip_rate(L0) = 0.0D0
        RSS(L0)       = var_slip(14+L0)
        CRSS(L0)      = var_slip(26+L0)
        rho_imm(L0)   = var_slip(38+L0)       
        TS_NS(L0)     = var_slip(62+L0)
        END DO
 
! Active slip rate
            active_slip    = var_slip(14)
!
! Number of active slip systems
            active = 0.0D0
            
!************************************************************************
! Flag_slip = 1  --> Power law for slip
!                    y_dot={ y0*[tau_RSS/tau_CRSS]^Xp } Sgn(tau_RSS)
!************************************************************************
          IF ( Flag_slip .eq. 1) THEN

            GD0 = var_slip(1)
            XP  = var_slip(2) 
       
            DO J=1,12
        
        TS_fric = T_cr-TS_NS(J)
       IF (TS_fric .lt. 0.0) then 
        TS_fric = 0.0
        endif 

             slip_rate(J)=GD0*( ABS( RSS(J)/(CRSS(J)+TS_fric) ))**XP

             slip_rate(J)=Sign(slip_rate(J),RSS(J))

! Active slip threshold
!              
              IF ( ABS(slip_rate(J)) .le. active_slip ) THEN
              slip_rate(J) = 0.0D0
                 active = active
              END IF
              
              IF ( ABS(slip_rate(J)) .gt. active_slip ) THEN
                 active = active + 1.0D0
              END IF
               
            END DO

            END IF

!
! Assign results from slip kinematics
!            
            DO L=1,12

            res_slip(L) = slip_rate(L)
            ENDDO

            res_slip(13) = active


!************************************************************************
! Flag_slip = 2 --> Thermally activated dislocation motion 
!                   y_dot=y0*exp{ (-DF/kT)x[1-(tau_RSS/tau_CRSS)] }
!                           *{ [tau_RSS/tau_CRSS]^Xp }*Sgn(tau_RSS)
!************************************************************************
          IF ( Flag_slip .eq. 2) THEN

            GD0 = var_slip(1)
            XP  = var_slip(2)
!
! deltF in (10^-20 J)
!            
            Qeff = var_slip(11)
!
! Temperature (in K)
!            
            Temp  = var_slip(10)
!
! Boltzmann constant (in 10-^20 J.K^-1)
!            
!            Boltzk = 1.3806503D0*0.001D0
             Boltzk = 8.62e-5
!
! Multiplicative factor in expon:  DF/(k*T)
!
            AF  = (-1.0D0)*Qeff/(Boltzk*Temp)
            
            DO J=1,12
            
         slip_rate(J)=GD0*exp(AF*(1.0D0-ABS(RSS(J)/CRSS(J))))        *  &
                      ( ABS( RSS(J)/CRSS(J) ) )**XP  

         slip_rate(J)=Sign(slip_rate(J),RSS(J))
              
              IF ( ABS(slip_rate(J)) .le. active_slip ) THEN
                       slip_rate(J) = 0.0D0
                       active = active
              END IF
              IF ( ABS(slip_rate(J)) .gt. active_slip ) THEN
                       active = active + 1.0D0
              END IF
            END DO

            END IF

! Assign results from slip kinematics
!            
            DO L=1,12

            res_slip(L) = slip_rate(L)
            ENDDO

            res_slip(13) = active
 


!************************************************************************
! Flag_slip = 4  --> Power law for slip incorporating rate and T effect
!                    y_dot={ y0*[tau_RSS/tau_CRSS]^Xp } Sgn(tau_RSS)
!************************************************************************
          IF ( Flag_slip .eq. 4) THEN
            GD0     = var_slip(1)
            XP      = var_slip(2) 
            Temp    = var_slip(10)
            SIG_EI0 = var_slip(6)
            SIG_LT0 = var_slip(7)
            HK      = var_slip(8)
            EDOT0   = var_slip(9)
            Boltzk  = 8.617e-5
    
         DO J=1,12
    
             SIG_EI=SIG_EI0*(1-Boltzk/HK*temp*log(EDOT0/var_slip(50+j)))**2
             SIG_LT=SIG_LT0*(1-(Boltzk/HK*temp*log(EDOT0/var_slip(50+j)))**0.5)
             T_cr=min(SIG_LT,SIG_EI)

             TS_fric = max(T_cr-TS_NS(J),0.0D0)

             slip_rate(J)=GD0*( ABS( RSS(J)/(CRSS(J)+TS_fric) ))**XP

             slip_rate(J)=Sign(slip_rate(J),RSS(J))

! Active slip threshold
!              
              IF ( ABS(slip_rate(J)) .le. active_slip ) THEN
              slip_rate(J) = 0.0D0
                 active = active
              END IF
    
              IF ( ABS(slip_rate(J)) .gt. active_slip ) THEN
                 active = active + 1.0D0
              END IF
               
            END DO

            END IF

! Assign results from slip kinematics
!            
    DO L=1,12
        res_slip(L) = slip_rate(L)
    ENDDO

            res_slip(13) = active
       end
       
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Harden_law
! Functionality: Defines the hardening law and calculate the
!                  current Critical Resolved Shear Stress (CRSS)
!
! Input(s):      var_Harden  : Hardening law variables
!                       (     1) = Initial Flow Stress    - T0
!                       (     2) = Shear Modulus          - G
!                       (     3) =                        - H1
!                       (     4) =                        - H2
!                       (     5) =                        - H3
!                       (     6) =                        - H4
!                       (     7) =                        - H5
!                       (     8) =                        - H6
!                       (     9) =                        - H7
!                       (    10) =                        - H8
!                       (    11) =                        - H9
!                       (    12) =                        - H10
!                       (    13) =                        - Eeffe_plast
!                       (14- 25) = Crit. Resolv. Shear    - CRSS   
!                       (26- 37) = Accumulated slip/syst. - cumul_slip
!                       (38- 49) = SSD density            - ssd
!                       (50- 61) = [Empty]
!
!                Flag_Harden : Determines which hardening model to use
!                            1  - Taylor Hardening Law 
!                            2  - Voce Hardening Law
!                            3  - Power-law Hardening Law
!                            4  - Slip based Hardening Law
!
! Output(s):     res_Harden  : Hardening law results
!                        ( 1-12) = Crit. Resolv. Shear    - CRSS_nxt
!                        (13-24) = SSD density            - ssd_nxt
!                        (25-36) = [Empty]
!       
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine Harden_law(res_Harden,var_Harden,Flag_Harden,Flag_Sys,sd,sn,res_slip)

        double precision res_Harden(24), var_Harden(61)
        double precision G110, burg, cKM1, cKM2, alpha, dy, d_ssd    ,  &
                         T0, Hv1, Hv2, H1, H2, Eeff_plast, E8, Em10  ,  &
                         H0, GS, A_exp, H_s, H_l, ARG,slip(12),H3    ,  &
                         H4,H5,H6,H7,Boltzk,SIG_EI,SIG_LT,SIG_KP
        double precision CRSS(12), cumul_slip(12), ssd(12),             &
                         CRSS_nxt(12), ssd_nxt(12), res_slip(13),         &
                         Fp(3,3), Eplast(6),cH(6), HAB(12,12),HB(12),   &
                         sd(12,3),sn(12,3)
        integer Flag_Harden, I, J, K, L, L0, Flag_Sys
!
          E8   = 1.0E+08
          Em10 = 1.0E-10
!        
          DO L0=1,12
        
             CRSS(L0) = var_Harden(L0+13)      
             res_Harden(L0) = 0.0D0
          
          END DO
!************************************************************************
! Flag_Harden = 1  --> Taylor Hardening WITHOUT GND density 
!             t_CRSS = A*G*b [ r_SSD ]^1/2
!************************************************************************    
          IF ( Flag_Harden .eq. 1) THEN
!
! Materials properties used in hardening model
!                  
               G110         = var_Harden(2)
               burg         = var_Harden(12)
               H1           = var_Harden(3)
               H2           = var_Harden(4)
               H3           = var_Harden(5)
               H4           = var_Harden(6)
               H5           = var_Harden(7)
               H6           = var_Harden(8)
               H7           = var_Harden(9)
               alpha        = var_Harden(11)               
               Boltzk       = 8.617e-5
               T0           = var_Harden(1)
!
            DO K=1,12
               CRSS(k)=var_Harden(k+13)
               cumul_slip(K) = var_Harden(K+25)
               ssd(K)        = var_Harden(K+37)
               ssd_nxt(K)    = 0.0D0
               CRSS_nxt(K)   = 0.0D0          
            END DO

!  SSD evolution equation

            call DD_evol(ssd_nxt,ssd,cumul_slip,H1,H2)
            call HAB_calc(sd,sn,HAB)

            DO I=1,12
               DO J=1,12

                !HAB(I,J)=1
                !IF (I.ne.J) THEN
                !HAB(I,J)=1.4
                !ENDIF

                CRSS_nxt(I)=CRSS_nxt(I)+ HAB(I,J)*ssd_nxt(J)
               !CRSS_nxt(i)=CRSS_nxt(i)+ ssd_nxt(J)
               ENDDO
            !CRSS_nxt(I)=CRSS_nxt(I)/12
            CRSS_nxt(I)=T0+(alpha*G110*burg )*Sqrt(CRSS_nxt(I))

! Temperature and strain rate dependence

            !IF (res_slip(I).eq.0.0D0) THEN
            !SIG_KP=0
            !ELSE

            !SIG_EI=H3*(1-Boltzk/H5*H7*log(H6/dabs(res_slip(I))) )**2
            !SIG_LT=H4*(1-(Boltzk/H5*H7*log(H6/dabs(res_slip(I))))**0.5)
            !SIG_KP=min(SIG_LT,SIG_EI)
            !SIG_KP=max(SIG_KP,0.0D0)
            !ENDIF

            !CRSS_nxt(I)=CRSS_nxt(I)+SIG_KP
            res_Harden(I+12) = ssd_nxt(I)

           END DO         
        END IF
       
!************************************************************************
! Flag_Harden = 2 --> Voce Hardening law
!             t_CRSS = t0_CRSS + A [ 1 - exp( (-n/A)*eps_plast ) ]
!************************************************************************          
          IF ( (Flag_Harden .eq. 2).OR.(Flag_Harden .eq. 3) ) THEN

            Eeff_plast = var_Harden(13)

          END IF


          IF ( Flag_Harden .eq. 2 ) THEN 

          T0  =var_Harden(1)
          H1 = var_Harden(3)
          H2 = var_Harden(4)
          DO K=1,12
          CRSS_nxt(K) = 0.0D0
          END DO
!
! Voce Hardening Law
! NB: 3.06 = Taylor factor adjustment
!
          DO J=1,12
          CRSS_nxt(J) =(T0+(H1*(1.0D0 - EXP((-1.0D0)*Eeff_plast*H2))))
          END DO
   
          END IF
!************************************************************************
! Flag_Harden = 3 --> Isotropic Power-law Hardening
!             t_CRSS = t0_CRSS + H1 (y)^H2
!************************************************************************
          IF ( Flag_Harden .eq. 3) THEN

          T0 = var_Harden(1)
          H1 = var_Harden(3)
          H2 = var_Harden(4)
          DO K=1,12
          CRSS_nxt(K) = 0.0D0
          END DO
!          
! Isotropic Power-law Hardening
!          
          DO J=1,12
          CRSS_nxt(J) = (T0 + H1*(Eeff_plast)**H2)
          END DO
          
          END IF

!************************************************************************
! Flag_Harden = 4 --> Slip-based Hardening Law
!             t_CRSS = hab * gamma
!************************************************************************
          IF (Flag_Harden.eq.4) THEN

          T0    = var_Harden(1)
          H0    = var_Harden(3)
          GS    = var_Harden(4)
          A_exp = var_Harden(5)
          H_s   = var_Harden(6)
          H_l   = var_Harden(7)

          DO k=1,12
          cumul_slip(K) = var_Harden(K+25)
          CRSS(k)=var_Harden(k+13)
          CRSS_nxt(K) = 0.0D0

          ARG = 1.- CRSS(k)/GS

            IF(ARG .GE. 0.D0) THEN
            HB(k) = H0*(DABS(ARG)**A_EXP)
            ELSE
            HB(k) = 0
            END IF  
          
          ENDDO

          DO J=1,12
           DO K=1,12
            IF (J.NE.K) THEN
             HAB(J,K)=H_l*HB(J)
            ELSE
             HAB(J,K)=H_s*HB(J)
            ENDIF

            CRSS_nxt(j)=CRSS_nxt(j)+HAB(j,k)*dabs( cumul_slip(j))
           ENDDO

            CRSS_nxt(j)=CRSS(j)+CRSS_nxt(j)
          ENDDO

         ENDIF
!************************************************************************
!
! Assign hardening results
!          
          DO L=1,12
               res_Harden(L) = CRSS_nxt(L)
          END DO
!
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          SID_evol
! Functionality: calculate the evolution of statistically immob.
! dislocations based on a source and sink term
!
! Input(s) :     sid     - Statistically immob. disloc. at time t
!                cslip   - Cumulative slip at time t
!                c(1)    - Source term coefficient
!                c(2)    - Sink term coefficient
!                c(3)    - Self hardening coefficient
!                c(4)    - Coplanar/Colinear hardening coefficient
!                c(5)    - Glissile junction hardening coefficient
!                c(6)    - Lomer-Cotrell locks
!
! Output(s):     sid_nxt - Statistically immob. disloc. at timet+Dt
!      
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine SID_evol(sid_nxt,sid,gnd,cslip)

        double precision sid_nxt(12),sid(12),gnd(12),cslip(12),c(6)
        double precision SL(12)
        double precision dy,d_sid,ptmp,a0,a1,a2,a3,rhotot,gndtot,sndtot
        integer J,K
!
            DO J=1,12
               sid_nxt(J) = 0.0D0
               SL(J) = 0.0D0
            END DO
!            
            a0 = c(3)
            a1 = c(4)
            a2 = c(5)
            a3 = c(6)
!
! The Hardening latent matrix has been expanded for computational
! efficiency
! Note that the colums and rows correspond to the 12 slip systems in
! following order: A2,A3,A6,B2,B4,B5,C1,C3,C5,D1,D4,D6
!
            rhotot  = 0.0D0
            
            DO J=1,12
            
            rhotot  =  rhotot + (sid(J) + gnd(J))
            
            ENDDO

            DO J=1,12
            
              dy = DABS(cslip(J))
              d_sid= c(1)*(sid(j)+gnd(J))**0.5 - c(2)*(rhotot)
              d_sid=d_sid*dy
              sid_nxt(J) = sid(J) + d_sid
            
            END DO

            end        
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Elasticity
! Functionality: calculate the stress based on Hooke's law
!
! Input(s) :     eps : Cauchy Green elastic strain
!                C   : Elastic stiffness matrix
!        
!
! Output(s):     sig : 2nd Piola-Kirchkoff stress 
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine Elasticity(sig,C,eps)

        double precision sig(6), eps(6), C(6,6)
        integer I0
!        
          DO I0=1,6
             sig(I0) =0.0D0
          END DO
!        
          sig(1) = C(1,1)*eps(1) + C(1,2)*eps(2) + C(1,3)*eps(3)     +  &
           2.0D0*( C(1,4)*eps(4) + C(1,5)*eps(5) + C(1,6)*eps(6) )
          sig(2) = C(1,2)*eps(1) + C(2,2)*eps(2) + C(2,3)*eps(3)     +  &
           2.0D0*( C(2,4)*eps(4) + C(2,5)*eps(5) + C(2,6)*eps(6) )
          sig(3) = C(1,3)*eps(1) + C(2,3)*eps(2) + C(3,3)*eps(3)     +  &
           2.0D0*( C(3,4)*eps(4) + C(3,5)*eps(5) + C(3,6)*eps(6) )
          sig(4) = C(1,4)*eps(1) + C(2,4)*eps(2) + C(3,4)*eps(3)     +  &
           2.0D0*( C(4,4)*eps(4) + C(4,5)*eps(5) + C(4,6)*eps(6) )
          sig(5) = C(1,5)*eps(1) + C(2,5)*eps(2) + C(3,5)*eps(3)     +  &
           2.0D0*( C(4,5)*eps(4) + C(5,5)*eps(5) + C(5,6)*eps(6) )
          sig(6) = C(1,6)*eps(1) + C(2,6)*eps(2) + C(3,6)*eps(3)     +  &
           2.0D0*( C(4,6)*eps(4) + C(5,6)*eps(5) + C(6,6)*eps(6) )
         
        end


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          stiffness 
! Functionality: calculates the rotated stiffness matrix
!
!                Crotated = R.R.C0.R^T.R^T
!
! Input(s) :     var_stiff : Elastic constants
!                var_stiff(1) - C11
!                var_stiff(2) - C22
!                var_stiff(3) - C33
!                var_stiff(4) - C12
!                var_stiff(5) - C23
!                var_stiff(6) - C13
!                var_stiff(7) - C44
!                var_stiff(8) - C55
!                var_stiff(9) - C66
!
!                Rot       : Rotation matrix
!
! Output(s):     Cel       : Elastic stiffness matrix
!
!                1 -> 11 ; 2-> 22 ; 3 ->33 ; 4 -> 12 ; 5-> 23 ; 6 ->13
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        subroutine stiffness(Cel,var_stiff,Rot)

        double precision Cel(6,6), var_stiff(9), Rot(3,3)            ,  &
                         Ccrystal(3,3,3,3),Ctmp(3,3,3,3)
        integer I, J, K, L, I0, J0, K0, L0
!
! Initialize variables
!
        DO I=1,3
        DO J=1,3
          DO K=1,3
          DO L=1,3
           Ccrystal(I,J,K,L) = 0.0D0
           Ctmp(I,J,K,L)     = 0.0D0
          END DO
          END DO
           Cel(I,J) = 0.0D0
        END DO
        END DO
!
! Single crystal stiffness tensor
!
        Ccrystal(1,1,1,1) = var_stiff(1)
        Ccrystal(2,2,2,2) = var_stiff(2)
        Ccrystal(3,3,3,3) = var_stiff(3)
        Ccrystal(1,1,2,2) = var_stiff(4)
        Ccrystal(2,2,1,1) = var_stiff(4)
        Ccrystal(2,2,3,3) = var_stiff(5)
        Ccrystal(3,3,2,2) = var_stiff(5)
        Ccrystal(1,1,3,3) = var_stiff(6)
        Ccrystal(3,3,1,1) = var_stiff(6)
        Ccrystal(1,2,1,2) = var_stiff(7)
        Ccrystal(2,1,2,1) = var_stiff(7)
        Ccrystal(1,2,2,1) = var_stiff(7)
        Ccrystal(2,1,1,2) = var_stiff(7)
        Ccrystal(2,3,2,3) = var_stiff(8)
        Ccrystal(3,2,2,3) = var_stiff(8)
        Ccrystal(2,3,3,2) = var_stiff(8)
        Ccrystal(3,2,3,2) = var_stiff(8)
        Ccrystal(1,3,1,3) = var_stiff(9)
        Ccrystal(1,3,3,1) = var_stiff(9)
        Ccrystal(3,1,1,3) = var_stiff(9)
        Ccrystal(3,1,3,1) = var_stiff(9)

        DO I=1,3
        DO J=1,3
          DO K=1,3
          DO L=1,3        
!
! Rotate elastic stiffness tensor Crot=R.R.C0.R^T.R^T
!
             DO I0=1,3
             DO J0=1,3
               DO K0=1,3
               DO L0=1,3
          
        Ctmp(I,J,K,L) = Ctmp(I,J,K,L)                                +  &
        Rot(I,I0)*Rot(J,J0)*Ccrystal(I0,J0,K0,L0)*Rot(K,K0)*Rot(L,L0)
          
               END DO
               END DO
             END DO
             END DO
!
          END DO
          END DO
        END DO
        END DO
!
! Assign elastic constant to upper triangle matrix
!
        Cel(1,1) = Ctmp(1,1,1,1)
        Cel(2,2) = Ctmp(2,2,2,2)
        Cel(3,3) = Ctmp(3,3,3,3)
        Cel(4,4) = Ctmp(1,2,1,2)
        Cel(5,5) = Ctmp(2,3,2,3)
        Cel(6,6) = Ctmp(1,3,1,3)
        Cel(1,2) = Ctmp(1,1,2,2)
        Cel(1,3) = Ctmp(1,1,3,3)
        Cel(1,4) = Ctmp(1,1,1,2)
        Cel(1,5) = Ctmp(1,1,2,3)
        Cel(1,6) = Ctmp(1,1,1,3)
        Cel(2,3) = Ctmp(2,2,3,3)
        Cel(2,4) = Ctmp(2,2,1,2)
        Cel(2,5) = Ctmp(2,2,2,3)
        Cel(2,6) = Ctmp(2,2,1,3)
        Cel(3,4) = Ctmp(3,3,1,2)
        Cel(3,5) = Ctmp(3,3,2,3)
        Cel(3,6) = Ctmp(3,3,1,3)
        Cel(4,5) = Ctmp(1,2,2,3)
        Cel(4,6) = Ctmp(1,2,1,3)
        Cel(5,6) = Ctmp(2,3,1,3)
        
        end  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          pk2tocauchy
! Functionality: moves the 2nd Piola-Kirchhoff stress to the current
!                config. and convert to Cauchy stress.
!
!                cauchy(i,j) = 1/J(Fe) [Fe.Sig_pk2.Fe^T]
!                                  + (1-cos(phi))/(phi)^2 (A.A).(Dt)^2         
!
!                cauchy matrix = cauchy(1) cauchy(4) cauchy(6)
!                                cauchy(4) cauchy(2) cauchy(5)
!                                cauchy(6) cauchy(5) cauchy(3)        
!
!
! Input(s) :     sig_pk2 : 2nd P-K stress
!                Fe      : Elastic Deformation Gradient
!
! Output(s):     cauchy  : Cauchy stress       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        subroutine pk2tocauchy(cauchy,sig_pk2,Fe)

        double precision cauchy(6), sig_pk2(6), Fe(3,3), A(3,3)
        double precision det_fe, det_fe_inv
        integer I0
!
           DO I0=1,6
             cauchy(I0) = 0.0D0
           END DO
!
!
! Jacobian of the Elastic Deformation Gradient
!
          det_fe = Fe(1,1)*Fe(2,2)*Fe(3,3) + Fe(1,2)*Fe(2,3)*Fe(3,1) +  &
                   Fe(1,3)*Fe(2,1)*Fe(3,2) - Fe(1,3)*Fe(2,2)*Fe(3,1) -  &
                   Fe(1,1)*Fe(2,3)*Fe(3,2) - Fe(1,2)*Fe(2,1)*Fe(3,3) 


!
! Inverse of Jacobian                      
!
          detfe_inv = (1.0D0/det_fe)

          A(1,1) = sig_pk2(1)*Fe(1,1) + sig_pk2(4)*Fe(1,2)           +  &
                   sig_pk2(6)*Fe(1,3)
          A(1,2) = sig_pk2(1)*Fe(2,1) + sig_pk2(4)*Fe(2,2)           +  &
                   sig_pk2(6)*Fe(2,3)
          A(1,3) = sig_pk2(1)*Fe(3,1) + sig_pk2(4)*Fe(3,2)           +  &
                   sig_pk2(6)*Fe(3,3)
          A(2,1) = sig_pk2(4)*Fe(1,1) + sig_pk2(2)*Fe(1,2)           +  &
                   sig_pk2(5)*Fe(1,3)
          A(2,2) = sig_pk2(4)*Fe(2,1) + sig_pk2(2)*Fe(2,2)           +  &
                   sig_pk2(5)*Fe(2,3)
          A(2,3) = sig_pk2(4)*Fe(3,1) + sig_pk2(2)*Fe(3,2)           +  &
                   sig_pk2(5)*Fe(3,3)
          A(3,1) = sig_pk2(6)*Fe(1,1) + sig_pk2(5)*Fe(1,2)           +  &
                   sig_pk2(3)*Fe(1,3)
          A(3,2) = sig_pk2(6)*Fe(2,1) + sig_pk2(5)*Fe(2,2)           +  &
                   sig_pk2(3)*Fe(2,3)
          A(3,3) = sig_pk2(6)*Fe(3,1) + sig_pk2(5)*Fe(3,2)           +  &
                   sig_pk2(3)*Fe(3,3)
                  
          cauchy(1) = detfe_inv*(Fe(1,1)*A(1,1) + Fe(1,2)*A(2,1)     +  &
                                 Fe(1,3)*A(3,1))
                                       
          cauchy(2) = detfe_inv*(Fe(2,1)*A(1,2) + Fe(2,2)*A(2,2)     +  &
                                 Fe(2,3)*A(3,2))
      
          cauchy(3) = detfe_inv*(Fe(3,1)*A(1,3) + Fe(3,2)*A(2,3)     +  &
                                 Fe(3,3)*A(3,3))
                                 
          cauchy(4) = detfe_inv*(Fe(1,1)*A(1,2) + Fe(1,2)*A(2,2)     +  &
                                 Fe(1,3)*A(3,2))

          cauchy(5) = detfe_inv*(Fe(2,1)*A(1,3) + Fe(2,2)*A(2,3)     +  &
                                 Fe(2,3)*A(3,3))
                                 
          cauchy(6) = detfe_inv*(Fe(3,1)*A(1,1) + Fe(3,2)*A(2,1)     +  &
                                 Fe(3,3)*A(3,1))
    
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          cauchytosig
! Functionality: Rotate stress from the current config.
!                to unrotated config.
!
!                cauchy(i,j) = Rot^T.cauchy.Rot
!
!                cauchy matrix = cauchy(1) cauchy(4) cauchy(6)
!                                cauchy(4) cauchy(2) cauchy(5)
!                                cauchy(6) cauchy(5) cauchy(3)        
!
!
! Input(s) :    cauchy    : rotated stress
!               Rot       : rotation matrix
!
! Output(s):    sig_unrot : Cauchy stress       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        subroutine cauchytosig(sig_unrot,cauchy,Rot)

        double precision sig_unrot(6), cauchy(6), Rot(3,3), A(3,3)
        integer I0
!
        DO I0=1,6
           sig_unrot(I0) = 0.0D0
        END DO
!        
            A(1,1) = cauchy(1)*Rot(1,1) + cauchy(4)*Rot(2,1)         +  &
                     cauchy(6)*Rot(3,1)
            A(1,2) = cauchy(1)*Rot(1,2) + cauchy(4)*Rot(2,2)         +  &
                     cauchy(6)*Rot(3,2)
            A(1,3) = cauchy(1)*Rot(1,3) + cauchy(4)*Rot(2,3)         +  &
                     cauchy(6)*Rot(3,3)
            A(2,1) = cauchy(4)*Rot(1,1) + cauchy(2)*Rot(2,1)         +  &
                     cauchy(5)*Rot(3,1)
            A(2,2) = cauchy(4)*Rot(1,2) + cauchy(2)*Rot(2,2)         +  &
                     cauchy(5)*Rot(3,2)
            A(2,3) = cauchy(4)*Rot(1,3) + cauchy(2)*Rot(2,3)         +  &
                     cauchy(5)*Rot(3,3)
            A(3,1) = cauchy(6)*Rot(1,1) + cauchy(5)*Rot(2,1)         +  &
                     cauchy(3)*Rot(3,1)
            A(3,2) = cauchy(6)*Rot(1,2) + cauchy(5)*Rot(2,2)         +  &
                     cauchy(3)*Rot(3,2)
            A(3,3) = cauchy(6)*Rot(1,3) + cauchy(5)*Rot(2,3)         +  &
                     cauchy(3)*Rot(3,3)
     
            sig_unrot(1) = Rot(1,1)*A(1,1) + Rot(2,1)*A(2,1)         +  &
                           Rot(3,1)*A(3,1)
                       
            sig_unrot(2) = Rot(1,2)*A(1,2) + Rot(2,2)*A(2,2)         +  &
                           Rot(3,2)*A(3,2)
                       
            sig_unrot(3) = Rot(1,3)*A(1,3) + Rot(2,3)*A(2,3)         +  &
                           Rot(3,3)*A(3,3)

            sig_unrot(4) = Rot(1,1)*A(1,2) + Rot(2,1)*A(2,2)         +  &
                           Rot(3,1)*A(3,2)
                       
            sig_unrot(5) = Rot(1,2)*A(1,3) + Rot(2,2)*A(2,3)         +  &
                           Rot(3,2)*A(3,3)
                           
            sig_unrot(6) = Rot(1,3)*A(1,1) + Rot(2,3)*A(2,1)         +  &
                           Rot(3,3)*A(3,1)
        
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          calc_elast_grad
! Functionality: Calculate the elastic deformation gradient
!
!                Fe(t+Dt) = F(t+Dt).[Fp(t+Dt)]^(-1)
!        
! Input(s) :     Ftot   : Total Deformation Gradient
!                Fplast : Plastic Deformation Gradient 
!
! Output(s):     Fel    : Elastic Deformation Gradient
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine calc_elast_grad(Fel,Ftot,Fplast)

        double precision Fel(3,3), Fplast(3,3), Ftot(3,3)            ,  &
                         Fplast_inv(3,3)
        integer I0, J0
!
        DO I0=1,3
        DO J0=1,3       
           Fel(I0,J0) = 0.0D0
        END DO
        END DO
!
! Calculate inverse of Plastic Deformation Gradient
!
          call inversemat(Fplast_inv,Fplast)

!
! Calculate Elastic Deformation gradient
!
          Fel(1,1) = Ftot(1,1)*Fplast_inv(1,1)                       +  &
                     Ftot(1,2)*Fplast_inv(2,1)                       +  &
                     Ftot(1,3)*Fplast_inv(3,1)
          Fel(1,2) = Ftot(1,1)*Fplast_inv(1,2)                       +  &
                     Ftot(1,2)*Fplast_inv(2,2)                       +  &
                     Ftot(1,3)*Fplast_inv(3,2)
          Fel(1,3) = Ftot(1,1)*Fplast_inv(1,3)                       +  &
                     Ftot(1,2)*Fplast_inv(2,3)                       +  &
                     Ftot(1,3)*Fplast_inv(3,3)
          Fel(2,1) = Ftot(2,1)*Fplast_inv(1,1)                       +  &
                     Ftot(2,2)*Fplast_inv(2,1)                       +  &
                     Ftot(2,3)*Fplast_inv(3,1)
          Fel(2,2) = Ftot(2,1)*Fplast_inv(1,2)                       +  &
                     Ftot(2,2)*Fplast_inv(2,2)                       +  &
                     Ftot(2,3)*Fplast_inv(3,2)
          Fel(2,3) = Ftot(2,1)*Fplast_inv(1,3)                       +  &
                     Ftot(2,2)*Fplast_inv(2,3)                       +  &
                     Ftot(2,3)*Fplast_inv(3,3)
          Fel(3,1) = Ftot(3,1)*Fplast_inv(1,1)                       +  &
                     Ftot(3,2)*Fplast_inv(2,1)                       +  &
                     Ftot(3,3)*Fplast_inv(3,1)
          Fel(3,2) = Ftot(3,1)*Fplast_inv(1,2)                       +  &
                     Ftot(3,2)*Fplast_inv(2,2)                       +  &
                     Ftot(3,3)*Fplast_inv(3,2)
          Fel(3,3) = Ftot(3,1)*Fplast_inv(1,3)                       +  &
                     Ftot(3,2)*Fplast_inv(2,3)                       +  &
                     Ftot(3,3)*Fplast_inv(3,3)
        end 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          calc_plast_grad
! Functionality: Calculate the plastic deformation gradient
!
!   The numerical integration over the time step done using expon. fnc.
!
!   Increment found by integrating dF_plast = plast_Vel_grad xF_plast
!
!      --> F_plast(t+Dt) = exp{ plast_Vel_grad(t) x Dt } x F_plast(t)
!        
! Input(s) :     Fplast     : Plastic Deformation Gradient at time t
!                Vp         : Plastic Velocitity Gradient
!                time_step  : Time step Dt 
!
! Output(s):     Fplast_nxt : Plastic Deformation Gradient at time t+Dt
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine calc_plast_grad(Fp_nxt,Fp,Vp,time_step)

        double precision time_step
        double precision Fp_nxt(3,3), Fp(3,3), Vp(3,3), expon(3,3)
        integer I0, J0
!
        DO I0=1,3
        DO J0=1,3
           Fp_nxt(I0,J0) = 0.0D0
        END DO
        END DO
!
! Calculate exponential of plast_Vel_grad(t) x Dt
!

!write(92,*) expon(1,1:3),Vp(1,1),time_step

          call expo_mat(expon,Vp,time_step)



!
! Calculate plastic gradient increment
!
          Fp_nxt(1,1) = expon(1,1)*Fp(1,1) + expon(1,2)*Fp(2,1)      +  &
                        expon(1,3)*Fp(3,1)
          Fp_nxt(1,2) = expon(1,1)*Fp(1,2) + expon(1,2)*Fp(2,2)      +  &
                        expon(1,3)*Fp(3,2)
          Fp_nxt(1,3) = expon(1,1)*Fp(1,3) + expon(1,2)*Fp(2,3)      +  &
                        expon(1,3)*Fp(3,3)
          Fp_nxt(2,1) = expon(2,1)*Fp(1,1) + expon(2,2)*Fp(2,1)      +  &
                        expon(2,3)*Fp(3,1)
          Fp_nxt(2,2) = expon(2,1)*Fp(1,2) + expon(2,2)*Fp(2,2)      +  &
                        expon(2,3)*Fp(3,2)
          Fp_nxt(2,3) = expon(2,1)*Fp(1,3) + expon(2,2)*Fp(2,3)      +  &
                        expon(2,3)*Fp(3,3)
          Fp_nxt(3,1) = expon(3,1)*Fp(1,1) + expon(3,2)*Fp(2,1)      +  &
                        expon(3,3)*Fp(3,1)
          Fp_nxt(3,2) = expon(3,1)*Fp(1,2) + expon(3,2)*Fp(2,2)      +  &
                        expon(3,3)*Fp(3,2)
          Fp_nxt(3,3) = expon(3,1)*Fp(1,3) + expon(3,2)*Fp(2,3)      +  &
                        expon(3,3)*Fp(3,3)

        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          cauchygreen_strn
! Functionality: Calculate the Green Elastic strain tensor
!
!                E_CauchyGreen(i,j)=0.5[ (F_e^T.F_e - I ]
!        
!                Cauchy Green matrix  E_CG(1) E_CG(4) E_CG(6)
!                                     E_CG(4) E_CG(2) E_CG(5)
!                                     E_CG(6) E_CG(5) E_CG(3)
!
! Input(s) :     Fel           : Deformation Gradient
!
! Output(s):     E_CauchyGreen : Cauchy Green tensor
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine cauchygreen_strn(E_CauchyGreen,Fel)

        double precision E_CauchyGreen(6), Fel(3,3)
        integer I0
!
        DO I0 =1,6
        E_CauchyGreen(I0) = 0.0D0
        END DO
!
          E_CauchyGreen(1) = 0.5D0*( Fel(1,1)**2 + Fel(2,1)**2       +  &
                                     Fel(3,1)**2 - 1.0D0)
          E_CauchyGreen(2) = 0.5D0*( Fel(1,2)**2 + Fel(2,2)**2       +  &
                                     Fel(3,2)**2 - 1.0D0)
          E_CauchyGreen(3) = 0.5D0*( Fel(1,3)**2 + Fel(2,3)**2       +  &
                                     Fel(3,3)**2 - 1.0D0)
          E_CauchyGreen(4) = 0.5D0*( Fel(1,1)*Fel(1,2)               +  &
                                     Fel(2,1)*Fel(2,2)               +  &
                                     Fel(3,1)*Fel(3,2) )
          E_CauchyGreen(5) = 0.5D0*( Fel(1,2)*Fel(1,3)               +  &
                                     Fel(2,2)*Fel(2,3)               +  &
                                     Fel(3,2)*Fel(3,3) )
          E_CauchyGreen(6) = 0.5D0*( Fel(1,1)*Fel(1,3)               +  &
                                     Fel(2,1)*Fel(2,3)               +  &
                                     Fel(3,1)*Fel(3,3) )
 
end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          expo_mat
! Functionality: calculates exponential of a matrix up to the
!                second order.
!
!                exp(A x Dt) = I + sin(phi)/phi A.Dt
!                                + (1-cos(phi))/(phi)^2 (A.A).(Dt)^2         
!
!                phi = Dt x [0.5* (A:A)]^1/2         
!
!
! Input(s) :     A     :   3x3 matrix
!                Dt    :   time step
!
! Output(s):     expon : exponent matrix --> expon = exp(A x Dt)       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine expo_mat(expon,A,Dt)
          
        double precision A(3,3), expon(3,3), A2(3,3)
        double precision phi, Aexp1, Aexp2, Dt
        integer I0,J0,J,K
!
        DO I0=1,3
        DO J0=1,3
           expon(I0,J0) = 0.0D0
           A2(I0,J0)    = 0.0D0
        END DO
        END DO
!        
          phi = SQRT( 0.5D0*(A(1,1)**2 + A(1,2)**2 + A(1,3)**2       +  &
                             A(2,1)**2 + A(2,2)**2 + A(2,3)**2       +  &
                             A(3,1)**2 + A(3,2)**2 + A(3,3)**2) )*Dt

          IF( abs(phi).lt.1e-10 )THEN
            Aexp1 = 0.0D0
            Aexp2 = 0.0D0
          ELSE
            Aexp1 = (SIN(phi)/phi) * Dt
            Aexp2 = ((1.0D0-COS(phi))*(Dt**2)/(phi**2))
          END IF
!
! Components of:  A(i,k) x A(k,j)
!
          DO J=1,3
            DO K=1,3
              A2(J,K) = A(J,1)*A(1,K) + A(J,2)*A(2,K) + A(J,3)*A(3,K)
            END DO
          END DO   
!
! Components of exponential matrix
!          
          expon(1,1) = Aexp1*A(1,1) + Aexp2*A2(1,1) + 1.0D0
          expon(2,2) = Aexp1*A(2,2) + Aexp2*A2(2,2) + 1.0D0
          expon(3,3) = Aexp1*A(3,3) + Aexp2*A2(3,3) + 1.0D0
          expon(1,2) = Aexp1*A(1,2) + Aexp2*A2(1,2)
          expon(1,3) = Aexp1*A(1,3) + Aexp2*A2(1,3)
          expon(2,1) = Aexp1*A(2,1) + Aexp2*A2(2,1)
          expon(2,3) = Aexp1*A(2,3) + Aexp2*A2(2,3)
          expon(3,1) = Aexp1*A(3,1) + Aexp2*A2(3,1)
          expon(3,2) = Aexp1*A(3,2) + Aexp2*A2(3,2)

!write(92,*) expon(1,1),Aexp2, phi
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          converge_table
! Functionality: defines convergence criterion for slip rate
!                  
!
! Input(s) :     time_step : time step
!
! Output(s):     tol_int   : tolerance criterion       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine converge_table(tol_int,time_step)

        double precision tol_int, time_step

            IF( time_step.ge.0.001D0 )THEN
              tol_int = 100.0D0*time_step
            ELSE IF( (time_step.lt.0.001D0).and.                        &
                     (time_step.ge.0.0002D0) )THEN
              tol_int = 10.0D0*time_step              
            ELSE IF( (time_step.lt.0.0002D0).and.                       &
                     (time_step.ge.0.0001D0) )THEN
              tol_int = 0.003D0
            ELSE IF( (time_step.lt.0.0001D0).and.                       &
                     (time_step.ge.0.00001D0) )THEN
              tol_int = 0.05D0
            ELSE IF( (time_step.lt.0.00001D0).and.                      &
                     (time_step.ge.0.000001D0) )THEN
              tol_int = 0.0075D0
            ELSE IF( (time_step.lt.0.000001D0).and.                     &
                     (time_step.ge.0.0000001D0) )THEN
              tol_int = 0.025D0
            ELSE IF( (time_step.lt.0.0000001D0).and.                    &
                     (time_step.ge.0.00000001D0) )THEN
              tol_int = 0.02D0
            ELSE IF( (time_step.lt.0.00000001D0).and.                   &
                     (time_step.ge.0.0000000001D0) )THEN
              tol_int = 0.2D0
            ELSE  
              tol_int = 50.0D0
            END IF

        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          stable_inc
! Functionality: checks if subincrement is stable or not by looking at
!                the magnitude of slip rate increment 
!                  
! Magnitude:||y(t+Dt)|| = [ ( y(t+Dt) -y(t) )^2 +  (Dt)^2 ]^1/2
! Criterion: Unstable time subincrement if ||y(t+Dt)|| > y_tolerance
!
! Input(s) :     slip    : slip rate
!                ck_slip : check slip rate
!                Dt      : time step            
!
! Output(s):     subinc  : Logical variable indicating if increment
!                          stable or not       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine stable_inc1(subinc,slip,ck_slip,Dt)

        double precision tol_int, Dt
        double precision slip_magnitude(12), slip(12), ck_slip(13)
        integer J,conv
        logical subinc
!
! ||y(t+Dt)|| = [ ( y(t+Dt) -y(t) )^2 + (Dt)^2 ]^1/2
!
        call converge_table(tol_int,Dt)

        DO J=1,12
           slip_magnitude(J) = DSQRT( (ck_slip(J)-slip(J))**2 + Dt**2)
        END DO
!
! Unstable time subincrement if ||y(t+Dt)|| > y_tolerance
!
        DO J=1,12

            IF( (ABS(slip_magnitude(J)) .gt. tol_int) ) THEN
                     subinc = .true.
            END IF

        END DO
!                                
        end        
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          stable_inc2
! Functionality: checks if subincrement is stable or not by looking at
!                the error in a weighted level function. 
!                  
! ERR =(1/Nsys) SQRT( [( y(t+Dt)/ymax )*f ]^2 )
! f   = y_predictor(t+DT) - y0*ABS(RSS(t+Dt)/CRSS(t+DT))*sgn(RSS(t+Dt)       
! Criterion: Unstable time subincrement if ERR > tolerance
!
! Input(s) :     sta_slip    : slip rate from predictor and flow rule
!
! Output(s):     subinc  : Logical variable indicating if increment
!                          stable or not       
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine stable_inc2(subinc,sta_slip,active_rate)

        double precision tol_int, ERROR, Ymax, weigh_sum, GD0
        double precision sta_slip(24), slip(12), slip_1(12)
        double precision fslip(12), wJ
        integer J,I0 
        logical subinc
	    real(8) active_rate
!
! sta_slip( 1-12) = slip rate from predictor
! sta_slip(13-24) = slip rate from flow rule
!
        DO I0=1,12
           slip(I0)   = sta_slip(I0)
           slip_1(I0) = sta_slip(I0+12) 
           fslip(I0)  = 0.0D0
        END DO
!
        tol_int     = 0.5D0
        weigh_sum   = 0.0D0
        Ymax        = ABS(slip(1))
!
! Compute level function and maximum slip rate
!        
        DO J=1,12
        
        fslip(J) = slip(J) - slip_1(J)
!        
        IF( ABS(slip(J)).ge.Ymax )THEN        
        Ymax = ABS(slip(J))
        END IF
!
        END DO
!
! Case scenario where no slip system is active
! (avoid level function to go to infinity)
!
        IF( ABS(Ymax).lt.active_rate )THEN
        Ymax = 1.0D0
        END IF        
!
! Calculate weighted sum
!        
        DO J=1,12     
        weigh_sum = weigh_sum + ( fslip(J) )**2
!        wJ = ABS(slip(J))/Ymax
!        weigh_sum = weigh_sum + ( wJ*fslip(J) )**2 
        END DO
!
! Calculate convergence criterion
! ERROR = 1/Nsys*SQRT( SUM((y/ymax*f)^2) )
!        
        ERROR = (1.0D0/12.0D0)*SQRT(weigh_sum)
!
! Unstable time subincrement if ERROR > y_tolerance
!            
            IF( ERROR .gt. tol_int ) THEN
              subinc = .true.
            END IF
!                                
        end


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          predict_rate
! Functionality: predicts slip rate based on stress update using Taylor
!                expansion
!                  
! y_n+1 = y_n + [dy/dRSS]_n.RSS_dot_n.Dt + [dy/dCRSS]_n.CRSS_dot_n.Dt
! <=>
! y_n+1 = y_n + [dy/dRSS]_n.[RSS_n+1-RSS_n]
!             + [dy/dCRSS]_n.[CRSS_n+1-CRSS_n]     
!
! Input(s) :     slip      : slip rate at time tn
!                dRSS      : RSS for time tn and tn+1
!                dCRSS     : CRSS for time tn and tn+1 
!                Flow      : Flow rule parameters
!
! Output(s):     slip_1    : Predicted slip rate 
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine predict_rate(slip_1,slip,RSS,CRSS,rimm,rmob,Flow  ,  &
                                Flag,dNS)
        DOUBLE PRECISION slip_1(12),slip(12),RSS(24),CRSS(24)        ,  &
                         Flow(12),rmob(24),rimm(24)
        DOUBLE PRECISION Qeff, Temp, AF, Boltzk,GD0,V0,XP,active_rate,  &
                         dydR,dydC,dydI,TK,c1,c2,Burgers,dIdM,E6,A1,A2, &
                         Em10,Em20,T_cr,TS_fric,dNS(24),AF1,AF2,AF3,    &
                         SIG_EI,SIG_LT,SIG_EI0,SIG_LT0,HK,EDOT0
        INTEGER J,Flag
!
!  RSS( 1-12) = RSS at time tn  ;  RSS(13-24) = RSS at time tn+1
! CRSS( 1-12) = CRSS at time tn ; CRSS(13-24) = CRSS at time tn+1
! TS_NS( 1-12) = NS at time tn ; TS_NS(13-24) = NS at time tn+1
!---------------------------------------
! Power law flow rule
        IF(Flag.eq.1) THEN
!
! Flow parameter
!
        GD0         = Flow(1)
        XP          = Flow(2)
        active_rate = Flow(4)
        T_cr        = Flow(8)
!        
        DO J=1,12

        TS_fric  = T_cr - dNS(J)

        IF (TS_fric .lt. 0.0) then
 
        TS_fric = 0.0
        endif
! Derivative of slip rate w.r.t. resolved shear stress
!        
        dydR  = GD0*(XP/(CRSS(J)+TS_fric))*( ABS(RSS(J)/(CRSS(J)+TS_fric)) )**(XP-1.0D0)
        dydR  = SIGN( dydR,RSS(J) )
        
        IF (ABS(slip(J)).le.active_rate) THEN
        dydR = 0.0D0
        END IF
!
! Derivative of slip rate w.r.t. critical resolved shear stress

        dydC  = GD0*(XP/RSS(J))*( ABS(RSS(J)/(CRSS(J)+TS_fric)) )**(XP+1.0D0)
        dydC  = SIGN( dydC,RSS(J) )
        
        IF ( ABS(slip(J)).le.active_rate ) THEN
        dydC = 0.0D0
        END IF
!
! Taylor approximation of slip rate at time tn+1
!
! NB: "-" sign come from derivative of dy/dCRSS (cf. exponent)        
        slip_1(J) = slip(J) + dydR*( RSS(J+12)-RSS(J) )              -  &
                              dydC*( CRSS(J+12)-CRSS(J) )

        
        END DO

        END IF
!---------------------------------------
! Exponential flow rule
!      
        IF(Flag.eq.2) THEN
!
! Flow parameter
!
        GD0         = Flow(1)
        XP          = Flow(2)
        active_rate = Flow(4)
! Temperature (in K)         
            Temp  = Flow(5)
! Activation energy in (eV)          
            Qeff  = Flow(6)
!
! Boltzmann constant (in 10-^20 J.K^-1)
!            
!            Boltzk = 1.3806503D0*0.001D0
             Boltzk = 8.62e-5
!
! Multiplicative factor in expon:  DF/(k*T)
!
            AF  = (-1.0D0)*Qeff/(Boltzk*Temp)

        DO J=1,12
!
! Derivative of slip rate w.r.t. resolved shear stress
!        
        dydR  = GD0*(exp(AF*(1.0D0-ABS(RSS(J)/CRSS(J)))))            *  &
                ( ABS( RSS(J)/CRSS(J) ) )**(XP-1.0D0)                *  &
                ( XP-AF*ABS(RSS(J)/CRSS(J)) )/(CRSS(J))
        
        dydR  = SIGN( dydR,RSS(J) )
        
        IF (ABS(slip(J)).le.active_rate) THEN
        dydR = 0.0D0
        END IF
!
! Derivative of slip rate w.r.t. critical resolved shear stress
!
        dydC  = GD0*exp( AF*(1.0D0-ABS(RSS(J)/CRSS(J))) )            *  &
                ( ( ABS(RSS(J)/CRSS(J)) )**(XP+1.0D0) )              *  &
                ( (XP/(RSS(J))) - (AF/CRSS(J)) )


        dydC  = SIGN( dydC,RSS(J) )
        
        IF ( ABS(slip(J)).le.active_rate ) THEN
        dydC = 0.0D0
        END IF
!
! Taylor approximation of slip rate at time tn+1
!
! NB: "-" sign come from derivative of dy/dCRSS (cf. exponent)        
        slip_1(J) = slip(J) + dydR*( RSS(J+12)-RSS(J) )              -  &
                              dydC*( CRSS(J+12)-CRSS(J) )
        
        END DO
        END IF

!----------------------------------------------------------------------------
! Power law flow rule
        IF(Flag.eq.4) THEN
!
! Flow parameter
!
        GD0         = Flow(1)
        XP          = Flow(2)
        active_rate = Flow(4)
        T_cr        = Flow(8)
        Temp        = Flow(5)      
        SIG_EI0     = Flow(9)
        SIG_LT0     = Flow(10)
        HK          = Flow(11)
        EDOT0       = Flow(12)
        Boltzk      = 8.617e-5 
 
        DO J=1,12

        SIG_EI=SIG_EI0*(1-Boltzk/HK*temp*log(EDOT0/slip(j)))**2
        SIG_LT=SIG_LT0*(1-(Boltzk/HK*temp*log(EDOT0/slip(j)))**0.5)
        T_cr=min(SIG_LT,SIG_EI)

        TS_fric = max(T_cr-dNS(J),0.0D0)

! Derivative of slip rate w.r.t. resolved shear stress
!        
        dydR  = GD0*(XP/(CRSS(J)+TS_fric))*( ABS(RSS(J)/(CRSS(J)+TS_fric)) )**(XP-1.0D0)
        dydR  = SIGN( dydR,RSS(J) )
        
        IF (ABS(slip(J)).le.active_rate) THEN
        dydR = 0.0D0
        END IF
!
! Derivative of slip rate w.r.t. critical resolved shear stress

        dydC  = GD0*(XP/RSS(J))*( ABS(RSS(J)/(CRSS(J)+TS_fric)) )**(XP+1.0D0)
        dydC  = SIGN( dydC,RSS(J) )
        
        IF ( ABS(slip(J)).le.active_rate ) THEN
        dydC = 0.0D0
        END IF
!
! Taylor approximation of slip rate at time tn+1
!
! NB: "-" sign come from derivative of dy/dCRSS (cf. exponent)        
        slip_1(J) = slip(J) + dydR*( RSS(J+12)-RSS(J) )              -  &
                              dydC*( CRSS(J+12)-CRSS(J) )
       

 END DO

        END IF

        end       


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          misorient
! Functionality: Calculate the misorientation between 2 orientations
!
! Input(s) :     R1  : Rotation matrix 1
!                R2  : Rotation matrix 2
!
! Output(s):     phi : misorientation angle between R1 and R2
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine misorient(phi,R1,R2)

        double precision R1(3,3), R2(3,3), R1inv(3,3), Q(3,3,24)     ,  &
                         Qtmp(3,3)
        double precision phi, tr_tmp, trmaxi, Pi
        integer I, J, L, Lmax
!
        Pi = 3.14159265D0
!
        DO I=1,3
        DO J=1,3
           Qtmp(I,J) = 0.0D0
        END DO
        END DO        
!
! Inverse R1
!        
        call inversemat(R1inv,R1)
!
! Call symmetry group rotation matrices
!        
        call symmetrygroup(Q)
!
! Find L / Max[tr(R2.QL.R1^-1)], L = symmetry group       
!
        Lmax = 1
        DO I=1,3
        DO J=1,3
           Qtmp(I,J) = Q(I,J,Lmax)
        END DO
        END DO

        call traceRQR(tr_tmp,R2,Qtmp,R1inv)
        trmax = tr_tmp

        DO L=2,24
        
          DO I=1,3
          DO J=1,3
             Qtmp(I,J) = Q(I,J,L)
          END DO
          END DO
        
          call traceRQR(tr_tmp,R2,Qtmp,R1inv)
        
          IF (tr_tmp.gt.trmax) THEN
             trmax  = tr_tmp
          END IF
        
        END DO
!
! Calculate misorientation angle phi:z
! 
        prod = 0.5D0*(trmax - 1.0D0)
        phi  = (180.0D0/Pi)*ACOS(prod)

        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          symmetrygroup
! Functionality: Defines the 24 symmetry group rotation matrix Qi,i=1,24
!
!         Q(i,j) = cos(phi) + [1-cos(phi)]pi.pj + sin(phi).eijk.pk 
!        
! Input(s) :     
!
! Output(s):     Q(I,J,K) : Rotation matrix for symmetry group K
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine symmetrygroup(Q)

        double precision Q(3,3,24), gr(5,24), e(3,3,3), d(3,3)
        integer I, J, K, L

        
        DO L=1,24
           DO I=1,3
           DO J=1,3
              Q(I,J,L) = 0.0D0
           END DO
           END DO
       END DO
!
! gr(I,K) : I=1,3 -> component of rotation axis
!           I=4   -> cos(phi)
!           I=5   -> sin(phi)        
!        
! gr 1  [0 0 1] and phi = 0 deg.
        gr(1,1) = 0.0D0
        gr(2,1) = 0.0D0
        gr(3,1) = 1.0D0
        gr(4,1) = 1.0D0
        gr(5,1) = 0.0D0
! gr 2  [0 0 1] and phi = 90 deg.
        gr(1,2) = 0.0D0
        gr(2,2) = 0.0D0
        gr(3,2) = 1.0D0
        gr(4,2) = 0.0D0
        gr(5,2) = 1.0D0
! gr 3  [0 0 1] and phi = 180 deg.
        gr(1,3) = 0.0D0
        gr(2,3) = 0.0D0
        gr(3,3) = 1.0D0
        gr(4,3) = -1.0D0
        gr(5,3) = 0.0D0
! gr 4  [0 0 1] and phi = 270 deg.
        gr(1,4) = 0.0D0
        gr(2,4) = 0.0D0
        gr(3,4) = 1.0D0
        gr(4,4) = 0.0D0
        gr(5,4) = -1.0D0
! gr 5  [0 1 0] and phi = 90 deg.
        gr(1,5) = 0.0D0
        gr(2,5) = 1.0D0
        gr(3,5) = 0.0D0
        gr(4,5) = 0.0D0
        gr(5,5) = 1.0D0
! gr 6  [0 1 0] and phi = 180 deg.
        gr(1,6) = 0.0D0
        gr(2,6) = 1.0D0
        gr(3,6) = 0.0D0
        gr(4,6) = -1.0D0
        gr(5,6) = 0.0D0
! gr 7  [0 1 0] and phi = 270 deg.
        gr(1,7) = 0.0D0
        gr(2,7) = 1.0D0
        gr(3,7) = 0.0D0
        gr(4,7) = 0.0D0
        gr(5,7) = -1.0D0
! gr8  [1 0 0] and phi = 90 deg.
        gr(1,8) = 1.0D0
        gr(2,8) = 0.0D0
        gr(3,8) = 0.0D0
        gr(4,8) = 0.0D0
        gr(5,8) = 1.0D0
! gr 9  [1 0 0] and phi = 180 deg.
        gr(1,9) = 1.0D0
        gr(2,9) = 0.0D0
        gr(3,9) = 0.0D0
        gr(4,9) = -1.0D0
        gr(5,9) = 0.0D0
! gr 10  [1 0 0] and phi = 270 deg.
        gr(1,10) = 1.0D0
        gr(2,10) = 0.0D0
        gr(3,10) = 0.0D0
        gr(4,10) = 0.0D0
        gr(5,10) = -1.0D0
! gr 11  [1 1 0] and phi = 180 deg.
        gr(1,11) = 1.0D0/sqrt(2.0D0)
        gr(2,11) = 1.0D0/sqrt(2.0D0)
        gr(3,11) = 0.0D0
        gr(4,11) = -1.0D0
        gr(5,11) = 0.0D0
! gr 12  [1 -1 0] and phi = 180 deg.
        gr(1,12) = 1.0D0/sqrt(2.0D0)
        gr(2,12) = -1.0D0/sqrt(2.0D0)
        gr(3,12) = 0.0D0
        gr(4,12) = -1.0D0
        gr(5,12) = 0.0D0
! gr 13  [0 1 1] and phi = 180 deg.
        gr(1,13) = 0.0D0
        gr(2,13) = 1.0D0/sqrt(2.0D0)
        gr(3,13) = 1.0D0/sqrt(2.0D0)
        gr(4,13) = -1.0D0
        gr(5,13) = 0.0D0
! gr 14  [0 1 -1] and phi = 180 deg.
        gr(1,14) = 0.0D0
        gr(2,14) = 1.0D0/sqrt(2.0D0)
        gr(3,14) = -1.0D0/sqrt(2.0D0)
        gr(4,14) = -1.0D0
        gr(5,14) = 0.0D0
! gr 15  [1 0 1] and phi = 180 deg.
        gr(1,15) = 1.0D0/sqrt(2.0D0)
        gr(2,15) = 0.0D0
        gr(3,15) = 1.0D0/sqrt(2.0D0)
        gr(4,15) = -1.0D0
        gr(5,15) = 0.0D0
! gr 16  [1 0 -1] and phi = 180 deg.
        gr(1,16) = 1.0D0/sqrt(2.0D0)
        gr(2,16) = 0.0D0
        gr(3,16) = -1.0D0/sqrt(2.0D0)
        gr(4,16) = -1.0D0
        gr(5,16) = 0.0D0
! gr 17  [1 1 1] and phi = 120 deg.
        gr(1,17) = 1.0D0/sqrt(3.0D0)
        gr(2,17) = 1.0D0/sqrt(3.0D0)
        gr(3,17) = 1.0D0/sqrt(3.0D0)
        gr(4,17) = -0.5D0
        gr(5,17) = (sqrt(3.0D0))*0.5D0
! gr 18  [1 1 1] and phi = 240 deg.
        gr(1,18) = 1.0D0/sqrt(3.0D0)
        gr(2,18) = 1.0D0/sqrt(3.0D0)
        gr(3,18) = 1.0D0/sqrt(3.0D0)
        gr(4,18) = -0.5D0
        gr(5,18) = -(sqrt(3.0D0))*0.5D0
! gr 19  [-1 1 1] and phi = 120 deg.
        gr(1,19) = -1.0D0/sqrt(3.0D0)
        gr(2,19) = 1.0D0/sqrt(3.0D0)
        gr(3,19) = 1.0D0/sqrt(3.0D0)
        gr(4,19) = -0.5D0
        gr(5,19) = (sqrt(3.0D0))*0.5D0
! gr 20  [-1 1 1] and phi = 240 deg.
        gr(1,20) = -1.0D0/sqrt(3.0D0)
        gr(2,20) = 1.0D0/sqrt(3.0D0)
        gr(3,20) = 1.0D0/sqrt(3.0D0)
        gr(4,20) = -0.5D0
        gr(5,20) = -(sqrt(3.0D0))*0.5D0
! gr 21  [1 -1 1] and phi = 120 deg.
        gr(1,21) = 1.0D0/sqrt(3.0D0)
        gr(2,21) = -1.0D0/sqrt(3.0D0)
        gr(3,21) = 1.0D0/sqrt(3.0D0)
        gr(4,21) = -0.5D0
        gr(5,21) = (sqrt(3.0D0))*0.5D0
! gr 22  [1 -1 1] and phi = 240 deg.
        gr(1,22) = 1.0D0/sqrt(3.0D0)
        gr(2,22) = -1.0D0/sqrt(3.0D0)
        gr(3,22) = 1.0D0/sqrt(3.0D0)
        gr(4,22) = -0.5D0
        gr(5,22) = -(sqrt(3.0D0))*0.5D0
! gr 23  [-1 -1 1] and phi = 120 deg.
        gr(1,23) = -1.0D0/sqrt(3.0D0)
        gr(2,23) = -1.0D0/sqrt(3.0D0)
        gr(3,23) = 1.0D0/sqrt(3.0D0)
        gr(4,23) = -0.5D0
        gr(5,23) = (sqrt(3.0D0))*0.5D0
! gr 24  [-1 -1 1] and phi = 240 deg.
        gr(1,24) = -1.0D0/sqrt(3.0D0)
        gr(2,24) = -1.0D0/sqrt(3.0D0)
        gr(3,24) = 1.0D0/sqrt(3.0D0)
        gr(4,24) = -0.5D0
        gr(5,24) = -(sqrt(3.0D0))*0.5D0
!        
! Permutation tensor / Kronecker delta
!        
        DO I=1,3
          DO J=1,3
            DO K=1,3
               e(I,J,K) = 0.0D0
            END DO
               d(I,J)   = 0.0D0
            END DO
        END DO 

        e(1,2,3) = 1.0D0
        e(2,3,1) = 1.0D0
        e(3,1,2) = 1.0D0
        e(2,1,3) = -1.0D0
        e(3,2,1) = -1.0D0
        e(1,3,2) = -1.0D0
        d(1,1)   = 1.0D0
        d(2,2)   = 1.0D0
        d(3,3)   = 1.0D0
!
! Qij = cos(phi)*dij + [1-cos(phi)]*pi*pj + sin(phi)*eijk*pk
!        
        DO L=1,24
        
        DO I=1,3
        DO J=1,3
            DO K=1,3
        Q(I,J,L) = Q(I,J,L) + (gr(5,L))*e(I,J,K)*gr(K,L)
            END DO
        Q(I,J,L) = Q(I,J,L) + (gr(4,L))*d(I,J)                       +  &
                              ( 1.0D0 - gr(4,L) )*gr(I,L)*gr(J,L)
        END DO
        END DO

        END DO
        end 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          traceRQR
! Functionality: calculate the trace of the matricial product R2xQxR1
!                function needed when defining misorientation...
!
! Input(s) :     R1 : Rotation matrix R1
!                (note that R1 has to be the inverse of
!                 the actual R1 matrix)
!                R2 : Rotation matrix R2
!                Q  : Symmetry group rotation matrix
!        
!
! Output(s):     tr : trace of R2xQxR1 
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine traceRQR(tr,R2,Q,R1)

        double precision R2(3,3), R1(3,3), Q(3,3), R12tmp(3,3), R12(3,3)
        double precision tr
        integer I, J, K
        
        DO I=1,3
        DO J=1,3
          R12tmp(I,J) = 0.0D0
          R12(I,J)    = 0.0D0
        END DO
        END DO
        
        DO I=1,3
        DO J=1,3
           DO K=1,3
             R12tmp(I,J) = R12tmp(I,J) + Q(I,K)*R1(K,J)
           END DO
        END DO
        END DO

        DO I=1,3
        DO J=1,3
           DO K=1,3
             R12(I,J) = R12(I,J) + R2(I,K)*R12tmp(K,J)
           END DO
        END DO
        END DO
         
        tr = R12(1,1) + R12(2,2) + R12(3,3)
        
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          polar_decomp
! Functionality: Polar decomposition algorithm
!                written by Rebecca Brannon
!
! Input(s) :     Fe : Elastic Deformation Gradient 
!
! Output(s):     Re : Polar Rotation of Fe
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine polar_decomp(Re,Fe)

        double precision ONE, TWO, THREE, HALF, TX
        double precision Fe(3,3), Re(3,3)
        double precision A11, A21, A31, A12, A22, A32, A13, A23, A33
        double precision C_11, C_22, C_33, C_23, C_31, C_12, S, ERR
        double precision X11, X21, X31, X12, X22, X32, X13, X23, X33
        integer N
            
! Step 1: initialize counter
         N=0

         ONE   = 1.0D0
         TWO   = 2.0D0
         THREE = 3.0D0
         HALF  = 0.5D0
         TX    = 0.0001D0
         
! Step 2: Set initial guess for the polar rotation tensor:
         A11 = Fe(1,1)
         A21 = Fe(2,1)
         A31 = Fe(3,1)
         A12 = Fe(1,2)
         A22 = Fe(2,2)
         A32 = Fe(3,2)
         A13 = Fe(1,3)
         A23 = Fe(2,3)
         A33 = Fe(3,3)

! Step 3: Compute [C]=Transpose[A].[A]
         C_11 =  A11*A11 + A21*A21 + A31*A31
         C_22 =  A12*A12 + A22*A22 + A32*A32
         C_33 =  A13*A13 + A23*A23 + A33*A33
         C_23 =  A12*A13 + A22*A23 + A32*A33
         C_31 =  A13*A11 + A23*A21 + A33*A31
         C_12 =  A11*A12 + A21*A22 + A31*A32

! begin scaling block of code
         S=THREE/(C_11+C_22+C_33)

         C_11 =  S*C_11
         C_22 =  S*C_22
         C_33 =  S*C_33
         C_23 =  S*C_23
         C_31 =  S*C_31
         C_12 =  S*C_12
         S    =  SQRT(S)
         A11  =  S*A11
         A21  =  S*A21
         A31  =  S*A31
         A12  =  S*A12
         A22  =  S*A22
         A32  =  S*A32
         A13  =  S*A13
         A23  =  S*A23
         A33  =  S*A33

! end scaling block of code
    3 CONTINUE

! Step 4: Compute a scalar measure of the error. ( [E] = [C] - [I] )
         ERR =  SQRT( (C_11-ONE)**2 + (C_22-ONE)**2                  +  &
                      (C_33-ONE)**2 + TWO*( C_12*C_12                +  &
                       C_23*C_23 + C_31*C_31) )

! Step 5: Check for convergence
         IF   ( (ERR*TX+ONE.EQ.ONE) .OR. (N.GT.99)  ) GO TO 9

! Step 6: Getting here means not yet converged. Improve the solution.
         X11 = A11*(THREE-C_11) - A12*C_12 - A13*C_31
         X21 = A21*(THREE-C_11) - A22*C_12 - A23*C_31
         X31 = A31*(THREE-C_11) - A32*C_12 - A33*C_31
         X12 = A12*(THREE-C_22) - A13*C_23 - A11*C_12
         X22 = A22*(THREE-C_22) - A23*C_23 - A21*C_12
         X32 = A32*(THREE-C_22) - A33*C_23 - A31*C_12
         X13 = A13*(THREE-C_33) - A11*C_31 - A12*C_23
         X23 = A23*(THREE-C_33) - A21*C_31 - A22*C_23
         X33 = A33*(THREE-C_33) - A31*C_31 - A32*C_23

! Now finish the fixed point function to improve the solution.
! [A] = .5 [X]
         A11 = HALF * X11
         A21 = HALF * X21
         A31 = HALF * X31
         A12 = HALF * X12
         A22 = HALF * X22
         A32 = HALF * X32
         A13 = HALF * X13
         A23 = HALF * X23
         A33 = HALF * X33
    
! Step 7 and step 8: Increment the counter and reiterate
         N=N+1
         C_11 =  A11*A11 + A21*A21 + A31*A31
         C_22 =  A12*A12 + A22*A22 + A32*A32
         C_33 =  A13*A13 + A23*A23 + A33*A33
         C_23 =  A12*A13 + A22*A23 + A32*A33
         C_31 =  A13*A11 + A23*A21 + A33*A31
         C_12 =  A11*A12 + A21*A22 + A31*A32
      GO TO 3

! Step 9: Reaching here means convergence! Save rotation and compute stretch
    9 CONTINUE

         Re(1,1) = A11
         Re(2,1) = A21
         Re(3,1) = A31
         Re(1,2) = A12
         Re(2,2) = A22
         Re(3,2) = A32
         Re(1,3) = A13
         Re(2,3) = A23
         Re(3,3) = A33

        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Symm_Schmid
! Functionality: Calculate the symmetric part of the Schmid tensor
!
! Input(s) :     vect_d : vector of plane directions
!                vect_n : vector of normal directions
!
! Output(s):     Symm_s  : Symmetric part of the Schmid tensor
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine Symm_Schmid(Symm_s,vect_d,vect_n)

        double precision vect_d(12,3), vect_n(12,3), Symm_s(12,3,3)    ,  &
                         Schmid(12,3,3)
        integer I, J, K 
!
        DO I=1,12
           DO J=1,3
           DO K=1,3
              Symm_s(I,J,K) = 0.0D0
           END DO
           END DO
       END DO
!        
          call Schmid_tensor(Schmid,vect_d,vect_n)

          DO K=1,12

             DO I=1,3
             DO J=1,3

             Symm_s(K,I,J) = 0.50D0*( Schmid(K,I,J) + Schmid(K,J,I) )
             
             END DO
             END DO
          
          END DO
  
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Schmid_tensor
! Functionality: Calculate the Schmid tensor
!
! Input(s) :     vect_d : vector of plane directions
!                vect_n : vector of normal directions
!
! Output(s):     Schmid : Schmid tensor
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       subroutine Schmid_tensor(Schmid,vect_d,vect_n)

        double precision vect_d(12,3), vect_n(12,3), Schmid(12,3,3)
        integer I, J, K
!
        DO K=1,12
           DO I=1,3
           DO J=1,3
              Schmid(K,I,J) = 0.0D0
           END DO
           END DO
        END DO
!      
         DO I=1,3
           DO J=1,3
            Schmid( 1,I,J) = vect_d(1,I)*vect_n(1,J)
            Schmid( 2,I,J) = vect_d(2,I)*vect_n(2,J)
            Schmid( 3,I,J) = vect_d(3,I)*vect_n(3,J)
            Schmid( 4,I,J) = vect_d(4,I)*vect_n(4,J)
            Schmid( 5,I,J) = vect_d(5,I)*vect_n(5,J)
            Schmid( 6,I,J) = vect_d(6,I)*vect_n(6,J)
            Schmid( 7,I,J) = vect_d(7,I)*vect_n(7,J)
            Schmid( 8,I,J) = vect_d(8,I)*vect_n(8,J)
            Schmid( 9,I,J) = vect_d(9,I)*vect_n(9,J)
            Schmid( 10,I,J) = vect_d(10,I)*vect_n(10,J)
            Schmid( 11,I,J) = vect_d(11,I)*vect_n(11,J)
            Schmid( 12,I,J) = vect_d(12,I)*vect_n(12,J)
           END DO
         END DO
  
       end 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          push_forwN
! Functionality: push forward to current config. 4 normal direction
!                vectors
!
!                N(i) = n(j) x Fe^-1(j,i) for 4 normal vectors
!                [to keep orthogonality of slip planes and normals]
!
! Input(s) :     vect_n : 4 normal directions to push forward
!                Fe     : Elastic Deformation Gradient 
!
! Output(s):     curr_n : 4 normal directions in current configuration
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine push_forwN(curr_n,vect_n,Fe)

        double precision val_normN
        double precision curr_n(12,3), vect_n(12,3), Fe(3,3), Feinv(3,3)
        integer I, J
!
        DO J=1,12
         DO I=1,3
          curr_n(J,I) = 0.0D0
         END DO
        END DO
!
        call inversemat(Feinv,Fe) 
!        
          DO J=1,12
              curr_n(J,1) = Feinv(1,1)*vect_n(J,1)                   +  &
                            Feinv(2,1)*vect_n(J,2)                   +  &
                            Feinv(3,1)*vect_n(J,3) 
              curr_n(J,2) = Feinv(1,2)*vect_n(J,1)                   +  &
                            Feinv(2,2)*vect_n(J,2)                   +  &
                            Feinv(3,2)*vect_n(J,3) 
              curr_n(J,3) = Feinv(1,3)*vect_n(J,1)                   +  &
                            Feinv(2,3)*vect_n(J,2)                   +  &
                            Feinv(3,3)*vect_n(J,3) 
            END DO
!
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          push_forwD
! Functionality: push forward to current config. 6  plane direction
!                vectors
!
!                D(i) = Fe(i,j) x d(j) for 6 plane direction vectors
!
! Input(s) :     vect_d : 6 plane directions to push forward
!                Fe     : Elastic Deformation Gradient 
!
! Output(s):     curr_d : 6 plane directions in current configuration
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine push_forwD(curr_d, vect_d, Fe)

        double precision val_normD
        double precision curr_d(12,3),vect_d(12,3),Fe(3,3)
        integer I, J
!
        DO J=1,12
           DO I=1,3
             curr_d(J,I) = 0.0D0
           END DO
        END DO
!
            DO J=1, 12
              curr_d(J,1) = vect_d(J,1)*Fe(1,1)                      +  &
                            vect_d(J,2)*Fe(1,2)                      +  &
                            vect_d(J,3)*Fe(1,3)
              curr_d(J,2) = vect_d(J,1)*Fe(2,1)                      +  &
                            vect_d(J,2)*Fe(2,2)                      +  &
                            vect_d(J,3)*Fe(2,3)
              curr_d(J,3) = vect_d(J,1)*Fe(3,1)                      +  &
                            vect_d(J,2)*Fe(3,2)                      +  &
                            vect_d(J,3)*Fe(3,3)
            END DO                 
!
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          inversemat
! Functionality: Inverse 3x3 matrix
!
! Input(s) :     A    : Matrix to inverse
!
! Output(s):     Ainv : Inversed matrix
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine inversemat(Ainv,A)

        double precision A(3,3), Ainv(3,3), detA, detA_inv
!
        DO I=1,3
        DO J=1,3
           Ainv(I,J) = 0.0D0
        END DO
        END DO
!
! Determinant        
!
        detA = A(1,1)*A(2,2)*A(3,3) + A(1,2)*A(2,3)*A(3,1)           +  &
               A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1)           -  &
               A(1,1)*A(2,3)*A(3,2) - A(1,2)*A(2,1)*A(3,3)

        detA_inv =  (1.0D0/detA)
!
! Inverse matrix
!          
        Ainv(1,1) = detA_inv *( A(2,2)*A(3,3) - A(2,3)*A(3,2) )
        Ainv(1,2) = detA_inv *( A(1,3)*A(3,2) - A(1,2)*A(3,3) )
        Ainv(1,3) = detA_inv *( A(1,2)*A(2,3) - A(1,3)*A(2,2) )
        Ainv(2,1) = detA_inv *( A(2,3)*A(3,1) - A(2,1)*A(3,3) )
        Ainv(2,2) = detA_inv *( A(1,1)*A(3,3) - A(1,3)*A(3,1) )
        Ainv(2,3) = detA_inv *( A(1,3)*A(2,1) - A(1,1)*A(2,3) )
        Ainv(3,1) = detA_inv *( A(2,1)*A(3,2) - A(2,2)*A(3,1) )
        Ainv(3,2) = detA_inv *( A(1,2)*A(3,1) - A(1,1)*A(3,2) )
        Ainv(3,3) = detA_inv *( A(1,1)*A(2,2) - A(1,2)*A(2,1) )

        end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Slip_syst
! Functionality: 12 slip systems from 12 slip plane normals and 12 directions 
!
! Input(s) :     vect_d, vect_n,  vect_n1
!   
! Output(s):     sn_temp,sd_temp,sn1_temp
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        subroutine Slip_syst(vect_d,vect_n,sd_tp,sn_tp)
 
        double precision sd_tp(12,3), sn_tp(12,3), &
                         vect_d(12,3), vect_n(12,3)
        integer I,J,K 
        
        DO I=1,3
           DO J=1,12
              sn_tp(J,I)    =  vect_n(J,I)
              sd_tp(J,I)    =  vect_d(J,I)
           ENDDO
        ENDDO
        
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          Non_Schmid
! Functionality: Calculate Ptot, Pns
!
! Input(s) :     vect_d, vect_n, Ps
!
! Output(s):     Pns
!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        subroutine Non_Schmid(Ptot,Pns,Ps,vect_d,vect_n,bcc_var)

        double precision vect_d(12,3), vect_n(12,3), Ps(12,3,3)        ,& 
                         Ptot(12,3,3), Pns(12,3,3),temp1(12,3,3)      ,&
                         NSch1(12,3,3), NSch2(12,3,3), NSch3(12,3,3)  ,&
                         NSch4(12,3,3), bcc_var(5), dxn(12,3)         ,&
                         sn_tp(12,3),sd_tp(12,3),NSch5(12,3,3)
                         
        integer I, J, K 

        call Slip_syst(vect_d,vect_n,sd_tp,sn_tp)

       DO I=1,12
         dxn(I,1) = sd_tp(I,2) * sn_tp(I,3) - sd_tp(I,3) * sn_tp(I,2)
         dxn(I,2) = sd_tp(I,3) * sn_tp(I,1) - sd_tp(I,1) * sn_tp(I,3)
         dxn(I,3) = sd_tp(I,1) * sn_tp(I,2) - sd_tp(I,2) * sn_tp(I,1)
        ENDDO

       DO I=1,12
         DO J=1,3
         DO K=1,3
       
         NSch1(I,J,K) = 0.5*( dxn(I,J)*sd_tp(I,K)+ dxn(I,K) * sd_tp(I,J))
         NSch2(I,J,K) = 0.5*( dxn(I,J)*sn_tp(I,K)+ dxn(I,K) * sn_tp(I,J))
         NSch3(I,J,K) = sn_tp(I,J)*sn_tp(I,K)
         NSch4(I,J,K) = dxn(I,J)*dxn(I,K)
         NSch5(I,J,K) = sd_tp(I,J)*sd_tp(I,K)
        
         Pns(I,J,K) = bcc_var(1)*NSch1(I,J,K) + bcc_var(2)*NSch2(I,J,K) +  &
                      bcc_var(3)*NSch3(I,J,K) + bcc_var(4)*NSch4(I,J,K) +  & 
                      bcc_var(5)*NSch5(I,J,K)                 

         ENDDO
         ENDDO
        ENDDO

        END
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          DD_evol
! Functionality: calculate the evolution of statistically immob.
! dislocations based on a source and sink term
!
! Input(s) :     sid     - Statistically immob. disloc. at time t
!                cslip   - Cumulative slip at time t
!
! Output(s):     sid_nxt - Statistically immob. disloc. at timet+Dt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Subroutine DD_evol(ssd_nxt,ssd,cumul_slip,H1,H2)
           
        double precision ssd_nxt(12),ssd(12),cumul_slip(12),H1,H2
        double precision rhotot,dy,d_sid

           rhotot  = 0.0D0
            
            DO J=1,12
                rhotot  =  rhotot + (ssd(J))
            ENDDO
                !rhotot = rhotot/12

            DO J=1,12
              dy = DABS(cumul_slip(J))
!              d_sid= H1*(rhotot)**0.5 - H2*(rhotot)

              d_sid= H1*(rhotot)**0.5 - H2*(ssd(j))
              d_sid=max(d_sid,0.0E0)

              d_sid=d_sid*dy
              ssd_nxt(J) = ssd(J) + d_sid          
            END DO

            END

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          HAB_CAL
! Functionality: calculate hardening matrix
!
! Input(s) :     sd   - slip direction
!                sn   - slip normal
!
! Output(s):     HAB - hardening matrix
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Subroutine  HAB_calc(sd,sn,HAB)
           
      double precision sd(12,3),sn(12,3),HAB(12,12),sn_tp(12,3),sd_tp(12,3), &
                        eta(12,3)

    DO I=1,12
        DO J=1,12
            HAB(I,J)  = 0.0D0
        ENDDO
    ENDDO

    DO I=1,3
         DO J=1,12
         sn_tp(J,I)    =  sn(J,I)
         sd_tp(J,I)    =  sd(J,I)
         ENDDO
    ENDDO

    DO I=1,12
      eta(I,1) = sd_tp(I,2) * sn_tp(I,3) - sd_tp(I,3) * sn_tp(I,2)
      eta(I,2) = sd_tp(I,3) * sn_tp(I,1) - sd_tp(I,1) * sn_tp(I,3)
      eta(I,3) = sd_tp(I,1) * sn_tp(I,2) - sd_tp(I,2) * sn_tp(I,1)
    ENDDO

    DO I=1,12
       DO J=1,12
         HAB(I,J)=sn_tp(I,1)*eta(J,1)+sn_tp(I,2)*eta(J,2)+sn_tp(I,3)*eta(J,3)
         HAB(I,J)=DABS(HAB(I,J))
       ENDDO
    ENDDO
 
    END

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Name:          SLIPSYS
! Functionality: calculate hardening matrix
!
! Input(s) :     sd   - slip direction
!                sn   - slip normal
!
! Output(s):     HAB - hardening matrix
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Subroutine SLIPSYS(Flag_Sys,sd,sn)

    double precision sd(12,3),sn(12,3)
    integer I, J, K, L, L0, Flag_Sys

    DO I=1,12
      DO J=1,3
       sd(I,J)=0.0D0
       sn(I,J)=0.0D0
      ENDDO
    ENDDO

!         Define the 8 <111> BCC slip directions

IF (Flag_sys.EQ.1.OR.Flag_sys.EQ.2) THEN 
          sd(1,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]
          sd(1,2)   =  1.0/SQRT(3.0)
          sd(1,3)   =  1.0/SQRT(3.0)
          sd(2,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]
          sd(2,2)   =  1.0/SQRT(3.0)
          sd(2,3)   =  1.0/SQRT(3.0)
          sd(3,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]
          sd(3,2)   =  1.0/SQRT(3.0)
          sd(3,3)   =  1.0/SQRT(3.0)
          sd(4,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]
          sd(4,2)   =  1.0/SQRT(3.0)
          sd(4,3)   =  1.0/SQRT(3.0)
          sd(5,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]   
          sd(5,2)   =  1.0/SQRT(3.0)
          sd(5,3)   =  1.0/SQRT(3.0)
          sd(6,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]
          sd(6,2)   =  1.0/SQRT(3.0)
          sd(6,3)   =  1.0/SQRT(3.0)        
          sd(7,1)   = -1.0/SQRT(3.0) ![-1,-1, 1]
          sd(7,2)   = -1.0/SQRT(3.0)
          sd(7,3)   =  1.0/SQRT(3.0)
          sd(8,1)   = -1.0/SQRT(3.0) ![-1,-1, 1] 
          sd(8,2)   = -1.0/SQRT(3.0)
          sd(8,3)   =  1.0/SQRT(3.0)
          sd(9,1)   = -1.0/SQRT(3.0) ![-1,-1, 1]   
          sd(9,2)   = -1.0/SQRT(3.0)
          sd(9,3)   =  1.0/SQRT(3.0)
          sd(10,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1]
          sd(10,2)  = -1.0/SQRT(3.0)
          sd(10,3)  =  1.0/SQRT(3.0)
          sd(11,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1]
          sd(11,2)  = -1.0/SQRT(3.0)
          sd(11,3)  =  1.0/SQRT(3.0)
          sd(12,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1] 
          sd(12,2)  = -1.0/SQRT(3.0)
          sd(12,3)  =  1.0/SQRT(3.0)
ENDIF

!      Define the 12 {110} BCC slip plane normals 

IF (Flag_sys.EQ.1) THEN   
          sn(1,1) =  0              !( 0 1-1)
          sn(1,2) =  1.0/SQRT(2.0)
          sn(1,3) = -1.0/SQRT(2.0) 
          sn(2,1) = -1.0/SQRT(2.0)  !(-1 0 1) 
          sn(2,2) =  0
          sn(2,3) =  1.0/SQRT(2.0)
          sn(3,1) =  1.0/SQRT(2.0)  !( 1-1 0)
          sn(3,2) = -1.0/SQRT(2.0)
          sn(3,3) =  0
          sn(4,1) = -1.0/SQRT(2.0)  !(-1 0-1)
          sn(4,2) =  0
          sn(4,3) = -1.0/SQRT(2.0)   
          sn(5,1) =  0              !( 0-1 1)
          sn(5,2) = -1.0/SQRT(2.0)
          sn(5,3) =  1.0/SQRT(2.0) 
          sn(6,1) =  1.0/SQRT(2.0)  !( 1 1 0)
          sn(6,2) =  1.0/SQRT(2.0)
          sn(6,3) =  0
          sn(7,1) =  0              !( 0-1-1)
          sn(7,2) = -1.0/SQRT(2.0)
          sn(7,3) = -1.0/SQRT(2.0)  
          sn(8,1) =  1.0/SQRT(2.0)  !( 1 0 1)
          sn(8,2) =  0
          sn(8,3) =  1.0/SQRT(2.0)
          sn(9,1) = -1.0/SQRT(2.0)  !(-1 1 0) 
          sn(9,2) =  1.0/SQRT(2.0)
          sn(9,3) =  0
          sn(10,1) =  1.0/SQRT(2.0) !( 1 0-1) 
          sn(10,2) =  0
          sn(10,3) = -1.0/SQRT(2.0)
          sn(11,1) =  0             !( 0 1 1)
          sn(11,2) =  1.0/SQRT(2.0)
          sn(11,3) =  1.0/SQRT(2.0)
          sn(12,1) = -1.0/SQRT(2.0) !(-1-1 0) 
          sn(12,2) = -1.0/SQRT(2.0)
          sn(12,3) =  0
endif

!      Define the 12 {112} BCC slip plane normals 
IF (Flag_sys.EQ.2) THEN
          sn(1,1) =  1.0/SQRT(6.0)  !( 1 1-2)
          sn(1,2) =  1.0/SQRT(6.0)
          sn(1,3) = -2.0/SQRT(6.0) 
          sn(2,1) = -2.0/SQRT(6.0)  !(-2 1 1)
          sn(2,2) =  1.0/SQRT(6.0)
          sn(2,3) =  1.0/SQRT(6.0)
          sn(3,1) =  1.0/SQRT(6.0)  !( 1-2 1)
          sn(3,2) = -2.0/SQRT(6.0)
          sn(3,3) =  1.0/SQRT(6.0)
          sn(4,1) = -1.0/SQRT(6.0)  !(-1 1-2)
          sn(4,2) =  1.0/SQRT(6.0)
          sn(4,3) = -2.0/SQRT(6.0)
          sn(5,1) = -1.0/SQRT(6.0)  !(-1-2 1)
          sn(5,2) = -2.0/SQRT(6.0)
          sn(5,3) =  1.0/SQRT(6.0) 
          sn(6,1) =  2.0/SQRT(6.0)  !( 2 1 1)
          sn(6,2) =  1.0/SQRT(6.0)
          sn(6,3) =  1.0/SQRT(6.0)
          sn(7,1) = -1.0/SQRT(6.0)  !(-1-1-2)
          sn(7,2) = -1.0/SQRT(6.0)
          sn(7,3) = -2.0/SQRT(6.0)
          sn(8,1) =  2.0/SQRT(6.0)  !( 2-1 1)
          sn(8,2) = -1.0/SQRT(6.0)
          sn(8,3) =  1.0/SQRT(6.0)
          sn(9,1) = -1.0/SQRT(6.0)  !(-1 2 1)
          sn(9,2) =  2.0/SQRT(6.0)
          sn(9,3) =  1.0/SQRT(6.0)
          sn(10,1) = 1.0/SQRT(6.0)  !( 1-1-2)
          sn(10,2) =-1.0/SQRT(6.0)
          sn(10,3) =-2.0/SQRT(6.0)
          sn(11,1) = 1.0/SQRT(6.0)  !( 1 2 1)
          sn(11,2) = 2.0/SQRT(6.0)
          sn(11,3) = 1.0/SQRT(6.0)
          sn(12,1) =-2.0/SQRT(6.0)  !(-2-1 1)
          sn(12,2) =-1.0/SQRT(6.0)
          sn(12,3) = 1.0/SQRT(6.0)
ENDIF

!      Define the 12 {111} FCC slip plane normals 
IF (Flag_sys.EQ.4) THEN
          sd(1,1) =  0              !( 0 1-1)
          sd(1,2) =  1.0/SQRT(2.0)
          sd(1,3) = -1.0/SQRT(2.0)          
          sd(2,1) =  1.0/SQRT(2.0)  !( 1 0 1) 
          sd(2,2) =  0
          sd(2,3) =  1.0/SQRT(2.0)
          sd(3,1) =  1.0/SQRT(2.0)  !( 1 1 0)
          sd(3,2) =  1.0/SQRT(2.0)
          sd(3,3) =  0
          sd(4,1) =  0              !( 0-1 1)
          sd(4,2) = -1.0/SQRT(2.0)
          sd(4,3) =  1.0/SQRT(2.0) 
          sd(5,1) = -1.0/SQRT(2.0)  !(-1 0 1)
          sd(5,2) =  0
          sd(5,3) =  1.0/SQRT(2.0) 
          sd(6,1) = -1.0/SQRT(2.0)  !(-1 1 0)
          sd(6,2) =  1.0/SQRT(2.0)
          sd(6,3) =  0
          sd(7,1) =  0              !( 0 1 1)
          sd(7,2) =  1.0/SQRT(2.0)
          sd(7,3) =  1.0/SQRT(2.0)  
          sd(8,1) =  1.0/SQRT(2.0)  !( 1 0 1)
          sd(8,2) =  0
          sd(8,3) =  1.0/SQRT(2.0)
          sd(9,1) = -1.0/SQRT(2.0)  !(-1 1 0) 
          sd(9,2) =  1.0/SQRT(2.0)
          sd(9,3) =  0
          sd(10,1) =  0             !( 0 1 1) 
          sd(10,2) =  1.0/SQRT(2.0)
          sd(10,3) =  1.0/SQRT(2.0)
          sd(11,1) = -1.0/SQRT(2.0) !(-1 0 1)
          sd(11,2) =  0
          sd(11,3) =  1.0/SQRT(2.0)
          sd(12,1) =  1.0/SQRT(2.0) !( 1 1 0) 
          sd(12,2) =  1.0/SQRT(2.0)
          sd(12,3) =  0

          sn(1,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]
          sn(1,2)   =  1.0/SQRT(3.0)
          sn(1,3)   =  1.0/SQRT(3.0)
          sn(2,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]
          sn(2,2)   =  1.0/SQRT(3.0)
          sn(2,3)   =  1.0/SQRT(3.0)
          sn(3,1)   = -1.0/SQRT(3.0) ![-1, 1, 1]
          sn(3,2)   =  1.0/SQRT(3.0)
          sn(3,3)   =  1.0/SQRT(3.0)
          sn(4,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]
          sn(4,2)   =  1.0/SQRT(3.0)
          sn(4,3)   =  1.0/SQRT(3.0)
          sn(5,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]   
          sn(5,2)   =  1.0/SQRT(3.0)
          sn(5,3)   =  1.0/SQRT(3.0)
          sn(6,1)   =  1.0/SQRT(3.0) ![ 1, 1, 1]
          sn(6,2)   =  1.0/SQRT(3.0)
          sn(6,3)   =  1.0/SQRT(3.0)        
          sn(7,1)   = -1.0/SQRT(3.0) ![-1,-1, 1]
          sn(7,2)   = -1.0/SQRT(3.0)
          sn(7,3)   =  1.0/SQRT(3.0)
          sn(8,1)   = -1.0/SQRT(3.0) ![-1,-1, 1] 
          sn(8,2)   = -1.0/SQRT(3.0)
          sn(8,3)   =  1.0/SQRT(3.0)
          sn(9,1)   = -1.0/SQRT(3.0) ![-1,-1, 1]   
          sn(9,2)   = -1.0/SQRT(3.0)
          sn(9,3)   =  1.0/SQRT(3.0)
          sn(10,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1]
          sn(10,2)  = -1.0/SQRT(3.0)
          sn(10,3)  =  1.0/SQRT(3.0)
          sn(11,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1]
          sn(11,2)  = -1.0/SQRT(3.0)
          sn(11,3)  =  1.0/SQRT(3.0)
          sn(12,1)  =  1.0/SQRT(3.0) ![ 1,-1, 1] 
          sn(12,2)  = -1.0/SQRT(3.0)
          sn(12,3)  =  1.0/SQRT(3.0)

ENDIF

END
