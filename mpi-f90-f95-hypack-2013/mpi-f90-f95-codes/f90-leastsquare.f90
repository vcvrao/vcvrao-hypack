!
!************************************************************************
!
!          C-DAC Tech Workshop : hyPACK-2013 
!                Oct 15 - 18, 2013
!
!   Example 6  : f90-leastsquare.f90
!
!   Objective  : Data fitting by Least Square approximation
!                This example demonstrates the use of
!                Allocatable arrays and Intrinsic function MATMUL
!               
!   Input      : Read Data set from file 'least_sq.inp'
!
!   Output     : The unstressed length of the wire and its 
!                Young's modulus value.
!
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!****************************************************************************
!
MODULE constants
	IMPLICIT NONE

! This module contains the physical and other constants for use 
! with the program youngs_modulus

! Define a real kind type q with at least 6 decimal digits and an 
! exponent range from 10**30 to 10** (-30)

		INTEGER, PARAMETER :: q = SELECTED_REAL_KIND(P=6, R=30)

		! Define pi
      REAL(KIND=q), PARAMETER :: pi = 3.1415926536_q

		! Define the mass to weight conversion factor
      REAL(KIND=q), PARAMETER :: g = 386.0_q

		! Define the size of the largest problem set that can be processed

		INTEGER, PARAMETER :: max_dat=100

END MODULE constants

PROGRAM youngs_modulus
	USE constants
	IMPLICIT NONE

		! This program calculates Young's modulus for a piece of wire using experimental data, and also calculates the 
		! unstretched lenght of the wire

		! Input variables
		REAL (KIND=q), DIMENSION(max_dat) :: wt, len
		REAL (KIND=q) :: diam
		INTEGER ::n_sets

		! Other variables
		REAL (KIND=q) :: k, l, e
		INTEGER::i

		! Read data
      OPEN (UNIT=5, FILE = "least_sq.inp")
      READ(UNIT=5,FMT=*) n_sets
      PRINT *,"Number of Data Sets is ", n_sets

		! End execution if too much or too little data
		SELECT CASE (n_sets)
		CASE (max_dat+1:)
			PRINT*,"Too much data!"
			PRINT*,"Maximum permitted is",max_dat,"data sets"
			STOP
		CASE (:1)
			PRINT *,"Not enough data!"
			PRINT *,"There must be at least 2 data sets"
			STOP
      END SELECT

      DO i= 1, n_sets
         READ (UNIT=5, FMT=*)wt(i), len(i)
      END DO
      READ (UNIT=5,FMT=*) diam

      CLOSE(UNIT=5)

      PRINT *,"data in pairs, weight (in lbs), &
               &length (in inches)"
      DO i= 1, n_sets
         PRINT *,wt(i), len(i)
      END DO
      PRINT *,"The diameter of the wire (in ins.)?",diam

! Convert mass to weight
wt=g*wt

! Calculate mass to weight
wt = g*wt

!Calculate least square fit
CALL least_squares_line(n_sets,wt,len,k,l)

! Calculate Young's modulus
e= (4.0_q*1)/(pi*diam*diam*k)

! PRINT results

!PRINT '(//,5x, "The unstressed lenght of the wire is",F7.3,"ins.")',1
!PRINT '(5x,"Its Young's modulus is ", E10.4,"lbs/in/sec/sec"//)',e

PRINT '("The unstressed lenght of the wire is",F7.3,"ins.")',l
PRINT '("Its Youngs modulus is ", E10.4,"lbs/in/sec/sec")',e
END PROGRAM youngs_modulus

SUBROUTINE least_squares_line(n,x,y,a,b)
USE constants
IMPLICIT NONE

! This subroutine calculates the least square fit line ax+b to the x-y 
! data pairs

! Dummy arguments
INTEGER, INTENT (IN) :: n
REAL (KIND=q), DIMENSION(n), INTENT(IN) :: x,y
REAL (KIND=q), INTENT(OUT) :: a,b

! Local variables
REAL(KIND=q) :: sum_x, sum_y, sum_xy, sum_x_sq

! Calculate sums
sum_x = SUM(x)
sum_y = SUM(y)
sum_xy = DOT_PRODUCT(x,y)
sum_x_sq = DOT_PRODUCT(x,x)

! Calculate coefficients of least squares fit line
a = (sum_x*sum_y - n*sum_xy) / (sum_x*sum_x - n*sum_x_sq)
b =   (sum_y - a*sum_x)/n

END SUBROUTINE least_squares_line
