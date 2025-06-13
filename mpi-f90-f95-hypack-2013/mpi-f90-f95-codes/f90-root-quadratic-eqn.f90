!
!********************************************************************
!           C-DAC Tech Workshop : hyPACK-2013 
!                Oct 15 - 18, 2013
!
!   Example 5  : f90-root-quadratic-eqn.f90
!
!   Objective  : Roots of a quadratic equation This example demonstrates 
!                the use of CASE statement
!               
!   Input      : Coefficients of a quadratic equation
!
!   Output     : Roots of a quadratic equation
!
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!**********************************************************************
!
PROGRAM Root_Quad_Eqan_CASE
! A program to solve a quadratic equation using a CASE
! statement to distinguish between the three cases

! Constant declaration
REAL, PARAMETER :: epsilon=1E-6

! Variable declarations
REAL :: a,b,c,d,sqrt_d,x1,x2
INTEGER :: selector

! Read coefficients
    PRINT *,"Please type the three coefficients a, b, and c"
    READ *, a, b, c

! Calculate b**2-4*a*c and resulting case selector
    d=b**2 - 4.0*a*c
    selector = d/epsilon

! Calculate and print roots, if any
    SELECT CASE (selector)
       CASE (1:)
	       ! Two roots
	       sqrts_d=SQRT(d)
	       x1=(-b+sqrts_d)/(2*a)
	       x2=(-b-sqrts_d)/(2*a)
	       PRINT *, "The equation has two roots;",x1," and ",x2

       CASE (0)
	       ! One root
	       x1=-b/(a+a)
	       PRINT *, "The equation has one root;",x1

	    CASE (:-1)
		    ! No roots
		    PRINT *, "The equation has no real roots"

    END SELECT

END PROGRAM Root_Quad_Eqan_CASE
