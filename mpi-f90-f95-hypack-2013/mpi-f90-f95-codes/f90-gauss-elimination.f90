!******************************************************************************
!             C-DAC Tech Workshop : hyPACK-2013 
!                     Oct 15 - 18, 2013
!
!   Example 8  : f90-gauss-elimination.f90
!
!   Objective  : Solution of linear equation using Gauss Elimination
!                This example demonstrates the use of
!                Procedures, and how to encapsulate these procedures 
!                in a module, which will make only the solving procedure 
!                public.
!               
!   Input      : Real Symmetric Positive definite Matrix a and
!                vector b. Read file gauss.inp for Matrix 
!                and Vector.
!
!   Output     : The solution of  Ax = b
!
!  
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!******************************************************************************


PROGRAM GaussElimination
   
   USE linear_equations
   IMPLICIT NONE

! Allocate array for coefficients 

   REAL, ALLOCATABLE, DIMENSION(:, :) :: a
   REAL, ALLOCATABLE, DIMENSION(:) :: b

! Size of arrays 

  INTEGER :: n

! Loop variables and error flags

   INTEGER :: i, j, error

! Read Matrix A & Vector B from file

   OPEN(UNIT=5, FILE = "gauss.inp")
   READ(UNIT=5,FMT=*) n
   PRINT *, n
   ALLOCATE (a(n, n), b(n))
   DO i = 1,n
      READ(UNIT=5,FMT=*) (a(i,j),j=1,n)
      PRINT *, (a(i,j),j=1,n)
   ENDDO
   DO i = 1,n
      READ(UNIT=5,FMT=*) b(i)
      PRINT *, b(i)
   ENDDO
   CLOSE(UNIT=5)

! Attempt to solve system of equations 

   CALL gaussian_solve(a, b, error)

! Check to see if there were any errors

   IF(error <= -1 .AND. error >= -3) THEN
     PRINT *, "Error in call to gaussian_solve"
   ELSE IF(error == -4) THEN
     PRINT *, "System is degenerate"
   ELSE 
     PRINT *, " "
     PRINT *, "The solution is"
     PRINT '(1X,"x(",I2,")=", F6.2)',(i,b(i), i=1,n)
   END IF

END PROGRAM GaussElimination

