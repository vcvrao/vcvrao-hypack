!
!*******************************************************************
!             C-DAC Tech Workshop : hyPACK-2013 
!                     Oct 15 - 18, 2013
!
!   Example 7  : f90-newton-raphson-root.f90
!
!   Objective  : Newton's method for solving non-linear equations
!                This example demonstrates the use of
!                an external function to define the equation and its 
!                first deravitive
!               
!   Input      : Starting point for interploation
!
!   Output     : Root of the equation f(x) = 0 and values of f(x).
!
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!  
!**********************************************************************

PROGRAM NewtonRaphson

   IMPLICIT NONE
   
   ! Find root of the equation f(x) = 0 using
   ! Newton-Raphson method

   ! Input variables
   REAL, EXTERNAL :: f, f_prime
   INTEGER :: max_iter, error
   REAL :: epsilon, start, root
   REAL :: f_val

   PRINT *, "Input the starting value"
   READ *, start
    
!   PRINT *, "Input maximum number of iterations"
!   READ *, max_iter

   epsilon = 0.000001    
   max_iter = 100
   CALL Newton_Raphson(f, f_prime, start, epsilon, max_iter, root, error)

   f_val = f(root)
   PRINT '(2(A,E15.6))',"A root was found at x =", root,"  f(x) =", f_val 


END PROGRAM NewtonRaphson

REAL FUNCTION f(x)
   IMPLICIT NONE
   REAL, INTENT(IN) :: x
   f = x + EXP(x)
!   PRINT *,"f=",f
END FUNCTION f

REAL FUNCTION f_prime(x)
   IMPLICIT NONE
   REAL, INTENT(IN) :: x
   f_prime = 1.0 + EXP(x)
!   PRINT *,"f_prime=",f_prime
END FUNCTION f_prime


! ***************************************************************************

SUBROUTINE Newton_Raphson(f, f_prime, start, epsilon, max_iter, root, error)
 
   IMPLICIT NONE
   
   ! This subroutine finds a root of the equation f(x) = 0
   ! using Newton-Raphson method
   ! The function f_prime returns the value of the derivatives of
   ! the function f(x)

   ! Dummy arguments

   REAL, EXTERNAL :: f, f_prime
   REAL, INTENT(IN) :: start, epsilon
   INTEGER, INTENT(IN) :: max_iter

   REAL, INTENT(INOUT) :: root
   INTEGER, INTENT(OUT) :: error

   ! error indicates the result of the processing as follows :
   ! = 0 a root is found
   ! = -1 no root found after max_iter
   ! = -2 the first derivstive becomes zero, and so no further iterations possible
   ! = -3 the value of epsilon supplied is negative or zero.

   ! Local variables
 
   INTEGER :: i
   REAL :: f_val, f_der

   ! Check validity of epsilon
   IF(epsilon <= 0.0) THEN
     error = -3
     root = HUGE(root)
     RETURN
   END IF

   ! Begin the iteration up to the maximum number specified
   root = start

   DO i = 1, max_iter
      f_val = f(root)
      ! Output latest estimate of root value while testing
       PRINT '(2(A,E15.6))', "root =", root,"  f(root) =", f_val 
      IF(ABS(f_val ) <= epsilon) THEN
        ! A root has been found
        error = 0
        RETURN
      END IF
      f_der = f_prime(root)
      IF(f_der == 0.0) THEN
        ! f'(x)=0 so no more iterations are possible
        error = -2
        RETURN
      END IF
      
      ! Use Newton's iteration to obtain next approximation
      root = root - f_val/f_der
!     PRINT '(2(A,E15.6))', "New_root =", root
   END DO

   ! Process has not converged after max_iter iterations
   error = -1
 
END SUBROUTINE Newton_Raphson


