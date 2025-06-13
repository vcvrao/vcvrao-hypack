!
!***********************************************************************
!             C-DAC Tech Workshop : hyPACK-2013 
!                     Oct 15 - 18, 2013
!
!  Example 8. Gauss_Modules.f90
!
!  Description : Two subroutines 'gaussian_elemination' and 
!                'back_substitution' are called by another subroutine 
!                'gaussian_solve'. We do not want a user to be able to 
!                call 'gaussian_elimination' or 'back_substitution' directly. 
!                As well as the subroutine 'gaussian_solve' will use 
!                assumed-shape dummy arguments, it must have an explicit 
!                interface in the main program. To accomplish both of these 
!                aims, all three subroutines are put in a module called 
!                'Linear_Equations' being public.
!
!  Input         : Input Matrices 
!
!  Output        : Soluiton of Matrix System of Linear Equations 
!
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!************************************************************************

MODULE Linear_Equations
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: gaussian_solve

  CONTAINS


  !******************************************************
  !Subroutine gaussian_solve

  SUBROUTINE gaussian_solve(a, b, error)
      ! This subroutine solves the linear system Ax=b
      ! where the coefficients of A are stored in the array a
      ! The solution is put in the array b
      ! error indicates is errors are found

      ! Dummy arguments
      REAL, DIMENSION(:,:), INTENT(INOUT) :: a 
      REAL, DIMENSION(:), INTENT(INOUT) :: b 
      INTEGER, INTENT(OUT) :: error

      ! Reduce the equation by gaussian elimination

      CALL gaussian_elimination(a,b,error)

      ! If reduction was succesful, calculate solution by
      ! back substitution

      IF(error == 0) CALL back_substitution(a,b,error)
  END SUBROUTINE gaussian_solve


  !******************************************************
  ! Subroutine gaussian elimination

  SUBROUTINE gaussian_elimination(a,b,error)
     ! The subroutine performs Gaussian elimination on a
     ! system of linear equations


     ! Dummy arguments
     ! a contains the coefficients
     ! b contains the right-hand side
    
     REAL, DIMENSION (:,:), INTENT(INOUT) :: a
     REAL, DIMENSION (:) :: b
     INTEGER, INTENT(INOUT) :: error


     ! Local variables
     REAL, DIMENSION(SIZE(a,1)) :: temp_array ! Automatic array
     INTEGER, DIMENSION(1) :: ksave
     INTEGER :: i,j,k,n
     REAL :: temp, m

     ! Validity checks

     n = SIZE(a,1)
     IF( n == 0) THEN
       error = -1              ! No problem to solve
       RETURN
     END IF
 
     IF( n /= SIZE(a,2)) THEN
       error = -2              ! a is not square
       RETURN
     END IF

     IF( n /= SIZE(b)) THEN
       error = -3              ! Size of b does not match
       RETURN
     END IF

     ! Dimension of arrays are OK, so go ahead with Gaussian 
     ! elimination

     error = 0
     DO i=1, n-1
       ! Find row with largest value of | a(j,i)|, j=i, ...,n
       ksave = MAXLOC(ABS(a(i:n, i)))

       ! Check whether largest |a(j,i)| is zero
       k = ksave(1) + i - 1
       IF(ABS(a(k, i)) <= 1E-5) THEN
          error = -4           ! No solution possible
          RETURN
       END IF

       ! Interchange row i and k, if necessary
       IF(k/=i) THEN
         temp_array = a(i,:)
         a(i,:) = a(k,:)
         a(k,:) = temp_array

         ! Interchanging corresponding elements of b

         temp = b(i)
         b(i) = b(k)
         b(k) = temp
       END IF

       ! Subtract multiples of row i from subsequent rows to
       ! zero all subsequent coefficients of x sub i
 
       DO j = i+1, n
          m = a(j, i) / a(i, i)
          a(j,:) = a(j,:) - m*a(i, :)
          b(j) = b(j) - m*b(i)
       END DO
     END DO

  END SUBROUTINE gaussian_elimination


  !******************************************************
  ! Subroutine back_substitution         
  
  SUBROUTINE back_substitution(a, b, error)
    ! This subroutine performs back substitution once a system
    ! of equations has been reduced by Gaussian elimination

    ! Dummy arguments
    ! The array a contains the coefficients
    ! The array b contains right-hand side coefficients
    ! error will be set to non-zero if an error occurs

    REAL, DIMENSION(:,:), INTENT(IN) :: a
    REAL, DIMENSION(:), INTENT(INOUT) :: b
    INTEGER, INTENT(OUT) :: error

    ! Local variables
    REAL :: sum
    INTEGER :: i,j,n

    error = 0
    n = SIZE(b)
 
    ! Solve for each variable in turn
    DO i=n,1,-1
       ! Check for zero coefficient
       IF(ABS(a(i,i)) <= 1E-5) THEN
          error = -4
          RETURN
       ENDIF
       
       sum = b(i)
       DO j = i+1,n
          sum = sum - a(i,j)*b(j)
       END DO
       b(i) = sum / a(i,i)
    END DO
  
  END SUBROUTINE back_substitution

END MODULE Linear_Equations
