!
!****************************************************************************
!
!		C-DAC Tech Workshop : hyPACK-2013
!                           October 15-18, 2013
!
!   Objective : Read the coordinates of two distinct points, 
!               calculate the line joining them and print the 
!               equation of line.
!               This example demonstrates the use of
!               Creation of your own data types
!               
!   Input     : Coordinates of two distinct points.
!
!   Output    : Equation of line joining two points.
!
!   Created             : August-2013
!
!   E-mail              : hpcfte@cdac.in     
!
!
!***************************************************************************

PROGRAM geometry
	IMPLICIT NONE
	
	! A program to use derived types for two-dimensional geometric calculations

	! Type definitions
	TYPE point
		REAL :: x,y	! Cartesian coordinates of the point
	END TYPE point

	TYPE line
		REAL :: a, b, c ! coefficients of defining equation
	END TYPE line

	! Variable declarations
	TYPE (point) :: p1, p2
	TYPE (line)   :: p1_to_p2

	! Read data
	PRINT *, "Please type co-ordinates of first point"
	READ *, p1
	PRINT *, "Please type co-ordinates of second point"
	READ *, p2

	! Calculate coefficients of equation representing the line
	p1_to_p2%a = p2%y - p1%y
	PRINT *,  "where p2 = ",p2
	p1_to_p2%b = p1%x - p2%x
	p1_to_p2%c = p1%y*p2%x  - p2%y*p1%x 

	! Print result
	PRINT *, "The equation of the line joining these two points is"
	PRINT *,  "ax + by + c = 0"
	PRINT *,  "where a = ",p1_to_p2%a
	PRINT *,  "where b = ",p1_to_p2%b
	PRINT *,  "where c = ",p1_to_p2%c

END PROGRAM geometry  


	
