!
!******************************************************************************
!             C-DAC Tech Workshop : hyPACK-2013 
!                     Oct 15 - 18, 2013
!
!   Example 2  : f90-welcome.f90
!
!   Objective  : Print welcome message using the full name and first name
!                This example demonstrates the use of
!                simple character manipulation
!               
!   Input      : Title, first name and last name of user.
!
!   Output     : Welcome messsge using both full name and first name.
!
!  
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!************************************************************************

        PROGRAM Welcome
	IMPLICIT NONE
	! This program manipulates character strings to produce a 
	! properly formatted welcome message

	! Variable declarations
	CHARACTER (LEN=20) :: title, first_name, last_name
	CHARACTER (LEN=40) ::  full_name

	! Ask for name, etc
	PRINT *, "Please give your full name n the form requested"
	PRINT *, "Title (Mr./Mrs./Ms./Professor/etc:)"
	READ *, title

	PRINT *, "First name:"
	READ *, first_name

	PRINT *,"Last name:"
	READ *, last_name

	! Create full name
	full_name=TRIM(title)//" "//TRIM(first_name)//" "//last_name

	! Print messages
	PRINT *, "Welcome ", full_name
	PRINT *, "May I call you ", TRIM(first_name), "?"

        END PROGRAM Welcome
