!
! ***********************************************************************
!          C-DAC Tech Workshop : hyPACK-2013 
!                 Oct 15 - 18, 2013
!
!   Example 1  : f90-hello-world.f90
!
!   Objective  : Fortran90 program to print "Hello WorldC" 
!               
!   Input      : None
!
!   Output     : Message "Hello World ... " on screen.
!   
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
! **********************************************************************
!

program HelloWorld

 IMPLICIT NONE

 character(len=12) :: String

 String = "Hello World...!"

 print*,String

end program HelloWorld
