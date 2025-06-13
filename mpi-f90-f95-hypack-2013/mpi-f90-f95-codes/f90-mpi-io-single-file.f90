!
! *********************************************************************
!           C-DAC Tech Workshop : hyPACK-2013 
!                  Oct 15 - 18, 2013
!
! Example 10 	       : Fortran 90 MPI I/O to a single file
!
! Objective            : MPI  Parallel I/O to Single I/O 
!
! Input                : None 
!
! Description          : MPI  Parallel I/O to Multiple files
!                        Based on the number of processes spawned the program
!                        Each process use  MPI 2.0 I/O library calls. 
!                        and writes to s single file                     
!
! Output               : Output files with random data written 
!
! Necessary conditions : Number of Processes should be less than or equal to 8
!  
! Created              : August 2013  
!
! E-mail               : hpcfte@cdac.in                                          
!
!********************************************************************
!

! Example of parallel MPI write into a single file, in Fortran

 PROGRAM main

	! Fortran 90 users can (and should) use
	! use mpi
	! instead of include 'mpif,h' if their MPI implementation provides a
	! mpi module.

	include 'mpif.h'

	integer ierr, i, myrank, numprocs, BUFSIZE, thefile
	Parameter (BUFSIZE = 100)
	integer buf(BUFSIZE)
	integer (kind = MPI_OFFSET_KIND) disp

	call MPI_INIT(ierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

	do i = 0, BUFSIZE
	     buf(i) = myrank * BUFSIZE + i
	enddo

	call MPI_FILE_OPEN(MPI_COMM_WORLD, 'testfile', &
               MPI_MODE_WRONLY + MPI_MODE_CREATE,  &
    	       MPI_INFO_NULL, thefile, ierr)

	! assume 4-byte integers

	disp = myrank * BUFSIZE * 4
	call MPI_FILE_SET_VIEW(thefile, disp, MPI_INTEGER, &
				   MPI_INTEGER, 'native', &
				   MPI_INFO_NULL, ierr)

	call MPI_FILE_WRITE(thefile, buf, BUFSIZE, MPI_INTEGER, &
			       MPI_STATUS_IGNORE, ierr)

	call MPI_FILE_CLOSE(thefile, ierr)

	call MPI_FINALIZE(ierr)

END PROGRAM main

