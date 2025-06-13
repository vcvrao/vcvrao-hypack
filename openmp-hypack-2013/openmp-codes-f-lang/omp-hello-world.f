
C*****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 1.2           : omp_hello_world.f
C
C Objective             : OpenMP program to print "Hello World"
C               		 This example demonstrates the use of
C              		 OpenMP PARALLEL PRIVATE directive and
C               		 run-time library routines
C               		 OMP_GET_THREAD_NUM
C               		 OMP_GET_NUM_THREADS
C
C Input                 : User has to set OMP_NUM_THREADS environment variable for
C               		 n number of threads
C
C Output                : Each thread prints a message "Hello World" and its
C               		 identifier.	                                            
C                                                                        
c
c  Created             : August-2013
c
c  E-mail              : hpcfte@cdac.in     
c
C*********************************************************************************

       program HelloWorld

       integer NoofThreads, ThreadID, OMP_GET_NUM_THREADS
       integer OMP_GET_THREAD_NUM

C     Fork a team of threads giving them their own copies of variables
C$OMP PARALLEL PRIVATE(NoofThreads, ThreadID)

C     Obtain and print thread identifier
      ThreadID = OMP_GET_THREAD_NUM()
      print *, 'Hello World from thread = ', ThreadID

C     Only master thread does this
      if (ThreadID .EQ. 0) then
        NoofThreads = OMP_GET_NUM_THREADS()
        print *, 'Number of threads = ', NoofThreads
      end if

C     All threads join master thread
C$OMP END PARALLEL

       stop
       end
