

C***************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                         October 15-18, 2013
c
C Example 1.1      : omp_unique_threadid.f
C
C Objective        : Write a simple OpenMP program to print unique number for
C                    each thread using the !$OMP PARALLEL directive.
C                    This example demonstrates the use of OpenMP PARALLEL
C                    directive and OMP_GET_THREAD_NUM run-time library routine
C 
C Input            : User has to set OMP_NUM_THREADS environment variable
C	             for n number of threads
C
C Output           : Each thread prints its thread identifier and master
C                    thread prints total number of threads used in the
C              	     program.	                                            
C                                                                        
c   Created        : August-2013
c
c   E-mail         : hpcfte@cdac.in     
c
C*******************************************************************************


       program UniqueThreadID
       integer OMP_GET_THREAD_NUM, OMP_GET_NUM_THREADS
       integer ThreadID, NoofThreads

C$OMP PARALLEL PRIVATE (ThreadID, NoofThreads)

       ThreadID = OMP_GET_THREAD_NUM()
       print *, "My Thread ID is", ThreadID

       if(ThreadID .eq. 0) then
         NoofThreads = OMP_GET_NUM_THREADS()
         print*,"Number of threads = ", NoofThreads
       endif

C$OMP END PARALLEL

       end
