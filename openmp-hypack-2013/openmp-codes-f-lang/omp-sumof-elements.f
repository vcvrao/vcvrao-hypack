
C************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 1.4           : omp_sumof_elements.f
C
C Objective             : Sum of 'n' elements of one-dimensional real array.
C               	  This example demonstrates the use of
C               	  PARALLEL DO directive and PRIVATE, CRITICAL and SHARED clause.
C 
C Input                 : Size of an array
C
C Output                : Sum of 'n' elements	                                            
C                                                                        
c Created               : August-2013
c
c E-mail                : hpcfte@cdac.in     
c
C**************************************************************************

       program SumOfElements
       integer SIZE
       parameter (SIZE=100)
       real Sum
       real Array(SIZE)

       integer OMP_GET_THREAD_NUM, ThreadID
       integer ArraySize

       print*,"Input the Size of Array"
       read(*,*) ArraySize
       if((ArraySize .le. 0) .or. (ArraySize .gt. SIZE)) then
          print*,"Input is Wrong."
          print*,"Please try with another number less than",SIZE+1
          print*,"Or greater than 0"
          goto 100
       endif

       Sum = 0.0
       do j = 1, ArraySize
         Array(j) = float(j)
       enddo

C$OMP PARALLEL DO PRIVATE(j) SHARED(Sum, Array)
       do j = 1, ArraySize
	   !$OMP CRITICAL
          Sum = Sum + Array(j)
		!$OMP END CRITICAL
       enddo

       print *, 'Sum:  ', Sum
100    end
