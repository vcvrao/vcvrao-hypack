C********************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 3.1 : omp-maxof-elements-critical.f
C
C Objective   : Write an OpenMP program to print Largest element of
C               an array
C               This example demonstrates the use of
C               OMP_CRITICAL section and PARALLEL DIRECTIVE
C
C Input       : Number of elements of an array
C               User has to set OMP_NUM_THREADS environment variable for
C               n number of threads
C
C Output      : Each thread checks with its available iterations and
C               finally master thread prints the maximum value in the 
C  		array    
C                                                                      
c
c Created     : August-2013
c
c E-mail      : hpcfte@cdac.in     
c
C******************************************************************

      program MaxOfElements

      integer SIZE
      parameter(SIZE=100000)
      integer NoofElements
c      double precision Seed
      real Array(SIZE), LargeNumber, CheckValue
      integer  Seed, iseed

       print*,"Enter the number of elements"
       read(*,*) NoofElements

       if((NoofElements .le. 0) .or. (NoofElements .gt. SIZE)) then
          print*,"Input is Wrong."
          print*,"Please try with another number less than",SIZE+1
          print*,"Or greater than 0"
          goto 100
       endif

       Seed = 80629.0
       call surand(Seed, Size, NoofElements,Array)
c       do i = 1, NoofElements
c          print*,"Array(",i,")=",Array(i)
c       enddo

       LargeNumber = Array(1)
C$OMP PARALLEL DO
       do i = 1, Noofelements
          if (Array(i) .gt. LargeNumber) then
C$OMP CRITICAL
             if (Array(i) .gt. LargeNumber) then
                LargeNumber = Array(i)
             endif
C$OMP END CRITICAL
          endif
       enddo

       CheckValue = Array(1)
       do i = 2, Noofelements
          if (Array(i) .gt. CheckValue) then
             CheckValue = Array(i)
          endif
       enddo

       if (CheckValue .eq. LargeNumber) then
          print*,"The Maximum value is same using serial computation"
          print*,"and openmp directive"
       else
       print*,"The Maximum value is not same using serial computation"
          print*,"and openmp directive"
       endif


       print*,"The largest number in the array is", LargeNumber


 100  stop
      end
c
c     **********************************************************
      subroutine surand(Seed, SIZE, NoofElements,Array)
c     *********************************************************
c
      integer iseed
      integer SIZE
      real Array(SIZE)

      do i = 1, NoofElements
          iseed = (i+1) * Seed
          Array(i) = ran(iseed)
          write(6,150) i, Array(i);
150       format(5x, i8, f16.7)
       enddo
      return
      end
c     *******************************************************
