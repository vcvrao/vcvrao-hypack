c
c************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.7           : single-core-prg-tlb-miss.f
c
c Objective 	        :  Write a program to demonstrate the execution time for 
c                          the following loop with/without re-organizing it to avoid
c                          TLB misses.
c
c                            real x(1001000)
c                            common x,y
c                            do 10 i = 0, 1000
c                            do 20 j=1, 1000000,10000
c                               y = x(j+i)
c                           20 continue
c                          10 continue
c
c Input                 : None
c
c Output                : The time taken by the two different implementations of
c                         the same loop without/with Loop Optimizations to avoid
c                         TLB misses.
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c******************************************************************************/

      program TLBmiss
      double precision timetakenorig,timetakenmod
      integer M,N,INCR
      parameter(N=1001000,M=1000,INCR=10000)
      double precision MatrixXorig(N), MatrixXmod(N), Y
      integer time(2), CounterI,CounterJ
      external dtime

c  Populating the Matrix

      do 10, CounterI = 1,N
      MatrixXorig(CounterI)=1.0d0
      MatrixXmod(CounterI)=MatrixXorig(CounterI)
   10 continue


c  Measuring the time taken without Optimizations
    
      timetakenorig=dtime(time)
      do 20, CounterI = 0, N
      do 30, CounterJ = 1, M, INCR
         Y=MatrixXorig(CounterJ+CounterI)
   30 continue
   20 continue
      timetakenorig=dtime(time)


c  Measuring the time taken on avoiding TLB misses    

      timetakenmod=dtime(time)
      do 40, CounterJ = 1, M, INCR
      do 50, CounterI = 0, N
         Y=MatrixXorig(CounterJ+CounterI)
   50 continue
   40 continue
      timetakenmod=dtime(time)

      do 60, CounterI = 1,N
      if (MatrixXorig(CounterI) .ne. MatrixXmod(CounterI)) then
      print *,CounterI
      print *,MatrixXorig(CounterI),MatrixXmod(CounterI)
      print *,'The modified construct is not performing same as original
     $ one'
      stop
      end if
   60 continue      

      if(timetakenorig .eq. 0.0) then
      print *,'Time taken without Loop optimizations is lower than a mic
     +rosecond'
      print *,'Try increasing values of N,M,INCR in code for results'
      else
      print *,'Time taken in seconds without Loop Optimizations        
     $            :',timetakenorig
      end if

      print *
      print *

      if(timetakenmod .eq. 0.0) then
      print *,'Time taken with Loop optimizations is lower than a micros
     +econd'
      print *,'Try increasing values of N,M,INCR in code for results'
      else
      print *,'Time taken in seconds with Loop Optimizations to avoid TL
     $B misses    :',timetakenmod
      end if

      print *
      print *

      if(timetakenorig <= timetakenmod) then
      print *,'Try for larger values of N,M,INCR to see the effect of Lo
     +op Optimizations'
      end if

      print *
      print *

      stop
      end
