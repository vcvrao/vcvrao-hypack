c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c
c Example 1.4           : single-core-prg-loop-interchange.f
c
c Objective 	        : Write a program to demonstrate the execution time for 
c                         the following loop with/without applying "LOOP 
c                         INTERCHANGE" to ease the memory access for the . Explain
c                         the reasons behind to improve the memory access for the
c                         arrays A and B.
c
c                         do 10, i = 1, n
c                         do 20, j = 1, n
c                             A(j,i) = B(i,j) + A(i,j)
c                         20 continue
c                         10 continue
c
c Input                 : None
c
c Output                :The time taken by the two different implementations of
c                        the same loop without/with Loop Interchange.
c                  
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c******************************************************************************/

      program LoopInterchange
      double precision timetakenorig,timetakenmod
      integer N
      parameter(N=1000) 
      double precision MatrixAorig(N,N), MatrixAmod(N,N)
      double precision MatrixBorig(N,N), MatrixBmod(N,N)
      integer time(2), CounterI,CounterJ
      external dtime

      print *,'The order of the matrices=',N
      print *,'Change N in code to change order of matrices'

c  Populating the Matrix

      do 10, CounterI = 1,N
      do 20, CounterJ = 1,N
      MatrixAorig(CounterI,CounterJ)=dble(CounterI*(CounterJ+2))
      MatrixAmod(CounterI,CounterJ)=MatrixAorig(CounterI, CounterJ)
      MatrixBorig(CounterI,CounterJ)=dble(CounterI*(CounterJ+1))
      MatrixBmod(CounterI,CounterJ)=MatrixBorig(CounterI, CounterJ)
   20 continue
   10 continue


c  Measuring the time taken without Loop Interchange
    
      timetakenorig=dtime(time)
      do 30, CounterI = 1, N
      do 40, CounterJ = 1, N
         MatrixAorig(CounterJ, CounterI) = MatrixBorig(CounterI, Counter
     $J) * MatrixAorig(CounterI, CounterJ) 
   40 continue
   30 continue
      timetakenorig=dtime(time)


c  Measuring the time taken with Loop Interchange    

      timetakenmod=dtime(time)
      do 50, CounterJ = N, 1, -1
      do 60, CounterI = 1, N
         MatrixAmod(CounterJ, CounterI) = MatrixBmod(CounterI, Counter
     $J) * MatrixAmod(CounterI, CounterJ) 
   60 continue
   50 continue
      timetakenmod=dtime(time)


      do 70, CounterI = 1,N
      do 80, CounterJ = 1,N
      if (MatrixAorig(CounterI, CounterJ) .ne. MatrixAmod(CounterI, Coun
     $terJ)) then
      print *,CounterI
      print *,MatrixAorig(CounterI,CounterJ),MatrixAmod(CounterI, Counte
     $rJ)
      print *,'The modified construct is not performing same as original
     $ one'
      stop
      end if
   80 continue      
   70 continue      


      if(timetakenorig .eq. 0.0) then
      print *,'Time taken without Loop Interchange is lower than a micro
     +second'
      print *,'Try large sizes of N for results'
      else
      print *,'Time taken in seconds without Loop Interchange :',timetak
     $enorig
      end if
      print *
      print *
      if(timetakenmod .eq. 0.0) then
      print *,'Time taken with Loop Interchange is lower than a microsec
     +ond'
      print *,'Try large sizes of N for results'
      else
      print *,'Time taken in seconds with Loop Interchange    :',timetak
     $enmod
      end if

      print *
      print *

      if(timetakenmod >= timetakenorig) then
      print *,'Try for larger sizes of N to see the effect of Loop Inter
     +change'
      end if
      print *
      print *
      stop
      end
