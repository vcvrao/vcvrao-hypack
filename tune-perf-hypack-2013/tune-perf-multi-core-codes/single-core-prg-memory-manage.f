c
c************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c
c Example 1.2         : single-core-prg-memory-manage.fc

c Objective 	      : Identify the bottlenecks for performance in Memory 
c                        References in the following fragment of the code and 
c
c                        write a program to demonstrate the execution time for 
c                        the given fragment with/without better memory accesses.
c
c                        do 10, i = 1, 10000000
c                          do 20, j= i+1, 20000000
c                           A(i) = A(i)*B(j) + A(j)*B(i)
c                           B(i) = A(i)*B(i) + A(j)*B(j)
c                          20 continue
c                         10 continue
c
c Input               : None
c
c Output              : The time taken by the two different implementations of
c                       the same loop.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c******************************************************************************/
c
      program MemoryManage
      double precision timetakenorig,timetakenmod
      integer M,N
      parameter(M=20000)
      parameter(N=40000) 
      double precision VectorAorig(N),VectorBorig(N)
      double precision VectorAmod(N),VectorBmod(N)
      double precision tempA,tempB
      integer time(2), CounterI,CounterJ
      external dtime

      if((M .lt. 10000) .or. (N .lt. 20000)) then
      print *,'To see the effect of Memory Management'
      print *,'Try M=10000,N=20000'
      end if
      
c  Populating the Vectors

      do 10, CounterI=1,N
      VectorAorig(CounterI)=0.01d0
      VectorBorig(CounterI)=0.01d0
      VectorAmod(CounterI)=0.01d0
      VectorBmod(CounterI)=0.01d0
   10 continue


c  Measuring the time taken by the inefficient loop      
      timetakenorig=dtime(time)
      do 20,CounterI = 1, M
      do 30, CounterJ= CounterI+1, N
      VectorAorig(CounterI) = VectorAorig(CounterI)*VectorBorig(CounterJ
     $)+VectorAorig(CounterJ)*VectorBorig(CounterI)
      VectorBorig(CounterI) = VectorAorig(CounterI)*VectorBorig(CounterI
     $)+VectorAorig(CounterJ)*VectorBorig(CounterJ)
   30 continue
   20 continue
      timetakenorig=dtime(time)


c  Measuring the time taken by the efficient loop      

      timetakenmod=dtime(time)
      do 40, CounterI = 1, M
      tempA = VectorAmod(CounterI)  
      tempB = VectorBmod(CounterI)
      do 50,CounterJ=CounterI+1, N
      tempA = tempA*VectorBmod(CounterJ) + VectorAmod(CounterJ)*tempB
      tempB = tempA*tempB + VectorAmod(CounterJ)*VectorBmod(CounterJ)
   50 continue                  
      VectorAmod(CounterI) = tempA 
      VectorBmod(CounterI) = tempB
   40 continue

      timetakenmod=dtime(time)


      do 60, CounterI = 1,N
      if ((VectorAorig(CounterI) .ne. VectorAmod(CounterI)) .or. (Vector
     $Borig(CounterI) .ne. VectorBmod(CounterI))) then
      print *,CounterI 
      print *,VectorAorig(CounterI),VectorAmod(counterI)
      print *,VectorBorig(CounterI),VectorBmod(counterI)
      print *,'The modified construct is not performing same as original
     $ one'
      stop
      end if
   60 continue
      

      if(timetakenorig .eq. 0.0) then
      print *,'Time taken by original construct without efficient Memory
     + Management is too lower than a microsecond'
      print *,'Increase M and N in code for results'
      else
      print *,'Time taken in seconds by the original construct without e
     +fficient Memory Management is:',timetakenorig
      end if

      print *
      print *

      if(timetakenmod .eq. 0.0) then
      print *,'Time taken by the modified construct with efficient Memor
     +y Management is too lower than a microsecond'
      print *,'Increase M and N in code for results'
      else
      print *,'Time taken in seconds by the modified construct with effi
     +cient Memory Management is   :',timetakenmod
      end if

      print *
      print *

      if(timetakenmod >= timetakenorig) then
          print *,'Try increasing sizes of M and N in code for seeing th
     +e effect of Memory Management'
      end if 

      print *
      print *

      stop
      end
