c***************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.3           : single-core-prg-if-goto.f
c
c Objective 	        : Write a program to avoid "IF" and "GOTO" from the following
c                         loop for better performance.
c
c                         i=0
c                         10 i = i+1
c                          if(i .gt. 100000) goto 30
c                           a(i) = a(i) + b(i) * c(i)
c                          go to 10
c                         30 continue
c
c Input                 : None
c
c Output                : Message showing whether the modified construct is performing
c                         same operations as original construct.
c                  
c
c   
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/


      program IfGoto
      integer N
      parameter (N=100)
      integer VectorAorig(N),VectorAmod(N)
      integer VectorB(N),VectorC(N)      

c Populating the vectors A and B

      do 2, CounterI=1,N,1
      VectorAorig(CounterI)=10
      VectorAmod(CounterI)=10
      VectorB(CounterI)=20
      VectorC(CounterI)=30
    2 continue

c Original construct using If and Goto

      CounterI=0
   10 CounterI=CounterI+1
      if(CounterI .gt. N) goto 30
      VectorAorig(CounterI)=VectorAorig(CounterI)+ 
     $VectorB(CounterI)*VectorC(CounterI)
      goto 10
   30 continue 

c Modified construct avoiding use of If and Goto

      do 20, CounterI=1,N,1
      VectorAmod(CounterI)=VectorAmod(CounterI)+
     $VectorB(CounterI)*VectorC(CounterI)
   20 continue

c Checking for the correctness of modifications

      do 40, CounterI=1,N,1
      if(VectorAorig(CounterI) .ne. VectorAmod(CounterI)) then
      print *,'Not successfully rewritten avoiding Goto and If'
      stop
      endif
   40 continue

      print *
      print *,'Successfully rewritten avoiding If and Goto'
      print *
     
 
      stop
      end

