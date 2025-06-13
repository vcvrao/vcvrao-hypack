c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.8         : single-core-prg-loop-interchange-center.f
c
c Objective 	      : Write a program to demonstrate the execution time for the
c                 	following fragment of code with/without applying "LOOP 
c                       interchange" to move the computations to the center in the 
c                       following loop for better performance.
c
c                       parameter (idim = 1000, jdim = 1000, kdim = 4)
c                            ...
c                          do 10 i = 1, idim
c                          do 20 j = 1, jdim
c                          do 30 k = 1,kdim
c                                d(i,j,k) = d(i,j,k) + v(i,j,k) * dt
c                          30 continue
c                          20 continue
c                          10 continue
c
c Input              : None
c
c Output             : The time taken by the two different implementations of
c                      the same loop without/with Loop Interchange (making it
c                      possible to apply Loop Unrolling to the inner loop). 
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/
c
      program LoopInterchange_Center
      integer IDIM, JDIM, KDIM, dt
      parameter (IDIM = 1000, JDIM = 1000, KDIM = 4)
      parameter (dt = 1)
      integer CounterI, CounterJ, CounterK
      double precision MatrixDorig(IDIM, JDIM, KDIM)
      double precision MatrixVorig(IDIM, JDIM, KDIM)
      double precision MatrixDmod(IDIM, JDIM, KDIM)
      double precision MatrixVmod(IDIM, JDIM, KDIM)


c     Populating the matrices      

      do 10, CounterI = 1,IDIM
      do 20, CounterJ = 1,JDIM
      do 30, CounterK = 1,KDIM
      MatrixDorig(CounterI,CounterJ,CounterK)=
     $        dble(CounterI*(CounterJ+CounterK+1))
      MatrixDmod(CounterI,CounterJ,CounterK)=
     $        MatrixDorig(CounterI,CounterJ,CounterK)
      MatrixVorig(CounterI,CounterJ,CounterK)=
     $        dble(CounterI*(CounterJ+CounterK+2))
      MatrixVmod(CounterI,CounterJ,CounterK)=
     $        MatrixVorig(CounterI,CounterJ,CounterK)
   30 continue
   20 continue
   10 continue


c  Measuring the time taken without Loop Optimizations
    
      timetakenorig=dtime(time)
      do 40, CounterI = 1, IDIM
      do 50, CounterJ = 1, JDIM
      do 60, CounterK = 1, KDIM
         MatrixDorig(CounterI,CounterJ,CounterK) = MatrixDorig(CounterI,
     $CounterJ,CounterK) + MatrixVorig(CounterI,CounterJ,CounterK) * dt
   60 continue
   50 continue
   40 continue
      timetakenorig=dtime(time)


c  Measuring the time taken with Loop Optimizations    

      timetakenmod=dtime(time)
      do 70, CounterK = 1, KDIM
      do 80, CounterJ = 1, JDIM
      do 90, CounterI = 1, IDIM,4
        MatrixDmod(CounterI,CounterJ,CounterK) = MatrixDmod(CounterI,
     $CounterJ,CounterK) + MatrixVmod(CounterI,CounterJ,CounterK)*dt
        MatrixDmod(CounterI+1,CounterJ,CounterK) = MatrixDmod(CounterI+
     $1,CounterJ,CounterK)+MatrixVmod(CounterI+1,CounterJ,CounterK)*dt
        MatrixDmod(CounterI+2,CounterJ,CounterK) = MatrixDmod(CounterI+
     $2,CounterJ,CounterK)+MatrixVmod(CounterI+2,CounterJ,CounterK)*dt
        MatrixDmod(CounterI+3,CounterJ,CounterK) = MatrixDmod(CounterI+
     $3,CounterJ,CounterK)+MatrixVmod(CounterI+3,CounterJ,CounterK)*dt
   90 continue
   80 continue
   70 continue
      timetakenmod=dtime(time)

      do 2, CounterI = 1,IDIM
      do 3, CounterJ = 1,JDIM
      do 4, CounterK = 1,KDIM
      if (MatrixDorig(CounterI,CounterJ,CounterK) .ne. MatrixDmod(Counte
     $rI,CounterJ,CounterK)) then
      print *,CounterI,CounterJ,CounterK
      print *,MatrixDorig(CounterI,CounterJ,CounterK),MatrixDmod(Counter
     $I,CounterJ,CounterK)
      print *,'The modified construct is not performing same as original
     $ one'
      stop
      end if
    4 continue      
    3 continue      
    2 continue      


      if(timetakenorig .eq. 0.0) then
      print *,'Time taken without Loop Interchange is too lower than a mi
     +crosecond'
      print *,'Try for larger values of IDIM, JDIM, KDIM for results'
      else
      print *,'Time taken in seconds without Loop Interchange :',timetak
     $enorig
      end if

      print *
      print *

      if(timetakenmod .eq. 0.0) then
      print *,'Time taken with Loop Interchange is too lower than a micro
     +crosecond'
      print *,'Try for larger values of IDIM, JDIM, KDIM for results'
      else
      print *,'Time taken in seconds with Loop Interchange    :',timetak
     $enmod
      end if

      print *
      print *

      if(timetakenorig <= timetakenmod) then
      print *,'Try for larger values of IDIM,JDIM,KDIM to see the effect
     + of Loop Interchange'
      end if

      print *
      print *

      stop
      end


