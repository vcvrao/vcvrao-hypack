c
c***************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.6           : single-core-prg-loop-unroll-interchange-block.f
c
c Objective 	        : Try un-rolling, interchanging or blocking the loop in 
c                         given subroutine tryparam to increase the performance. 
c                         What method or combination of methods work best ? (NOTE : 
c                         Compile the main routine and tryparam separately. Use 
c                         compiler's default optimization level.)
c
c Input                 : None
c
c Output                : The time taken by the two different implementations of
c                         the same loop without/with Loop Optimizations.
c
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c******************************************************************************/
c

      program main
      integer m,n,ntimes,i,j
c      parameter (n=51, m=64, ntimes = 50)

c     Uncomment the following line to increase problem sizes 
c     to see improvement in performance on optimization
      parameter (n=100, m=100, ntimes =100)

      double precision q(n,m), r(n,m)
      double precision timetakenorig, timetakenmod
      integer time(2)
      external dtime
      external tryparamorig
      external tryparammod


c Populating the matrices

      do 10 i = 1, m
      do 10 j = 1, n
         q(j,i) = 1.0d0
         r(j,i) = 1.0d0
   10 continue

c Time taken for unoptimized tryparam subroutine

      timetakenorig=dtime(time) 
      do 20 i = 1, ntimes
      call tryparamorig(q,r,n,m)
   20 continue
      timetakenorig=dtime(time) 

c Time taken for optimized tryparam subroutine

      timetakenmod=dtime(time) 
      do 30 i = 1, ntimes
      call tryparammod(q,r,n,m)
   30 continue
      timetakenmod=dtime(time) 

      if(timetakenorig .eq. 0.0) then
      print *,'Time taken without optimization is lower than a microseco
     +nd'
      print *,'Try increasing n,m,ntimes in code for results'
      else
      print *,'Time taken in seconds for function without optimization :
     +',timetakenorig
      end if
      if(timetakenmod .eq. 0.0) then
      print *,'Time taken with optimization is lower than a microsecond'
      print *,'Try increasing n,m,ntimes in code for results'
      else
      print *,'Time taken in seconds for function with optimization    :
     +',timetakenmod
      end if  

      if(timetakenorig <= timetakenmod) then
      print *,'Try for large sizes of n,m,ntimes for seeing the effect o
     +f optimization'
      end if 

      end

