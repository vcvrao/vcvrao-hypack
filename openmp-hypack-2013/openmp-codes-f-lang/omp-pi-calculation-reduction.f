
C****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 2.2           : omp-pi-calculation-reduction.f
C
C Objective             : Write an OpenMP program to compute the value of pi
C               		 function by numerical integration of a function
C               		 f(x) = 4/(1+x*x ) between the limits 0 and 1.
C              		 This example demonstrates the use of
C               			OMP_SET_DYNAMIC library routine,
C               			PARALLEL DO loop, PRIVATE and SHARED clauses
C               			and REDUCTION operation '+'.
C 
C Input                 : Number of intervals
C
C Output                : Computed value of PI
C                                                                        
c
c Created               : August-2013
c
c E-mail                : hpcfte@cdac.in     
c
C*********************************************************************************



      program PICalculation

      double precision pi, Sum, x, step
      double precision LocalSum, PartialSum
      double precision pi_true
      double precision dum, time

      PARAMETER(pi_true = 3.1415926535897932384626433)
      integer i, NoofIntervals

C      real tt1(2), tt2(2)
C      real toh1, toh2, toh

      external  userinpt
      integer*4 userinpt
      call OMP_SET_DYNAMIC(.FALSE.)

C       toh1 = etime(tt1)
C       toh2 = etime(tt2)
C       toh  = toh2 - toh1

 120    j = userinpt(NoofIntervals)
       if (j.le.0) goto 200
       step = 1.0 / NoofIntervals
       Sum = 0.0

C       t1 = etime(tt1)

       LocalSum = 0.0
C$OMP PARALLEL DO PRIVATE (x) SHARED (step)
C$OMP& REDUCTION (+:LocalSum)

       do 100 i=1, NoofIntervals
          x = (i-0.5) * step
          LocalSum = LocalSum + 4.0/(1.0+x*x)
100    continue

       pi = step * LocalSum

C       t2 = etime(tt2)

       print*,"for", NoofIntervals, "Steps PI=", pi
       print*,"error = ",dabs(pi - pi_true)

C       write(6,*) 'Time = ',t2-t1-toh

       goto 120

 200   stop
       end

C   Function which prompts the user for the integration parameters
C

       function userinpt (NoofIntervals)

       integer*4 userinpt

       integer*4 NoofIntervals
        integer*4 i

       print *
       print *, '******************** INTEGRATION OF PI **************'
       print *
       print *, 'This Example calculate PI value by integrating    '
       print *, 'the function:'
       print *, '     f(x) = 4 / (1 + x**2)'
       print *, 'between x=0 and x=1.'
       print *
       write(6,99)
 99    format(' How many points do you want (0 or neg. value quits)? ')
       read (5, 110) NoofIntervals
 110   format(i10)

        i = 0

      if (NoofIntervals .gt. 0) then
        i = 1
      else
        i = 0
      endif

      userinpt = i
      return
      end
