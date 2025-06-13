c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.3           : tools-omp-pi-calculation.f
c
c Objective 	        : Write an OpenMP program to compute the value of pi
C                         function by numerical integration of a function
C                         f(x) = 4/(1+x*x ) between the limits 0 and 1.
C                         This example demonstrates the use of
C                         OMP_SET_DYNAMIC
C                         OMP_GET_THREAD_NUM
C                         OMP_GET_NUM_THREADS
C                         and PARALLEL and CRITICAL directives
C
c Input                 : Number of intervals
c
c Output                : Computed value of PI
c             
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/

      program PICalculation

      double precision pi, sum, x, step
      double precision local_sum
      double precision pi_true
      double precision dum, time

      PARAMETER(pi_true = 3.1415926535897932384626433)
      integer i, NoofIntervals

C      real tt1(2), tt2(2)
C      real toh1, toh2, toh

      integer ThreadID, NoofThreads
      integer OMP_GET_THREAD_NUM, OMP_GET_NUM_THREADS
      external  userinpt
      integer*4 userinpt

      call OMP_SET_DYNAMIC(.FALSE.)

C       toh1 = etime(tt1)
C       toh2 = etime(tt2)
C       toh  = toh2 - toh1

 120    j = userinpt(NoofIntervals)
       if (j.le.0) goto 200
       step = 1.0 / NoofIntervals
       sum = 0.0

C       t1 = etime(tt1)

C$OMP PARALLEL PRIVATE (i, x, local_sum)

       local_sum = 0.0
       ThreadID = OMP_GET_THREAD_NUM()
       NoofThreads = OMP_GET_NUM_THREADS()

       do 100 i=(ThreadID+1), NoofIntervals, NoofThreads
          x = (i-0.5) * step
          local_sum = local_sum + 4.0/(1.0+x*x)
100    continue

C$OMP CRITICAL

       sum = sum + local_sum

C$OMP END CRITICAL
C$OMP END PARALLEL

       pi = sum * step
C       t2 = etime(tt2)

       print*,"for", NoofIntervals, "Steps PI=", pi
       print*,"error = ",dabs(pi - pi_true)

C       write(6,*) 'Time = ',t2-t1-toh,"With",NoofThreads,"Threads"
        write(6,*)  'This ='."With",NoofThreads,"Threads"

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
 99   format(' How many points do you want (0 or neg. value quits)? ')
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
