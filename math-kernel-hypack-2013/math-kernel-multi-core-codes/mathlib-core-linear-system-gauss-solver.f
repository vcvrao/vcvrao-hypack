c
c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.8           : mathlib-core-linear-system-gauss-solver.f
c
c Objective 	        : Write your own program to solve the matrix system of 
c                         linear equations in which A is symmetric positive definite 
c                         and use compiler optimizations to extract the performance.
c
c Input                 : The input for this program is the order of the Matrix 
c                         and number of elements in Vector,i.e. coefficient matrix
c                         and variable vector for a linear system of equations 
c                         respectively.
c
c Output                : The time taken in seconds for finding the solution matrix and 
c                         performance in MFLOPS.
c                 
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/
c
       subroutine solvegauss(Matrix,Vector,Result,n,operations,time,
     $ timetaken,mflops)
       integer n
       double precision Matrix(n,n), Vector(n),Result(n)
       double precision Temp(n,2*n),timetaken,mflops
       double precision time(2),operations
       external dtime

c      initialize the reduction matrix
       n2 = 2*n
       do 1 i = 1,n
       do 2 j = 1,n
       Temp(i,j) = Matrix(i,j)
       Temp(i,n+j) = 0.
 2     continue
       Temp(i,n+i) = 1.
 1     continue

c      do the reduction 

       timetaken = dtime(time)
       do 3 i = 1,n
       alpha = Temp(i,i)
       if(alpha .eq. 0.) go to 300
       do 4 j = 1,n2
       Temp(i,j) = Temp(i,j)/alpha
 4     continue
       do 5 k = 1,n
       if((k-i).eq.0) go to 5
       beta = Temp(k,i)
       do 6 j = 1,n2
       Temp(k,j) = Temp(k,j) - beta*Temp(i,j)
 6     continue
 5     continue
 3     continue

c      Temp matrix contains the inverse of Matrix in right 
c      topmost n*n part

c      find the solution matrix

       do 9 i = 1,n
       do 10 j = 1,n
       Result(i)=Result(i)+Temp(i,j+n)*Vector(j)
 10    continue
 9     continue
       timetaken = dtime(time)

       if(timetaken .eq. 0.0) then
       print *,'Time taken is too lower Than a microsecond'
       stop
       end if

c      Calculate the number of operations and performance

       operations=6*n*n*n
       mflops=dble(operations)/(timetaken*1000000)
       return
 300   print *,'*** ERROR: Singular matrix ***'
       stop
       end


      program SeqPrg_LinSysEq_Orig
      integer Size
      parameter (Size=100)
      double precision Matrix(Size,Size),Vector(Size),ResultVector(Size)
      double precision timetaken, mflops
      integer operations,time(2),i,j


      print *,'Change Size in code for other sizes of Matrix and Vector'

c     Initialisation of Matrix and Vector to arbitrary values

      do 90, i=1,Size
      do 80, j=1,Size
      if(i .eq. j) then
      Matrix(i,j)=2.0
      else
      Matrix(i,j)=1.0
      end if
  80  continue
  90  continue
      do 70, i=1,Size
      Vector(i)=Size+1
  70  continue
      do 60, i=1,Size
      ResultVector(i)=0.0d0
  60  continue

c     Solve the linear system of equations using Gauss-Jordan method 
c     and note the time taken and performance

      call solvegauss(Matrix,Vector,ResultVector,Size, operations, time,
     $timetaken,mflops)

c     Print the performance in Mflops/s

      print *,'  '
      print *,'Size of Matrix       :',Size,' *',Size
      print *,'Size of Vector       :',Size
      print *,'No. of Operations art:',operations
      print *,'Time taken in sec.   :',timetaken
      print *,'Mflop/s              :',mflops

      stop
      end
