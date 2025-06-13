c
c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.5           : mathlib-core-mat-mat-mult-daxpy.f
c
c Objective 	        : Write matrix-matrix multiplication program with daxpy 
c                         inner loop and use compiler optimizations to extract the 
c                         performance. Assume that the arrays' dimension is of 2 
c                         the power i where i = 4, 8.
c
c Input                 : The order of the matrices
c
c Output                : The time taken in seconds for the multiplication 
c                         and performance in MFLOPS.
c                
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/

      program SeqPrg_MatMatMult_Daxpy

      real MatrixA(1000,1000),MatrixB(1000,1000),MatrixC(1000,1000)
      integer Size,CounterI, CounterJ, CounterK
      double precision timetaken, mflops
      integer operations,time(2)
      external dtime

c     Read and print the size of matrices being taken

      print *,'Read the size of matrices'
      read *,Size
      if (Size .lt. 1) then
      print *,'The size of the matrices should be positive'
      stop
      endif
      print *,'The size of the matrices being multiplied is',Size

c     Initialisation of Matrices A and B and the Result Matrix C

      do 10, CounterI=1,Size
      do 20, CounterJ=1,Size
      MatrixA(CounterI,CounterJ)=float(CounterI*(CounterJ+1))
      MatrixB(CounterI,CounterJ)=float(CounterI*(CounterJ+2))
      MatrixC(CounterI,CounterJ)=0.0
   20 continue
   10 continue

c     Measure the time for Multiplication of the Matrices A and B

      timetaken=dtime(time)
      do 30, CounterK=1,Size
      do 40, CounterJ=1,Size
      do 50, CounterI=1,Size
      MatrixC(CounterI,CounterJ)=MatrixC(CounterI,CounterJ) +
     $MatrixA(CounterI,CounterK)*MatrixB(CounterK,CounterJ)
   50 continue
   40 continue
   30 continue 
      timetaken=dtime(time)

      if(timetaken .eq. 0.0) then
      print *,'Time taken is lower than a microsecond'
      print *,'Try for larger sizes for results'
      stop
      end if

      operations=3*Size*Size*Size
      mflops=dble(operations)/(timetaken*1000000)

c     Print the arrays

c      print *,'The Matrix A is:'
c      do 60, CounterI=1,Size
c      print *,(MatrixA(CounterI, CounterJ),CounterJ=1,Size)
c   60 continue
c
c      print *,'The Matrix B is:'
c      do 70, CounterI=1,Size
c      print *,(MatrixB(CounterI, CounterJ),CounterJ=1,Size)
c   70 continue
c
c      print *,'The Matrix C is:'
c      do 80, CounterI=1,Size
c      print *,(MatrixC(CounterI, CounterJ),CounterJ=1,Size)
c   80 continue

c     Print the performance in Mflops/s

      print *,'  '
      print *,'Size of Matrices       :',Size
      print *,'No. of Operations are  :',operations
      print *,'Time taken in sec.     :',timetaken
      print *,'Mflops/sec             :',mflops


      stop
      end
