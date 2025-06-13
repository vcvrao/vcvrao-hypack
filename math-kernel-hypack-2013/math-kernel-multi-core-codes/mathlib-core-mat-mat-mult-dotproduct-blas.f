c
c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.6           : mathlib-core-mat-mat-mult-dotproduct-blas.f
c
c Objective 	        : Write matrix-matrix multiplication program with dot 
c                         product inner loop, and use  BLAS libraries and compiler 
c                         optimizations to extract the performance. (You have to 
c                         download the BLAS libraries in your home directory).
c
c Input                 : Number of Rows and Columns of the two matrices.
c
c Output                : The time taken for the matrix-matrix multiplication 
c                         in seconds and the performance in MFLOPS.
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/

      program MatMatMultDot_Blas
      double precision MatrixA(1000,1000),MatrixB(1000,1000)
      double precision ResultMatrix(1000,1000)
      double precision VectorATemp(1000),VectorBTemp(1000)
      double precision timetaken, mflops
      integer operations,time(2),MatARowSize,MatAColSize,i,j
      integer MatBRowSize,MatBColSize,temp
      external dtime
      double precision ddot
      external ddot

c     Read the sizes of Matrix and Vector

      print *,'Enter the number of rows and columns of the Matrix A'
      read *,MatARowSize,MatAColSize
      if ((MatARowSize .lt. 1) .or. (MatAColSize .lt. 1)) then
      print *,'The number of rows or columns should be positive'
      stop
      endif
      print *,'Enter the number of rows and columns of the Matrix B'
      read *,MatBRowSize,MatBColSize
      if ((MatBRowSize .lt. 1) .or. (MatBColSize .lt. 1)) then
      print *,'The number of rows or columns should be positive'
      stop
      endif

c     Check if multiplication can be performed, else give error message and quit

      if(MatAColSize.ne.MatBRowSize) then
      print *,'The number of columns of Matrix A should be equal to numb
     $er of columns of the Matrix B'
      stop
      endif

c     Initialisation of Matrices to arbitrary values

      do 100, i=1,MatARowSize
      do 90, j=1,MatAColSize
      MatrixA(i,j)=float(i*(j+1))
  90  continue
 100  continue
      do 80, i=1,MatBRowSize
      do 70, j=1,MatBColSize
      MatrixB(i,j)=float(i*(j+2))
  70  continue
  80  continue
      do 60, i=1,MatARowSize
      do 50, j=1,MatBColSize
      ResultMatrix(i,j)=0.0d0
   50 continue
   60 continue

c     Measure the time for Multiplication of Matrices in Rowwise
c     fashion

      timetaken=dtime(time)
      do 40, i=1,MatARowSize
      do 30, j=1,MatBColSize
      do 20, temp=1,MatAColSize
      VectorATemp(temp)=MatrixA(i,temp)
      VectorBTemp(temp)=MatrixB(temp,j)
   20 continue
      ResultMatrix(i,j)=ddot(MatAColSize,VectorATemp,1,VectorBTemp,1)
   30 continue
   40 continue    
      timetaken=dtime(time)

      if(timetaken .eq. 0.0) then
      print *,'Timettaken is too lower than a microsecond'
      stop
      end if

c     Find the number of operations and performance in Mflops/s
      
      operations=2*MatARowSize*MatBColSize*MatAColSize
      mflops=dble(operations)/(timetaken*1000000)

c     Print the matrices and performance in Mflops/s

      print *,'  '
      print *,'Size of Matrix A     :',MatARowSize,' *',MatAColSize
      print *,'Size of Matrix B     :',MatBRowSize,' *',MatBColSize
      print *,'No. of Operations are:',operations
      print *,'Time taken in sec.   :',timetaken
      print *,'Mflop/s             :',mflops

      stop
      end
