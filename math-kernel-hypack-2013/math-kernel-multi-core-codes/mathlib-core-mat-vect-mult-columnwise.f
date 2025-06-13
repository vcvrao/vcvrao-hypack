c
c*****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c Example 1.3           : mathlib-core-mat-vect-mult-columnwise.f
c
c Objective 	        : Write a program for Matrix-Vector Multiplication. The 
c                         Matrix entries should be accessed in Column-wise fashion.
c                         Explain the reasons for the performance.
c
c Input                 : The number of rows and columns in matrix and the number of
c                         rows in vector.
c
c Output                : The time taken in seconds for the multiplication in
c                         columnwise fashion and performance in MFLOPS.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/
c
      program SeqPrg_MatVecMult_Col
      real Matrix(1000,1000),Vector(1,1000),ResultVector(1,1000)
      double precision timetaken, mflops
      integer operations,time(2),MatRowSize,MatColSize,VecRowSize,i,j
      external dtime

c     Read the sizes of Matrix and Vector 

      print *,'Enter the number of rows and columns of the Matrix'
      read *,MatRowSize,MatColSize
      if ((MatRowSize .lt. 1) .or. (MatColSize .lt. 1)) then
      print *,'The number of rows or columns should be positive'
      stop
      endif
      print *,'Enter the number of rows of the Vector'
      read *,VecRowSize
      if (VecRowSize .lt. 1) then
      print *,'The number of rows of the Vector should be positive'
      stop
      endif

c     Check if multiplication can be performed, else give error message and quit

      if(MatColSize.ne.VecRowSize) then
      print *,'The number of columns of Matrix should be equal to number  
     $ of columns of the Vector'
      stop
      endif

c     Initialisation of Matrix and Vector to arbitrary values

      do 90, i=1,MatRowSize
      do 80, j=1,MatColSize
      Matrix(i,j)=float(i*(j+1))
  80  continue
  90  continue
      do 70, i=1,VecRowSize
      Vector(i,1)=float(i+2)
  70  continue
      do 60, i=1,MatRowSize
      ResultVector(i,1)=0.0d0
   60 continue

c     Measure the time for Multiplication of Matrix and Vector in Rowwise 
c     fashion

      timetaken=dtime(time)
      do 50, j=1,MatColSize
      do 40, i=1,MatRowSize
      ResultVector(i,1)=ResultVector(i,1)+Matrix(i,j)*Vector(j,1)
   40 continue
   50 continue
      timetaken=dtime(time)

      if(timetaken .eq. 0.0) then
      print *,'Time taken is lower than a microsecond'
      print *,'Increase the matrix and vector dimensions to get results'
      stop
      end if

      operations=2*VecRowSize*VecRowSize
      mflops=dble(operations)/(timetaken*1000000)

c     Print the performance in MFLOPS

      print *,'  '
      print *,'Size of Matrix       :',MatRowSize,' *',MatColSize
      print *,'Size of Vector       :',VecRowSize
      print *,'No. of Operations are:',operations
      print *,'Time taken in sec.   :',timetaken
      print *,'Mflops               :',mflops

      stop
      end
