c
c*************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.9           : mathlib-core-linear-system-gauss-intel-mkl.f
c
c Objective 	        : Write your own program to solve the matrix system of linear 
c                         equations in which A is symmetric postiive definite using 
c                         IBM ESSL libraries and compiler optimizations.
c                         Use linker -lessl while compilation. For using ESSL-SMP
c                         libraries, use -lesslsmp option.
c
c Input                 : The input for this program is the order of the Matrix 
c                         and number of elements in Vector,i.e. coefficient matrix
c                         and variable vector for a linear system of equations
c                         which is hardcoded.
c
c Output                : The time taken for finding the solution matrix and 
c                         performance.
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/

      program SeqPrg_LinSysEq_Essl
      integer Size
      parameter (Size=1000)
      integer lda,ipivot(2000)
      double precision Matrix(Size,Size), Vector(Size,1)
      double precision timetaken, mflops
      integer operations,i,j

      print *,'Change Size and ipivot dimensions in code for other sizes 
     +of Matrix and Vector'    

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
      Vector(i,1)=Size+1
  70  continue

      lda=Size

c     Solve the linear system of equations using IBM ESSL/ESSLSMP
c     libraries and note the time taken and performance

      timetaken=dtime(time)
      call dges(Matrix,lda,Size,ipivot,Vector,0)
      timetaken=dtime(time)

      if(timetaken .eq. 0.0) then
      print *,'Time taken is too lower than a microsecond'
      stop
      end if

      operations=6*Size*Size*Size
      mflops=dble(operations)/(timetaken*1000000)

c     Print the performance in Mflops/s

      print *,'  '
      print *,'Size of Matrix       :',Size,' *',Size
      print *,'Size of Vector       :',Size
      print *,'No. of Operations art:',operations
      print *,'Time taken in sec.   :',timetaken
      print *,'Mflops               :',mflops

      stop
      end
