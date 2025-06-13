
C*****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example  2.3          : omp_matrix-transpose.f
C
C Objective             : Write an OpenMP Program for Transpose of a Matrix
C                  	 and measure the performance
C                  	 This example demonstrates the use of
C                  	 PARALLEL DO directive and Private clause.
C
C Input                 : Size of Matrix
C
C Output                : Each thread transposes assaigned Row of Matrix,
C                  	 time taken for this computation and Mflop/s	                                            
C                                                                        
c Created               : August-2013
c
c E-mail                : hpcfte@cdac.in     
c
C***********************************************************

      program MatTranspose

      integer DATA_SIZE
      parameter (DATA_SIZE = 1001)

      integer  Matrix_size, NoofRows, NoofCols, i, j
      real     Matrix_A(DATA_SIZE, DATA_SIZE)
      real     Trans(DATA_SIZE, DATA_SIZE)
      real     Checkoutput(DATA_SIZE, DATA_SIZE), flops
C      real     Mflops
C      integer  NoofOperations, ValidOutput
C      real     tt1(2), tt2(2)
C      real     toh1, toh2, toh

C      toh1 = etime(tt1)
C      toh2 = etime(tt2)
C      toh  = toh2 - toh1

      print *,'Input Size of Matrix'
      read(*,*) MatrixSize
      Iterations = 100
      if((MatrixSize .le. 0) .or. (MatrixSize .gt. DATA_SIZE)) then
        print *,'Invalid MatrixSize.'
        print *,'Please try with another number less than', DATA_SIZE+1
        goto 100
      endif


C   Matrix Elements

      do i = 1, MatrixSize
         do j = 1, MatrixSize
            Matrix_A(i, j) = 1.0 * i
            Checkoutput(i,j) = 0.0
            Trans(i,j) = 0.0
         enddo
      enddo

C      t1 = etime(tt1)

C  OpenMP Parallel For Directive

C$OMP PARALLEL DO PRIVATE(j)
      do i = 1, MatrixSize
         do j = 1, MatrixSize
            Trans(j,i) = Matrix_A(i,j)
         enddo
      enddo

C      t2 = etime(tt2)

C  Serial Computation

      do i = 1, MatrixSize
         do j = 1, MatrixSize
            CheckOutput(j,i) = Matrix_A(i,j)
         enddo
      enddo

      do j = 1, MatrixSize
         do i = 1, MatrixSize
            if(Trans(i,j) .ne. CheckOutput(i,j)) then
               ValidOutput = 0
               stop
            else
               ValidOutput = 1
            endif
         enddo
      enddo

      if(ValidOutput .eq. 1) then
        print*,"------ Correct Result -----"
      else
        print*,"------ Wrong Result -----"
      endif

C      print*,"Input Matrix"
C      do i = 1, MatrixSize
C         write (6,10)(Matrix_A(i,j),j=1,MatrixSize)
C      enddo
C
C      print*,"Traspose of Matrix"
C      do i = 1, MatrixSize
C         write (6,10)(Trans(i,j),j=1,MatrixSize)
C      enddo
C 10   format(8(2x,f8.2))

C      NoofOperations = 2*MatrixSize
C      TimeOverhead = (t2-t1-toh) / Iterations
C      Mflops = NoofOperations / (TimeOverhead * 1000000)

      print *,'******************************'
C      print *,'Size of Matrix=',MatrixSize
C      print *,'Time Taken (seconds) =',(t2-t1-toh)
C      print *,'Mflop/s =',Mflops
      print *,'******************************'

 100  stop
      end
