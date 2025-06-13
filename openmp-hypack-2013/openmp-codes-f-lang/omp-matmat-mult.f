C***************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 2.5           : omp_matmat_mult.f
C
C Objective             : Matrix matrix multiplication
C               		 This example demonstrates the use of
C              			PARALLEL section, SHARED, PRIVATE clause and
C               			DO directive
C 
C Input                 : Size of matrix.
C
C Output                : Result of Matrix matrix computation, time taken for this
C	                 computation and Mflop/s	                                   C        
C                                                                        
c
c Created             : August-2013
c
c E-mail              : hpcfte@cdac.in     
c
C*********************************************************************************

      program MatMatMult

      integer DATA_SIZE
      parameter (DATA_SIZE = 1001)

      integer   MatrixSize, Iterations
      real       FinalMatrix(DATA_SIZE, DATA_SIZE)
      real       CheckMatrix(DATA_SIZE, DATA_SIZE)
      real    Matrix_A(DATA_SIZE, DATA_SIZE),
     $        Matrix_B(DATA_SIZE, DATA_SIZE)
C      real    Mflops
      integer ValidOutput
C      integer NoofOperations, ValidOutput
C      real tt1(2), tt2(2)
C      real toh1, toh2, toh

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

      do i = 1, MatrixSize
         do j = 1, MatrixSize
            Matrix_A(i, j) = 1.0 * i
            Matrix_B(i, j) = Matrix_A(i, j)
         enddo
      enddo

C      t1 = etime(tt1)
      Iterations = 1

C$OMP PARALLEL PRIVATE(i,j,k)
C$OMP& SHARED(Iterations, MatrixSize,Matrix_A,Matrix_B,FianlMatrix)
      do iter = 1, Iterations
C$OMP DO
         do j = 1, MatrixSize
            do i = 1, MatrixSize
               FinalMatrix(i, j) = 0.0
               do k = 1, MatrixSize
                  FinalMatrix(i, j) = FinalMatrix(i, j) +
     $                 Matrix_A(i, k)*Matrix_B(k, j)
               enddo
            enddo
         enddo
      enddo
C$OMP END PARALLEL

C      t2 = etime(tt2)
C ...... Serial Computation .......

      do i = 1, MatrixSize
         do j = 1, MatrixSize
            CheckMatrix(i, j) = 0.0
            do k = 1, MatrixSize
               CheckMatrix(i, j) = CheckMatrix(i, j) +
     $                 Matrix_A(i, k)*Matrix_B(k, j)
            enddo
         enddo
      enddo


      do i = 1, MatrixSize
         do j = 1, MatrixSize
            if(FinalMatrix(i,j) .ne. CheckMatrix(i,j)) then
              ValidOutput = 0
              stop
            else
              ValidOutput = 1
            endif
         enddo
      enddo

      if(ValidOutput .eq. 1) then
        print*," .... Correct Result ......"
C        write(6,*) 'Time = ',t2-t1-toh
      else
        print*," .... Wrong Result ......"
      endif

C      NoofOperations = 2*MatrixSize**3
C      TimeOverhead = (t2-t1-toh) / Iterations
C      Mflops = NoofOperations / (TimeOverhead * 1000000)

C      print *,'******************************'
C      print *,'Size of Matrix=',MatrixSize
C      print *,'Time Taken (seconds) =',(t2-t1-toh)
C      print *,'Mflop/s =',Mflops
C      print *,'******************************'

100   stop
      end

