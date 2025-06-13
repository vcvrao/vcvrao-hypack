C************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C Example 2.4           : omp_matvect_mult.f
C
C Objective             : Matrix vector multiplication
C               	This example demonstrates the use of
C               	 	OMP_GET_THREAD_NUM
C               	        PARALLEL directive, SHARED, PRIVATE clause and
C               		ORDERED section
C               	 It uses loop work-sharing construct i.e. distribution of
C               	 columns of matrix
C 
C Input                 : Size of matrix.
C
C Output                : Each thread computes the matrix vector multiplication and
C               	 prints its part of result vector	                                            
C                                                                        
c
c Created               : August-2013
c
c E-mail                : hpcfte@cdac.in     
c
C*************************************************************************
      program MatVectMult

      integer  DATA_SIZE
      parameter (DATA_SIZE=1000)
      integer i, j, ThreadID, ValidOutput, OMP_GET_THREAD_NUM
      real Matrix_A(DATA_SIZE,DATA_SIZE), Vector(DATA_SIZE)
      real C(DATA_SIZE), Check_C(DATA_SIZE)

      print *,'Input Size of Matrix'
      read(*,*) MatrixSize

      if((MatrixSize .le. 0) .or. (MatrixSize .gt. DATA_SIZE)) then
        print *,'Invalid MatrixSize.'
        print *,'Please try with another number less than', DATA_SIZE+1
        goto 100
      endif


C     Initializations
      do i = 1, MatrixSize
        do j = 1, MatrixSize
          Matrix_A(i,j) = i * 1.0
        enddO
        Vector(i) = i
        C(i) = 0.0
      enddo
C      print *, 'Starting values of matrix and vector'
C      do i = 1, MatrixSize
C        write(*,10) i
C  10    format('Matrix_A(',I2,')=',$)
C        do j = 1, MatrixSize
C          write(*,20) Matrix_A(i,j)
C  20      format(F6.2,$)
C        enddo
C        write(*,30) i,Vector(i)
C  30    format('  Vector(',I2,')=',F6.2)
C        enddo
      print *, ' '
      print *, 'Results by thread/column: '

C     Create a team of threads and scope variables
C$OMP PARALLEL SHARED(Matrix_A,Vector,C) PRIVATE(i,ThreadID)
      ThreadID = OMP_GET_THREAD_NUM()

C     Loop work-sharing construct - distribute columns of matrix
C$OMP DO PRIVATE(j) ORDERED
      do i = 1, MatrixSize
        do j = 1, MatrixSize
          C(i) = C(i) + (Matrix_A(j,i) * Vector(i))
        enddo

C$OMP ORDERED

        print *,"thread",ThreadID,"did column",i,"C(",i,")= ",c(i)

C$OMP END ORDERED

      enddo
C$OMP END DO

C$OMP END PARALLEL

      print *, ' '

      do i = 1, MatrixSize
        do j = 1, MatrixSize
          Check_C(i) = Check_C(i) + (Matrix_A(j,i) * Vector(i))
        enddo
      enddo
      do i = 1, MatrixSize
         if(C(i) .ne. Check_C(i)) then
            ValidOutput = 0
            stop
         else
            ValidOutput = 1
         endif
      enddo

      if(ValidOutput .eq. 1) then
        print*,"------ Correct Result -----"
      else
        print*,"------ Wrong Result -----"
      endif

100   stop
      end
