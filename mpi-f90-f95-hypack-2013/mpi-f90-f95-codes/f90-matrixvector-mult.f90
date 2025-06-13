!**************************************************************************
!             C-DAC Tech Workshop : hyPACK-2013 
!                     Oct 15 - 18, 2013
!
!   Example 4  : f90-matvect-mult.f90
!
!   Objective  : Matrix vector multiplication
!               This example demonstrates the use of
!               Allocatable arrays and Intrinsic function MATMUL
!               
!   Input      : Size of matrix.  
!
!   Output     : matrix vector multiplication product
!
!  
!  Created     : August 2013  
!
!   E-mail     : hpcfte@cdac.in                                          
!
!************************************************************************

      program MatVectMult

      integer  DATA_SIZE
      parameter (DATA_SIZE=800)
      integer :: i, j, ValidOutput, MatrixSize
      integer :: alloc_error, dealloc_error

       real, ALLOCATABLE, DIMENSION(:, :) :: Matrix_A
       real, ALLOCATABLE, DIMENSION(:) :: Vector
       real, ALLOCATABLE, DIMENSION(:) :: C, Check_C


      print *,'Input Size of Matrix'
		read(*,*) MatrixSize

      if((MatrixSize .le. 0) .or. (MatrixSize .gt. DATA_SIZE)) then
        print *,'Invalid MatrixSize.'
        print *,'Please try with another number less than', DATA_SIZE+1
        goto 100
	   endif

      ALLOCATE (Matrix_A(MatrixSize, MatrixSize), STAT=alloc_error)
      if(alloc_error /= 0) then
        print*,"Insufficient space to allocate Matrix_A when MatrixSize is", &
                MatrixSize
      endif

      ALLOCATE (Vector(MatrixSize), STAT=alloc_error)
      if(alloc_error /= 0) then
        print*,"Insufficient space to allocate Vector when MatrixSize is", &
                MatrixSize
      endif

      ALLOCATE (C(MatrixSize), STAT=alloc_error)
      if(alloc_error /= 0) then
        print*,"Insufficient space to allocate C when MatrixSize is", &
                MatrixSize
      endif

      ALLOCATE (Check_C(MatrixSize), STAT=alloc_error)
      if(alloc_error /= 0) then
        print*,"Insufficient space to allocate Check_C when MatrixSize is", &
                MatrixSize
      endif

!     Initializations
      do i = 1, MatrixSize
        do j = 1, MatrixSize
          Matrix_A(i,j) = i * 1.0
        enddO
        Vector(i) = i
        C(i) = 0.0
      enddo

!      print *, 'Starting values of matrix and vector'
!      write(*,*) 'Matrix_A'
!      do i = 1, MatrixSize
!        print*,(Matrix_A(i,j),j=1,MatrixSize)
!      enddo

!      write(*,*) 'Vector'
!      do i = 1, MatrixSize
!        write(*,30) Vector(i)
!  30    format(F6.2)
!      enddo

     C = MATMUL(Matrix_A, Vector)


      do i = 1, MatrixSize
        do j = 1, MatrixSize
          Check_C(i) = Check_C(i) + (Matrix_A(j,i) * Vector(i))
        enddo
      enddo
      do i = 1, MatrixSize
         if(C(i) .ne. Check_C(i)) then
            ValidOutput = 0
         else
            ValidOutput = 1
         endif
      enddo

      if(ValidOutput .eq. 1) then
        print*,"------ Correct Result -----"
!      write(*,*) 'Product of Matrix and Vector'
!      do i = 1, MatrixSize
!          print *, C(i)
!      enddo
      else
        print*,"------ Wrong Result -----"
      endif

      DEALLOCATE (Matrix_A, STAT=dealloc_error)
      if(dealloc_error /= 0) then
        print*,"Unexpected deallocation error"
      endif

      DEALLOCATE (Vector, STAT=dealloc_error)
      if(dealloc_error /= 0) then
        print*,"Unexpected deallocation error"
      endif

      DEALLOCATE (C, STAT=dealloc_error)
      if(dealloc_error /= 0) then
        print*,"Unexpected deallocation error"
      endif

      DEALLOCATE (Check_C, STAT=dealloc_error)
      if(dealloc_error /= 0) then
        print*,"Unexpected deallocation error"
      endif


100   stop
      end
