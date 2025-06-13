c
c
c*******************************************************************
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c   Example 5.4		 : mat_infnorm_blkstp.f
c
c   Objective            : Find Infinity norm of a matrix 
c	
c   Input                : File : infndata.inp Real Matrix 
c                          Read files (infndata) for Matrix A
c
c   Description          : Input matrix is stored in n by n format.Rowwise 
c                          block striped partitioning matrix is used. 
c
c   Output               : The infinity norm of matrix A on process with Rank 0
c
c   Necessary conditions : Number of Processes should be less than or equal
c                          to 8 Input Matrix Should be Square Matrix. 
c                          The size for Matrix A should be exactly dividible 
c                          by the number of processes used.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************


	 program main
	
	 include 'mpif.h'

         integer ROW_SIZE_A, COL_SIZE_A,TOTAL_SIZE_A
	 parameter (ROW_SIZE_A = 100, COL_SIZE_A = 100)
	 parameter (TOTAL_SIZE_A = ROW_SIZE_A*COL_SIZE_A)

c 	 ......Variables Initialisation ......
  	 integer n_size, NoofRows_Bloc, NoofRows, NoofCols
  	 integer Numprocs, MyRank, Root
  	 integer irow, icol, index, j

  	 double precision Matrix_A(ROW_SIZE_A,COL_SIZE_A), 
     $	        Input_A(TOTAL_SIZE_A)
	 double precision ARecv(TOTAL_SIZE_A),output(ROW_SIZE_A)
	 double precision infnorm, val_max

c	 ........MPI Initialisation .......
  	 call MPI_INIT(ierror) 
  	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
  	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
  
c	 .......Read the Matrix From Input file ......
	 Root = 0
	 if(MyRank .eq. Root) then
	   open(unit=12, file = './data/infndata.inp')
           read(12,*) NoofRows, NoofCols
           write(6,*) NoofRows, NoofCols
           do i = 1,NoofRows
              read(12,*) (Matrix_A(i,j), j=1,NoofCols)
              write(6, 75) (Matrix_A(i,j), j=1,NoofCols)
           enddo
75	 format(8(2x,f8.3)/)
     	   n_size=NoofRows
	   close(12)

c	   ..... Convert Matrix_A into 1-D array Input_A ...
	  index = 1
	  do irow=1, n_size
	     do icol=1, n_size
		Input_A(index) = Matrix_A(irow, icol)
		index = index + 1
	     enddo
	  enddo
	endif

c   .... Broad cast the size of the matrix to all ....
	call MPI_BCAST(NoofRows, 1, MPI_INTEGER, Root, 
     $	               MPI_COMM_WORLD, ierror) 
  	call MPI_BCAST(NoofCols, 1, MPI_INTEGER, Root, 
     $		       MPI_COMM_WORLD, ierror) 

  	if(NoofRows .ne. NoofCols) then
	   if(MyRank .eq. Root) 
     $	     print*,"Input Matrix Should Be Square Matrix .."
	   goto 100
	endif

c	..... n_size = size of the square matrix
	if(MyRank .ne. Root) n_size = NoofRows

  	if(mod(n_size, Numprocs) .ne. 0) then
	   if(MyRank .eq. 0) 
     $    print*,"Matrix Can not be Striped Evenly ..."
	   goto 100
	endif

  	NoofRows_Bloc = n_size/Numprocs

c  ......Scatter the Input Data to all process ......
  	call MPI_SCATTER (Input_A, NoofRows_Bloc * n_size, 
     $     MPI_DOUBLE_PRECISION, ARecv, NoofRows_Bloc*n_size, 
     $     MPI_DOUBLE_PRECISION, Root, MPI_COMM_WORLD,ierror)

c
c    Perform local rowwise maximum sum of absolute values of 
c    each row and accumlate local maximum 
c
      index = 0
      do irow = 1, NoofRows_Bloc
	     index = (irow-1)*NoofCols 
        sum = 0.0
          do j = 1, NoofCols 
	       sum = sum + abs(ARecv(index+j))
          end do 
          output(irow) = sum
      end do

c     Accumalate local maximum on each process 
c
      val_max = output(1)
      do irow =1, NoofRows_Bloc
	     if(output(irow) .gt. val_max) val_max = output(irow)
      enddo
		write(6,*) 'MyRank =', Myrank, 'Value_max =', val_max

c     .......Output vector .....
	call MPI_Reduce(val_max,infnorm,1,MPI_Real,MPI_Max, Root, 
     $		  MPI_COMM_WORLD, ierror)
	
	if(MyRank .eq. Root) Print *, "Infinity Norm of Matrix ", infnorm

 100   call MPI_FINALIZE(ierror)

	stop
	end
c
c	********************************************************
c
c
