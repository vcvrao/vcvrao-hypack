c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c       Example 6.1	     : jacobi-mpi-code-clang.f
c
c       Objective            : Jacobi method to solve AX = b matrix 
c                              system of linear equations.
c	
c       Input                : Real Symmetric Positive definite Matrix 
c                    	       and the real vector - Input_A and Vector_B
c                              Read files (mdatjac.inp) for Matrix A
c                     	       and (vdatjac.inp) for Vector b
c
c       Description          : Input matrix is stored in n by n format.
c                              Diagonal preconditioning matrix is used.
c                              Rowwise block striped partitioning matrix
c                              is used. Maximum iterations is given by 
c                              MAX_ITERATIONS.Tolerance value is given 
c                              by EPSILON
c
c       Output               : The solution of  Ax=b on process with 
c                              Rank 0 and the number of iterations 
c                              for convergence of the method.
c
c       Necessary conditions : Number of Processes should be less than
c                              or equal to 8 Input Matrix Should be 
c                              Square Matrix. Matrix size for Matrix A
c                              and vector size for vector b should be 
c                              equally striped, that is Matrix size and 
c                              Vector Size should be dividible by the 
c                              number of processes used.
c
c
c       Created             : August-2013
c
c       E-mail              : hpcfte@cdac.in     
c
c***********************************************************************


	 program main

	 include 'mpif.h'
	 include 'jacconst.h'

c 	 ......Variables Initialisation ......
  	 integer n_size, NoofRows_Bloc, NoofRows, NoofCols
  	 integer Numprocs, MyRank, Root
  	 integer irow, icol, index, Iteration, GlobalRowNo

	 double precision ARecv(TOTAL_SIZE_A), BRecv(ROW_SIZE_B)
  	 double precision Matrix_A(ROW_SIZE_A, COL_SIZE_A), 
     $	        Input_A(TOTAL_SIZE_A), Input_B(ROW_SIZE_B)
  	 double precision X_New(ROW_SIZE_A), X_Old(ROW_SIZE_A), 
     $		Bloc_X(ROW_SIZE_A)
	 double precision Tolerance

c	 ........MPI Initialisation .......
  	 call MPI_INIT(ierror) 
  	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
  	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
  
c	 .......Read the Matrix From Input file ......
	 Root = 0
	 if(MyRank .eq. Root) then
	   open(unit=12, file = './marix-data-jacobi.inp')
           read(12,*) NoofRows, NoofCols
           do j = 1,NoofCols
              read(12,*) (Matrix_A(i,j),i=1,NoofRows)
           enddo

c    ....... Read vector from input file .......
	   open(unit=13, file = './vector-data-jacobi.inp')
           read(13,*) NoofRows
           read(13,*) (Input_B(i), i=1,NoofRows)
     	   n_size=NoofRows
	   close(12)
	   close(13)

c	   ..... Convert Matrix_A into 1-D array Input_A ...
	  index = 1
	  do irow=1, n_size
	     do icol=1, n_size
		Input_A(index) = Matrix_A(irow, icol)
		index = index + 1
	     enddo
	  enddo
	endif

	call MPI_BCAST(NoofRows, 1, MPI_INTEGER, Root, 
     $	               MPI_COMM_WORLD, ierror) 
  	call MPI_BCAST(NoofCols, 1, MPI_INTEGER, Root, 
     $		       MPI_COMM_WORLD, ierror) 

  	if(NoofRows .ne. NoofCols) then
	   if(MyRank .eq. Root) 
     $	     print*,"Input Matrix Should Be Square Matrix .."
	   goto 100
	endif


c   .... Broad cast the size of the matrix to all ....
  	call MPI_BCAST(n_size, 1, MPI_INTEGER, Root, 
     $		 MPI_COMM_WORLD, ierror) 

  	if(mod(n_size, Numprocs) .ne. 0) then
	   if(MyRank .eq. 0) 
     $	      print*,"Matrix Can not be Striped Evenly ..."
	   goto 100
	endif

  	NoofRows_Bloc = n_size/Numprocs

c  ......Scatter the Input Data to all process ......
  	call MPI_SCATTER (Input_A, NoofRows_Bloc * n_size, 
     $	     MPI_DOUBLE_PRECISION, ARecv, NoofRows_Bloc*n_size, 
     $	     MPI_DOUBLE_PRECISION, Root, MPI_COMM_WORLD,ierror)


  	call MPI_SCATTER (Input_B, NoofRows_Bloc, 
     $ 	     MPI_DOUBLE_PRECISION,BRecv, NoofRows_Bloc, 
     $	     MPI_DOUBLE_PRECISION, Root, MPI_COMM_WORLD, ierror)

c  ....... Initailize X[i] = B[i] ....... 
	do irow=1, NoofRows_Bloc
	   Bloc_X(irow) = BRecv(irow)
	enddo

	call MPI_ALLGATHER(Bloc_X, NoofRows_Bloc, 
     $ 	    MPI_DOUBLE_PRECISION,X_New, NoofRows_Bloc, 
     $      MPI_DOUBLE_PRECISION,MPI_COMM_WORLD, ierror)

  	Iteration = 1
  	do Iteration =1, MAX_ITERATIONS-1

	   do irow=1, n_size
	      X_Old(irow) = X_New(irow)
	   enddo

      	   do irow=1, NoofRows_Bloc
              GlobalRowNo = (MyRank * NoofRows_Bloc) + irow
	      Bloc_X(irow) = BRecv(irow)
	      index = (irow-1) * n_size + 1

	      do icol = 1, GlobalRowNo-1
	 	 Bloc_X(irow) = Bloc_X(irow) - 
     $		     X_Old(icol) * ARecv(index + icol - 1)
	      enddo
	      do icol = GlobalRowNo+1, n_size
	 	 Bloc_X(irow) = Bloc_X(irow) - 
     $		      X_Old(icol) * ARecv(index + icol - 1)
 	      enddo
	      iindex = (irow-1)*n_size
              Bloc_X(irow)=Bloc_X(irow)/ARecv(iindex+GlobalRowNo)

	   enddo

  	   call MPI_ALLGATHER(Bloc_X, NoofRows_Bloc, 
     $		MPI_DOUBLE_PRECISION, X_New, NoofRows_Bloc, 
     $		MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierror)

c   ... Check for Tolerance ....

	   call Distance(X_Old, X_New, n_size, Tolerance)
	   if(Tolerance .le. EPSILON) then
	      goto 90
	   endif

	enddo
  
c	.......Output vector .....

 90     if(MyRank .eq. 0 )print*,"Solution converged"

	if (MyRank .eq. Root) then
     	    print*,"Results of Jacobi Method"
	    print*,"Number of iterations = ",Iteration
     	    print*,"Solution vector"
     	    do irow = 1, n_size
               print*,'X(',irow,') =', X_New(irow)
	    enddo
	endif
  
 100    call MPI_FINALIZE(ierror)

	stop
	end
c
c	********************************************************
c


	subroutine Distance(X_Old, X_New, n_size, Tolerance)

	include 'jacconst.h'
  	double precision X_New(ROW_SIZE_A), X_Old(ROW_SIZE_A)
	double precision Tolerance
	integer n_size
	integer index
			  
	Tolerance = 0.0
	do index=1, n_size
	    Tolerance = Tolerance + 
     $	    (X_New(index) - X_Old(index))*(X_New(index)-X_Old(index))
	enddo

	return
	end





