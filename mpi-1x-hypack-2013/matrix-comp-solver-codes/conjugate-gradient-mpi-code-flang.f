c
c*****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c*******************************************************************
c
c
c       Example 6.3	     : conjugate-gradient-mpi-code-flang.f
c
c       Objective            : Conjugate Gradient Method to solve 
c                              AX = b matrix system of linear
c                              equations.
c	
c       Input                : Real Symmetric Positive definite 
c                              Matrix and the real vector - Input_A
c                              and Vector_B. Read files (mdatcg.inp) 
c                              for Matrix A and (vdatcg.inp) for 
c                              Vector b
c
c       Description          : Input matrix is stored in n by n
c                              format.Diagonal preconditioning
c                              matrix is used.Rowwise block striped
c                              partitioning matrix is used.Maximum
c                              iterations is given by MAX_ITERATIONS
c                              Tolerance value is given by EPSILON
c                              Header file used  : cgconst.h 
c
c       Output               : The solution of  Ax=b on process 
c                              with Rank 0 and the number of 
c                              iterations for convergence of the 
c                              method.
c
c       Necessary conditions : Number of Processes should be less
c                              than or equal to 8 Input Matrix 
c                              Should be Square Matrix. Matrix size 
c                              for Matrix A and vector size for
c                              vector b should be equally striped,
c                              that is Matrix size and Vector Size 
c                              should be dividible by the number of 
c                              processes used.
c
c   Created                  : August-2013
c
c   E-mail                   : hpcfte@cdac.in     
c
c********************************************************************


	program main

	include 'mpif.h'
	include 'cgconst.h'

	integer  NumProcs, MyRank, Root
	integer  NoofRows, NoofCols, VectorSize
	integer  NoofRows_Bloc, Bloc_MatrixSize
	integer  Bloc_VectorSize
	integer	 irow, icol, index, local_index, Iteration 

	double precision  Matrix_A(TOTAL_SIZE), 
     $	                  Input_A(MATRIX_SIZE, MATRIX_SIZE)
	double precision  Vector_B(VECTOR_SIZE), 
     $ 	                  Vector_X(VECTOR_SIZE)
	double precision  Bloc_Matrix_A(TOTAL_SIZE), 
     $	                  Bloc_Vector_X(VECTOR_SIZE)
	double precision  Direction_Vector(VECTOR_SIZE), 
     $	                  Bloc_Direction_Vector(VECTOR_SIZE)

	double precision  Bloc_Residue_Vector(VECTOR_SIZE)
	double precision  Bloc_Precond_Matrix(TOTAL_SIZE)
	double precision  Buffer(VECTOR_SIZE)
	double precision  Bloc_HVector(VECTOR_SIZE)

        double precision Delta0,Delta1,Bloc_Delta0,Bloc_Delta1
	double precision Tau, val, temp, Beta

	Root = 0
	Iteration = 1

C  	......Initialising MPI 
	call MPI_INIT(ierror)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, NumProcs, ierror)
	call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)

C       .....Read Input Matrix_A and Vector_B from file
	if(MyRank .eq. Root) then

	   open(unit = 12, file = './matrix-data-cg.inp')
         write(6,*) 'Input Matrix' 
	   read(12,*) NoofRows, NoofCols
  	   do i = 1, NoofRows
	    read(12,*) (Input_A(i,j), j=1, NoofCols)
         write(6,75) (Input_A(i,j),j=1,NoofCols)
  	   end do	

75    format(8(2x,f8.3))
C          ......Read vector from input file .......
	   open(unit=13, file = './vector-data-cg.inp')
         write(6,*) 'Input Vector' 
         read(13,*) VectorSize
         read(13,*) (Vector_B(i), i=1,VectorSize)
         write(6,75) (Vector_B(i), i = 1, NoofRows)

     	   n_size = VectorSize

C 	   .......Convert Input_A into 1-D array Matrix_A 
	   index = 1
	   do irow=1, NoofRows
	      do icol=1, NoofCols
		 Matrix_A(index) = Input_A(irow, icol)
		 index = index + 1
	      enddo
	   enddo
	endif
c
c  	Broadcast Matrix and Vector size and perform 
c	input validation tests
c
	call MPI_BCAST(NoofRows, 1, MPI_INTEGER, Root, 
     $	MPI_COMM_WORLD, ierror)

	call MPI_BCAST(NoofCols, 1, MPI_INTEGER, Root, 
     $	MPI_COMM_WORLD, ierror)

	call MPI_BCAST(VectorSize, 1, MPI_INTEGER, Root, 
     $	MPI_COMM_WORLD, ierror)

   	if(NoofRows .ne. NoofCols) then
	  if(MyRank .eq. Root) 
     $	  print*,"Error : Input Matrix Should be square matrix"
	  goto 100
	endif

   	if(NoofRows .ne.  VectorSize) then
	   if(MyRank .eq. Root) print*,
     $	      "Error : Matrix Size should be equal to VectorSize"
	   goto 100
	endif

	if(mod(NoofRows, NumProcs) .ne. 0) then
	   if(MyRank .eq. Root) print*, 
     $	     "Matrix cannot be evenly striped among processes"
	   goto 100
	endif

c	.....BroadCast Vector_B.......
	call MPI_BCAST(Vector_B, VectorSize, MPI_DOUBLE_PRECISION, 
     $	     Root, MPI_COMM_WORLD, ierror)

c	....... Scatter Matrix_A .......
   	NoofRows_Bloc   = NoofRows / NumProcs
	Bloc_VectorSize = NoofRows_Bloc
	Bloc_MatrixSize = NoofRows_Bloc * NoofCols

	call MPI_SCATTER(Matrix_A, Bloc_MatrixSize, 
     $	    MPI_DOUBLE_PRECISION, Bloc_Matrix_A, Bloc_MatrixSize, 
     $	    MPI_DOUBLE_PRECISION, Root, MPI_COMM_WORLD, ierror)

c	........Intialise Solution vector it to zero...... 
	do i = 1, VectorSize
       	   Vector_X(i) = 0.0
	end do

c	....Calculation of RESIDUE = Ax-B
	call CalculateResidueVector(Bloc_Residue_Vector,
     $ 	Bloc_Matrix_A, Vector_B, Vector_X, NoofRows_Bloc, 
     $  VectorSize, MyRank)
		
c	......Precondtion Matrix is identity matrix ......

	call GetPreconditionMatrix( Bloc_Precond_Matrix, 
     $	NoofRows_Bloc, NoofCols)

	call SolvePrecondMatrix(Bloc_HVector, Bloc_Residue_Vector, 
     $	Bloc_VectorSize)

c       .......Initailise Bloc Direction Vector = -(Bloc_HVector)...
	do index=1, Bloc_VectorSize
	   Bloc_Direction_Vector(index) = 0 - Bloc_HVector(index)
	enddo

c 	........Calculate Delta0 and check for convergence ..

	Bloc_Delta0 = 0.0
        do index = 1, Bloc_VectorSize 
	   Bloc_Delta0 = Bloc_Delta0 + 
     $	   Bloc_Residue_Vector(index) * Bloc_HVector(index)
	enddo

	call MPI_Allreduce(Bloc_Delta0, Delta0, 1, 
     $	  MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD,ierror)

	if(Delta0 .le. EPSILON) then
      	   if(MyRank .eq. Root) write(6,77) 
77	format(3x, / '**** Algorithm Converged '/)
	   goto 100
	endif

	Iteration = 0			

 400    Iteration = Iteration + 1

c	......Gather Direction Vector on all processes.......
	call MPI_Allgather(Bloc_Direction_Vector,Bloc_VectorSize, 
     $	  MPI_DOUBLE_PRECISION, Direction_Vector,Bloc_VectorSize, 
     $	  MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierror)

c   	Compute Tau value 
c	Tau = Delta0/(DirVector Transpose*Matrix_A*DirVector)..

	do irow=1, NoofRows_Bloc
	   index = (irow-1) * NoofCols + 1
	   Buffer(irow) = 0.0
	   do icol=1, NoofCols
	      Buffer(irow) = Buffer(irow) +  
     $	      Bloc_Matrix_A(index) * Direction_Vector(icol)
	      index = index + 1
	   enddo
	enddo

	temp = 0.0
	do index=1, Bloc_VectorSize
	 temp = temp+Bloc_Direction_Vector(index)*Buffer(index)
	enddo

	call MPI_Allreduce(temp, val, 1, MPI_DOUBLE_PRECISION,
     $	     MPI_SUM, MPI_COMM_WORLD, ierror)

	Tau = Delta0 / val

c       Compute new vector Xnew = Xold + Tau*Direction
c       Compute BlocResidueVector which is given by
c       BlocResidueVect+Tau*Bloc_MatA*DirVector
c
	do index = 1, Bloc_VectorSize
	   local_index = MyRank*Bloc_VectorSize + index
	   Bloc_Vector_X(index) = Vector_X(local_index) + 
     $	                       Tau*Bloc_Direction_Vector(index)
	   Bloc_Residue_Vector(index)=Bloc_Residue_Vector(index) 
     $					+ Tau*Buffer(index)
	enddo

c	...Gather New Vector X at all  processes......
	call MPI_Allgather(Bloc_Vector_X, Bloc_VectorSize, 
     $	     MPI_DOUBLE_PRECISION, Vector_X, Bloc_VectorSize, 
     $	     MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, ierror)


	call SolvePrecondMatrix(Bloc_HVector, Bloc_Residue_Vector, 
     $	     Bloc_VectorSize)
		
	Bloc_Delta1 = 0.0
       	do index = 1, Bloc_VectorSize 
	   Bloc_Delta1 = Bloc_Delta1 + 
     $	   Bloc_Residue_Vector(index) * Bloc_HVector(index)
	enddo

	call MPI_Allreduce(Bloc_Delta1, Delta1, 1, 
     $	     MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierror)

C	   If(MyRank .eq. 0) then 
C	     write(6,76) Iteration, Delta1
C 76	     format(1x, 'Iteration =', i6, 2x, 'Residue for 
C     $      successive iterations = ', E16.8)
C	   endif

	if(Delta1 .le. EPSILON) then
	   goto 90
	endif

	Beta   = Delta1 / Delta0
	Delta0 = Delta1

	do index=1, Bloc_VectorSize
	   Bloc_Direction_Vector(index) = -Bloc_HVector(index) 
     $		           + Beta*Bloc_Direction_Vector(index)
	enddo

	if(Iteration .lt. MAX_ITERATIONS) then
	   go to 400
	else
	   goto 80
	endif

 80	if(MyRank .eq. Root) print*,'Iterations Exhausted'
	   go to 200
 90	if(MyRank .eq. Root)  write(6,77)
	   go to 200

 200	if(MyRank .eq. 0) then
	   print*,'Number of iterations : ',Iteration
     	   print*,'Number of processors : ',NumProcs
	   print*,'Results for solution of matrix system of equations'
	   do index = 1,NoofRows
	     print*,'X(',index,')= ',Vector_X(index)
	   enddo
	endif

 100	call MPI_FINALIZE(ierror)

	stop
	end	

c 	**************************************************************
c
	subroutine CalculateResidueVector(Bloc_Residue_Vector, 
     $ 	           Bloc_Matrix_A, Vector_B, Vector_X, NoofRows_Bloc, 
     $             VectorSize, MyRank)
c 	**************************************************************
c
	include 'cgconst.h'

	double precision Bloc_Residue_Vector(VECTOR_SIZE)
	double precision Bloc_Matrix_A(TOTAL_SIZE)
	double precision Vector_B(VECTOR_SIZE)
	double precision Vector_X(VECTOR_SIZE)
	integer NoofRows_Bloc, VectorSize, MyRank
 
C	... Computes residue = AX - b .......

	integer   irow, index, GlobalVectorIndex
	double precision value

	GlobalVectorIndex = MyRank * NoofRows_Bloc + 1
	do irow=1, NoofRows_Bloc
	   value = 0.0
	   index = irow * VectorSize
	  do jcol = 1, VectorSize
	    value = value + Bloc_Matrix_A(index)*Vector_X(jcol)
	   index = index + 1
	  enddo
	  Bloc_Residue_Vector(irow) =
     $	       value-Vector_B(GlobalVectorIndex)
	  GlobalVectorIndex = GlobalVectorIndex + 1
	enddo

        return
        end
c
c 	*********************************************************
	subroutine GetPreconditionMatrix( Bloc_Precond_Matrix, 
     $             NoofRows_Bloc, NoofCols)
c
c 	********************************************************
c
	include 'cgconst.h'
	double precision Bloc_Precond_Matrix(TOTAL_SIZE) 

	integer NoofRows_Bloc, NoofCols
	integer	Bloc_MatrixSize
	integer	irow, icol, index

	Bloc_MatrixSize = NoofRows_Bloc*NoofCols

c	......Preconditioned Martix is identity matrix .......
   	index = 1
	do irow=1, NoofRows_Bloc
	   do icol=1, NoofCols
		Bloc_Precond_Matrix(index) = 1.0
		index = index + 1
	   enddo
	enddo

	return
	end

c 	************************************************************
	subroutine SolvePrecondMatrix(HVector, Bloc_Residue_Vector, 
     $	           Bloc_VectorSize)
c 	************************************************************
c
	include 'cgconst.h'
	double precision HVector(VECTOR_SIZE)
	double precision Bloc_Residue_Vector(VECTOR_SIZE) 

	integer Bloc_VectorSize
	integer	index

c	Calculate the HVector 
c	HVector=Inverse of Bloc_Precond_Matrix*Bloc_Residue_Vector.

	do index = 1, Bloc_VectorSize
		HVector(index) = Bloc_Residue_Vector(index)
	enddo

	return
	end

c ********************************************************************

