c
c
c******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c******************************************************************
c
c                        Example 6 (Tools_MM_Mult_Cartesian.f)
c
c  Objective           : Matrix Matrix multiplication(Using Cartesian Topology)
c
c  Input               : Read files (mdata1.inp) for first input matrix 
c                        and (mdata2.inp) for second input matrix  
c
c  Output              : Result of matrix matrix multiplication on Processor 0.
c
c  Necessary Condition : Number of Processes should be less than
c                        or equal to 8. Vector size for Vectors A and
c                        B should be properly striped. that is Vector
c                        size should be properly divisible by
c                        Number of processes used.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c*********************************************************************
	 
	 
	 program main 

	 include 'mpif.h'

  	 integer A_Bloc_MatrixSize, B_Bloc_MatrixSize
  	 integer NoofRows_A, NoofCols_A, NoofRows_B, NoofCols_B
  	 integer NoofRows_BlocA, NoofCols_BlocA
  	 integer NoofRows_BlocB, NoofCols_BlocB

  	 integer Matrix_Size(0:4)
  	 integer Local_Index, Global_Row_Index, Global_Col_Index
  	 integer Local_varb, irow, icol, jrow 
	 integer Local_iproc, Local_irow	   	
  	 integer iproc, jproc, index, Proc_Id, Root
         integer MAX_ROWS,MAX_COLS
         parameter(MAX_ROWS=9,MAX_COLS=9)

	 integer Numprocs, MyRank, MyGlobalRank, root_p

  	real Matrix_A(MAX_ROWS,MAX_COLS),
     $       Matrix_B(MAX_ROWS,MAX_COLS),
     $       Matrix_C(MAX_ROWS,MAX_COLS)

  	real MatA_array(MAX_ROWS*MAX_COLS),
     $       MatB_array(MAX_ROWS*MAX_COLS), 
     $ 	     MatC_array(MAX_ROWS*MAX_COLS)
  	real A_Bloc_Matrix(MAX_ROWS*MAX_COLS), 
     $       B_Bloc_Matrix(MAX_ROWS*MAX_COLS), 
     $ 	     C_Bloc_Matrix(MAX_ROWS*MAX_COLS)

  	real LocalMatrix_A(MAX_ROWS*MAX_COLS)
  	real LocalMatrix_B(MAX_ROWS*MAX_COLS)
  	real RowBlock_A(MAX_ROWS*MAX_COLS), ColBlock_B(MAX_ROWS*MAX_COLS)

c        For Setting Cartesian Topology 
c	 Periods     - For Wraparound in each dimension.           
c	 Dimensions  - Number of processors in each dimension. 
c	 Coordinates - processor Row and Column identification 
c	 Remain_dims - For row and column communicators.      

   	 integer Periods(2)          
   	 integer Dimensions(2)       
   	 integer Coordinates(2)     
   	 integer Remain_dims(2)      
	 integer Comm2D, Row_comm, Col_comm

c        .....Initialising 
  	 call MPI_INIT (ierror)
   	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
   	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyGlobalRank, ierror)

c	 ........Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY 
   	 Dimensions(1) = sqrt(dble(Numprocs))
	 Dimensions(2) = sqrt(dble(Numprocs))
	 root_p = sqrt(dble(Numprocs))

c	 ........Wraparound mesh in both dimensions. 
   	 Periods(1) = 1
	 Periods(2) = 1    

c	 ........Create Cartesian topology in two dimnesions and 
c                Cartesian decomposition of the processes   

	 call MPI_DIMS_CREATE( Numprocs, 2, Dimensions, ierr )
   	 call MPI_CART_CREATE(MPI_COMM_WORLD, 2, Dimensions, 
     $             Periods, .TRUE., Comm2D, ierror)
		
   	 call MPI_COMM_RANK(Comm2D,MyRank,ierror)
   	 call MPI_CART_COORDS(Comm2D,MyRank,2,Coordinates,ierror)

   	 Row = Coordinates(1)
   	 Col = Coordinates(2)

c        Construction of row communicator and column communicators 
c        (use cartesian row and columne machanism to get
c	 Row and Col Communicators) 

   	 Remain_dims(1) = 0            
   	 Remain_dims(2) = 1 

c        The output communicator represents the column 
c	 containing the process 

   	 call MPI_CART_SUB(Comm2D, Remain_dims, Row_comm, ierror)
   
   	 Remain_dims(1) = 1
   	 Remain_dims(2) = 0

c        The output communicator represents the row containing 
c	 the process 
   	 call MPI_CART_SUB(Comm2D, Remain_dims, Col_comm, ierror)
	
c        Reading Input 
	 Root = 0
         if ( MyRank .eq. Root) then

c            ......Read the Matrix From Input file .....
	      open(unit=12, file = './data/mdata1.inp')
	     
             read(12,*) NoofRows_A, NoofCols_A
             write(6,*) 'Input matrix - A  ' 
             do i = 1,NoofRows_A
               read(12,*) (Matrix_A(i,j),j=1,NoofCols_A)
               write(6,75) (Matrix_A(i,j),j=1,NoofCols_A)
              enddo

	     open(unit=13, file = './data/mdata2.inp')
             read(13,*) NoofRows_B, NoofCols_B
             write(6,*) 'Input matrix - B' 
             do i = 1,NoofRows_B
               read(13,*) (Matrix_B(i,j),j=1,NoofCols_B)
               write(6,75) (Matrix_B(i,j),j=1,NoofCols_B)
             enddo

	     close(12)
	     close(13)

     	     Matrix_Size(0) = NoofRows_A
     	     Matrix_Size(1) = NoofCols_A
     	     Matrix_Size(2) = NoofRows_B
     	     Matrix_Size(3) = NoofCols_B

       endif

c      ..........Send Matrix Size to all processors  
	call MPI_BCAST (NoofRows_A, 1, MPI_INTEGER, Root, 
     $ 		Comm2D, ierror)
  	call MPI_BCAST (NoofCols_A, 1, MPI_INTEGER, Root, 
     $ 		Comm2D, ierror)
  	call MPI_BCAST (NoofRows_B, 1, MPI_INTEGER, Root,
     $  	Comm2D, ierror)
  	call MPI_BCAST (NoofCols_B, 1, MPI_INTEGER, Root, 
     $ 		Comm2D, ierror)

        if(NoofCols_A .ne. NoofRows_B) then
	 if(MyRank .eq. Root)
     $ 	   print*," Matrix Dimension is in correct for Multiplication"
	 goto 100
	endif

c  	 rem_Rows_A = mod(NoofRows_A, Dimensions(1))
c      	 rem_Cols_A = mod(NoofCols_A, Dimensions(1))
c
c  	 rem_Rows_B = mod(NoofRows_B, Dimensions(1))
c      	 rem_Cols_B = mod(NoofCols_B, Dimensions(1))

c	 

  	 if((mod(NoofRows_A, Dimensions(1)) .ne. 0) .or. 
     $ 	    (mod(NoofCols_A, Dimensions(1)) .ne. 0) .or. 
     $      (mod(NoofRows_B, Dimensions(1)) .ne. 0) .or. 
     $      (mod(NoofCols_B, Dimensions(1)) .ne. 0)) then

	  if(MyRank .eq. Root) 
     $     print*,"Matrices can't be divided among processors equally"
	  goto 100
	 endif

  	 NoofRows_BlocA = NoofRows_A / Dimensions(1)
  	 NoofCols_BlocA = NoofCols_A / Dimensions(1)

  	 NoofRows_BlocB = NoofRows_B / Dimensions(1)
  	 NoofCols_BlocB = NoofCols_B / Dimensions(1)

  	 A_Bloc_MatrixSize = NoofRows_BlocA * NoofCols_BlocA
  	 B_Bloc_MatrixSize = NoofRows_BlocB * NoofCols_BlocB

c	 print*,'MyRank',MyRank,NoofRows_BlocA, NoofCols_BlocA

c        ....Memory allocating for Bloc Matrices 
c        ....Rearrange the input matrices in one dimensional 
c           arrays by approriate order  

   	 if(MyRank .eq. Root) then

c            ...... Convert Matrix into one-dimensional array 
c	            according to Checkerboard 
	     Local_Index = 1
     	     do iproc = 0, root_p-1
               do jproc = 0, root_p-1

	       	do irow = 1, NoofRows_BlocA
		  Global_Row_Index = iproc * NoofRows_BlocA + irow 

	          do icol = 1, NoofCols_BlocA
		   Global_Col_Index = jproc * NoofCols_BlocA + icol
	           MatA_array(Local_Index) = 
     $		   Matrix_A(Global_Row_Index, Global_Col_Index)
	   	   Local_Index = Local_Index + 1
	         enddo 
	       enddo 
	      enddo 
	    enddo 

	    Local_Index = 1
     	    do iproc = 0, root_p-1
              do jproc = 0, root_p-1
	        do irow = 1, NoofRows_BlocB
	         Global_Row_Index = iproc * NoofRows_BlocB + irow 
	         do icol = 1, NoofCols_BlocB
		  Global_Col_Index = jproc * NoofCols_BlocB + icol
	          MatB_array(Local_Index) = 
     $		  Matrix_B(Global_Row_Index, Global_Col_Index)
	   	  Local_Index = Local_Index + 1
  		 enddo 
	       enddo 
	     enddo 
	   enddo 
        endif

c	Scatter the Data to all processes by MPI_SCATTER 

	call MPI_SCATTER(MatA_array, A_Bloc_MatrixSize, MPI_REAL, 
     $ 	    A_Bloc_Matrix, A_Bloc_MatrixSize, MPI_REAL, 
     $      Root, Comm2D, ierror)

	call MPI_SCATTER(MatB_array, B_Bloc_MatrixSize, MPI_REAL, 
     $      B_Bloc_Matrix, B_Bloc_MatrixSize, MPI_REAL, 
     $      Root, Comm2D, ierror)

c  	.........Broadcasting Data amongst rows and columns 
c
        call MPI_ALLGATHER(A_Bloc_Matrix, A_Bloc_MatrixSize,
     $ 	     MPI_REAL, RowBlock_A, A_Bloc_MatrixSize, MPI_REAL, 
     $       Row_comm, ierror)
  
  	call MPI_ALLGATHER (B_Bloc_Matrix, B_Bloc_MatrixSize, 
     $ 	     MPI_REAL, ColBlock_B, B_Bloc_MatrixSize, MPI_REAL, 
     $       Col_comm, ierror)

  	call MPI_BARRIER(Comm2D, ierror)

c	..Rearrange Data to do matrix multiplication for RowBlock_A  
  	index = 1
  	do irow = 0, NoofRows_BlocA-1
	  Local_irow = irow*NoofCols_BlocA

     	  do iproc = 0, root_p-1
	    Local_iproc = iproc * A_Bloc_MatrixSize

            do icol = 1, NoofCols_BlocA
	      Local_varb = icol + Local_irow + Local_iproc 
	      LocalMatrix_A(index) = RowBlock_A(Local_varb)
	      index = index + 1
	   enddo

	 enddo
	enddo

c      ... Rearrange Data to do matrix multiplication for ColBlock_B  
       index = 1
       do icol = 1, NoofCols_BlocB

          do iproc = 0, root_p-1
 	    Local_iproc = iproc*B_Bloc_MatrixSize

            do irow = 0, NoofRows_BlocB - 1
 	      Local_varb = icol + Local_iproc + irow*NoofCols_BlocB
              LocalMatrix_B(index) = ColBlock_B(Local_varb)
	      index = index + 1
	    enddo

	 enddo
       enddo

C     ........Multiply LocalBlock matrices to get C_Bloc_matrix 
      index = 1
      do irow = 0, NoofRows_BlocA-1
    	do icol = 0, NoofCols_BlocB-1
   	  C_Bloc_Matrix(index) = 0.0
	  do jrow = 1, NoofRows_B
	   C_Bloc_Matrix(index) = C_Bloc_Matrix(index) + 
     $     LocalMatrix_A(irow*NoofCols_A + jrow) * 
     $     LocalMatrix_B(icol*NoofRows_B+jrow)
	enddo
        index = index + 1 
        enddo
       enddo

c     ........Gather output block matrices at processor 0 

       call MPI_GATHER(C_Bloc_Matrix, NoofRows_BlocA*NoofCols_BlocB, 
     $      MPI_REAL, MatC_array, NoofRows_BlocA * NoofCols_BlocB, 
     $      MPI_REAL, Root, Comm2D, ierror)

c     ...Rearranging the output matrix in a array by approriate order  

      if (MyRank .eq. Root) then

       do iproc = 0, root_p-1
     	do jproc = 0, root_p-1
 	  Proc_Id = iproc * root_p + jproc
	  do irow = 1, NoofRows_BlocA
	     Global_Row_Index = iproc * NoofRows_BlocA + irow
	     do icol = 1, NoofCols_BlocB
		Global_Col_Index = jproc * NoofCols_BlocB + icol

	       	Local_Index  = (Proc_Id*NoofRows_BlocA*NoofCols_BlocB) 
     $ 		                + ((irow-1) * NoofCols_BlocB) + icol

		Matrix_C(Global_Row_Index,Global_Col_Index) = 
     $ 		MatC_array(Local_Index) 
             enddo
          enddo
	enddo
       enddo

         write(6,*) 'Output matrix ' 
         do i = 1,NoofRows_A
            write(6,75) (Matrix_C(i,j),j=1,NoofCols_B)
         enddo
 75	    format(8(2x,f8.3))

      endif

 100  call MPI_FINALIZE(ierror)

      stop
      end

