c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 5.7	       : mv_mult_checkerboard.f
c
c  Objective           : Matrix_Vector Multiplication
c                       (Using Block CheckerBoard Partitioning)
c
c  Input               : Process 0 reads files (mdata.inp) for Matrix
c                        and (vdata.inp) for Vector
c
c  Output              : Process 0 prints the result of Matrix_Vector
c                        Multiplication
c
c  Necessary Condition : Number of processors should be perfect
c                        square and less than or equal to 8
c
c   Created           : August-2013
c
c   E-mail            : hpcfte@cdac.in     
c
c******************************************************************
c
c
	program main

	include "mpif.h"

        integer ROW_SIZE, COL_SIZE, TOTAL_SIZE, VECTOR_SIZE

	parameter (ROW_SIZE = 9, COL_SIZE = 9)
	parameter (TOTAL_SIZE = ROW_SIZE*COL_SIZE)
	parameter (VECTOR_SIZE = ROW_SIZE*ROW_SIZE)

C       .......Variables Initialisation ......

  	integer Numprocs, MyRank, root_p, Root
  	integer irow, icol, iproc, jproc, index
  	integer NoofRows, NoofCols, NoofRows_Bloc, NoofCols_Bloc
  	integer Bloc_MatrixSize, Bloc_VectorSize, VectorSize
  	integer Local_Index, Global_Row_Index, Global_Col_Index
  	real Matrix(ROW_SIZE, COL_SIZE), Matrix_Array(TOTAL_SIZE)
	real Bloc_Matrix(TOTAL_SIZE) 
	real Vector(VECTOR_SIZE), Bloc_Vector(VECTOR_SIZE)
  	real FinalResult(VECTOR_SIZE), MyResult(ROW_SIZE)
	real FinalVector(VECTOR_SIZE)

  	integer colsize, colrank, rowsize, rowrank, MYRow
  	integer row_comm, col_comm

C   ........ MPI Initialisation ........  

  	call MPI_INIT(ierror)
  	call MPI_COMM_GROUP(MPI_COMM_WORLD,MPI_GROUP_WORLD, ierror)
  	call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
  	call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)

	Root = 0
  	root_p = sqrt(dble(Numprocs))
C	PRINT*, root_p

  	if(Numprocs .ne. (root_p * root_p)) then
	    if( MyRank .eq. Root) print*,"Error : Number 
     $	       of processors should be perfect square"
	      goto 100
	    endif		

       if(MyRank .eq. Root) then

C        .......Read the Matrix From Input file ......
	 open(unit=12, file = './data/mdata.inp')
         read(12,*) NoofRows, NoofCols
             write(6,*) 'Input matrix   ' 
         do i = 1,NoofRows
            read(12,*) (Matrix(i,j),j=1,NoofCols)
c            write(6,75) (Matrix(i,j),j=1,NoofCols)
         enddo

C        Read vector from input file 
	 open(unit=13, file = './data/vdata.inp')
         read(13,*) VectorSize
         read(13,*) (Vector(i), i=1,VectorSize)
             write(6,*) 'Input vector ' 
c         write(6,75) (Vector(i), i=1,VectorSize)

	close(12)
	close(13)
      endif

   	 call MPI_BCAST (NoofRows, 1, MPI_INTEGER, 
     $   	Root, MPI_COMM_WORLD, ierror)
   	 call MPI_BCAST (NoofCols, 1, MPI_INTEGER, 
     $   	Root, MPI_COMM_WORLD, ierror)
   	 call MPI_BCAST (VectorSize, 1, MPI_INTEGER, 
     $   	Root, MPI_COMM_WORLD, ierror)

   	 if(NoofCols .ne. VectorSize) then
	    if(MyRank .eq. Root) print*,"Matrice and Vector 
     $         Dimensions incompatible for Multiplication"
	       goto 100
	    endif		

   	 if((mod(NoofRows,root_p) .ne. 0) .or.
     $      (mod(NoofCols,root_p) .ne. 0)) then
	   if(MyRank .eq. Root) 
     $ 	      print*,"Matrix can't be divided among 
     $        processors equally"
	     goto 100
	   endif		

   	 NoofRows_Bloc = NoofRows / root_p
   	 NoofCols_Bloc = NoofCols / root_p

   	 if(MyRank .eq. Root) then
c          Convert Matrix into 1-D array according to Checkerboard 
	   Local_Index = 1
     	   do iproc = 0, root_p-1
           do jproc = 0, root_p-1
	     do irow = 1, NoofRows_Bloc
	   	Global_Row_Index = iproc * NoofRows_Bloc + irow 
	        do icol = 1, NoofCols_Bloc
	  	  Global_Col_Index = jproc * NoofCols_Bloc + icol
	          Matrix_Array(Local_Index) = 
     $		  Matrix(Global_Row_Index, Global_Col_Index)
	   	  Local_Index = Local_Index + 1
	        end do 
	     end do 
	   end do 
	   end do 
         endif
  	Bloc_VectorSize = VectorSize / root_p
  	Bloc_MatrixSize = NoofRows_Bloc * NoofCols_Bloc

  	call MPI_SCATTER(Matrix_Array, Bloc_MatrixSize, MPI_REAL, 
     $  	 Bloc_Matrix, Bloc_MatrixSize, MPI_REAL, 
     $	         Root, MPI_COMM_WORLD, ierror)

c       write(6,750) MyRank, (Bloc_Matrix(i),i=1,Bloc_matrixSize)

  	call MPI_BARRIER(MPI_COMM_WORLD)

c	........Creating groups of procesors row wise 
  	myrow = MyRank/root_p
  	call MPI_COMM_SPLIT(MPI_COMM_WORLD, myrow, 
     $ 	                    MyRank, row_comm, ierror)
  	call MPI_COMM_SIZE(row_comm, rowsize, ierror)
  	call MPI_COMM_RANK(row_comm, rowrank, ierror)

c       ........Creating groups of procesors column wise 
  	myrow=MOD(MyRank, root_p)
  	call MPI_COMM_SPLIT(MPI_COMM_WORLD, myrow, 
     $  	           MyRank, col_comm, ierror)
  	call MPI_COMM_SIZE(col_comm, colsize, ierror)
  	call MPI_COMM_RANK(col_comm, colrank, ierror)

c	Scatter part of vector to all row master processors 
  	if(MyRank/root_p .eq. 0) then
	call MPI_SCATTER(Vector, Bloc_VectorSize, MPI_REAL, 
     $  	Bloc_Vector, Bloc_VectorSize, 
     $          MPI_REAL, Root, row_comm, ierror)

	endif


  
c       Row master broadcasts the its vector part to
c       processors in its column

  	call MPI_BCAST(Bloc_Vector, Bloc_VectorSize, MPI_REAL, 
     $ 	               Root, col_comm, ierror)
   
c       write(6,751) MyRank, (Bloc_Vector(i),i=1,Bloc_VectorSize)

C       Multiplication done by all procs 
   	index = 1
   	do irow=1, NoofRows_Bloc
	   MyResult(irow)=0
      	  do icol=1, NoofCols_Bloc
	    MyResult(irow) = MyResult(irow) + 
     $ 	    Bloc_Matrix(index) * Bloc_Vector(icol)
	    index = index + 1
	 enddo
	enddo


c     Collect partial product from all procs on to master 
c   	processor and add it to get final answer                             

      if(NofRows_Bloc*Numprocs .ge. VECTOR_SIZE) then
	     write(6,*) 'error in dimension of the array FinalResults'
	     go to 100
	endif

   	 call MPI_GATHER (MyResult, NoofRows_Bloc, MPI_REAL, 
     $ 	 FinalResult, NoofRows_Bloc, MPI_REAL, 
     $   Root, MPI_COMM_WORLD, ierror)

c   	 if(MyRank .eq. Root) then
c           write(6,*) ' * Output vector ' 
c           write(6,75) (FinalResult(i),i=1,NoofRows_Bloc*Numprocs)
c	 endif

      if(MyRank .eq. Root) then
c	   ............
     	do iproc = 1, root_p
         do jproc = 1, root_p
	      do irow = 1, NoofRows_Bloc
		 
	   	 Global_Row_Index  = (iproc-1) * NoofRows_Bloc*root_p
     $ 	      	              + (jproc-1) * NoofRows_Bloc + irow 
		       Local_index = (iproc-1) * NoofRows_Bloc  + irow

               if(Global_Row_Index .ge. VECTOR_SIZE) then
		     write(6,*) 'error in dimension of the array FinalResults'
		     stop
		 endif

	        FinalVector(Local_Index) = FinalVector(Local_Index) 
     $	 	                       + FinalResult(Global_Row_Index)
c		  write(6,*) 'iproc,jproc,irow,Global_Row_Index,Local_index'
c		  write(6,*) iproc,jproc,irow,Global_Row_Index,Local_index
	     end do 
	    end do 
	   end do 

        write(6,*) ' *** Output final  - vector ' 
        write(6,75) (FinalVector(i),i=1,NoofRows)
c
   	endif

C	Free the groups formed 

 	call MPI_COMM_FREE(row_comm, ierror)
	call MPI_COMM_FREE(col_comm, ierror)

100	call MPI_FINALIZE(ierror)

75	   format(8(2x,f8.3))
750	   format(3x, ' MyRank =', i6/, 4(2x,f6.2))
751	   format(3x, ' ** MyRank =', i6/, 4(2x,f6.2))
752	   format(3x, ' $$ MyRank =', i6/, 4(2x,f6.2))

	stop
	end



