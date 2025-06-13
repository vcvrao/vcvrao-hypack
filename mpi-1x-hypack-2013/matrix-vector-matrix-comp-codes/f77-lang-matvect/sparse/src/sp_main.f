c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c *******************************************************************
c
c                        Example 27
c
c  Objective : Simple  parallel formulation of Sparse Matrix-Vector 
c               Multiplication algorithm. 
c
c  Input     : Generated input matrix by random numbers 
c  Output    : Process 0 prints the result of Matrix-Vector 
c              Multiplication 
c
c  Description : Generate the input matrix randomly and convert it 
c               into the CSR format scheme. 
c	        Use the strip parttitioning algorithm to distribute 
c		    the n/p rows to respective processors. 
c	        Develop Off Processor Communication library. 
c	        Compute Serial  Sparse Matrix vector multiplication 
c		        algorithm concurrently on every processes. 
c	        Gather the final data on every process.  
c
c *******************************************************************


	program main
        implicit none

	include 'mpif.h'
	include 'define.h'

 	integer Root, index, ierror, Numprocs, MyRank
 	integer Myrows, Mycols, MyTotalRowCount, MyTotalColCount
	integer Mycolind(MAX_NONZEROS), Myrowptr(MATRIX_SIZE + 1) 

	double precision Myvalues(TOTAL_SIZE)
	double precision global_output(MATRIX_SIZE) 

c	Total number of non zero entries. 
 	integer nvtxs         

c  	Total number of rows of the sparse matrix. 
 	integer nsize         

c  	First and last row on each processes 
 	integer MyFirstRow(MAX_PROCS)  
 	integer MyLastRow(MAX_PROCS)  

c 	CSR rowptr 
 	integer  rowptr(MATRIX_SIZE + 1)       

c	CSR Column index 
 	integer  colind(MAX_NONZEROS)

c  	CSR Sparse matrix non-zero entries 
 	double precision   values(TOTAL_SIZE), Myoutput(MATRIX_SIZE)   

c  	Vector values in the sparse matrix-vector multiplication 
 	double precision   vector(VECTOR_SIZE) 

 	call MPI_INIT(ierror)
	call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
 	call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)


 	if(MyRank .eq. Root) then
	  call ReadSparseInput(nvtxs,nsize,vector,rowptr,colind,values)
	endif

 	call MPI_BARRIER(MPI_COMM_WORLD, ierror)

 	call MPI_BCAST(nsize,1,MPI_INTEGER,Root,MPI_COMM_WORLD,ierror)
 	call MPI_BCAST(nvtxs,1,MPI_INTEGER,Root,MPI_COMM_WORLD,ierror)

c       Root sends info about first and last row on each process
c	to all process 

 	call MatrixRowsCommunication(nsize, MyFirstRow, MyLastRow, 
     $	     MPI_COMM_WORLD)

        call SparseMatrixScatter(MyFirstRow, MyLastRow, rowptr, colind,
     $	     values, MyTotalRowCount, MyTotalColCount, Myrowptr, 
     $       Mycolind, Myvalues, MPI_COMM_WORLD)

 	Myrows   = MyTotalRowCount
 	Mycols   = MyTotalColCount

 	call MPI_BCAST(vector, nsize, MPI_DOUBLE_PRECISION, Root, 
     $                 MPI_COMM_WORLD, ierror)

         call SerialSparseMatrixVector(Myrows, Myrowptr, Mycolind, 
     $        MyFirstRow, MyLastRow, Myvalues, vector, Myoutput, 
     $        MPI_COMM_WORLD)

         call OutputGather(nsize, Myrows, MyFirstRow, MyLastRow, 
     $                 Myoutput, global_output, MPI_COMM_WORLD)


	if(MyRank .eq. Root) then
	   do index = 1, nsize
	      print*,'FinalResult(',index,') =',global_output(index)
	   enddo
	endif

 100	call MPI_Finalize(ierror) 

	stop
	end

c	*************************************************************
c

