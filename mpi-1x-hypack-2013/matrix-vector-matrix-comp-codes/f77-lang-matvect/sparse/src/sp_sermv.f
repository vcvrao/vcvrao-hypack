c
c	*******************************************************

	 subroutine SerialSparseMatrixVector(Myrows,Myrowptr,
     $ 	     Mycolind, MyFirstRow, MyLastRow, Myvalues, vector, 
     $       Myoutput, comm) 
c
c	********************************************************



	include 'define.h'

	integer Myrowptr(MAX_PROCS+1), Mycolind(MAX_NONZEROS) 
	integer MyFirstRow(MAX_PROCS), MyLastRow(MAX_PROCS) 

	double precision Myvalues(TOTAL_SIZE) 
	double precision vector(VECTOR_SIZE) 
	double precision Myoutput(MATRIX_SIZE), Product 

  	integer  i, j, count, npes, mype, comm, Myrows 

  	  call MPI_COMM_SIZE(comm, npes, ierror)
  	  call MPI_COMM_RANK(comm, mype, ierror)

  	  count = 1

  	  do i = 1, Myrows-1
    	    Myoutput(i) = 0.0
    	    do j = Myrowptr(i)+1, Myrowptr(i+1)
                 Product = Myvalues(count)*vector(Mycolind(count))   
             Myoutput(i) = Myoutput(i) + Product
	           count = count + 1
   	    enddo
  	 enddo  
			 
 	 return
	 end
	 
c	***********************************************************




