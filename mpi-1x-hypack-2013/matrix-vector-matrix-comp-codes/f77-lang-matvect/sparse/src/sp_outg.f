c
c	***********************************************************	
	subroutine OutputGather(nsize, Myrows, MyFirstRow, 
     $ 	           MyLastRow, Myoutput, global_output, comm) 
c	**********************************************************	
	include 'mpif.h'
	include 'define.h'
	integer nsize 
	integer Myrows
	integer MyFirstRow(MAX_PROCS)
	integer MyLastRow(MAX_PROCS)

	double precision Myoutput(MATRIX_SIZE)
	double precision global_output(MATRIX_SIZE)

	integer comm 
  	integer Root
  	integer Numprocs, MyRank
  	integer start_row, end_row, iproc
  	integer Displacementvector(MAX_PROCS)
  	integer RecvCountvector(MAX_PROCS)
	integer TotalLocalRowCount(MAX_PROCS), MyLocalRowCount

	Root = 0

  	call MPI_Comm_size(comm, Numprocs, ierror)
  	call MPI_Comm_rank(comm, MyRank, ierror)
			
  	do iproc=1, Numprocs
    	     start_row                = MyFirstRow(iproc)
    	     end_row                  = MyLastRow(iproc) 
    	    TotalLocalRowCount(iproc) = end_row - start_row 
    	    Displacementvector(iproc) = start_row-1
    	    RecvCountvector(iproc)    = TotalLocalRowCount(iproc) 
  	enddo

	MyLocalRowCount = Myrows - 1

	call MPI_Gatherv(Myoutput, MyLocalRowCount,  
     $        MPI_DOUBLE_PRECISION, global_output, RecvCountvector, 
     $        Displacementvector,  MPI_DOUBLE_PRECISION, Root, comm, 
     $        ierror)

c	if (MyRank .eq. Root) then
c       do index = 1, nsize
c	     print*,'After Gather Result',index, global_output(index)
c	   enddo
c	endif

	return
	end

c	***************************************************************
c



