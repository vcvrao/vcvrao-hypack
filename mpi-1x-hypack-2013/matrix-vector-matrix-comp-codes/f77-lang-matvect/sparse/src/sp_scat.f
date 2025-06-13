c
c
c	*******************************************************
c
	subroutine SparseMatrixScatter(MyFirstRow,MyLastRow,
     $     rowptr,colind, values, MyTotalRowCount, 
     $     MyTotalColCount,Myrowptr, Mycolind, Myvalues,comm)
c
c	*******************************************************
c
	include 'mpif.h'
	include 'define.h'

	integer MyFirstRow(MAX_PROCS), MyLastRow(MAX_PROCS) 
	integer rowptr(MAX_PROCS+1), colind(MAX_NONZEROS)
	integer MyTotalRowCount, MyTotalColCount 
	integer Myrowptr(MAX_PROCS+1), Mycolind(MAX_NONZEROS)

	double precision values(TOTAL_SIZE)
	double precision Myvalues(TOTAL_SIZE)

     	integer Numprocs, MyRank, Root, iproc, comm
     	integer  start_row, end_row

     	integer Destination, Source, Destination_tag,Source_tag
     	integer status(MPI_STATUS_SIZE)

     	integer TotalRowCount(MAX_PROCS),
     $ 	        TotalColCount(MAX_PROCS), 
     $          TotalValuesCount(MAX_PROCS)
     	integer DisplacementRowptr(MAX_PROCS), 
     $          SendCountRowptr(MAX_PROCS)
     	integer Displacementcolind(MAX_PROCS), 
     $          SendCountcolind(MAX_PROCS)
     	integer Displacementvalues(MAX_PROCS), 
     $          SendCountvalues(MAX_PROCS)

    	call MPI_COMM_SIZE(comm, Numprocs, ierror)
     	call MPI_COMM_RANK(comm, MyRank, ierror)

	if(MyRank .eq. Root) then

       	 do iproc = 1, Numprocs

	         start_row     = MyFirstRow(iproc)
	         end_row       = MyLastRow(iproc) 
	  TotalRowCount(iproc) =(end_row - start_row) + 1
	  TotalColCount(iproc) =rowptr(end_row)-rowptr(start_row)
	  TotalValuesCount(iproc)=rowptr(end_row)-rowptr(start_row)

	  DisplacementRowptr(iproc) = start_row-1
	  SendCountRowptr(iproc)    = TotalRowCount(iproc) 

	  Displacementcolind(iproc) = rowptr(start_row)
	  SendCountcolind(iproc)    = TotalColCount(iproc) 

	  Displacementvalues(iproc) = rowptr(start_row)
	  SendCountvalues(iproc)    = TotalColCount(iproc) 

	 enddo

	  MyTotalRowCount    = TotalRowCount(1)
	  MyTotalColCount    = TotalColCount(1) 
	  MyTotalValuesCount = TotalValuesCount(1) 

	endif

     	if(MyRank .eq. Root) then
	   do iproc = 1, Numprocs-1
	      Destination     = iproc
	     Destination_tag  = 99

             call MPI_SEND(TotalRowCount(iproc+1),1,MPI_INTEGER,
     $            Destination,Destination_tag,comm, ierror)
	     call MPI_SEND(TotalColCount(iproc+1), 1, MPI_INTEGER, 
     $            Destination, Destination_tag, comm, ierror)
	     call MPI_SEND(TotalValuesCount(iproc+1), 1, MPI_INTEGER, 
     $            Destination, Destination_tag, comm, ierror)
	    enddo

     	  else

	    Source     = Root
	    Source_tag = 99

	    call MPI_RECV(MyTotalRowCount, 1, MPI_INTEGER, Source, 
     $           Source_tag, comm, status, ierror)
	    call MPI_RECV(MyTotalColCount, 1, MPI_INTEGER, Source, 
     $           Source_tag, comm, status, ierror)
	    call MPI_RECV(MyTotalValuesCount, 1, MPI_INTEGER, Source, 
     $           Source_tag, comm, status, ierror)

    	endif

     	call MPI_SCATTERV(rowptr, SendCountRowptr, DisplacementRowptr, 
     $       MPI_INTEGER, Myrowptr, MyTotalRowCount, MPI_INTEGER, 
     $       Root, comm, ierror) 

     	call MPI_SCATTERV(colind, SendCountcolind, Displacementcolind, 
     $       MPI_INTEGER, Mycolind, MyTotalColCount, MPI_INTEGER, 
     $       Root, comm, ierror) 

	call MPI_SCATTERV(values,  SendCountvalues, Displacementvalues, 
     $       MPI_DOUBLE_PRECISION, Myvalues, MyTotalValuesCount, 
     $       MPI_DOUBLE_PRECISION, Root, comm, ierror) 

	return
	end
c
c	**************************************************************
c
c
c
