c
c	********************************************************

	subroutine MatrixRowsCommunication(nsize, MyFirstRow, 
     $             MyLastRow, comm)
c
c	********************************************************
c
	include 'mpif.h'
	include 'define.h'

	integer Root, comm
	integer i, j, k, iproc
	integer Numprocs,  MyRank, nsize 
	integer Destination, Source, Destination_tag, Source_tag
	integer DistributeRows, MyStart, MyLast, remaining_rows

	integer LocalProcRows(MAX_PROCS), temp(MAX_PROCS*2)
	integer MyFirstRow(MAX_PROCS),  MyLastRow (MAX_PROCS)

	Root = 0

C	MPI - communication calls 
	call MPI_Comm_rank(comm, MyRank, ierror)
	call MPI_Comm_size(comm, Numprocs, ierror)

	if(MyRank .eq. Root) then

C	  Find first and last of row distribution of sparse matrix 
	  DistributeRows = nsize/Numprocs
	  remaining_rows = mod(nsize, Numprocs)

	  do i = 1, Numprocs
	   LocalProcRows(i) = DistributeRows
	  enddo

	  do i = remaining_rows, 1, -1 
	   LocalProcRows(i) = LocalProcRows(i) + 1
	  enddo

	  do i = 1, Numprocs

	   MyStart = 1
	   do j = 1, i-1
	     MyStart = MyStart + LocalProcRows(j)
	   enddo

  	          MyLast = MyStart + LocalProcRows(j)
	   MyFirstRow(i) = MyStart
	   MyLastRow(i)  = MyLast
	  enddo

	  do j = 1, Numprocs
	   temp(j) = MyFirstRow(j)
	         k = j+Numprocs
	   temp(k) = MyLastRow(j)
	  enddo

	  do iproc = 1, Numprocs-1
	     Destination     = iproc
	     Destination_tag = 100

             call MPI_SEND(temp,2*Numprocs,MPI_INTEGER,
     $ 	     Destination, Destination_tag,comm, ierror)
	  enddo
	else
	      Source  = Root
	  Source_tag  = 100

          call MPI_RECV(temp, 2*Numprocs, MPI_INTEGER,Source, 
     $         Source_tag,comm, status, ierror)

	  do j = 1, Numprocs
	   MyFirstRow(j) = temp(j)
	  enddo

	  do j = 1, Numprocs
	   MyLastRow(j)  = temp(j+Numprocs)
	  enddo

	endif

	return
	end
c
c	**********************************************************
c 
c
