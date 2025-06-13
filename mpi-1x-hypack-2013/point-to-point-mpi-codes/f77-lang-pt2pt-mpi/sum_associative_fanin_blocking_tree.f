c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 1.6         : sum_associative_fanin_blocking_tree.f
c
c  Objective           : To find sum of 'n' integers on 'p' processors 
c                        using 'Associative Fan-in' rule.
c                        MPI Blocking Communication library calls are used. 
c
c                        This example demonstrates the use of
c                        MPI_Init
c                        MPI_Comm_rank
c                        MPI_Comm_size
c                        MPI_Recv
c                        MPI_Send
c                        MPI_Finalize
c
c  Input               : Automatic input data generation  
c                        The rank of each process is input on each process. 
c
c  Output              : Process with Rank 0 should print the sum of 
c                        'n' values. 
c
c  Necessary Condition : Number of Processes should be 
c                        less than or equal to 8.
c
c   Created           : August-2013
c
c   E-mail            : hpcfte@cdac.in     
c
c*********************************************************************

       program main
       include 'mpif.h'

       integer 	   MyRank, Numprocs
       integer     Root
       integer     ilevel, value
       integer	   Source, Source_tag
       integer     Destination, Destination_tag
       integer	   Level, NextLevel
       real        NoofLevels
       integer     sum, ValidInput
       integer     status(MPI_STATUS_SIZE)

C      ....MPI initialisation....
       call MPI_INIT(ierror)
       call MPI_COMM_SIZE(MPI_COMM_WORLD,Numprocs, ierror)
       call MPI_COMM_RANK(MPI_COMM_WORLD,MyRank, ierror)

C      Check for the no. of processors to be a power of 2 

	 Root = 0
	 sum = 0
	 ValidInput = 0
	 do iproc = 1, Numprocs
		if((2 ** iproc) .eq. Numprocs) then
			NoofLevels = iproc
			ValidInput = 1
			goto 11
		endif
	 enddo

 11	 continue

	 if(ValidInput .ne. 1) then
   	    if(MyRank .eq. Root) 
     $      print*,"Number of processors should be power of 2"
	    goto 100
	 endif

    	 sum = MyRank 
    	 Source_tag = 0
    	 Destination_tag = 0

    	 do ilevel = 0, NoofLevels-1
	     Level = 2 ** ilevel
	
	    if(mod(MyRank, Level) .eq. 0) then
	       NextLevel = 2 ** (ilevel+1)

	     if(mod(MyRank, NextLevel) .eq. 0) then
	        Source = MyRank + Level
	        call MPI_RECV(value, 1, MPI_INTEGER, Source, Source_tag, 
     $                 MPI_COMM_WORLD, status, ierror)
	     		  sum = sum + value
	     else

C 0123456789012345678901234567890123456789012345678901234567890123456789
	     	Destination = MyRank - Level
	        call MPI_SEND(sum, 1, MPI_INTEGER, Destination, 
     $                        Destination_tag, MPI_COMM_WORLD, ierror)
	     endif
	     endif
    	 enddo

	 if(MyRank .eq. Root) print*,'MyRank',MyRank,'Final SUM ', sum

 100    call MPI_FINALIZE(ierror)

	     stop
	     end



