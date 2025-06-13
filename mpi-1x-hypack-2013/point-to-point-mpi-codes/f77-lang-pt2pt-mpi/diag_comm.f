c
c******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 12		: diag_comm.f 
c
c
c  Objective            : Mpi program to create communicator consisting of 
c                         processes that lie on the diagonal in the square 
c                         grid of processors, arranged in the form of a matrix.
c
c                         This example demonstrates the use of
c                         MPI_Init
c                         MPI_Comm_rank
c                         MPI_Comm_size
c                         MPI_Comm_group
c                         MPI_Group_incl
c                         MPI_Comm_create
c                         MPI_Finalize
c
c  Input                : Communicator consits of all processors.
c
c  OutPut               : process Id's of  communicator
c
c
c  Necessary Condition  : Number of Processers should be perfect square 
c	                       and less than or equal to 8.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c***********************************************************************

	 program main

         include 'mpif.h'
	 integer	i,iproc, index
	 integer	Numprocs, MyRank
	 integer	New_numprocs, NewMyRank
	 integer	process_rank(3), Group_Diag

	 integer	GroupWorld,new_group
	 integer	Diag_Group


C   ....MPI Initialisation....

  	 call MPI_INIT(ierror)
  	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
  	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)

	 Group_Diag= sqrt(dble(Numprocs))
	 if((Group_Diag * Group_Diag) .ne. Numprocs) then
	     if(MyRank .eq. Root) 
     * 	     print*,'Numprocs',Numprocs,'  is not a perfect square ...' 
	     goto 100
	 endif

         iproc=0	
	 do 10 index = 1, Group_Diag
		process_rank(index) = iproc
		iproc = iproc + Group_Diag + 1
 10	 continue

	 if(MyRank .eq. Root) then
		print *, "Processors forming the diagonal group are" 	
	   do 20 i = 1, Group_Diag
		print*, 'Processor', process_rank(i) 
 20        continue
	 endif


C       .... To create new group....

	
	 call MPI_COMM_GROUP(MPI_COMM_WORLD, GroupWorld, ierror)
	 call MPI_GROUP_INCL(GroupWorld, Group_Diag,process_rank,
     $			new_group, ierror)
	 call MPI_COMM_CREATE(MPI_COMM_WORLD,new_group,Diag_Group, ierror) 

         if ( mod(MyRank, Group_Diag+1) .eq. 0) then
	        call MPI_COMM_RANK(Diag_Group,NewMyRank, ierror)
  		call MPI_COMM_SIZE(Diag_Group,New_numprocs, ierror)

		print *, 'newnumprocs=',New_numprocs
		print*, 'NewRank in the diagonal group =', NewMyRank
  	 endif

100	 call MPI_FINALIZE(ierror) 
		 stop
		 end



