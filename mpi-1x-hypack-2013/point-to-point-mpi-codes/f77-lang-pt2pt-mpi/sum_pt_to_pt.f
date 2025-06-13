c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 1.3:        : sum_pt_to_pt.f
c
c  Objective           : MPI program to Sum n integers using point-to-point 
c                        communications.
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
c  Output              : Process with rank 0 should print the sum of 
c                        Rank values. 
c
c  Necessary Condition : Number of Processes should be 
c                        less than or equal to 8.
c
c   Created            : August-2013
c
c   E-mail            : hpcfte@cdac.in     
c
c************************************************************************
c
       program main
 
       include "mpif.h"

       integer MyRank, Numprocs
       integer status(MPI_STATUS_SIZE)
       integer value, sum, Root 
       integer Source, Source_tag, Destination, Destination_tag

       call MPI_INIT(ierror)
       call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
       call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)

       Root = 0
       if(MyRank .eq. Root) then

	    sum = MyRank
	    do i = 1, Numprocs -1 
              Source = i  
              Source_tag = 0
              call MPI_RECV(value,1,MPI_INTEGER,Source,Source_tag,
     &                    MPI_COMM_WORLD,status,ierror)
              sum = sum + value

	    end do 

	    print *,  MyRank, sum

	else

            Destination = Root 
            Destination_tag = 0

            call MPI_SEND(MyRank, 1, MPI_INTEGER,Destination,
     &			Destination_tag, MPI_COMM_WORLD,ierror)

        endif

        call MPI_FINALIZE(ierror)

        stop
        end


           
