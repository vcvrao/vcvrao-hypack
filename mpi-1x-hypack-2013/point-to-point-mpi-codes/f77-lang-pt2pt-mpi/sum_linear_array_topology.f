c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 1.4:	      : sum_linear_array_topology.f
c
c  Objective          :  MPI program to calculate sum of 'n' integers using 
c                        Point-to-Point Communication library calls and linear 
c                        array topology.
c
c                        This example demonstrates the use of
c                        MPI_Init
c                        MPI_Comm_rank
c                        MPI_Comm_size
c                        MPI_Send
c                        MPI_Recv
c                        MPI_Finalize
c
c  Input              :  Automatic input data generation 
c                        The rank of each process is input on each process. 
c
c  Output             :  Process with Rank 0 should print the sum of 
c                        'n' values. 
c
c  Necessary Condition : Number of Processes should be 
c                        less than or equal to 8.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c***********************************************************************
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
       if( MyRank .eq. Root) then
           Destination = 1
           Destination_tag = 0
           call MPI_SEND(MyRank, 1, MPI_INTEGER, Destination,
     &                   Destination_tag, MPI_COMM_WORLD, ierror)
	    else 
         if( MyRank .lt. Numprocs - 1) then
            Source = MyRank - 1
            Source_tag = 0
            call MPI_RECV(value,1,MPI_INTEGER,Source,Source_tag,
     &                    MPI_COMM_WORLD,status,ierror)
            sum = MyRank + value
            Destination = MyRank +1
            Destination_tag = 0
            call MPI_SEND(sum, 1, MPI_INTEGER,Destination,
     &			Destination_tag, MPI_COMM_WORLD,ierror)
          else
            Source = MyRank - 1
            Source_tag = 0
            call MPI_RECV(value,1,MPI_INTEGER,Source,Source_tag,
     &		MPI_COMM_WORLD ,status,ierror)
            sum = MyRank + value
            write(6,*) 'Rank of the Process ', MyRank
            write(6,*) 'SUM ',sum
          endif
        endif

        call MPI_FINALIZE(ierror)
        stop
        end


           
