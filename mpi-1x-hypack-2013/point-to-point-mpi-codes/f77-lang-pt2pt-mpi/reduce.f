c
c*****************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c      Example 10      : reduce.f 
c
c
c      Objective       : To find sum of 'n' integers on 'p' processors 
c                        using MPI collective communication and communciation 
c                        library call ( MPI_REDUCE ).
c
c                        This example demonstrates the use of
c                        MPI_Init
c                        MPI_Comm_rank
c                        MPI_Comm_size
c                        MPI_Reduce
c                        MPI_Finalize
c
c    Input             : Automatic generation of input 
c                        The rank of each proceess is input on each process. 
c
c   Output             : Process with Rank 0 should print the sum of 
c                        Rank values. 
c
c  Necessary Condition : Number of Processes should be 
c                        less than or equal to 8.
c
c   Created            : August-2013
c
c   E-mail             : hpcfte@cdac.in     
c
c*****************************************************************

      program main

      include "mpif.h"
      
      integer  MyRank, Numprocs
      integer  sum, Root

C....MPI initialisation....

       call MPI_INIT( ierror )
       call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)
       call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)

       Root = 0
       sum  = 0

C....The REDUCE function of MPI....

       call MPI_REDUCE(MyRank, sum, 1, MPI_INTEGER, MPI_SUM, 
     $			Root, MPI_COMM_WORLD, ierror)

      if( MyRank .eq. Root) then
       write(6,*) "FINAL SUM IS ", sum
      endif

       call MPI_FINALIZE( ierror )

      stop
      end
