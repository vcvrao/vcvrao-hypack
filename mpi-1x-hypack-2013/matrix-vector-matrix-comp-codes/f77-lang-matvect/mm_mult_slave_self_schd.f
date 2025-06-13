c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 5.8 :  mm_mult_slave_sschd.f 
c
c  Objective   : Matrix_Matrix Multiplication
c                (Self_Scheduling - slave Program)
c
c  Input       : From Master
c                (Mpi_Master_SelfScheduling_Matrix_Matrix.c)
c
c  Output      : To Master
c                (Mpi_Master_SelfScheduling_Matrix_Matrix.c)
c
c  Description : This is the worker program which will be executed
c                by the rest of the processors.
c                The workers does the computations on the data
c                distributed by the master & indicates to the master
c                for which row it has done the computation.
c
c   Created    : August-2013
c
c   E-mail     : hpcfte@cdac.in     
c
c**********************************************************************
      
      program worker

      implicit none

      include 'mpif.h'

      integer MAX_ROWS, MAX_COLS, NoofColsA,NoofRowsB,NoofColsB, Source
      parameter (MAX_ROWS = 100, MAX_COLS = 100)

      double precision Matrix_B(MAX_ROWS,MAX_COLS)
      double precision RowA(MAX_COLS), ResultRow(MAX_ROWS)

      integer MyRank,Root,Numprocs,ierror,status(MPI_STATUS_SIZE)
      integer i, j
      integer rowtype, anstype, donetype, row

      call MPI_INIT( ierror )
      call MPI_COMM_RANK( MPI_COMM_WORLD, MyRank, ierror )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, Numprocs, ierror )

      rowtype  = 1
      anstype  = 2
      donetype = 3

      Root   = 0

	call MPI_Barrier(MPI_COMM_WORLD, ierror)
	call MPI_Bcast (NoofColsA, 1, MPI_INTEGER, Root, 
     $   	 MPI_COMM_WORLD, ierror)

	call MPI_Bcast (NoofRowsB, 1, MPI_INTEGER, Root, 
     $  	MPI_COMM_WORLD, ierror)

        if(NoofColsA .ne. NoofRowsB) then
 	  goto 500
        endif

        call MPI_Bcast (NoofColsB, 1, MPI_INTEGER, Root, 
     $  	MPI_COMM_WORLD, ierror)

C
C  	slave receive B, then compute rows of C until done message.
C
 	do 85 i = 1, NoofColsB

         call MPI_BCAST(Matrix_B(1,i),NoofColsA,MPI_DOUBLE_PRECISION, 
     $            Root,MPI_COMM_WORLD,ierror)
85      continue   
c
        Source = Root
90      call MPI_RECV(RowA,NoofColsA,MPI_DOUBLE_PRECISION,Source,
     $        MPI_ANY_TAG, MPI_COMM_WORLD, status, ierror)

c
        if (status(MPI_TAG) .eq. 0) then 
            go to 200
        else
           row = status(MPI_TAG)
            do 100 i = 1, NoofColsB
               ResultRow(i) = 0.0
               do 95 j = 1, NoofColsA
                 ResultRow(i) = ResultRow(i) + RowA(j)*Matrix_B(j,i)
 95	       continue
100	    continue               
c
            call MPI_SEND(ResultRow, NoofColsB, MPI_DOUBLE_PRECISION, 
     $                    Root, row, MPI_COMM_WORLD, ierror)
            go to 90
         endif
200	continue

 500	call MPI_FINALIZE(ierror)

        stop
        end
c	
c	*********************************************************
              
