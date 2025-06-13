c     
c*********************************************************************
c
c	C-DAC Tech Workshop : hyPACK-2013
c                 October 15-18, 2013
c
c   Example 17		: mv_mult_slave_sschd.f
c
c   Objective           : Matrix_Vector Multiplication
c                         (Self_scheduling algorithm slave Program)
c
c   Input               : A matrix row from master
c
c   Output              : Computed result to master
c
c
c   Necessary Condition : Number of processors should be greater than
c                         2 and less than or equal to 8
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c*********************************************************************
c
      
      program worker

      include 'mpif.h'

      integer MAX_ROWS, MAX_COLS 
      parameter (MAX_ROWS = 1000, MAX_COLS = 1000)

      double precision Vector(MAX_COLS)
      double precision MyBuffer(MAX_COLS), ans
c
      integer MyRank, Root, Numprocs, ierr, status(MPI_STATUS_SIZE)
      integer i, Destination_tag, VectorSize, ValidInput

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, MyRank, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, Numprocs, ierr )

      Root   = 0
      ValidInput = 1

      if(Numprocs .lt. 2) then
	 goto 400
      endif

      call MPI_Barrier(MPI_COMM_WORLD, ierr)

      call MPI_Bcast(ValidInput, 1, MPI_INTEGER, Root, 
     $          MPI_COMM_WORLD, ierr)

       if(ValidInput .eq. 0) then
	 goto 400
       endif


C 	Slavereceives VacorSize and Vector

 	 call MPI_BCAST(VectorSize, 1, MPI_INTEGER, Root,
     $        MPI_COMM_WORLD, ierr)

 	 call MPI_BCAST(Vector,VectorSize,MPI_DOUBLE_PRECISION,Root,
     $        MPI_COMM_WORLD, ierr)

c  	Slave receive B, then compute dot product until done message.

90      call MPI_RECV(MyBuffer,VectorSize,MPI_DOUBLE_PRECISION,Root,
     $        MPI_ANY_TAG, MPI_COMM_WORLD, status, ierr)

            flag = status(MPI_TAG)
         if (flag .eq. 0) then 
             go to 200
         else
	      Destination_tag = flag
	      ans = 0.0
              do 100 i = 1, VectorSize
                ans = ans + MyBuffer(i)*Vector(i)
100	      continue               

              call MPI_SEND(ans, 1, MPI_DOUBLE_PRECISION, Root,
     $                 Destination_tag, MPI_COMM_WORLD, ierr)
              go to 90
          endif
200	continue

        write(6,*) 'slave  successful  ', MyRank, ' is done'
400	call MPI_FINALIZE(ierr)

        stop
        end
              

c	********************************************************************

