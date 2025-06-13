c
c ******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 17		: mv_mult_master_sschd.f
c
c  Objective           : Matrix_Vector Multiplication
c                        (Self_scheduling algorithm master program)
c
c  Input               : Process 0 (master) reads files (mdata.inp)
c                        for Matrix and (vdata.inp) for Vector
c
c  Output              : Process 0 prints the result of Matrix_Vector
c                        Multiplication
c
c  Necessary Condition : Number of processors should be greater than
c                        2 and less than or equal to 8
c
c   Created            : August-2013
c
c   E-mail             : hpcfte@cdac.in     
c
c********************************************************************
c
      
      program main

      include 'mpif.h'

      integer MAX_ROWS, MAX_COLS 
      parameter (MAX_ROWS = 1000, MAX_COLS = 1000)
      double precision Matrix(MAX_ROWS,MAX_COLS), Vector(MAX_COLS)
		double precision FinalVector(MAX_COLS)
      double precision Buffer(MAX_COLS), ans

      real*8 Starttime, Endtime, Time
      integer MyRank, Root, Numprocs, ierr, status(MPI_STATUS_SIZE)
      integer i, j, Destination
      integer RowtoSend, Source_tag
      integer NoofRows, NoofCols, VectorSize
      integer Destination_tag, ValidInput

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, MyRank, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, Numprocs, ierr )

      Root   = 0
      ValidInput = 1

C    /* .......Read the Matrix From Input file ......*/
      open(unit=12, file = './data/mdata.inp')
      read(12,*) NoofRows, NoofCols
      do i = 1, NoofRows
            read(12,*) (Matrix(i,j),  j=1,NoofCols)
C            write(6,*) (Matrix(i,j), j=1,NoofCols)
      enddo

C    /* Read vector from input file */
      open(unit=13, file = './data/vdata.inp')
      read(13,*) VectorSize
      read(13,*) (Vector(i), i=1,VectorSize)
      do i = 1,VectorSize
C            write(6,*) Vector(i)
      enddo

	close(12)
	close(13)

      if(Numprocs .lt. 2) then
	    print*,"Numprocs less than two .."
	    goto 100
       endif

      if((VectorSize.ne.NoofCols).or.(NoofRows.lt.(Numprocs-1)))then
            ValidInput = 0
      endif

       call MPI_Barrier(MPI_COMM_WORLD, ierr)

       call MPI_Bcast(ValidInput, 1, MPI_INTEGER, Root, 
     $          MPI_COMM_WORLD, ierr)

	 if(ValidInput .eq. 0) then
		 print*,"Invalid input data..... "
		 print*,"NoofCols should be equal to VectorSize"
		 goto 100
         endif

c        send VectorSize and Vector to each other process
         call MPI_BCAST(VectorSize, 1, MPI_INTEGER, Root,
     $        MPI_COMM_WORLD, ierr)

         call MPI_BCAST(Vector, VectorSize, MPI_DOUBLE_PRECISION,
     $        Root, MPI_COMM_WORLD, ierr)

c        send Matrix row to each other process
	      RowToSend = 0
         do 40 i = 1,Numprocs-1
            do 30 j = 1,NoofCols
               Buffer(j) = Matrix(i,j)
30         continue
            call MPI_SEND(Buffer, NoofCols, MPI_DOUBLE_PRECISION, i,
     $           RowtoSend+1, MPI_COMM_WORLD, ierr)
            RowtoSend = RowtoSend+1
40      continue
         
         do 70 i = 1,NoofRows
            call MPI_RECV(ans,1,MPI_DOUBLE_PRECISION,MPI_ANY_SOURCE,
     $           MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
            Destination = status(MPI_SOURCE)
            Source_tag = status(MPI_TAG)
            FinalVector(Source_tag) = ans

            if (RowtoSend .lt. NoofRows) then
	      Destination_tag = RowtoSend + 1
               do 50 j = 1,NoofCols
                  Buffer(j) = Matrix(RowtoSend+1,j)
50            continue
               call MPI_SEND(Buffer,NoofCols,MPI_DOUBLE_PRECISION, 
     $              Destination,Destination_tag,MPI_COMM_WORLD,ierr)
               RowtoSend     = RowtoSend+1
            else
            call MPI_SEND(1, 1, MPI_INTEGER, Destination, 0,
     $           MPI_COMM_WORLD, ierr)
            endif
	Endtime = MPI_Wtime()
	Time    = EndTime - Starttime
70      continue

	do i = 1,NoofCols,1
	    write(*, *) 'FinalAns[',i,'] = ',FinalVector(i)
	enddo
	print*,' Time Taken = ',Time

 100   	call MPI_FINALIZE(ierr)

      	stop
      	end

