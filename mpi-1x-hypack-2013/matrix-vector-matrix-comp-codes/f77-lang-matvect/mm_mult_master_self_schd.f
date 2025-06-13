c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c  Example 5.8		: mm_mult_master_sschd.f
c
c  Objective           : Matrix_Matrix_Multiplication
c                        (Self_Scheduling algorithm master program)
c
c  Input               : Process 0 (master) reads files (mdata1.inp) 
c                        for Matrix_A and (mdata2.inp) for Matrix_B.
c
c  Output              : Process 0 prints the result of Matrix-Matrix 
c                        Multiplication 
c
c  Necessary Condition : Number of Processes should be less than 
c                        or equal to 8
c
c   Created           : August-2013
c
c   E-mail            : hpcfte@cdac.in     
c
c*******************************************************************

      program main

      implicit none

      include 'mpif.h'

      integer MAX_ROWS, MAX_COLS
      parameter (MAX_ROWS = 100, MAX_COLS = 100)
c

      integer Dest, RowtoSend
      integer NoofRowsA, NoofColsA
      integer NoofRowsB, NoofColsB

      double precision Matrix_A(MAX_ROWS,MAX_COLS), 
     $	               Matrix_B(MAX_ROWS,MAX_COLS), 
     $                 ResultMatrix(MAX_ROWS,MAX_COLS)
      double precision RowA(MAX_COLS),  ResultRow(MAX_COLS)

      real*8 Starttime, Endtime, Time
c
      integer MyRank, Root, Numprocs
      integer status(MPI_STATUS_SIZE)
      integer i, j, k, iproc, ierror
      integer rowtype, anstype, donetype
      integer Source_tag, Destination_tag
c
c      open(unit = 23, file = 'mat.out')

      call MPI_INIT(ierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror )
      call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror )
c
      rowtype  = 1
      anstype  = 2
      donetype = 0
      Root     = 0

c     Initialize Matrix_A and Matrix_B

      Starttime = MPI_WTIME()

C        .......Read the Matrix From Input file ......
	 open(unit=12, file = './data/mdata1.inp')
         read(12,*) NoofRowsA, NoofColsA
         do i = 1,NoofRowsA
            read(12,*) (Matrix_A(i,j),j=1,NoofColsA)
C            write(6,*) (Matrix_A(i,j),j=1,NoofColsA)
         enddo

	 open(unit=13, file = './data/mdata2.inp')
         read(13,*) NoofRowsB, NoofColsB
         do i = 1,NoofRowsB
            read(13,*) (Matrix_B(i,j),j=1,NoofColsB)
C            write(6,*) (Matrix_B(i,j),j=1,NoofColsB)
         enddo

 	 close(12)
	 close(13)


	call MPI_Barrier(MPI_COMM_WORLD, ierror)
	call MPI_Bcast(NoofColsA, 1, MPI_INTEGER, Root,
     $ 	               MPI_COMM_WORLD,ierror)

	call MPI_Bcast(NoofRowsB, 1, MPI_INTEGER, Root,
     $  	       MPI_COMM_WORLD,ierror)

	if(NoofColsA .ne. NoofRowsB) then
  	 print*,"Incompatible dimensions of Matrices for Mutlipication"
   	 goto 100
        endif

	call MPI_Bcast(NoofColsB, 1, MPI_INTEGER, Root, 
     $ 	               MPI_COMM_WORLD,ierror)

c        send Matrix_B to each other process
	 do 86 i = 1, NoofColsB
          call MPI_BCAST(Matrix_B(1,i),NoofRowsB,MPI_DOUBLE_PRECISION, 
     $          Root,MPI_COMM_WORLD,ierror)
86	continue
c
c        send a row of Matrix_A to each other process
c
	 RowtoSend = 0
         do 40 iproc = 1, Numprocs-1
            do 30 j = 1,NoofColsA
               RowA(j) = Matrix_A(iproc,j)
 30         continue
c
            Dest = iproc
               call MPI_SEND(RowA, NoofColsA, MPI_DOUBLE_PRECISION,
     $              Dest, RowtoSend+1, MPI_COMM_WORLD, ierror)
               RowtoSend = RowtoSend+1
 40      continue
c
         do 70 i = 1, NoofRowsA
          call MPI_RECV( ResultRow, NoofColsA, MPI_DOUBLE_PRECISION,
     $      MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierror)
c
             Dest = status(MPI_SOURCE)
             Source_tag = status(MPI_TAG)
c
             do k = 1, NoofColsA 
	             ResultMatrix(Source_tag,k) =  ResultRow(k)
             end do
c
               if(RowtoSend .lt. NoofRowsA) then
               do 50 j = 1,NoofColsA
                  RowA(j) = Matrix_A(RowtoSend+1,j)
 50            continue
c
	     Destination_tag = RowtoSend + 1
             call MPI_SEND(RowA,NoofColsA,MPI_DOUBLE_PRECISION,Dest,
     $                       Destination_tag, MPI_COMM_WORLD,ierror)
               
               RowtoSend     = RowtoSend+1
            else
               RowA(1) = 1.0d0
               call MPI_SEND(RowA, 1, MPI_DOUBLE_PRECISION, 
     $                     Dest, donetype, MPI_COMM_WORLD, ierror)
            endif
	Endtime = MPI_Wtime()
	Time    = EndTime - Starttime
 70      continue
         
c      print out the answer
       write(6,75)((ResultMatrix(i,j), j=1, NoofColsA), i=1, NoofRowsA)
	print*, "  Time = ", Time

 100  	call MPI_FINALIZE(ierror)
75	   format(8(2x,f8.3))
      stop
      end


