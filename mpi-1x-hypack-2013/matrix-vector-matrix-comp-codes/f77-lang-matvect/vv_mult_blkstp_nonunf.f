c
c********************************************************************
c
c	C-DAC Tech Workshop : hyPACK-2013
c                October 15-18, 2013
c
c  Example 5.2         : vv_mult_blkstp_nonunf.f
c
c  Objective           : Vector_Vector Multiplication 
c                        (Using blockstriped Non-uniform Data Partitioning) 
c
c  Input               : Read files vdata1.inp for Vector_A and 
c                        vdata2.inp Vector_B
c
c  Output              : Process with Rank 0 prints the Vector-Vector 
c                        Multiplication results 
c
c  Necessary Condition : Number of Processes should be
c                        less than or equal to 8
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c********************************************************************
c
	 program main

	 include 'mpif.h'

         integer DATA_SIZE, PROC_SIZE
         real Epsilon

	 parameter (Epsilon = 1.0E-10, DATA_SIZE = 16)
	 parameter (PROC_SIZE = 8)

  	 integer   Numprocs, MyRank
  	 integer   VectorSize
  	 integer   VectorSize_A, VectorSize_B 
  	 integer   index, iproc
  	 integer   Root, DataSize
	 integer   Destination, Destination_tag
	 integer   Source, Source_tag
  	 integer   Distribute_Cols, Remaining_Cols 
     	 integer   Displacement(PROC_SIZE)
     	 integer   RecvCount, SendCount(PROC_SIZE)

  	 real    Mybuffer_A(DATA_SIZE), Mybuffer_B(DATA_SIZE)  
	 real	   MyFinalVector, FinalAnswer
  	 real 	   Vector_A(DATA_SIZE), Vector_B(DATA_SIZE)
	 integer   status(MPI_STATUS_SIZE)

C   ........MPI Initialisation .......

   	 call MPI_INIT(ierror) 
   	 call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank, ierror)
   	 call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs, ierror)

	 Root = 0

   	 if(MyRank .eq. Root) then 
 	  open(unit=12, file = './data/vdata1.inp')
	  read(12,*) VectorSize_A
	  read(12,*) (Vector_A(i), i=1,VectorSize_A)

 	  open(unit=12, file = './data/vdata2.inp')
	  read(12,*) VectorSize_B
	  read(12,*) (Vector_B(i), i=1,VectorSize_B)
	 endif

C     .......Allocate memory and read data for vector A .....

      call MPI_BCAST(VectorSize_A, 1, MPI_INTEGER, Root, 
     $		 MPI_COMM_WORLD, ierror)

      call MPI_BCAST(VectorSize_B, 1, MPI_INTEGER, Root, 
     $		 MPI_COMM_WORLD, ierror)

       if(VectorSize_A .ne. VectorSize_B) then
	  if(MyRank .eq. Root) then
          print *,"Two Vectors can not be multiplied - exit"
	  endif
	  goto 100
       else
        VectorSize = VectorSize_A
	endif
	 
      if(VectorSize .lt. Numprocs) then
        if(MyRank .eq. Root) then
          print*,"VectorSize should be more than No of Processors"
        endif
        goto 100
      endif

C	 ....... Initial arrangement for Scatterv operation ....

        if (MyRank .eq. Root) then
        	Distribute_Cols = VectorSize/Numprocs
		Remaining_Cols  = mod(VectorSize, Numprocs)
		do iproc = 1, Numprocs
		  SendCount(iproc) = Distribute_Cols
		enddo

		do iproc=Remaining_Cols,1,-1
		  SendCount(iproc) = SendCount(iproc) + 1
		enddo

		Displacement(1) = 1
		do iproc = 1, Numprocs
	   	  disp_value = 0
	      	  do jproc = 1, iproc-1
	           disp_value = disp_value + SendCount(jproc)
		  enddo
		  Displacement(iproc) = disp_value
		enddo

	endif

C	 .......Send RecvCount to each process ........ 

		 Destination_tag = 0
		 Source_tag = 0
	 if(MyRank .eq. Root) then
		do iproc = 1, Numprocs-1
   		Destination = iproc
	   	DataSize = 1
	   	RecvCount = SendCount(iproc+1)
	 call MPI_SEND(RecvCount,DataSize,MPI_INTEGER,Destination, 
     $	      Destination_tag, MPI_COMM_WORLD, ierror)
		enddo
      		RecvCount = SendCount(1)
	 else
	 	Source = Root
	 	DataSize = 1
	 call MPI_RECV(RecvCount, DataSize, MPI_INTEGER, Source, 
     $        Source_tag, MPI_COMM_WORLD, status, ierror)
	 endif

C   ......... Scatter Vector A and Vector B .........

      call MPI_SCATTERV(Vector_A, SendCount, Displacement, MPI_REAL, 
     $     Mybuffer_A,RecvCount,MPI_REAL,Root,MPI_COMM_WORLD,ierror)

      call MPI_SCATTERV(Vector_B, SendCount, Displacement, MPI_REAL, 
     $	   Mybuffer_B,RecvCount,MPI_REAL,Root,MPI_COMM_WORLD,ierror)

C	  ........ Calculate partial sum ........ 
	 MyFinalVector = 0.0
	 do index = 1, RecvCount
           MyFinalVector = MyFinalVector + 
     $	                   Mybuffer_A(index)*Mybuffer_B(index)
	 enddo

C	 .... Collective computation : Final answer on process 0 ..
        call MPI_REDUCE(MyFinalVector, FinalAnswer, 1, MPI_REAL, 
     $	     MPI_SUM, Root, MPI_COMM_WORLD, ierror)

	 if(MyRank .eq. Root) then
	  print*,'FinalAnswer', FinalAnswer
	 endif

 100	 call MPI_FINALIZE(ierror)
	 stop
	 end



