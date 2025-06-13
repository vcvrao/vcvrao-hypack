c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c    Example 9           : allgather.f
c
c    Objective           : To gather data from all the processes
c                          This example demonstrates the use of
c                          MPI_Init
c                          MPI_Comm_rank
c                          MPI_Comm_size
c                          MPI_Allgather
c                          MPI_Barrier
c                          MPI_Finalize
c
c    Input               : Data from files (gdata0,gdata1,gdata2,gdata3,gdata4,
c                          gdata5,gdata6,gdata7) is available on respective 
c                          processes
c
c    Output              : Gathered data is printed on each processor.
c
c    Necessary Condition : Number of Processes should be 
c                          less than or equal to 8.
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************
	  
	  program main
  
	  include "mpif.h"


C  	   .......Variables Initialisation ......
	  parameter 	(MAX_SIZE = 1000) 
	  integer 	n_size, Numprocs, MyRank
	  integer 	i
	  integer 	Input_A(MAX_SIZE), Output(MAX_SIZE)


C	   ........MPI Initialisation .......
	  call MPI_INIT(ierror) 
	  call MPI_COMM_RANK(MPI_COMM_WORLD, MyRank,ierror)
	  call MPI_COMM_SIZE(MPI_COMM_WORLD, Numprocs,ierror)

C     .......Read the Input file ......

	  if(Numprocs .gt. 8) then
		 if(MyRank .eq. Root)
     *           print*,"Number of processor should be less than 9"
		 goto 100
	  endif

	  if( MyRank .eq. 0) then
    	      open (unit= 15, FILE = "./data/gdata0")
	  endif

	  if( MyRank .eq. 1) then
    	      open (unit= 16, FILE = "./data/gdata1")
	  endif

	  if( MyRank .eq. 2) then
    	      open (unit= 17, FILE = "./data/gdata2")
	  endif

	  if( MyRank .eq. 3) then
    	      open (unit= 18, FILE = "./data/gdata3")
	  endif

	  if( MyRank .eq. 4) then
    	      open (unit= 19, FILE = "./data/gdata4")
	  endif

	  if( MyRank .eq. 5) then
    	      open (unit= 20, FILE = "./data/gdata5")
	  endif

	  if( MyRank .eq. 6) then
    	      open (unit= 21, FILE = "./data/gdata6")
	  endif

	  if( MyRank .eq. 7) then
    	      open (unit= 22, FILE = "./data/gdata7")
	  endif

	  if( MyRank .eq. 0) then
    	     read(15,*) n_size
             read(15,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 1) then
    	     read(16,*) n_size
             read(16,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 2) then
    	     read(17,*) n_size
             read(17,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 3) then
    	     read(18,*) n_size
             read(18,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 4) then
    	     read(19,*) n_size
             read(19,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 5) then
    	     read(20,*) n_size
             read(20,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 6) then
    	     read(21,*) n_size
             read(21,*)(Input_A(i), i=1,n_size)     
	  endif

	  if( MyRank .eq. 7) then
    	     read(22,*) n_size
             read(22,*)(Input_A(i), i=1,n_size)     
	  endif

C    	  do i=1,n_size,1   
C	    print*,MyRank, i, Input_A(i)
C    	  enddo 

    	  call MPI_ALLGATHER (Input_A, n_size, MPI_INTEGER, Output, 
     &		n_size, MPI_INTEGER, MPI_COMM_WORLD,ierror) 

    	  write(6,*) "Results of Gathering data on processor", MyRank,
     &		 n_size*Numprocs



    	  do i=1,n_size*Numprocs,1
	  	     write(20+MyRank, *) I, Output(i)
          enddo

 100    call MPI_FINALIZE(ierror) 
   	  stop
   	  end




