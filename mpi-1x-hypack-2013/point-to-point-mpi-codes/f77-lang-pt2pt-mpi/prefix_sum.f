c
c**********************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c     Program     : Mpi_Prefix_Sum.f
c
c     Objective   : Calculate Prefix Sum on a Hyper-Cube using 
c                   global reduction on a d-dimensional hypercube.  
c
c     Description : The MPI TAGS have been used in this assignment 
c		    to get the desired output.  
c
c     Input       : None
c
c     Output      : Prefix Sum 
c
c
c   Created       : August-2013
c
c   E-mail        : hpcfte@cdac.in     
c
c**********************************************************************
c
      include 'mpif.h'

      integer ierror,Numprocs,my_partner,dimension
      integer req(1),rtag,stag,my_id,resultlen
      integer status(MPI_STATUS_SIZE)
      real result,msg,number,res(2),output(40)

      call MPI_INIT(ierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD,my_id,ierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,Numprocs,ierror)

C     ......... Set the value for each processor     .........
C     ......... and initialize the number variable   .........     

      do i = 0,Numprocs-1
        if(my_id .eq. i) then
           msg     =  real(my_id*1.)
           number  =  0.0
           result  =   msg      
        endif
      enddo

C     ......... Hypercube dimension dimension .........
c
      dimension = 3

      do 100 i  =  0, dimension-1

C     ............Find the my_partner 
      my_partner  =  IEOR(my_id, 2**i)          

C     ..........Tags for the sent (stag) and   
C     		received messages (rtag)     ......... 

      stag = my_id
      rtag = my_partner

C    ......... Non-blocking SEND by my_id process 
      call MPI_ISEND(msg,1,MPI_REAL,my_partner,stag,
     *     MPI_COMM_WORLD,req(1),ierror)

C     ...........Blocking RECV by my_partner process 
      call MPI_RECV(number,1,MPI_REAL,my_partner,rtag,MPI_COMM_WORLD,
     *     status,ierror)

C     .........Add number to result only           
C     ........if my_id > my_partner                   

      if(my_id .gt. my_partner) result = result + number

C     .........Add number to msg anyway             
      msg = msg + number     

 100  continue          

C     .........Gather the results to processor 0 
      res(1) = my_id
      res(2) = result


      call MPI_GATHER(res(1),2,MPI_REAL,output,2,MPI_REAL,0,
     *           MPI_COMM_WORLD,ierror)

C     ......... Print out the results by processor 0 
      if(my_id .eq. 0) then
       itotal = 1
        do i = 1, 2*Numprocs, 2
           write(*,*) 'Node:',INT(output(i)),'   ','Prefix sum:',
     *     output(i+1)
            itotal1 = itotal1 + 1
        enddo
      endif

      call MPI_FINALIZE(ierror)

      end
      


