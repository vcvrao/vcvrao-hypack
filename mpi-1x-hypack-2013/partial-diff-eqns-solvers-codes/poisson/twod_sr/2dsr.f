c
c
c	C-DAC Tech Workshop : hyPACK-2013
c                   October 15-18, 2013
c
c		    Example 29
c 		   MAIN PROGRAM
c
c*******************************************************************
c
c Objective : To obtain  a solution to the Poisson problem using Jacobi 
c   		     iteration on a 2-d decomposition
c Input     : Number of Grid points and Maximum number of Iterations
c		        are to be provided as arguments to the program.
c Output    : The Difference in each iteration , convergence result 
c		        along with number of iterations carried out and time 
c		        taken to complete the iterations.
c Necessary Condition : Number of processes should be a perfect square
c			and less than 8. Number of grid points should be 
c			divisible by the square root of processors.
c
c   The size of the domain is read by processor 0 and broadcast to
c   all other processors.  The Jacobi iteration is run until the 
c   change in successive elements is small or a maximum number of 
c   iterations is reached.  The difference is printed out at each 
c   step.
c
c
	PROGRAM MAIN

        include "mpif.h"

        integer 	maxn
        parameter (maxn = 128)
        parameter (iter = 20000)
        parameter (tolerance = 0.000001) 

      	real*8    a(maxn,maxn), b(maxn,maxn), f(maxn,maxn)
      	integer   nx, ny
      	integer   myid, Root, numprocs
      	integer   it, comm2d, ierr, stride, num_sqroot
      	integer   nbrleft, nbrright, nbrtop, nbrbottom
      	integer   sx, ex, sy, ey
      	integer   dims(2)
      	logical   periods(2)
      	real*8    diff2d, diffnorm, dwork, t1, t2

      	data periods/2*.false./

	Root = 0
      	CALL MPI_INIT( ierr )
      	CALL MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      	CALL MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )

        if(myid .eq. Root) then
          write(6,100)
 100      format(//3x,'Number of Cells in X- and Y- direction is same'/
     $    5x,' Give number of cells in X-direction')
	  read(5,*) nx
      	endif

      	CALL MPI_BCAST(nx,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

	if(nx .ge. maxn) then
         if(myid .eq. Root) write(6,103)
103    format(4x,'Number of cells exceeds the dimension of the array')
	  go to 30
	 endif
c
         num_sqroot = sqrt(dble(numprocs))
         if( mod(nx,num_sqroot) .ne. 0) then 
	 if( myid .eq. 0) then 
           write(6,104) 
104	   format(4x,'Number of points (X-direction)should be divisible',2x/
     $     5x,'by  square root  number of processors 2D decomposition')
	 endif
	 GO TO 30
	 endif
      	ny   = nx
c
c       Get a new communicator for a decomposition of the domain.  
c       Let MPI find a "good" decomposition
c
      	dims(1) = 0
      	dims(2) = 0
      	CALL MPI_DIMS_CREATE( numprocs, 2, dims, ierr )
      	CALL MPI_CART_CREATE( MPI_COMM_WORLD, 2, dims, periods, .true.,
     *                    comm2d, ierr )
c
c       Get my position in this communicator
c 
      	CALL MPI_COMM_RANK( comm2d, myid, ierr )
c
c       My neighbors are now +/- 1 with my rank.  Handle the case of the 
c       boundaries by using MPI_PROCNULL.
C
      	CALL fnd2dnbrs( comm2d, nbrleft, nbrright, nbrtop, nbrbottom )
c
c       Compute the decomposition
c     
      	CALL fnd2ddecomp( comm2d, nx, sx, ex, sy, ey )
c
c       Create a new, "strided" datatype for the exchange in 
c	the "non-contiguous"  direction
c
      	CALL MPI_Type_vector( ey-sy+1, 1, ex-sx+3, 
     $                      MPI_DOUBLE_PRECISION, stride, ierr )
      	CALL MPI_Type_commit( stride, ierr )
c
c
c       Initialize the right-hand-side (f) and the initial solution 
c       guess (a)
c
      CALL twodinit( a, b, f, nx, sx, ex, sy, ey )
c
c     Actually do the computation.  Note the use of a collective 
c     operation to check for convergence, and a do-loop to bound the 
c     number of iterations.
c
      CALL MPI_BARRIER( MPI_COMM_WORLD, ierr )
      t1 = MPI_WTIME()

      do 10 it = 1, iter 

	CALL exchng2( b, sx, ex, sy, ey, comm2d, stride, 
     $                nbrleft, nbrright, nbrtop, nbrbottom )
	CALL sweep2d( b, f, nx, sx, ex, sy, ey, a )
	CALL exchng2( a, sx, ex, sy, ey, comm2d, stride, 
     $                nbrleft, nbrright, nbrtop, nbrbottom )
	CALL sweep2d( a, f, nx, sx, ex, sy, ey, b )

	dwork = diff2d( a, b, nx, sx, ex, sy, ey )

	CALL MPI_Allreduce( dwork, diffnorm, 1, MPI_DOUBLE_PRECISION, 
     $                     MPI_SUM, comm2d, ierr )

        if (diffnorm .lt. tolerance) goto 20

        if (myid .eq. 0) print *, ' Difference is ', diffnorm

10     continue

       t2 = MPI_WTIME()
       if (myid .eq. Root) then
	write(6,105) it, t2-t1
105       format(5x,'Failed to converge after',2x, i8,
     $    'Iterations in',2x, E17.8, 'seconds ')
       endif
       go to 31
20    continue

       t2 = MPI_WTIME()

        if(myid .eq. Root) then
        write(6,106) it, t2 -t1
106      format(4x,'Converged after ',2x, i8,'  Iterations and the time
     $   taken is ', F20.10//15x, '*** Successful exit ***')
        endif

31     continue
C      Cleanup goes here.

      CALL MPI_Type_free( stride, ierr )
      CALL MPI_Comm_free( comm2d, ierr )

30      CALL MPI_FINALIZE(ierr)

      STOP
      END

c
c*********************************************************************
C 
C		      T=0
C 		*****************
C		*		*
C		*		*
C         T=1	*		* T=0
C		*		*
C		*		*
C		*****************	
C		      T=1
C
c*********************************************************************

      	SUBROUTINE twodinit( a, b, f, nx, sx, ex, sy, ey )
      	integer nx, sx, ex, sy, ey

      	real*8  a(sx-1:ex+1, sy-1:ey+1), 
     +	      	b(sx-1:ex+1, sy-1:ey+1),
     +          f(sx-1:ex+1, sy-1:ey+1)
c
      integer i, j
c
      	do 10 j=sy-1,ey+1
        do 10 i=sx-1,ex+1
            a(i,j) = 0.0d0
            b(i,j) = 0.0d0
            f(i,j) = 0.0d0
10     continue
c      
cHandle boundary conditions
C
C Left boundary Conditions
      	if (sx .eq. 1) then 
            do 20 j=sy,ey
               a(0,j) = 1.0d0
               b(0,j) = 1.0d0
 20         continue
        endif

C Right boundary Conditions
        if (ex .eq. nx) then
            do 21 j=sy,ey
               a(nx+1,j) = 0.0d0
               b(nx+1,j) = 0.0d0
 21         continue
        endif 

C Bottom Boundary conditions;
      	if (sy .eq. 1) then
            do 30 i=sx,ex
               a(i,0) = 1.0d0
               b(i,0) = 1.0d0
 30         continue 
      	endif
c
      	RETURN
      	END

c*********************************************************************
C
C This routine show how to determine the neighbors in a 2-d decomposition of
C the domain. This assumes that MPI_Cart_create has already been called 
C
c*********************************************************************

      subroutine fnd2dnbrs(comm2d,nbrleft,nbrright,nbrtop,nbrbottom )
      integer comm2d, nbrleft, nbrright, nbrtop, nbrbottom
c
      integer ierr

c Get the processor id of left and right processor.
C MPI returns MPI_PROC_NULL for left and right edges.
C
      CALL MPI_Cart_shift( comm2d, 0,  1, nbrleft,   nbrright, ierr )

c Get the processor id of bottom and top processor.
C MPI returns MPI_PROC_NULL for bottom and top edges.
C
      CALL MPI_Cart_shift( comm2d, 1,  1, nbrbottom, nbrtop,   ierr )
c
      RETURN
      END

c*********************************************************************
C 
C		      
C 			******************
C			*    *     *     *
C			*    *     *     *
C 			******************
C			*    *     *     *
C			*    *     *     *
C 			******************
C			*    *     *     *
C			*    *     *     *
C 			******************
C  		Example: Decomposition into 3X3 Process grid
C
c*********************************************************************
C 
      subroutine fnd2ddecomp( comm2d, n, sx, ex, sy, ey )
      integer 	comm2d
      integer 	n, sx, ex, sy, ey
      integer 	dims(2), coords(2), ierr
      logical 	periods(2)

c Get (i,j) position of a processor from Cartesian topology.
      CALL MPI_Cart_get( comm2d, 2, dims, periods, coords, ierr )

C Decomposition in first (ie. X) direction
      CALL MPE_DECOMP1D( n, dims(1), coords(1), sx, ex )

C Decomposition in second (ie. Y) direction
      CALL MPE_DECOMP1D( n, dims(2), coords(2), sy, ey )
c
      return
      end
      

c*********************************************************************
C
C  This file contains a routine for producing a decomposition of a 1-d array
C  into number of processors.  It may be used in "direct" product
C  decomposition.  The values returned assume a "global" domain in [1:n]
C
c*********************************************************************

      SUBROUTINE MPE_DECOMP1D( n, numprocs, myid, s, e )
      integer n, numprocs, myid, s, e, nlocal, deficit

      nlocal  = n / numprocs
      s	      = myid * nlocal + 1
      deficit = mod(n,numprocs)
      s	      = s + min(myid,deficit)

C Give one more slice to processors

      if (myid .lt. deficit) then
          nlocal = nlocal + 1
      endif

      e = s + nlocal - 1

      if (e .gt. n .or. myid .eq. numprocs-1) e = n

      return
      end

C

c*********************************************************************
c Perform a Jacobi sweep for a 2-d decomposition of a 2-d domain
c*********************************************************************


      	SUBROUTINE sweep2d( a, f, n, sx, ex, sy, ey, b )
      	integer 	n, sx, ex, sy, ey
      	real*8 	a(sx-1:ex+1, sy-1:ey+1), 
     +	  	f(sx-1:ex+1, sy-1:ey+1),
     +          b(sx-1:ex+1, sy-1:ey+1)

c Local Variables 
      	integer 	i, j
      	real*8	h
c
      	h = 1.0d0 / dble(n+1)
      	do 10 j=sy, ey
        do 10 i=sx, ex
            b(i,j) = 0.25 * (a(i-1,j)+a(i,j+1)+a(i,j-1)+a(i+1,j)) - 
     +               h * h * f(i,j)
 10   	continue

      	return
      	end
c*********************************************************************
c	exchng2.f
c	----------
c	Exchange data with neighbors for a 2-d decomposition of a 2-d 
c	domain, using MPI_SENDRECV and a "strided" data-type.
c*********************************************************************

       SUBROUTINE exchng2( a, sx, ex, sy, ey, 
     $                     comm2d, stridetype, 
     $                     nbrleft, nbrright, nbrtop, nbrbottom  )

       include "mpif.h"

       integer sx, ex, sy, ey, stridetype

       real*8  a(sx-1:ex+1, sy-1:ey+1)

       integer nbrleft, nbrright, nbrtop, nbrbottom, comm2d
       integer status(MPI_STATUS_SIZE), ierr, nx
c
       nx = ex - sx + 1
C
C  These are just like the 1-d versions, except for less data
C

	call MPI_SENDRECV( a(sx,ey),  nx, MPI_DOUBLE_PRECISION, 
     $ 	                  nbrtop, 0,
     $                    a(sx,sy-1), nx, MPI_DOUBLE_PRECISION, 
     $                    nbrbottom, 0, comm2d, status, ierr )
	call MPI_SENDRECV( a(sx,sy),  nx, MPI_DOUBLE_PRECISION,
     $                    nbrbottom, 1,
     $                    a(sx,ey+1), nx, MPI_DOUBLE_PRECISION, 
     $                    nbrtop, 1, comm2d, status, ierr )
C
C This uses the "strided" datatype
C
	call MPI_SENDRECV( a(ex,sy),  1, stridetype, nbrright, 0,
     $                     a(sx-1,sy), 1, stridetype, nbrleft, 0, 
     $                     comm2d, status, ierr )
	call MPI_SENDRECV( a(sx,sy),  1, stridetype, nbrleft,   1,
     $                     a(ex+1,sy), 1, stridetype, nbrright, 1, 
     $                     comm2d, status, ierr )

	return
	end

c*********************************************************************
c       diff2d.f
c	----------
c	Computes the difference between two successive approximate 
c	solutions, assuming a 2-d decomposition of a 2-d domain
c*********************************************************************


	real*8 function diff2d( a, b, nx, sx, ex, sy, ey )

      	integer  nx, sx, ex, sy, ey
      	real*8	 a(sx-1:ex+1, sy-1:ey+1), b(sx-1:ex+1, sy-1:ey+1)
c
      	double precision sum
      	integer i, j
c
      	sum = 0.0d0
      	do 10 j=sy,ey
        do 10 i=sx,ex
            sum = sum + (a(i,j) - b(i,j)) ** 2
10     continue
c      
      	diff2d = sum
      	return
      	end

C****************************************************************************
