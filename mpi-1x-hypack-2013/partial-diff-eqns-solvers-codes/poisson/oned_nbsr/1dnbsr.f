c
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c	                 Example 29 
c			MAIN PROGRAM
c
c*******************************************************************
c   oned.f - a solution to the Poisson problem using Jacobi 
c   interation on a 1-d decomposition
c     
c   The size of the domain is read by processor 0 and broadcast to
c   all other processors.  The Jacobi iteration is run until the 
c   change in successive elements is small or a maximum number of 
c   iterations is reached.  The difference is printed out at each 
c   step.
c*******************************************************************
c
      PROGRAM MAIN 
c
      include "mpif.h"
      integer maxn
      parameter (maxn = 500)
      parameter (iter = 20000)
      parameter (tolerance  = 0.000001)

      double precision  a(maxn,maxn), b(maxn,maxn), f(maxn,maxn)

      integer nx, ny
      integer Root, myid, numprocs, ierr
      integer comm1d, nbrbottom, nbrtop, s, e, it

      double precision diff, diffnorm, dwork
      double precision t1, t2

      Root = 0

      call MPI_INIT( ierr )
      call MPI_COMM_RANK( MPI_COMM_WORLD, myid, ierr )
      call MPI_COMM_SIZE( MPI_COMM_WORLD, numprocs, ierr )
c
      if (myid .eq. Root) then
c
c      Get the number of cells in X- and Y- direction
       write(6,100)
100    format(//3x,'Number of Cells in X- and Y- direction is same'/
     $ 5x,' Give number of cells in X-direction')
         read(5,*) nx
       endif

      call MPI_BCAST(nx,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

      if(nx .ge. maxn) then
      if(myid. eq. Root) write(6,103)
103     format(4x,'Number of cells exceeds the dimension of the array')
	go to 30
      endif
      ny = nx
c
c     Get a new communicator for a decomposition of the domain
c
      call MPI_CART_CREATE( MPI_COMM_WORLD, 1, numprocs, .false., 
     $                    .true., comm1d, ierr )
c
c     Get my position in this communicator, and my neighbors
c 
      call MPI_COMM_RANK( comm1d, myid, ierr )
      call MPI_Cart_shift( comm1d, 0,  1, nbrbottom, nbrtop, ierr   )
c
c     Compute the actual decomposition
c     
      call MPE_DECOMP1D( ny, numprocs, myid, s, e )
c
c     Initialize the right-hand-side (f) and the initial solution guess (a)
c
      call onedinit( a, b, f, nx, s, e )
c
c     Actually do the computation.  Note the use of a collective 
c     operation to check for convergence, and a do-loop to bound 
c     the number of iterations.
c
      call MPI_BARRIER( MPI_COMM_WORLD, ierr )
c
      t1 = MPI_WTIME()

      do 10 it = 1, iter 

	call exchng1( a, nx, s, e, comm1d, nbrbottom, nbrtop )
	call sweep1d( a, f, nx, s, e, b )
	call exchng1( b, nx, s, e, comm1d, nbrbottom, nbrtop )
	call sweep1d( b, f, nx, s, e, a )

	dwork = diff( a, b, nx, s, e )

	call MPI_Allreduce( dwork, diffnorm, 1, MPI_DOUBLE_PRECISION, 
     $                      MPI_SUM, comm1d, ierr )

        if (diffnorm .lt. tolerance) goto 20

        if (myid .eq. Root) print *, 2*it, ' Difference is ', diffnorm

10     continue
20    continue
      t2 = MPI_WTIME()

      if (myid .eq. Root) then
        write(6,105) it, t2 -t1
105     format(4x,'Converged after ',2x, i8,'  Iterations and the time
     $  taken is ', F20.10//
     $  15x, '*** Successful exit ***')
      endif
c
30      call MPI_FINALIZE(ierr)
      stop
      end
c
c     ------------------------------------------------------------
c
      subroutine onedinit( a, b, f, nx, s, e )
      integer nx, s, e
      double precision a(0:nx+1, s-1:e+1), b(0:nx+1, s-1:e+1),
     &                 f(0:nx+1, s-1:e+1)
c
      integer i, j
c
      do 10 j=s-1,e+1
         do 10 i=0,nx+1
            a(i,j) = 0.0d0
            b(i,j) = 0.0d0
            f(i,j) = 0.0d0
 10      continue
c      
c    Handle boundary conditions
c
      do 20 j=s,e
         a(0,j) = 1.0d0
         b(0,j) = 1.0d0
         a(nx+1,j) = 0.0d0
         b(nx+1,j) = 0.0d0
 20   continue
      if (s .eq. 1) then
         do 30 i=1,nx
            a(i,0) = 2.0d0
            b(i,0) = 2.0d0
 30      continue 
      endif
c
      return
      end
c
c     ------------------------------------------------------------
c
c     This routine is meant for producing a decomposition of a 1-d 
c     array when given a number of processors.  It may be used in 
c     "direct" product decomposition.  The values returned assume a 
c     "global" domain in [1:n]
c
      subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )

      integer n, numprocs, myid, s, e
      integer nlocal
      integer deficit
c
      nlocal  = n / numprocs
      s	      = myid * nlocal + 1
      deficit = mod(n,numprocs)
      s	      = s + min(myid,deficit)
      if (myid .lt. deficit) then
          nlocal = nlocal + 1
      endif
      e = s + nlocal - 1
      if (e .gt. n .or. myid .eq. numprocs-1) e = n

      return
      end
c
c     ------------------------------------------------------------
c
c     Perform a Jacobi sweep for a 1-d decomposition.
c     Sweep from a into b
c
      subroutine sweep1d( a, f, nx, s, e, b )
      integer nx, s, e
      double precision a(0:nx+1,s-1:e+1), f(0:nx+1,s-1:e+1),
     +                 b(0:nx+1,s-1:e+1)
c
      integer i, j
      double precision h
c
      h = 1.0d0 / dble(nx+1)
      do 10 j=s, e
         do 10 i=1, nx
            b(i,j) = 0.25 * (a(i-1,j)+a(i,j+1)+a(i,j-1)+a(i+1,j)) - 
     +               h * h * f(i,j)
 10   continue
      return
      end
c
c     ------------------------------------------------------------
c
c     ....THIS SUBROUTINE USES 1D ISEND AND IRECV MPI LIBRARY CALLS..

      subroutine exchng1( a, nx, s, e, comm1d,nbrbottom, nbrtop )

	include "mpif.h"
	integer nx, s, e
	integer comm1d, nbrbottom, nbrtop
	integer status_array(MPI_STATUS_SIZE,4), ierr, req(4)

	double precision a(0:nx+1,s-1:e+1)
c
        call MPI_IRECV ( 
     $       a(1,s-1), nx, MPI_DOUBLE_PRECISION, nbrbottom, 0, 
     $       comm1d, req(1), ierr )
        call MPI_IRECV ( 
     $       a(1,e+1), nx, MPI_DOUBLE_PRECISION, nbrtop, 1, 
     $       comm1d, req(2), ierr )
        call MPI_ISEND ( 
     $       a(1,e), nx, MPI_DOUBLE_PRECISION, nbrtop, 0, 
     $       comm1d, req(3), ierr )
        call MPI_ISEND ( 
     $       a(1,s), nx, MPI_DOUBLE_PRECISION, nbrbottom, 1, 
     $       comm1d, req(4), ierr )
c
        call MPI_WAITALL ( 4, req, status_array, ierr )
	return
	end
c
c     ------------------------------------------------------------
c
c     The function which calculates the difference 
c
      double precision function diff( a, b, nx, s, e )
      integer nx, s, e
      double precision a(0:nx+1, s-1:e+1), b(0:nx+1, s-1:e+1)
c
      double precision sum
      integer i, j
c
      sum = 0.0d0
      do 10 j=s,e
         do 10 i=1,nx
            sum = sum + (a(i,j) - b(i,j)) ** 2
 10      continue
c      
      diff = sum
      return
      end
c
c     ------------------------------------------------------------
c
