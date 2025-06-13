c

c	***********************************************************
	subroutine ReadSparseInput (nvtxs, nsize, vector, 
     $ 	           rowptr, colind, values)
c	************************************************************

	include 'define.h'

	integer nvtxs, nsize 
	integer rowptr(MATRIX_SIZE + 1), colind(TOTAL_SIZE) 
	double precision vector(VECTOR_SIZE) 

	integer irow, icol, index
	integer sparsity_value, nvtxs_local

	double precision Matrix_A(VECTOR_SIZE, VECTOR_SIZE)
	double precision values(TOTAL_SIZE)
	double precision randvect(VECTOR_SIZE)

C       ......Read vector from input file .......
	open(unit=13, file = '../data/vdat_sparse.inp')
	     write(6,*) 'Input vector' 
        read(13,*) nsize
        read(13,*) (vector(irow), irow=1,nsize)
	     write(6,76) (vector(irow), irow=1, nsize)
76	format(3x,8(2x,f8.3)//)
c
C       .....Read Input Matrix_A ........

        open(unit = 12, file = '../data/mdat_sparse.inp')
	     write(6,*) 'Input sparse matrix' 
	read(12,*) sparsity_value
	read(12,*) nsize
	do index = 1, nsize
	  randvect(index) = 0.0
	enddo

  	do irow = 1, nsize
	   read(12,*) (Matrix_A(irow,icol), icol=1, nsize)
	   write(6,75) (Matrix_A(irow,icol), icol=1, nsize)
75	format(3x,8(2x,f8.3))
  	end do	

	index = 1
        nvtxs_local = 0
	rowptr(1) = 0
	do irow=1, nsize
	   do icol=1, nsize
	      randvect(icol) = Matrix_A(irow, icol)
c	      print*, irow, randvect(icol)
	   enddo

	   do icol=1, nsize
		if(randvect(icol) .ge. EPSILON) then
		  colind(index) = icol
		  values(index) = randvect(icol)
		  nvtxs_local   = nvtxs_local + 1
		  index = index + 1
		endif
	   enddo

	   rowptr(irow+1) = nvtxs_local
	enddo

	return
	end

c	*********************************************************



