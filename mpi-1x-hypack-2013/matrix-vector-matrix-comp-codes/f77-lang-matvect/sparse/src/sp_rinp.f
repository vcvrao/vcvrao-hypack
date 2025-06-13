c


c	***********************************************************
	subroutine ReadSparseInput (nvtxs, nsize, vector, 
     $ 	           rowptr, colind, values)
c	************************************************************

	include 'define.h'

	integer nvtxs, nsize 
	integer rowptr(MATRIX_SIZE + 1), colind(TOTAL_SIZE) 
	double precision vector(VECTOR_SIZE) 

	integer MaxColNonZeros
	integer irow, icol, index, ColIndex
	integer TotalSize, nonzero
	integer nvtxs_local, nlocal_size
	integer Seed

	double precision values(TOTAL_SIZE)
	double precision randvect(VECTOR_SIZE), RandNo

	MaxColNonZeros    = nsize - nsize * SPARSITY+1
	print*,'MaxColNonZero', MaxColNonZeros
	TotalSize         = nsize* (MaxColNonZeros) 
	nlocal_size       = nsize

	print*, 'Rows', nsize,' SPARSITY', SPARSITY
	print*, 'Matrix size',nsize
	print*, 'TotalNonZero entries: ',TotalSize

	do index = 1, nsize
	  randvect(index) = 0.0
	enddo

	Seed = 760013
 	do nonzero  = 1, MaxColNonZeros
    	   RandNo = ran(Seed)
	   ColIndex = RandNo * nsize + 1
	   randvect(ColIndex) = RandNo
	enddo

	do index = 1, nsize
	   vector(index) = ran(Seed)
	enddo

	index = 1
        nvtxs_local = 0
	rowptr(1) = 0

	do irow=1, nsize

	   call RandomPermute(randvect, nsize, irow)   

	   do icol=1, nsize
		if(randvect(icol) .ne. 0) then
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

	subroutine RandomPermute (perm, nvtxs_local, row_no) 

c	**********************************************************

	include 'define.h'
c	parameter (VECTOR_SIZE = 8, MATRIX_SIZE = 8) 
c	parameter (TOTAL_SIZE = MATRIX_SIZE*MATRIX_SIZE) 

	integer nvtxs_local, row_no 
	integer index, ExcgIndex, center

	double precision perm(VECTOR_SIZE) 

	if (mod(nvtxs_local,2) .eq. 0) then
	   center =  nvtxs_local/2 
	else
	   center = nvtxs_local/2 + 1
	endif

       	do index = 1, center
	   ExcgIndex = mod((index + row_no), (nvtxs_local/2))+center
	   call swap (perm(index), perm(ExcgIndex))

	enddo

	return

	end

c	**********************************************************
c
	subroutine swap (a, b) 
c
c	************************************************************

	double precision  a, b
    
    	double precision tmp 

    	tmp = a 
    	  a = b 
    	  b = tmp 

	return
	end


c	*********************************************************	


