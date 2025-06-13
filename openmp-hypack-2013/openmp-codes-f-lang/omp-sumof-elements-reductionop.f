C **********************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
C  Example 1.5 : Omp_Sumof_Elements_ReductionOp.f
C
C   Objective : Sum of 'n' elements of one-dimensional real array.
C               This example demonstrates the use of
C               PARALLEL DO directive and REDUCTION operation.
C
C   Input     : Size of an array
C
C   Output    : Sum of 'n' elements
c
c   Created   : August-2013
c
c   E-mail    : hpcfte@cdac.in     
c
C **********************************************************************

       program SumOfElements
       integer SIZE
       parameter (SIZE=10000)
       real Sum, CheckSum
       real Array(SIZE)

       integer OMP_GET_THREAD_NUM, ThreadID
       integer ArraySize, ValidOutput

       print*,"Input the Size of Array"
       read(*,*) ArraySize

       if((ArraySize .le. 0) .or. (ArraySize .gt. SIZE)) then
          print*,"Input is Wrong."
          print*,"Please try with another number less than",SIZE+1
          print*,"Or greater than 0"
          goto 100
       endif

       Sum = 0.0
       do j = 1, ArraySize
         Array(j) = float(j)
       enddo

C$OMP PARALLEL DO REDUCTION(+:Sum)
       do j = 1, ArraySize
          Sum = Sum+Array(j)
       enddo

       print *, 'Sum:  ', Sum

       CheckSum = 0.0
       do j = 1, ArraySize
          CheckSum = CheckSum + Array(j)
       enddo

       if(CheckSum .ne. Sum) then
         ValidOutput = 0
       else
         ValidOutput = 1
       endif

       if(ValidOutput .eq. 1) then
         print*,"------ Correct Result -----"
       endif

100    end
