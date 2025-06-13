c
c*******************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c   Example   1.6  : single-core-prg-tryparammod.f
c
c   This is the subroutine tryparammod after applying loop optimizations
c   to the given loop and is used by the main program in
c   single-core-prg-loop-unroll-interchange-block.f.
c
c   Input    : None 
c  
c   Output   : Performance of program in terms of execution time
c
c  
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c****************************************************************************
c
      subroutine tryparammod(a,b,n,m)
      integer m,n,i,j
      double precision a(n,m), b(n,m)
      do 10 i = 1, n, 2
         do 20 j = 1, n/2, 2
             a(j,i) = a(j,i) * B(i,j)
             a(j+1,i) = a(j+1,i) * B(i,j+1)
             a(j,i+1) = a(j,i+1) * B(i+1,j)
             a(j+1,i+1) = a(j+1,i+1) * B(i+1,j+1)
   20    continue
   10 continue


      do 30 i = 1, n, 2
         do 40 j = n/2, n, 2
              a(j,i) = a(j,i) * B(i,j)
              a(j+1,i) = a(j+1,i) * B(i,j+1)
              a(j,i+1) = a(j,i+1) * B(i+1,j)
              a(j+1,i+1) = a(j+1,i+1) * B(i+1,j+1)
   40  continue
   30  continue

        return
        end
