c
c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c   Example   1.6  : single-core-prg-tryparamorig.f
c
c   This is the subroutine tryparamorig without applying any optimizations 
c   to the given loop and is used by the main program in 
c   single-core-prg-loop-unroll-interchange-block.f. 
c
c
c   Input         : None 
c
c   Outpur        : Performance in terms of Execution time 
c
c
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c
c****************************************************************************
c

      subroutine tryparamorig(a,b,n,m)
      integer m,n,i,j
      double precision a(n,m), b(n,m)
 
      do 10 i = 1, n
        do 20 j = 1, n
              a(i,j) = a(i,j)* b(i,j)
   20 continue
   10 continue

      return
      end
