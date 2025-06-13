
c****************************************************************************
c
c		C-DAC Tech Workshop : hyPACK-2013
c                           October 15-18, 2013
c
c Example 1.1           :  mathlib-core-horner-rule.f 
c
c Objective 	        :  Write a sequential program and estimate computational time for evaluation of function expressed in terms of 				   polynomial of degree 'p' by using direct method and Horner's rule. 
c
c Input                 : Degree of a polynomial.
c
c Output                : Time taken in seconds for computation of polynomial using 
c                         normal method and by Horner's Rule
c                  
c   Created             : August-2013
c
c   E-mail              : hpcfte@cdac.in     
c
c******************************************************************************/

      program HornerRule
      double precision timetakeng,timetakenh,powsum
      double precision value,coeff(10000),functord,functhorn
      integer timeg(2),timeh(2),degree,power,count
      external dtime

c     Read the degree and coefficients of the polynomial*/

      print *,'Enter the degree of the polynomial(<10000)'
      read *,degree
      if ((degree .lt. 1) .or. (degree .gt. 10000)) then
      print *,'The degree should be in range 1<degree<10000'
      stop
      endif

c     Prepare the polynomial and print it

      do 10, power=1,degree+1
      coeff(power)=dble(power+1)
   10 continue
      
      tempdegree=degree+1
      if(degree .le. 20) then
      print *,'The polynomial is'
      print *,(coeff(power),'X',power-1,power=1,tempdegree)
      endif

c     Read the value of variable in polynomial at which the value of the 
c     polynomial is to be computed

      print *,'Enter the value at which the function value is to be foun
     $d'
      read *,value

c     Perform the computation of the polynomial in the general method of 
c     multiplying the coefficients with the variable raised to the power of 
c     it and measure the time

      timetakeng=dtime(timeg)
      functord=0.0d0
      do 30,power=1,degree+1
      powsum=1.0d0
      do 40,count=1,power-1
      powsum=powsum*value
   40 continue
      functord=functord+coeff(power)*powsum
   30 continue 
      timetakeng=dtime(timeg)

c     Find out and print the time taken in seconds for the general method

      print *,'GENERAL METHOD OF POLYNOMIAL EVALUATION'
      print *,'---------------------------------------'
      print *,'  '
      if(timetakeng .eq. 0.0d0) then
      print *,'Time taken is lower than a microsecond'
      print *,'Try for larger order of polynomials to get results'
      else
      print *,'Function value for general method      :',functord
      print *,'Time taken in sec. for general method  :',timetakeng
      end if

c     Perform the computation of the polynomial using Horner's Rule
c     and measure the time

      timetakenh=dtime(timeh)
      functhorn=coeff(degree+1)
      do 50, power=degree,1,-1
      functhorn=functhorn*value+coeff(power)
   50 continue
      timetakenh=dtime(timeh)

c     Find out and print the time taken in seconds using Horner's Rule

      print *,'  '
      print *,'  '
      print *,'HORNERS RULE FOR POLYNOMIAL EVALUATION'
      print *,'--------------------------------------'
      print *,'  '
      if(timetakenh .eq. 0.0d0) then
      print *,'Time taken is lower than a microsecond'
      print *,'Try for larger order of polynomials to get results'
      else
      print *,'Function value using Horner      :',functhorn
      print *,'Time taken in sec. using Horner :',timetakenh
      end if


      if(functord .ne. functhorn) then
          print *
          print *
          print *,'Results using general method and Horners Rule are
     +different'
          print *
      end if
      stop
      end
